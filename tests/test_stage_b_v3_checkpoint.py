from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import optax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi" / "src"))

from tactile_vla.vla.stage_b_v3_checkpoint import delta_params  # noqa: E402
from tactile_vla.vla.stage_b_v3_checkpoint import cast_frozen_params  # noqa: E402
from tactile_vla.vla.stage_b_v3_checkpoint import merge_delta_params  # noqa: E402
from tactile_vla.vla.stage_b_v3_checkpoint import resume_state  # noqa: E402
from tactile_vla.vla.stage_b_v3_checkpoint import trainable_filter  # noqa: E402
from openpi.models import model as openpi_model  # noqa: E402
from openpi.training import utils as training_utils  # noqa: E402


class _OptionalParamModel(nnx.Module):
    def __init__(self) -> None:
        self.disabled_bias = nnx.Param(None)
        self.kernel = nnx.Param(jnp.ones((2, 2), dtype=jnp.float32))


def _load_train_script():
    spec = importlib.util.spec_from_file_location(
        "train_vla_stage_b_v3_checkpoint_test_module",
        PROJECT_ROOT / "scripts" / "train_vla_stage_b_v3.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LLM(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.base = nnx.Linear(2, 2, rngs=rngs)
        self.lora = nnx.Linear(2, 2, rngs=rngs)
        self.lora_1 = nnx.Linear(2, 2, rngs=rngs)


class _Backbone(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.llm = _LLM(rngs=rngs)


class _Model(openpi_model.BaseModel):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.action_dim = 2
        self.action_horizon = 1
        self.max_token_len = 4
        self.backbone = _Backbone(rngs=rngs)
        self.need_head = nnx.Linear(2, 2, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        del rng, observation, train
        return jnp.zeros(actions.shape[:-1])

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        return jnp.zeros((1, self.action_horizon, self.action_dim))


def test_delta_filter_overlay_and_resume_payload() -> None:
    train_script = _load_train_script()
    model = _Model(rngs=nnx.Rngs(0))
    params = nnx.state(model)
    filter_ = trainable_filter()
    tx = optax.adam(1e-3)
    state = training_utils.TrainState(
        step=7,
        params=params,
        model_def=nnx.graphdef(model),
        opt_state=tx.init(params.filter(filter_)),
        tx=tx,
        ema_decay=None,
        ema_params=None,
    )
    pure_delta = delta_params(state.params, filter_).to_pure_dict()
    assert set(pure_delta) == {"backbone", "need_head"}
    assert set(pure_delta["backbone"]["llm"]) == {"lora"}
    pure_resume = resume_state(state.params, filter_).to_pure_dict()
    assert "dropout" in pure_resume

    changed = delta_params(state.params, filter_).map(
        lambda _path, variable: variable.replace(jnp.ones_like(variable.value))
    )
    merged = merge_delta_params(state.params, changed).to_pure_dict()
    assert jnp.all(merged["backbone"]["llm"]["lora"]["kernel"] == 1)
    assert jnp.all(merged["need_head"]["kernel"] == 1)
    assert jnp.array_equal(
        merged["backbone"]["llm"]["base"]["kernel"],
        state.params.to_pure_dict()["backbone"]["llm"]["base"]["kernel"],
    )
    assert jnp.array_equal(
        merged["backbone"]["llm"]["lora_1"]["kernel"],
        state.params.to_pure_dict()["backbone"]["llm"]["lora_1"]["kernel"],
    )

    rng = jax.random.key(123)
    training_payload, saved_delta = train_script._split_checkpoint_state(
        state,
        filter_,
        rng,
    )
    restored_state, restored_rng = train_script._merge_checkpoint_state(
        state,
        training_payload,
        {"params": saved_delta},
    )
    assert int(restored_state.step) == 7
    assert jnp.array_equal(jax.random.key_data(restored_rng), jax.random.key_data(rng))
    restored_delta = delta_params(restored_state.params, filter_).to_pure_dict()
    expected_delta = delta_params(state.params, filter_).to_pure_dict()
    assert jax.tree.all(jax.tree.map(jnp.array_equal, restored_delta, expected_delta))


def test_evaluate_need_respects_exact_sample_limit(monkeypatch) -> None:
    train_script = _load_train_script()

    class FakeModel:
        def eval(self) -> None:
            pass

        def need_recovery_logits(self, observation):
            return observation

    monkeypatch.setattr(train_script.nnx, "merge", lambda *_args: FakeModel())
    monkeypatch.setattr(train_script.nnx_utils, "module_jit", lambda function: function)
    monkeypatch.setattr(
        train_script,
        "batch_to_jax",
        lambda batch, _task, _sharding: batch,
    )
    loader = [
        (
            jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            {"need": jnp.asarray([0, 1, 0, 1])},
        ),
        (
            jnp.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            {"need": jnp.asarray([1, 0, 1, 0])},
        ),
    ]

    fake_state = type("FakeState", (), {"model_def": None, "params": None})()
    metrics = train_script.evaluate_need(
        fake_state,
        loader,
        None,
        max_samples=5,
    )

    assert metrics["num_samples"] == 5
    assert metrics["accuracy"] == 1.0


def test_cast_frozen_params_preserves_optional_parameter() -> None:
    state = nnx.state(_OptionalParamModel())
    cast = cast_frozen_params(state, nnx.Param)
    pure = cast.to_pure_dict()

    assert pure["disabled_bias"] is None
    assert pure["kernel"].dtype == jnp.bfloat16
