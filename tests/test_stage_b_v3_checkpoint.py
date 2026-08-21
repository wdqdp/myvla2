from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import sys
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import pytest

os.environ.setdefault("USE_TF", "0")

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


def test_stage_b_v4_args_require_minimal_prompt_and_norm() -> None:
    train_script = _load_train_script()
    valid = SimpleNamespace(
        data_profile="rotation_v4",
        prompt_profile="minimal_v1",
        no_norm=False,
    )
    train_script.validate_v4_args(valid)
    with pytest.raises(ValueError, match="minimal_v1"):
        train_script.validate_v4_args(
            SimpleNamespace(data_profile="rotation_v4", prompt_profile="legacy", no_norm=False)
        )
    with pytest.raises(ValueError, match="norm stats"):
        train_script.validate_v4_args(
            SimpleNamespace(data_profile="rotation_v4", prompt_profile="minimal_v1", no_norm=True)
        )
    # V3 remains outside the V4 protocol guard.
    train_script.validate_v4_args(
        SimpleNamespace(data_profile="rotation_moderately_success_v1", prompt_profile="legacy", no_norm=True)
    )


def _v4_stage_b_protocol_args(**overrides):
    values = dict(
        data_profile="rotation_v4",
        prompt_profile="minimal_v1",
        batch_size=8,
        num_steps=4_000,
        lr=1e-4,
        eval_interval=500,
        save_interval=1_000,
        keep_period=1_000,
        action_horizon=30,
        action_dim=32,
        use_state_history=True,
        state_history_len=60,
        state_history_dim=7,
        state_history_fps=30.0,
        history_hidden_dim=256,
        max_token_len=200,
        reasoning_max_token_len=320,
        reasoning_window_frames=15,
        status_negative_ratio=3.0,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        grammar_profile="v3_full_v1",
        weight_decay=1e-4,
        grad_clip=1.0,
        seed=42,
        need_hidden_dim=512,
        need_dropout=0.1,
        action_loss_weight=1.0,
        need_loss_weight=1.0,
        failure_loss_weight=1.0,
        plan_loss_weight=1.0,
        log_interval=20,
        eval_max_need_samples=2048,
        action_loss_degradation_limit=0.10,
        fsdp_devices=1,
        video_backend="pyav",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stage_b_v4_protocol_resume_and_stage_a_step_are_strict() -> None:
    train_script = _load_train_script()
    args = _v4_stage_b_protocol_args()
    train_script.validate_v4_training_protocol(args)
    saved = vars(args) | {
        "precision": "float32",
        "trainable_components": list(train_script.V4_STAGE_B_TRAINABLE_COMPONENTS),
        "frozen_components": list(train_script.V4_STAGE_B_FROZEN_COMPONENTS),
    }
    train_script.validate_v4_resume_config(saved, args, actual_precision="float32")

    with pytest.raises(ValueError, match="protocol mismatch"):
        train_script.validate_v4_training_protocol(_v4_stage_b_protocol_args(action_dim=7))
    with pytest.raises(ValueError, match="resume config mismatch"):
        train_script.validate_v4_resume_config(
            saved | {"need_loss_weight": 2.0},
            args,
            actual_precision="float32",
        )
    assert train_script.validate_stage_a_checkpoint_step(
        "rotation_v4", Path("/run/15000")
    ) == 15_000
    assert train_script.validate_stage_a_checkpoint_step(
        "rotation_v4", Path("/run/15000/params")
    ) == 15_000
    assert train_script.validate_stage_a_checkpoint_step(
        "rotation_v4", Path("/run/10000")
    ) == 10_000
    with pytest.raises(ValueError, match="numeric checkpoint step"):
        train_script.validate_stage_a_checkpoint_step(
            "rotation_v4",
            Path("/run/latest"),
        )


def test_stage_b_v4_text_validation_is_full_and_covers_both_magnitudes() -> None:
    train_script = _load_train_script()
    assert train_script.text_eval_sample_limit("rotation_v4", 1) is None
    assert train_script.text_eval_sample_limit("rotation_moderately_success_v1", 32) == 32

    directions = ("left", "right", "front", "back")
    failure = {
        "by_direction": {direction: {"support": 1} for direction in directions},
    }
    plan = {
        "by_direction": {direction: {"support": 2} for direction in directions},
        "by_direction_magnitude": {
            f"{direction}/{magnitude}": {"support": 1}
            for direction in directions
            for magnitude in ("moderately", "slightly")
        },
    }
    train_script.validate_text_eval_coverage("rotation_v4", failure, plan)
    missing = dict(plan)
    missing["by_direction_magnitude"] = dict(plan["by_direction_magnitude"])
    missing["by_direction_magnitude"].pop("back/slightly")
    with pytest.raises(ValueError, match="direction/magnitude"):
        train_script.validate_text_eval_coverage("rotation_v4", failure, missing)


def test_stage_b_v4_index_is_never_rebuilt_by_training(tmp_path: Path) -> None:
    train_script = _load_train_script()
    common = dict(
        data_profile="rotation_v4",
        index_file=tmp_path / "missing.json",
        overwrite_index=False,
        reasoning_window_frames=15,
        action_horizon=30,
        status_negative_ratio=3.0,
        dataset_dir=tmp_path / "dataset",
    )
    with pytest.raises(FileNotFoundError, match="prepare_v4_training_index"):
        train_script.ensure_v3_index(SimpleNamespace(**common))
    common["overwrite_index"] = True
    with pytest.raises(ValueError, match="immutable"):
        train_script.ensure_v3_index(SimpleNamespace(**common))


def test_stage_b_v4_loader_routes_all_text_tasks_through_direct_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_script = _load_train_script()
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module

    direct_calls = []
    action_calls = []

    class RawDataset:
        def __init__(self, name):
            self.name = name

        def __len__(self):
            return 8

        def __getitem__(self, index):
            return index

    class SharedDataset:
        def __init__(self, repo_id, **kwargs):
            self.repo_id = repo_id

    def direct_factory(**kwargs):
        direct_calls.append(kwargs)
        return RawDataset(kwargs["task"])

    def action_factory(**kwargs):
        action_calls.append(kwargs)
        return RawDataset("action")

    def forbidden_v3(*args, **kwargs):
        raise AssertionError("rotation_v4 must not construct a V3 window dataset")

    monkeypatch.setattr(lerobot_dataset_module, "LeRobotDataset", SharedDataset)
    monkeypatch.setattr(train_script, "build_transform", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_script, "build_structured_text_transform", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_script, "build_structured_inference_transform", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_script, "TactileVLAFrameDataset", action_factory)
    monkeypatch.setattr(train_script, "V4DirectManifestDataset", direct_factory)
    monkeypatch.setattr(train_script, "V3StageBFrameDataset", forbidden_v3)
    monkeypatch.setattr(train_script, "V3RecoveryManifestDataset", forbidden_v3)
    monkeypatch.setattr(train_script, "TransformedTactileVLADataset", lambda dataset, transform: dataset)
    monkeypatch.setattr(train_script, "_loader", lambda dataset, **kwargs: dataset)
    args = SimpleNamespace(
        no_norm=True,
        norm_stats_dir=tmp_path,
        reasoning_max_token_len=320,
        data_profile="rotation_v4",
        dataset_dir=tmp_path,
        video_backend="pyav",
        state_history_fps=30.0,
        action_horizon=30,
        use_state_history=True,
        state_history_len=60,
        prompt_profile="minimal_v1",
        batch_size=8,
        num_workers=0,
    )
    split_row = {
        "execution_indices": [1, 2],
        "status_indices": [3, 4],
        "status_manifest_file": str(tmp_path / "need.jsonl"),
        "status_manifest_sha256": "need",
        "status_manifest_row_indices": [0, 1],
        "failure_reason_indices": [5],
        "failure_reason_manifest_file": str(tmp_path / "failure.jsonl"),
        "failure_reason_manifest_sha256": "failure",
        "failure_reason_manifest_row_indices": [14],
        "reasoning_indices": [5, 5],
        "reasoning_manifest_file": str(tmp_path / "plan.jsonl"),
        "reasoning_manifest_sha256": "plan",
        "reasoning_manifest_row_indices": [14, 29],
    }
    index = {"splits": {"train": dict(split_row), "val": dict(split_row)}}
    result = train_script.build_loaders(
        args,
        SimpleNamespace(max_token_len=200),
        index,
        [],
        object(),
        object(),
        object(),
    )
    assert set(result) == {"train", "val"}
    assert [call["task"] for call in direct_calls] == [
        "need", "failure", "plan", "need", "failure", "plan"
    ]
    assert all(call["dataset_repo_id"] == "tactile_vla_rotation_v4" for call in direct_calls)
    assert all(call["global_indices"] == split_row[
        {"need": "status_indices", "failure": "failure_reason_indices", "plan": "reasoning_indices"}[call["task"]]
    ] for call in direct_calls)
    assert [call["indices"] for call in action_calls] == [[1, 2], [1, 2]]
    assert all(call["dataset_repo_id"] == "tactile_vla_rotation_v4" for call in action_calls)
