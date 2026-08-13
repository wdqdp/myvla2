from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi" / "src"))

from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
from tactile_vla.vla import stage_b_v3_jax  # noqa: E402
from tactile_vla.vla.stage_b_v3_model import StageBV3Model  # noqa: E402
from tactile_vla.vla.structured_generation import (  # noqa: E402
    constrained_greedy_generate,
    constrained_greedy_generate_full_forward,
)
from tactile_vla.vla.structured_text import ConstrainedTokenGrammar  # noqa: E402


def _model_and_observation():
    config = Pi0Config(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=2,
        max_token_len=8,
        pi05=True,
        pytorch_compile_mode=None,
    )
    model = StageBV3Model(
        config.create(jax.random.key(0)),
        paligemma_width=64,
        need_hidden_dim=16,
        need_dropout=0.0,
        rngs=nnx.Rngs(jax.random.key(1)),
    )
    observation = config.fake_obs().replace(
        tokenized_prompt_mask=jnp.asarray(
            [[True, True, True, True, False, False, False, False]]
        ),
        token_ar_mask=jnp.zeros((1, 8), dtype=jnp.int32),
        token_loss_mask=jnp.zeros((1, 8), dtype=jnp.bool_),
    )
    return model, observation


def test_v3_need_and_compact_text_logits_have_expected_shapes() -> None:
    model, observation = _model_and_observation()

    assert model.need_recovery_logits(observation).shape == (1, 2)
    logits = model.structured_token_logits(
        observation,
        jnp.asarray([1, 2, 3], dtype=jnp.int32),
    )
    assert logits.shape == (1, 7, 3)


def test_v3_inference_methods_support_module_jit() -> None:
    model, observation = _model_and_observation()
    need_logits = nnx_utils.module_jit(model.need_recovery_logits)
    text_logits = nnx_utils.module_jit(model.structured_token_logits)
    assessment_prefill = nnx_utils.module_jit(model.assessment_prefill)

    assert need_logits(observation).shape == (1, 2)
    assert text_logits(
        observation,
        jnp.asarray([1, 2, 3], dtype=jnp.int32),
    ).shape == (1, 7, 3)
    need, first_token, kv_cache, prefix_mask, semantic_position = assessment_prefill(
        observation,
        jnp.asarray([1, 2, 3], dtype=jnp.int32),
    )
    assert need.shape == (1, 2)
    assert first_token.shape == (1, 3)
    assert kv_cache is not None
    assert prefix_mask.shape[0] == 1
    assert semantic_position.shape == (1,)
    generation_step = nnx_utils.module_jit(model.generation_step)
    next_logits, next_cache = generation_step(
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([1, 2, 3], dtype=jnp.int32),
        kv_cache,
        prefix_mask,
        semantic_position,
    )
    assert next_logits.shape == (1, 3)
    assert next_cache is not None


def test_v3_constrained_loss_masks_invalid_vocabulary() -> None:
    logits = jnp.asarray([[[10.0, 100.0, -2.0]]])
    targets = jnp.asarray([[0]], dtype=jnp.int32)
    allowed = jnp.asarray([[[True, False, False]]])

    loss = stage_b_v3_jax.constrained_token_cross_entropy(logits, targets, allowed)
    assert np.isclose(float(loss), 0.0)


def test_full_forward_generation_always_returns_a_legal_sequence() -> None:
    model, observation = _model_and_observation()
    grammar = ConstrainedTokenGrammar(
        ("a", "b"),
        encode=lambda text: [10 if text == "a" else 11, 1],
    )

    generated = constrained_greedy_generate_full_forward(
        observation,
        grammar,
        lambda obs, compact_ids: model.structured_token_logits(obs, compact_ids),
    )
    assert generated in grammar.texts


def test_kv_cache_generation_always_returns_a_legal_sequence() -> None:
    model, observation = _model_and_observation()
    grammar = ConstrainedTokenGrammar(
        ("a", "b"),
        encode=lambda text: [10 if text == "a" else 11, 1],
    )

    prefill = nnx_utils.module_jit(model.generation_prefill)
    step = nnx_utils.module_jit(model.generation_step)
    generated = constrained_greedy_generate(
        model.backbone,
        observation,
        grammar,
        prefill_fn=lambda _backbone, obs, compact_ids: prefill(obs, compact_ids),
        step_fn=lambda _backbone, *values: step(*values),
    )
    assert generated in grammar.texts
