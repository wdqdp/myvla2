"""JAX auxiliary VLA heads trained on top of the pi05 prefix encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from openpi.models import model as openpi_model
from openpi.models.pi0 import make_attn_mask


@dataclass(frozen=True)
class AuxiliaryHeadConfig:
    hidden_dim: int = 512
    dropout: float = 0.1
    num_failure_reasons: int = 4
    num_recovery_plans: int = 4
    layer_norm_eps: float = 1e-6


def _linear_init(rng: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    limit = jnp.sqrt(6.0 / float(in_dim + out_dim))
    kernel = jax.random.uniform(rng, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float32)
    return {"kernel": kernel, "bias": jnp.zeros((out_dim,), dtype=jnp.float32)}


def init_head_params(rng: jax.Array, *, input_dim: int, config: AuxiliaryHeadConfig) -> dict[str, Any]:
    shared_rng, need_rng, failure_rng, plan_rng = jax.random.split(rng, 4)
    return {
        "pool_norm": {
            "scale": jnp.ones((input_dim,), dtype=jnp.float32),
            "bias": jnp.zeros((input_dim,), dtype=jnp.float32),
        },
        "shared": _linear_init(shared_rng, input_dim, config.hidden_dim),
        "need_recovery": _linear_init(need_rng, config.hidden_dim, 2),
        "failure_reason": _linear_init(failure_rng, config.hidden_dim, config.num_failure_reasons),
        "recovery_plan": _linear_init(plan_rng, config.hidden_dim, config.num_recovery_plans),
    }


def _layer_norm(x: jax.Array, params: dict[str, jax.Array], eps: float) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(variance + eps)
    return x * params["scale"] + params["bias"]


def _linear(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    return x @ params["kernel"] + params["bias"]


def _dropout(x: jax.Array, *, rng: jax.Array | None, rate: float, train: bool) -> jax.Array:
    if not train or rate <= 0.0:
        return x
    if rng is None:
        raise ValueError("dropout rng is required when train=True and dropout > 0")
    keep_prob = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep_prob, x.shape)
    return jnp.where(mask, x / keep_prob, 0.0)


def encode_prefix(backbone: openpi_model.BaseModel, observation: openpi_model.Observation) -> jax.Array:
    """Run pi05 prefix-only forward and return mask-mean pooled features."""

    backbone.eval()
    observation = openpi_model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = backbone.embed_prefix(observation)
    attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), _ = backbone.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
    prefix_out = prefix_out.astype(jnp.float32)
    mask = prefix_mask[..., None].astype(prefix_out.dtype)
    return jnp.sum(prefix_out * mask, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1.0)


def forward_heads(
    head_params: dict[str, Any],
    features: jax.Array,
    *,
    config: AuxiliaryHeadConfig,
    rng: jax.Array | None = None,
    train: bool = False,
) -> dict[str, jax.Array]:
    x = _layer_norm(features, head_params["pool_norm"], config.layer_norm_eps)
    x = jax.nn.gelu(_linear(x, head_params["shared"]))
    x = _dropout(x, rng=rng, rate=config.dropout, train=train)
    return {
        "need_recovery": _linear(x, head_params["need_recovery"]),
        "failure_reason": _linear(x, head_params["failure_reason"]),
        "recovery_plan": _linear(x, head_params["recovery_plan"]),
    }


def forward(
    backbone: openpi_model.BaseModel,
    head_params: dict[str, Any],
    observation: openpi_model.Observation,
    *,
    config: AuxiliaryHeadConfig,
    rng: jax.Array | None = None,
    train: bool = False,
) -> dict[str, jax.Array]:
    features = jax.lax.stop_gradient(encode_prefix(backbone, observation))
    return forward_heads(head_params, features, config=config, rng=rng, train=train)
