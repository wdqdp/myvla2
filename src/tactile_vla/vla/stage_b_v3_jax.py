"""JAX helpers for V3 need-recovery and constrained language training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as openpi_model
from openpi.models.pi0 import make_attn_mask


@dataclass(frozen=True)
class NeedHeadConfig:
    hidden_dim: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6


def _linear_init(rng: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    limit = jnp.sqrt(6.0 / float(in_dim + out_dim))
    kernel = jax.random.uniform(
        rng,
        (in_dim, out_dim),
        minval=-limit,
        maxval=limit,
        dtype=jnp.float32,
    )
    return {"kernel": kernel, "bias": jnp.zeros((out_dim,), dtype=jnp.float32)}


def init_need_head_params(
    rng: jax.Array,
    *,
    input_dim: int,
    config: NeedHeadConfig,
) -> dict[str, Any]:
    shared_rng, output_rng = jax.random.split(rng)
    return {
        "pool_norm": {
            "scale": jnp.ones((input_dim,), dtype=jnp.float32),
            "bias": jnp.zeros((input_dim,), dtype=jnp.float32),
        },
        "shared": _linear_init(shared_rng, input_dim, config.hidden_dim),
        "need_recovery": _linear_init(output_rng, config.hidden_dim, 2),
    }


def _layer_norm(x: jax.Array, params: dict[str, jax.Array], eps: float) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) * jax.lax.rsqrt(variance + eps)
    return normalized * params["scale"] + params["bias"]


def _linear(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    return x @ params["kernel"] + params["bias"]


def _dropout(
    x: jax.Array,
    *,
    rng: jax.Array | None,
    rate: float,
    train: bool,
) -> jax.Array:
    if not train or rate <= 0:
        return x
    if rng is None:
        raise ValueError("dropout rng is required while training the need head")
    keep_probability = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep_probability, x.shape)
    return jnp.where(mask, x / keep_probability, 0.0)


def need_logits_from_features(
    head_params: dict[str, Any],
    features: jax.Array,
    *,
    config: NeedHeadConfig,
    rng: jax.Array | None = None,
    train: bool = False,
) -> jax.Array:
    hidden = _layer_norm(features, head_params["pool_norm"], config.layer_norm_eps)
    hidden = jax.nn.gelu(_linear(hidden, head_params["shared"]))
    hidden = _dropout(hidden, rng=rng, rate=config.dropout, train=train)
    return _linear(hidden, head_params["need_recovery"])


def encode_prefix(
    backbone: openpi_model.BaseModel,
    observation: openpi_model.Observation,
    *,
    preprocess_rng: jax.Array | None = None,
    train: bool = False,
) -> tuple[jax.Array, jax.Array, openpi_model.Observation]:
    observation = openpi_model.preprocess_observation(
        preprocess_rng,
        observation,
        train=train,
    )
    prefix_tokens, prefix_mask, prefix_ar_mask = backbone.embed_prefix(observation)
    attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_output, _), _ = backbone.PaliGemma.llm(
        [prefix_tokens, None],
        mask=attention_mask,
        positions=positions,
    )
    return prefix_output.astype(jnp.float32), prefix_mask, observation


def need_recovery_logits(
    backbone: openpi_model.BaseModel,
    head_params: dict[str, Any],
    observation: openpi_model.Observation,
    *,
    config: NeedHeadConfig,
    rng: jax.Array | None = None,
    train: bool = False,
) -> jax.Array:
    preprocess_rng, dropout_rng = (jax.random.split(rng) if rng is not None else (None, None))
    prefix_output, prefix_mask, _ = encode_prefix(
        backbone,
        observation,
        preprocess_rng=preprocess_rng,
        train=train,
    )
    mask = prefix_mask[..., None].astype(prefix_output.dtype)
    features = jnp.sum(prefix_output * mask, axis=1) / jnp.maximum(
        jnp.sum(mask, axis=1),
        1.0,
    )
    return need_logits_from_features(
        head_params,
        features,
        config=config,
        rng=dropout_rng,
        train=train,
    )


def structured_token_logits(
    backbone: openpi_model.BaseModel,
    observation: openpi_model.Observation,
    compact_token_ids: jax.Array,
    *,
    preprocess_rng: jax.Array | None = None,
    train: bool = False,
) -> jax.Array:
    """Return compact-vocabulary next-token logits for every language position."""

    observation = openpi_model.preprocess_observation(
        preprocess_rng,
        observation,
        train=train,
    )
    if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
        raise ValueError("Structured generation requires tokenized_prompt and mask")
    if observation.token_ar_mask is None:
        raise ValueError("Structured generation requires token_ar_mask")

    prefix_tokens, prefix_mask, _ = backbone.embed_prefix(observation)
    language_length = observation.tokenized_prompt.shape[1]
    image_length = prefix_tokens.shape[1] - language_length
    image_ar_mask = jnp.zeros(
        (observation.tokenized_prompt.shape[0], image_length),
        dtype=jnp.bool_,
    )
    autoregressive_mask = jnp.concatenate(
        [image_ar_mask, observation.token_ar_mask.astype(jnp.bool_)],
        axis=1,
    )
    attention_mask = make_attn_mask(prefix_mask, autoregressive_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_output, _), _ = backbone.PaliGemma.llm(
        [prefix_tokens, None],
        mask=attention_mask,
        positions=positions,
    )
    language_output = prefix_output[:, -language_length:, :]
    prediction_hidden = language_output[:, :-1, :].astype(jnp.float32)
    return backbone.PaliGemma.llm(
        prediction_hidden,
        compact_token_ids,
        method="decode",
    ).astype(jnp.float32)


def constrained_token_cross_entropy(
    logits: jax.Array,
    target_compact_ids: jax.Array,
    allowed_token_mask: jax.Array,
) -> jax.Array:
    if logits.shape != allowed_token_mask.shape:
        raise ValueError(
            f"logits/allowed mask shape mismatch: {logits.shape} != {allowed_token_mask.shape}"
        )
    if logits.shape[:-1] != target_compact_ids.shape:
        raise ValueError(
            "logits/target shape mismatch: "
            f"{logits.shape[:-1]} != {target_compact_ids.shape}"
        )
    valid = target_compact_ids >= 0
    safe_targets = jnp.where(valid, target_compact_ids, 0)
    masked_logits = jnp.where(allowed_token_mask, logits, jnp.finfo(logits.dtype).min)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1)[..., 0]
    valid_float = valid.astype(jnp.float32)
    return jnp.sum(losses * valid_float) / jnp.maximum(jnp.sum(valid_float), 1.0)


def assessment_prefill(
    backbone: openpi_model.BaseModel,
    observation: openpi_model.Observation,
    compact_token_ids: jax.Array,
) -> tuple[jax.Array, jax.Array, Any, jax.Array, jax.Array]:
    """Encode one assessment prefix for both need logits and text decoding.

    Returns pooled prefix features, first-token compact-vocabulary logits, the
    PaliGemma KV cache, the padded prefix mask, and the semantic next-token
    position. The caller can run the binary need head on ``features`` and only
    continue autoregressive decoding when recovery is actually required.
    """

    observation = openpi_model.preprocess_observation(None, observation, train=False)
    if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
        raise ValueError("Assessment generation requires tokenized_prompt and mask")
    prefix_tokens, prefix_mask, prefix_ar_mask = backbone.embed_prefix(observation)
    attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_output, _), kv_cache = backbone.PaliGemma.llm(
        [prefix_tokens, None],
        mask=attention_mask,
        positions=positions,
    )
    last_indices = jnp.sum(prefix_mask.astype(jnp.int32), axis=1) - 1
    last_hidden = jnp.take_along_axis(
        prefix_output,
        last_indices[:, None, None],
        axis=1,
    )[:, 0, :]
    logits = backbone.PaliGemma.llm(
        last_hidden,
        compact_token_ids,
        method="decode",
    ).astype(jnp.float32)
    mask = prefix_mask[..., None].astype(prefix_output.dtype)
    features = jnp.sum(prefix_output * mask, axis=1) / jnp.maximum(
        jnp.sum(mask, axis=1),
        1.0,
    )
    semantic_position = jnp.sum(prefix_mask.astype(jnp.int32), axis=1)
    return features.astype(jnp.float32), logits, kv_cache, prefix_mask, semantic_position


def generation_prefill(
    backbone: openpi_model.BaseModel,
    observation: openpi_model.Observation,
    compact_token_ids: jax.Array,
) -> tuple[jax.Array, Any, jax.Array, jax.Array]:
    """Encode an answer-free prompt and return first-token logits plus KV cache."""

    _, logits, kv_cache, prefix_mask, semantic_position = assessment_prefill(
        backbone,
        observation,
        compact_token_ids,
    )
    return logits, kv_cache, prefix_mask, semantic_position


def generation_step(
    backbone: openpi_model.BaseModel,
    token: jax.Array,
    compact_token_ids: jax.Array,
    kv_cache: Any,
    prefix_mask: jax.Array,
    semantic_position: jax.Array,
) -> tuple[jax.Array, Any]:
    """Advance one generated token while retaining padded-prefix cache slots."""

    embedded = backbone.PaliGemma.llm(token[:, None], method="embed")
    batch_size = token.shape[0]
    generated_count = int(kv_cache[0].shape[2] - prefix_mask.shape[1])
    generated_mask = jnp.ones((batch_size, generated_count + 1), dtype=jnp.bool_)
    key_mask = jnp.concatenate([prefix_mask, generated_mask], axis=1)
    attention_mask = einops.rearrange(key_mask, "b s -> b 1 s")
    output, kv_cache = backbone.PaliGemma.llm(
        [embedded, None],
        mask=attention_mask,
        positions=semantic_position[:, None],
        kv_cache=kv_cache,
    )
    logits = backbone.PaliGemma.llm(
        output[0][:, 0, :],
        compact_token_ids,
        method="decode",
    ).astype(jnp.float32)
    return logits, kv_cache
