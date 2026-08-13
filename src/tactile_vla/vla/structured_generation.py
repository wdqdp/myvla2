"""Host-side constrained greedy decoding for the fixed V3 text grammars."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tactile_vla.vla import stage_b_v3_jax
from tactile_vla.vla.structured_text import ConstrainedTokenGrammar


def constrained_greedy_generate(
    backbone,
    observation,
    grammar: ConstrainedTokenGrammar,
    *,
    prefill_fn: Callable[..., Any] | None = None,
    step_fn: Callable[..., Any] | None = None,
) -> str:
    """Generate one legal output for a batch-size-one observation."""

    batch_size = int(observation.state.shape[0])
    if batch_size != 1:
        raise ValueError(f"Constrained greedy generation requires batch size 1, got {batch_size}")
    compact_ids = jnp.asarray(grammar.compact_token_ids, dtype=jnp.int32)
    prefill_fn = prefill_fn or stage_b_v3_jax.generation_prefill
    step_fn = step_fn or stage_b_v3_jax.generation_step
    logits, kv_cache, prefix_mask, semantic_position = prefill_fn(
        backbone,
        observation,
        compact_ids,
    )
    return constrained_greedy_generate_from_prefill(
        backbone,
        grammar,
        logits=logits,
        kv_cache=kv_cache,
        prefix_mask=prefix_mask,
        semantic_position=semantic_position,
        step_fn=step_fn,
    )


def constrained_greedy_generate_from_prefill(
    backbone,
    grammar: ConstrainedTokenGrammar,
    *,
    logits: jax.Array,
    kv_cache: Any,
    prefix_mask: jax.Array,
    semantic_position: jax.Array,
    step_fn: Callable[..., Any] | None = None,
) -> str:
    """Continue constrained decoding from an already-computed prefix cache."""

    if int(logits.shape[0]) != 1:
        raise ValueError(
            f"Constrained greedy generation requires batch size 1, got {logits.shape[0]}"
        )
    compact_ids = jnp.asarray(grammar.compact_token_ids, dtype=jnp.int32)
    step_fn = step_fn or stage_b_v3_jax.generation_step
    compact_index = {
        token: index for index, token in enumerate(grammar.compact_token_ids)
    }
    generated: list[int] = []
    for _ in range(grammar.max_target_tokens):
        allowed_tokens = grammar.allowed_next(generated)
        if not allowed_tokens:
            raise RuntimeError(f"Grammar has no continuation for generated prefix: {generated}")
        allowed_indices = np.asarray(
            [compact_index[token] for token in allowed_tokens],
            dtype=np.int32,
        )
        host_logits = np.asarray(jax.device_get(logits[0]), dtype=np.float32)
        chosen_local = int(np.argmax(host_logits[allowed_indices]))
        token = int(allowed_tokens[chosen_local])
        generated.append(token)
        if grammar.is_complete(generated):
            return grammar.text_for_sequence(generated)
        logits, kv_cache = step_fn(
            backbone,
            jnp.asarray([token], dtype=jnp.int32),
            compact_ids,
            kv_cache,
            prefix_mask,
            semantic_position + len(generated) - 1,
        )
    raise RuntimeError(
        "Constrained generation reached grammar.max_target_tokens without a complete output"
    )


def constrained_greedy_generate_full_forward(
    observation,
    grammar: ConstrainedTokenGrammar,
    logits_fn: Callable[[Any, jax.Array], jax.Array],
) -> str:
    """Fixed-shape greedy decoding that is safe to JIT once for deployment.

    This intentionally recomputes the VLM prefix for each answer token. V3
    diagnosis/planning only run after motion has stopped, so the predictable
    fixed-shape implementation is preferred for the first version; KV-cache
    decoding can replace it later without changing the checkpoint or grammar.
    """

    if int(observation.state.shape[0]) != 1:
        raise ValueError("Full-forward constrained generation requires batch size 1")
    if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
        raise ValueError("Generation observation has no tokenized prompt")
    if observation.token_ar_mask is None or observation.token_loss_mask is None:
        raise ValueError("Generation observation has no AR/loss masks")

    compact_ids = jnp.asarray(grammar.compact_token_ids, dtype=jnp.int32)
    compact_index = {
        token: index for index, token in enumerate(grammar.compact_token_ids)
    }
    prefix_length = int(
        np.asarray(jax.device_get(observation.tokenized_prompt_mask[0])).sum()
    )
    max_length = int(observation.tokenized_prompt.shape[1])
    generated: list[int] = []
    for _ in range(grammar.max_target_tokens):
        if prefix_length + len(generated) >= max_length:
            raise RuntimeError(
                f"Generation exceeded token buffer: prefix={prefix_length}, max={max_length}"
            )
        tokens = observation.tokenized_prompt.at[
            0,
            prefix_length : prefix_length + len(generated),
        ].set(jnp.asarray(generated, dtype=jnp.int32))
        prompt_mask = observation.tokenized_prompt_mask.at[
            0,
            prefix_length : prefix_length + len(generated),
        ].set(True)
        ar_mask = observation.token_ar_mask.at[
            0,
            prefix_length : prefix_length + len(generated),
        ].set(1)
        current = observation.replace(
            tokenized_prompt=tokens,
            tokenized_prompt_mask=prompt_mask,
            token_ar_mask=ar_mask,
        )
        logits = logits_fn(current, compact_ids)
        prediction_index = prefix_length + len(generated) - 1
        host_logits = np.asarray(
            jax.device_get(logits[0, prediction_index]),
            dtype=np.float32,
        )
        allowed_tokens = grammar.allowed_next(generated)
        if not allowed_tokens:
            raise RuntimeError(f"Grammar has no continuation for generated prefix: {generated}")
        allowed_indices = np.asarray(
            [compact_index[token] for token in allowed_tokens],
            dtype=np.int32,
        )
        token = int(allowed_tokens[int(np.argmax(host_logits[allowed_indices]))])
        generated.append(token)
        if grammar.is_complete(generated):
            return grammar.text_for_sequence(generated)
    raise RuntimeError("Full-forward generation did not reach a complete grammar output")
