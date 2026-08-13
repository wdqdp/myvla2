"""Unified π0.5 + V3 monitor/language wrapper."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from tactile_vla.vla import stage_b_v3_jax


class NeedRecoveryHead(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, *, rngs: nnx.Rngs) -> None:
        self.norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self.hidden = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, 2, rngs=rngs)

    def __call__(self, features: jax.Array, *, train: bool) -> jax.Array:
        hidden = nnx.gelu(self.hidden(self.norm(features)))
        hidden = self.dropout(hidden, deterministic=not train)
        return self.output(hidden).astype(jnp.float32)


class StageBV3Model(nnx.Module):
    def __init__(
        self,
        backbone,
        *,
        paligemma_width: int,
        need_hidden_dim: int,
        need_dropout: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.backbone = backbone
        self.need_head = NeedRecoveryHead(
            paligemma_width,
            need_hidden_dim,
            need_dropout,
            rngs=rngs,
        )

    def need_recovery_logits(
        self,
        observation,
        *,
        rng: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        prefix_output, prefix_mask, _ = stage_b_v3_jax.encode_prefix(
            self.backbone,
            observation,
            preprocess_rng=rng,
            train=train,
        )
        mask = prefix_mask[..., None].astype(prefix_output.dtype)
        features = jnp.sum(prefix_output * mask, axis=1) / jnp.maximum(
            jnp.sum(mask, axis=1),
            1.0,
        )
        return self.need_head(features, train=train)

    def structured_token_logits(
        self,
        observation,
        compact_token_ids: jax.Array,
        *,
        rng: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        return stage_b_v3_jax.structured_token_logits(
            self.backbone,
            observation,
            compact_token_ids,
            preprocess_rng=rng,
            train=train,
        )

    def assessment_prefill(
        self,
        observation,
        compact_token_ids: jax.Array,
    ):
        """Share one PaliGemma prefix between need classification and diagnosis."""

        features, text_logits, kv_cache, prefix_mask, semantic_position = (
            stage_b_v3_jax.assessment_prefill(
                self.backbone,
                observation,
                compact_token_ids,
            )
        )
        need_logits = self.need_head(features, train=False)
        return need_logits, text_logits, kv_cache, prefix_mask, semantic_position

    def generation_prefill(self, observation, compact_token_ids: jax.Array):
        return stage_b_v3_jax.generation_prefill(
            self.backbone,
            observation,
            compact_token_ids,
        )

    def generation_step(
        self,
        token: jax.Array,
        compact_token_ids: jax.Array,
        kv_cache,
        prefix_mask: jax.Array,
        semantic_position: jax.Array,
    ):
        return stage_b_v3_jax.generation_step(
            self.backbone,
            token,
            compact_token_ids,
            kv_cache,
            prefix_mask,
            semantic_position,
        )
