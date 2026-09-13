"""V7.7 π0.5 wrapper with two phase classifiers and structured LM decoding."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp

from tactile_vla.vla import stage_b_v3_jax
from tactile_vla.vla.stage_b_v3_model import NeedRecoveryHead


class V77MultitaskModel(nnx.Module):
    def __init__(self, backbone, *, paligemma_width: int, need_hidden_dim: int,
                 need_dropout: float, rngs: nnx.Rngs) -> None:
        self.backbone = backbone
        self.need_head = NeedRecoveryHead(
            paligemma_width, need_hidden_dim, need_dropout, rngs=rngs
        )
        self.adjustment_head = NeedRecoveryHead(
            paligemma_width, need_hidden_dim, need_dropout, rngs=rngs
        )

    @staticmethod
    def _pooled(backbone, observation, *, rng=None, train=False):
        output, mask, _ = stage_b_v3_jax.encode_prefix(
            backbone, observation, preprocess_rng=rng, train=train
        )
        weights = mask[..., None].astype(output.dtype)
        return jnp.sum(output * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)

    def need_recovery_logits(self, observation, *, rng=None, train=False):
        return self.need_head(self._pooled(self.backbone, observation, rng=rng, train=train), train=train)

    def adjustment_end_logits(self, observation, *, rng=None, train=False):
        return self.adjustment_head(
            self._pooled(self.backbone, observation, rng=rng, train=train), train=train
        )

    def structured_token_logits(self, observation, compact_token_ids, *, rng=None, train=False):
        return stage_b_v3_jax.structured_token_logits(
            self.backbone, observation, compact_token_ids, preprocess_rng=rng, train=train
        )

    def phase_prefill(self, observation, *, active_head: str, compact_token_ids=None):
        if active_head == "need_recovery":
            if compact_token_ids is None:
                raise ValueError("need_recovery prefill requires failure compact token ids")
            features, text_logits, cache, mask, position = stage_b_v3_jax.assessment_prefill(
                self.backbone, observation, compact_token_ids
            )
            return self.need_head(features, train=False), text_logits, cache, mask, position
        if active_head == "adjustment_end":
            if compact_token_ids is not None:
                raise ValueError("adjustment_end prefill must not prepare text generation")
            output, mask, _ = stage_b_v3_jax.encode_prefix(self.backbone, observation, train=False)
            weights = mask[..., None].astype(output.dtype)
            features = jnp.sum(output * weights, axis=1) / jnp.maximum(
                jnp.sum(weights, axis=1), 1.0
            )
            return self.adjustment_head(features, train=False)
        raise ValueError(f"unsupported active_head={active_head!r}")

    def assessment_prefill(self, observation, compact_token_ids):
        return self.phase_prefill(
            observation, active_head="need_recovery", compact_token_ids=compact_token_ids
        )

    def generation_prefill(self, observation, compact_token_ids):
        return stage_b_v3_jax.generation_prefill(self.backbone, observation, compact_token_ids)

    def generation_step(self, token, compact_token_ids, kv_cache, prefix_mask, semantic_position):
        return stage_b_v3_jax.generation_step(
            self.backbone, token, compact_token_ids, kv_cache, prefix_mask, semantic_position
        )


def trainable_filter() -> nnx.filterlib.Filter:
    from openpi.shared import nnx_utils
    paligemma_lora = nnx.All(
        nnx_utils.PathRegex(".*backbone.*llm.*lora.*"),
        nnx.Not(nnx_utils.PathRegex(".*_1.*")),
    )
    heads = nnx.Any(
        nnx_utils.PathRegex(".*need_head.*"),
        nnx_utils.PathRegex(".*adjustment_head.*"),
    )
    return nnx.All(nnx.Param, nnx.Any(paligemma_lora, heads))


__all__ = ["V77MultitaskModel", "trainable_filter"]
