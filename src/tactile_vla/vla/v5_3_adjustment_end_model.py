"""Frozen Pi0.5 backbone plus the V5.3 adjustment-end head."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from tactile_vla.vla import stage_b_v3_jax


class AdjustmentEndHead(nnx.Module):
    """Architecture-identical copy of the V3 NeedRecoveryHead."""

    def __init__(self, input_dim: int, *, rngs: nnx.Rngs) -> None:
        self.norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self.hidden = nnx.Linear(input_dim, 512, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.output = nnx.Linear(512, 2, rngs=rngs)

    def __call__(self, features: jax.Array, *, train: bool) -> jax.Array:
        hidden = nnx.gelu(self.hidden(self.norm(features)))
        hidden = self.dropout(hidden, deterministic=not train)
        return self.output(hidden).astype(jnp.float32)


class AdjustmentEndModel(nnx.Module):
    def __init__(self, backbone, *, paligemma_width: int, rngs: nnx.Rngs) -> None:
        self.backbone = backbone
        self.adjustment_end_head = AdjustmentEndHead(paligemma_width, rngs=rngs)

    def adjustment_end_logits(
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
        return self.adjustment_end_head(features, train=train)
