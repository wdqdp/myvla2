"""Checkpoint helpers for the trainable part of the V3 Stage B model."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp

from openpi.shared import nnx_utils


CHECKPOINT_FORMAT = "stage_b_v3_delta_v1"


def trainable_filter() -> nnx.filterlib.Filter:
    """Select the only parameters updated by Stage B V3."""

    paligemma_lora = nnx.All(
        nnx_utils.PathRegex(".*backbone.*llm.*lora.*"),
        nnx.Not(nnx_utils.PathRegex(".*_1.*")),
    )
    need_head = nnx_utils.PathRegex(".*need_head.*")
    return nnx.All(nnx.Param, nnx.Any(paligemma_lora, need_head))


def cast_frozen_params(
    state: nnx.State,
    frozen_filter: nnx.filterlib.Filter,
    dtype: jnp.dtype = jnp.bfloat16,
) -> nnx.State:
    """Cast frozen array parameters while preserving optional ``None`` leaves.

    NNX modules can represent disabled parameters as explicit ``Param(None)``
    leaves.  In particular, ``GRUCell.dense_h.bias`` is ``None`` because the
    recurrent projection does not use a bias.  The leaf is part of the graph
    structure, but it cannot be cast with ``astype``.
    """

    def cast_parameter(parameter: nnx.VariableState) -> nnx.VariableState:
        if parameter.value is None:
            return parameter
        return parameter.replace(parameter.value.astype(dtype))

    return nnx_utils.state_map(state, frozen_filter, cast_parameter)


def delta_params(state: nnx.State, train_filter: nnx.filterlib.Filter) -> nnx.State:
    """Return the lightweight Stage B parameter overlay."""

    return state.filter(train_filter)


def resume_state(state: nnx.State, train_filter: nnx.filterlib.Filter) -> nnx.State:
    """Return trainable parameters plus tiny mutable model RNG state."""

    return state.filter(nnx.Any(train_filter, nnx.RngState))


def merge_delta_params(base_state: nnx.State, delta_state: nnx.State) -> nnx.State:
    """Overlay Stage B parameters on a complete Stage A-initialized state."""

    return nnx.State.merge(base_state, delta_state)
