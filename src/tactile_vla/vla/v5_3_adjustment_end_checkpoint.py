"""Delta-checkpoint helpers for V5.3 head-only and shared-LoRA classifiers."""

from __future__ import annotations

from flax import nnx
from flax import traverse_util
import hashlib
import jax
import numpy as np
from openpi.shared import nnx_utils


CHECKPOINT_FORMAT = "v5_3_adjustment_end_head_v1"
MULTITASK_CHECKPOINT_FORMAT = "v5_3_adjustment_end_paligemma_lora_v1"


def trainable_filter() -> nnx.filterlib.Filter:
    return nnx.All(nnx.Param, nnx_utils.PathRegex(".*adjustment_end_head.*"))


def multitask_trainable_filter() -> nnx.filterlib.Filter:
    """Train the shared PaliGemma LoRA and the adjustment-end head only."""

    paligemma_lora = nnx.All(
        nnx_utils.PathRegex(".*backbone.*llm.*lora.*"),
        nnx.Not(nnx_utils.PathRegex(".*_1.*")),
    )
    adjustment_end_head = nnx_utils.PathRegex(".*adjustment_end_head.*")
    return nnx.All(nnx.Param, nnx.Any(paligemma_lora, adjustment_end_head))


def delta_params(state: nnx.State, filter_: nnx.filterlib.Filter) -> nnx.State:
    return state.filter(filter_)


def resume_state(state: nnx.State, filter_: nnx.filterlib.Filter) -> nnx.State:
    return state.filter(nnx.Any(filter_, nnx.RngState))


def merge_delta_params(base_state: nnx.State, delta_state: nnx.State) -> nnx.State:
    return nnx.State.merge(base_state, delta_state)


def parameter_tree_sha256(state: nnx.State) -> str:
    """Hash only restored NNX ``Param`` leaves, excluding RNG/mutable state."""

    digest = hashlib.sha256()
    # TrainState.params is an NNX State and can also contain RngKey/RngCount
    # leaves required to reconstruct the module.  They are not model
    # parameters, and typed JAX PRNG keys intentionally cannot be converted to
    # NumPy.  Filtering here also makes the hash match its intended contract:
    # verifying that the frozen backbone *parameters* did not change.
    flat = traverse_util.flatten_dict(state.filter(nnx.Param).to_pure_dict())
    for path, value in sorted(flat.items(), key=lambda item: "/".join(item[0])):
        path_text = "/".join(path)
        digest.update(path_text.encode())
        # Some valid pi0.5 modules keep disabled parameters in the graph as
        # Param(None), for example a Dense bias which is intentionally absent.
        # np.asarray(None) has object dtype and cannot be viewed as raw bytes,
        # so give this graph leaf an explicit, stable representation.
        if value is None:
            digest.update(b"<none>")
            continue
        array = np.asarray(jax.device_get(value))
        if array.dtype.hasobject:
            raise TypeError(
                f"Cannot hash non-numeric parameter leaf {path_text!r}: "
                f"object of type {type(value).__name__}"
            )
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).view(np.uint8).tobytes())
    return digest.hexdigest()
