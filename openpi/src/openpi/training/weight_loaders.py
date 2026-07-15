import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Merge explicitly allowed new parameters (LoRA by default) from the reference initialization.
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref_value = flat_ref[k]
            # NNX parameter trees can contain explicit None leaves.  For example,
            # GRUCell.dense_h.bias is None when the recurrent projection does not
            # use a bias.  Orbax preserves that leaf in the checkpoint, so it is
            # part of the model structure rather than a missing parameter.
            if v is None or ref_value is None:
                if v is not None or ref_value is not None:
                    raise ValueError(
                        f"Checkpoint parameter structure mismatch at {k!r}: "
                        f"checkpoint value is {type(v).__name__}, "
                        f"reference value is {type(ref_value).__name__}."
                    )
                result[k] = None
                continue
            # Orbax serializes JAX typed PRNG keys as their raw uint32 key data
            # when restore_type=np.ndarray.  Re-wrap that data before comparing
            # against the typed key<fry> leaf in the NNX reference state.
            if jax.dtypes.issubdtype(ref_value.dtype, jax.dtypes.prng_key):
                restored_key = (
                    v
                    if jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key)
                    else jax.random.wrap_key_data(v)
                )
                if restored_key.shape != ref_value.shape or restored_key.dtype != ref_value.dtype:
                    raise ValueError(
                        f"Checkpoint PRNG key mismatch at {k!r}: "
                        f"checkpoint shape/dtype is {restored_key.shape}/{restored_key.dtype}, "
                        f"reference shape/dtype is {ref_value.shape}/{ref_value.dtype}."
                    )
                result[k] = restored_key
                continue
            result[k] = v.astype(ref_value.dtype) if v.dtype != ref_value.dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
