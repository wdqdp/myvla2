import jax
import numpy as np
import pytest

from openpi.training import weight_loaders


def test_merge_params_preserves_matching_none_leaf():
    loaded = {
        "history_gru": {
            "dense_h": {
                "bias": None,
                "kernel": np.ones((2, 3), dtype=np.float32),
            }
        }
    }
    reference = {
        "history_gru": {
            "dense_h": {
                "bias": None,
                "kernel": np.ones((2, 3), dtype=np.float64),
            }
        }
    }

    merged = weight_loaders._merge_params(loaded, reference, missing_regex=r"a^")  # noqa: SLF001

    assert merged["history_gru"]["dense_h"]["bias"] is None
    assert merged["history_gru"]["dense_h"]["kernel"].dtype == np.dtype(np.float64)


def test_merge_params_rejects_none_structure_mismatch():
    loaded = {"history_gru": {"dense_h": {"bias": None}}}
    reference = {"history_gru": {"dense_h": {"bias": np.zeros(3, dtype=np.float32)}}}

    with pytest.raises(ValueError, match="structure mismatch"):
        weight_loaders._merge_params(loaded, reference, missing_regex=r"a^")  # noqa: SLF001


def test_merge_params_wraps_raw_prng_key_data():
    key = jax.random.key(42)
    loaded = {"history_gru": {"rngs": {"default": {"key": np.asarray(jax.random.key_data(key))}}}}
    reference = {
        "history_gru": {
            "rngs": {
                "default": {
                    "key": jax.ShapeDtypeStruct(key.shape, key.dtype),
                }
            }
        }
    }

    merged = weight_loaders._merge_params(loaded, reference, missing_regex=r"a^")  # noqa: SLF001
    restored_key = merged["history_gru"]["rngs"]["default"]["key"]

    assert restored_key.shape == ()
    assert restored_key.dtype == key.dtype
    assert isinstance(jax.random.key_data(restored_key), jax.Array)
    assert restored_key.addressable_shards
    np.testing.assert_array_equal(jax.random.key_data(restored_key), jax.random.key_data(key))
