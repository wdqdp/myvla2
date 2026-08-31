from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

from flax import nnx
import jax
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/inference/agilex/inference"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/packages/openpi-client/src"))

from agilex_inference_forced_phase_anlation_5_3 import should_request_adjustment_end
from tactile_vla.vla.v5_3_adjustment_end_data import DeterministicOneToThreeBatchSampler
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256


def test_parameter_tree_hash_supports_disabled_none_parameters() -> None:
    first = nnx.State({
        "dense": {
            "bias": nnx.Param(None),
            "kernel": nnx.Param(np.asarray([[1.0, 2.0]], dtype=np.float32)),
        }
    })
    same = nnx.State({
        "dense": {
            "bias": nnx.Param(None),
            "kernel": nnx.Param(np.asarray([[1.0, 2.0]], dtype=np.float32)),
        }
    })
    changed = nnx.State({
        "dense": {
            "bias": nnx.Param(None),
            "kernel": nnx.Param(np.asarray([[1.0, 3.0]], dtype=np.float32)),
        }
    })

    assert parameter_tree_sha256(first) == parameter_tree_sha256(same)
    assert parameter_tree_sha256(first) != parameter_tree_sha256(changed)


def test_parameter_tree_hash_rejects_unexpected_object_leaf_with_path() -> None:
    state = nnx.State({"bad": nnx.Param(object())})

    with pytest.raises(TypeError, match="bad"):
        parameter_tree_sha256(state)


def test_parameter_tree_hash_excludes_typed_prng_state() -> None:
    first = nnx.State({
        "kernel": nnx.Param(np.asarray([1.0], dtype=np.float32)),
        "rng": nnx.RngKey(jax.random.key(1)),
    })
    different_rng = nnx.State({
        "kernel": nnx.Param(np.asarray([1.0], dtype=np.float32)),
        "rng": nnx.RngKey(jax.random.key(2)),
    })

    assert parameter_tree_sha256(first) == parameter_tree_sha256(different_rng)


def test_sampler_has_exact_two_to_six_batches_and_no_first_epoch_reuse() -> None:
    labels = [True] * 8 + [False] * 40
    sampler = DeterministicOneToThreeBatchSampler(
        labels=labels,
        num_batches=4,
        seed=42,
    )
    batches = list(sampler)
    assert len(batches) == 4
    positive_seen: list[int] = []
    negative_seen: list[int] = []
    for batch in batches:
        assert len(batch) == 8
        positives = [index for index in batch if labels[index]]
        negatives = [index for index in batch if not labels[index]]
        assert len(positives) == 2
        assert len(negatives) == 6
        positive_seen.extend(positives)
        negative_seen.extend(negatives)
    assert sorted(positive_seen) == list(range(8))
    assert len(set(negative_seen)) == len(negative_seen)


def test_sampler_resume_recreates_exact_suffix() -> None:
    labels = [True] * 8 + [False] * 40
    complete = list(
        DeterministicOneToThreeBatchSampler(labels=labels, num_batches=10, seed=42)
    )
    resumed = list(
        DeterministicOneToThreeBatchSampler(
            labels=labels,
            num_batches=10,
            seed=42,
            start_batch=6,
        )
    )
    assert resumed == complete[6:]


def test_adjustment_end_call_gate_is_phase_and_complete_h30_only() -> None:
    assert should_request_adjustment_end(phase="adjustment", completed_raw_actions=30)
    assert not should_request_adjustment_end(phase="execution", completed_raw_actions=30)
    assert not should_request_adjustment_end(phase="adjustment", completed_raw_actions=29)
    assert not should_request_adjustment_end(phase="adjustment", completed_raw_actions=0)


def test_sampler_is_seed_deterministic() -> None:
    labels = [True] * 8 + [False] * 40
    first = list(DeterministicOneToThreeBatchSampler(labels=labels, num_batches=6, seed=42))
    second = list(DeterministicOneToThreeBatchSampler(labels=labels, num_batches=6, seed=42))
    assert first == second
    assert not np.array_equal(np.asarray(first), np.asarray(list(
        DeterministicOneToThreeBatchSampler(labels=labels, num_batches=6, seed=43)
    )))
