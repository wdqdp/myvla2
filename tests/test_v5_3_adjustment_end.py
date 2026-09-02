from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
from types import SimpleNamespace
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
from agilex_inference_forced_phase_anlation_5_3 import validate_server_metadata
from scripts.serve_tactile_vla_v5_3 import _validate_configs
from scripts.serve_tactile_vla_v5_3 import _resolve_runtime_threshold
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v5_3_adjustment_end_data import DeterministicOneToThreeBatchSampler
from tactile_vla.vla.v5_3_adjustment_end_data import LABEL_POLICY
from tactile_vla.vla.v5_3_adjustment_end_data import is_adjustment_end_positive
from tactile_vla.vla.v5_3_adjustment_end_data import is_adjustment_end_valid
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v5_3_adjustment_end_evaluation import relative_probability_profile
from tactile_vla.vla.v5_3_adjustment_end_evaluation import select_max_recall_under_early_fpr
from tactile_vla.vla.v5_3_adjustment_end_evaluation import threshold_search


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


def test_conservative_probe_maximizes_recall_under_one_percent_early_fpr() -> None:
    rows = [
        {"label": 1, "probability": probability}
        for probability in (0.9, 0.8, 0.7, 0.6, 0.5)
    ]
    rows.extend(
        {"label": 0, "probability": probability}
        for probability in (0.69, 0.65, 0.64, *([0.1] * 97))
    )

    search = threshold_search(rows)

    assert not search["official_constraints_passed"]
    assert search["official"] is None
    assert search["conservative_probe"]["threshold"] == pytest.approx(0.7)
    assert search["conservative_probe"]["recall"] == pytest.approx(0.6)
    assert search["conservative_probe"]["early_false_positive_rate"] == 0.0
    assert (
        search["best_operating_point_with_recall_at_least_80_percent"]
        ["early_false_positive_rate"]
        == pytest.approx(0.03)
    )


def test_joint_policy_has_no_minimum_recall_gate() -> None:
    rows = [
        {"label": 1, "probability": probability}
        for probability in (0.9, 0.8, 0.7, 0.6, 0.5)
    ]
    rows.extend(
        {"label": 0, "probability": probability}
        for probability in (0.69, 0.65, 0.64, *([0.1] * 97))
    )

    result = select_max_recall_under_early_fpr(rows)

    assert result["minimum_recall"] is None
    assert result["selected"]["threshold"] == pytest.approx(0.7)
    assert result["selected"]["recall"] == pytest.approx(0.6)
    assert result["selected"]["early_false_positive_rate"] == 0.0


def test_r_minus_10_through_r_plus_5_label_and_valid_window() -> None:
    rexecution = 100
    assert not is_adjustment_end_positive(89, rexecution)
    assert is_adjustment_end_positive(90, rexecution)
    assert is_adjustment_end_positive(105, rexecution)
    assert not is_adjustment_end_positive(106, rexecution)
    assert is_adjustment_end_valid(105, rexecution)
    assert not is_adjustment_end_valid(106, rexecution)


def test_relative_probability_profile_covers_r_minus_30_through_r_plus_5() -> None:
    rows = []
    for episode_id, rexecution in ((1, 100), (2, 200)):
        for relative_frame in range(-30, 6):
            rows.append(
                {
                    "episode_id": episode_id,
                    "frame_index": rexecution + relative_frame,
                    "rexecution_frame": rexecution,
                    "label": int(relative_frame >= -10),
                    "probability": (relative_frame + 30) / 35,
                }
            )

    result = relative_probability_profile(rows)

    assert [(item["relative_frame_start_inclusive"], item["relative_frame_end_inclusive"])
            for item in result["bins"]] == [
        (-30, -26),
        (-25, -21),
        (-20, -16),
        (-15, -11),
        (-10, -6),
        (-5, -1),
        (0, 5),
    ]
    assert [item["sample_count"] for item in result["bins"]] == [10, 10, 10, 10, 10, 10, 12]
    assert result["bins"][-1]["sample_weighted_mean_probability"] == pytest.approx(
        np.mean(np.arange(30, 36) / 35)
    )


def test_relative_probability_profile_requires_explicit_rexecution_frame() -> None:
    rows = [
        {"episode_id": 1, "frame_index": frame, "label": 1, "probability": 0.5}
        for frame in range(90, 106)
    ]

    with pytest.raises(ValueError, match="missing rexecution_frame"):
        relative_probability_profile(rows)


def test_experimental_server_requires_explicit_client_acknowledgement(tmp_path: Path) -> None:
    captioner = tmp_path / "captioner.pt"
    captioner.write_bytes(b"probe")
    args = SimpleNamespace(
        phase_change_timeout_seconds=10.0,
        captioner_checkpoint=captioner,
        allow_experimental_adjustment_end=False,
    )
    metadata = {
        "supports_action_noise": True,
        "requires_action_noise": True,
        "supports_adjustment_end": True,
        "prompt_profile": "phase_v2",
        "data_profile": "rotation_phase_v5_adjustment_v2",
        "experiment_kind": "phase_prompt_h30_terminal_hold",
        "phase_change_prompt_profile": "phase_change_v1",
        "phase_change_max_token_len": 512,
        "qpos_h30_sample_offsets": [0, 3, 6, 10, 13, 16, 19, 23, 26, 29],
        "qpos_bin_count": 256,
        "qpos_discretization_extra_clip": False,
        "captioner_window_size": 30,
        "captioner_checkpoint_sha256": sha256_file(captioner),
        "action_horizon": 30,
        "action_dim": 32,
        "output_action_dim": 7,
        "state_history_len": 60,
        "state_history_dim": 7,
        "adjustment_end_threshold": 0.7,
        "phase_change_timeout_seconds": 10.0,
        "adjustment_end_experimental_override": True,
    }

    with pytest.raises(ValueError, match="allow-experimental-adjustment-end"):
        validate_server_metadata(args, metadata)
    args.allow_experimental_adjustment_end = True
    validate_server_metadata(args, metadata)


def test_server_accepts_multitask_step_8000_config(tmp_path: Path) -> None:
    stage_checkpoint = tmp_path / "stage-a" / "15000"
    stage_checkpoint.mkdir(parents=True)
    stage_config_path = stage_checkpoint.parent / "config.json"
    stage_config_path.write_text("{}")
    args = SimpleNamespace(stage_a_checkpoint=stage_checkpoint)
    stage_config = {
        "data_profile": "rotation_phase_v5_adjustment_v2",
        "prompt_profile": "phase_v2",
        "experiment_kind": "phase_prompt_h30_terminal_hold",
        "num_steps": 15000,
        "seed": 42,
    }
    adjustment_config = {
        "data_profile": "rotation_phase_v5_adjustment_end_v2",
        "prompt_profile": "phase_change_v1",
        "experiment_kind": "adjustment_end_action_multitask_v1",
        "num_steps": 8000,
        "phase_change_max_token_len": 512,
        "checkpoint_format": "v5_3_adjustment_end_paligemma_lora_v1",
        "stage_a_checkpoint": str(stage_checkpoint),
        "stage_a_config_sha256": sha256_file(stage_config_path),
        "label_policy": LABEL_POLICY,
    }
    metadata = {
        "official_step": 8000,
        "adjustment_end_threshold": 0.75,
        "label_policy": LABEL_POLICY,
    }

    threshold, checkpoint_format = _validate_configs(
        args,
        stage_config_path,
        stage_config,
        tmp_path / "adjustment-config.json",
        adjustment_config,
        metadata,
    )

    assert threshold == pytest.approx(0.75)
    assert checkpoint_format == "v5_3_adjustment_end_paligemma_lora_v1"


def test_server_manual_threshold_override_is_explicit_and_validated() -> None:
    assert _resolve_runtime_threshold(0.9696840643882751, None) == (
        pytest.approx(0.9696840643882751),
        False,
    )
    assert _resolve_runtime_threshold(0.9696840643882751, 0.7) == (
        pytest.approx(0.7),
        True,
    )
    with pytest.raises(ValueError, match=r"must be in \[0,1\]"):
        _resolve_runtime_threshold(0.9696840643882751, 1.01)
