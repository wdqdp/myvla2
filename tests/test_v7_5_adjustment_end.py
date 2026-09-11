from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.v7_5_adjustment_end_data import adjustment_end_label
from tactile_vla.vla.v7_5_adjustment_end_data import assign_idle_keep_counts
from tactile_vla.vla.v7_5_adjustment_end_data import DeterministicOneToThreeBatchSampler
from tactile_vla.vla.v7_5_adjustment_end_data import idle_overlap_indices
from tactile_vla.vla.v7_5_adjustment_end_data import sample_history_frame_indices
from tactile_vla.vla.v7_5_adjustment_end_data import valid_classification_frame
from scripts.train_vla_adjustment_end_multitask_v7_5 import _pad_eval_batch


def test_v7_5_label_and_sample_ranges_are_arm_stop_based() -> None:
    assert valid_classification_frame(130, 130, 200)
    assert valid_classification_frame(210, 130, 200)
    assert not valid_classification_frame(129, 130, 200)
    assert not valid_classification_frame(211, 130, 200)
    assert not adjustment_end_label(189, 200)
    assert adjustment_end_label(190, 200)
    assert adjustment_end_label(210, 200)


def test_idle_overlap_is_strictly_between_events_and_inside_h100() -> None:
    assert idle_overlap_indices(
        frame_index=130, gripper_motion_stop=100, arm_adjustment_start=130
    ) == list(range(101, 130))
    assert idle_overlap_indices(
        frame_index=220, gripper_motion_stop=100, arm_adjustment_start=130
    ) == list(range(121, 130))


def test_idle_keep_assignment_is_exact_811_for_complete_groups() -> None:
    candidates = [(index, 30) for index in range(10)]
    assignments, summary = assign_idle_keep_counts(candidates, seed=42)
    assert summary["actual_counts"] == {"0": 8, "1": 1, "2": 1}
    assert all(30 > keep * 10 for index, keep in assignments.items())


def test_history_sampler_keeps_current_and_only_k_idle_frames() -> None:
    sampled, retained, overlap = sample_history_frame_indices(
        frame_index=160,
        gripper_motion_stop=100,
        arm_adjustment_start=130,
        idle_keep_count=2,
        seed=7,
    )
    assert overlap == 29
    assert len(sampled) == 11
    assert len(set(sampled)) == 11
    assert sampled[-1] == 160
    assert len(retained) == 2
    assert [value for value in sampled if 101 <= value <= 129] == retained


def test_classifier_sampler_cycles_a_small_negative_pool() -> None:
    labels = [True, True, False, False, False, False, False, False]
    batches = list(
        DeterministicOneToThreeBatchSampler(
            labels=labels, num_batches=3, seed=42
        )
    )
    assert len(batches) == 3
    assert all(sum(labels[index] for index in batch) == 2 for batch in batches)


def test_eval_tail_is_padded_to_device_multiple_without_changing_valid_count() -> None:
    raw = {
        "adjustment_end_label": np.asarray([0, 1, 1], dtype=np.int32),
        "image": np.arange(6, dtype=np.float32).reshape(3, 2),
    }
    padded, valid_count = _pad_eval_batch(raw, multiple=4)
    assert valid_count == 3
    assert padded["adjustment_end_label"].tolist() == [0, 1, 1, 1]
    np.testing.assert_array_equal(padded["image"][-1], raw["image"][-1])
