from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.v7_5_adjustment_end_data import assign_idle_keep_counts
from tactile_vla.vla.v7_6_adjustment_end_data import (
    DeterministicNaturalBatchSampler,
    counterfactual_label,
    counterfactual_slightly_stop,
    select_train_slight_counterfactual_pairs,
)
from tactile_vla.vla.v7_6_adjustment_end_evaluation import counterfactual_pair_metrics


def test_counterfactual_midpoint_has_no_overlapping_label() -> None:
    assert counterfactual_slightly_stop(101, 160) == 130
    assert not counterfactual_label(
        frame_index=129,
        source_magnitude="moderately",
        arm_adjustment_start=101,
        arm_adjustment_stop=160,
    )
    assert counterfactual_label(
        frame_index=130,
        source_magnitude="moderately",
        arm_adjustment_start=101,
        arm_adjustment_stop=160,
    )
    assert not counterfactual_label(
        frame_index=160,
        source_magnitude="slightly",
        arm_adjustment_start=101,
        arm_adjustment_stop=160,
    )


def test_idle_keep_assignment_is_exact_1721() -> None:
    assignments, summary = assign_idle_keep_counts(
        [(index, 30) for index in range(20)], seed=42, weights=(17, 2, 1)
    )
    assert summary["actual_counts"] == {"0": 17, "1": 2, "2": 1}
    assert all(30 > assignments[index] * 10 for index in assignments)


def test_slight_counterfactual_reduction_keeps_hard_negatives_then_allocates() -> None:
    rows = []
    pair_id = 0
    for episode in range(2):
        for frame in range(100, 141):
            rows.append({
                "split": "train",
                "source_magnitude": "slightly",
                "episode_id": episode,
                "frame_index": frame,
                "arm_adjustment_stop_frame": 140,
                "pair_id": pair_id,
            })
            pair_id += 1
    selected, summary = select_train_slight_counterfactual_pairs(rows, keep_count=27, seed=42)
    hard = {
        int(row["pair_id"]) for row in rows
        if int(row["frame_index"]) >= int(row["arm_adjustment_stop_frame"]) - 10
    }
    assert hard <= selected
    assert len(selected) == 27
    assert summary["hard_negative_retained_count"] == 22
    assert summary["additional_retained_count"] == 5


def test_natural_sampler_carries_tail_and_resume_is_exact() -> None:
    labels = [True] * 5 + [False] * 10
    complete = list(
        DeterministicNaturalBatchSampler(labels=labels, num_batches=5, seed=42)
    )
    resumed = list(
        DeterministicNaturalBatchSampler(labels=labels, num_batches=5, seed=42, start_batch=2)
    )
    assert resumed == complete[2:]
    assert all(len(batch) == 8 for batch in complete)
    assert sorted(complete[0] + complete[1][:7]) == list(range(15))


def test_pair_metric_checks_the_prompt_conditioned_probability_order() -> None:
    rows = [
        {"pair_id": 1, "sample_variant_id": 0, "label": 0, "probability": 0.2},
        {"pair_id": 1, "sample_variant_id": 1, "label": 1, "probability": 0.8},
        {"pair_id": 2, "sample_variant_id": 0, "label": 1, "probability": 0.9},
        {"pair_id": 2, "sample_variant_id": 2, "label": 0, "probability": 0.1},
    ]
    metrics = counterfactual_pair_metrics(rows)
    assert metrics["paired_sample_count"] == 2
    assert metrics["label_disagreement_pair_count"] == 2
    assert metrics["label_disagreement_order_accuracy"] == 1.0
    assert metrics["label_disagreement_mean_signed_margin"] == pytest.approx(0.7)
