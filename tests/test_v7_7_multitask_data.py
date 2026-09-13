from __future__ import annotations

import numpy as np

from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v7_7_multitask_data import (
    NEGATIVE_SOURCES, allocate_balanced_without_replacement, select_need_rows,
)
from tactile_vla.vla.v7_7_phase_prompt import (
    TimelinePoint, build_phase_prompt, discretize_history, sample_episode_history,
)


def _point(attempt, frame, global_index, timestamp):
    return TimelinePoint(7, attempt, frame, global_index, timestamp, np.full(7, global_index))


def test_h100_crosses_attempt_but_not_episode_and_ends_at_current():
    timeline = [_point(1, i, i, float(i)) for i in range(90)]
    timeline += [_point(2, i, 90 + i, float(90 + i)) for i in range(20)]
    sampled, effective = sample_episode_history(timeline, 105)
    assert effective == 100
    assert sampled[-1].global_index == 105
    assert {point.attempt_id for point in sampled} == {1, 2}


def test_episode_start_left_padding_and_canonical_prompt():
    timeline = [_point(1, i, i, float(i)) for i in range(5)]
    sampled, effective = sample_episode_history(timeline, 4)
    assert effective == 5
    assert sampled[0].global_index == 0 and sampled[-1].global_index == 4
    stats = StateQuantileStats(q01=np.zeros(7), q99=np.full(7, 100.0))
    discrete = discretize_history(sampled, stats)
    prompt = build_phase_prompt(
        instruction="Pick.", tactile_caption="Touch[x]", recovery_plan="none",
        qpos_h100_11_discrete=discrete,
    )
    assert prompt.splitlines() == [
        "Mode: phase", "Task: Pick.", "Touch: Touch[x]", "Recovery plan: none",
        "State history H100 sampled to 11 points: " +
        __import__("json").dumps(discrete.tolist(), separators=(",", ":")),
    ]


def test_need_negative_allocation_is_3_to_1_and_balanced():
    positives = [{"global_index": i, "episode_id": 1, "attempt_id": 1,
                  "frame_index": i, "need_recovery": True} for i in range(4)]
    negatives = {name: [
        {"global_index": 1000 * (j + 1) + i, "episode_id": j + 1, "attempt_id": 1,
         "frame_index": i, "need_recovery": False, "source": name}
        for i in range(20)
    ] for j, name in enumerate(NEGATIVE_SOURCES)}
    rows, summary = select_need_rows(positives, negatives)
    assert len(rows) == 16
    assert summary["selected_counts"] == {name: 4 for name in NEGATIVE_SOURCES}


def test_need_allocation_redistributes_capacity_without_replacement():
    allocation = allocate_balanced_without_replacement({
        NEGATIVE_SOURCES[0]: 1, NEGATIVE_SOURCES[1]: 10, NEGATIVE_SOURCES[2]: 10,
    }, 12)
    assert allocation[NEGATIVE_SOURCES[0]] == 1
    assert sum(allocation.values()) == 12
    assert abs(allocation[NEGATIVE_SOURCES[1]] - allocation[NEGATIVE_SOURCES[2]]) <= 1
