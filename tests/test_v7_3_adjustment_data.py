from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.v7_3_adjustment_data import CROSS_PHASE_REASON
from tactile_vla.vla.v7_3_adjustment_data import STOP_H30_REASON
from tactile_vla.vla.v7_3_adjustment_data import build_v7_3_exclusions


def _row(frame: int, *, phase: str = "adjustment", pure: bool = True) -> dict:
    return {
        "global_index": 1_000 + frame,
        "episode_id": 7,
        "attempt_id": 2,
        "frame_index": frame,
        "phase": phase,
        "raw_chunk_phase_pure": pure,
    }


def _boundary(arm_start: int, arm_stop: int) -> dict:
    return {
        "attempts": [
            {
                "episode_id": 7,
                "attempt_id": 2,
                "events": {
                    "arm_adjustment_start": {"frame_index": arm_start},
                    "arm_adjustment_stop": {"frame_index": arm_stop},
                },
            }
        ]
    }


def test_long_adjustment_drops_full_h30_before_stop() -> None:
    rows = [_row(frame) for frame in range(10, 51)]
    exclusions, policies = build_v7_3_exclusions(
        rows,
        v7_2_exclusions={},
        boundary_payload=_boundary(10, 50),
        action_horizon=30,
    )

    assert 1_000 + 20 not in exclusions
    assert all(exclusions[1_000 + frame] == STOP_H30_REASON for frame in range(21, 51))
    assert policies[0]["mode"] == "full_pre_stop_h30"
    assert policies[0]["filter_start"] == 21


def test_short_adjustment_preserves_prefix_with_at_most_ten_post_stop_frames() -> None:
    rows = [_row(frame) for frame in range(30, 51)]
    exclusions, policies = build_v7_3_exclusions(
        rows,
        v7_2_exclusions={},
        boundary_payload=_boundary(30, 50),
        action_horizon=30,
    )

    assert 1_030 not in exclusions
    assert 1_031 not in exclusions
    assert all(exclusions[1_000 + frame] == STOP_H30_REASON for frame in range(32, 51))
    assert policies[0]["mode"] == "short_adjustment_preserve_prefix"
    assert policies[0]["retained_motion_start_count"] == 2
    assert policies[0]["max_retained_post_stop_frames"] == 10


def test_impossibly_short_adjustment_fails_instead_of_removing_all_motion() -> None:
    rows = [_row(frame) for frame in range(32, 51)]
    with pytest.raises(ValueError, match="cannot preserve an adjustment start"):
        build_v7_3_exclusions(
            rows,
            v7_2_exclusions={},
            boundary_payload=_boundary(32, 50),
            action_horizon=30,
        )


def test_residual_adjustment_crossing_is_dropped_but_execution_is_untouched() -> None:
    adjustment = _row(5, pure=False)
    execution = _row(60, phase="execution", pure=False)
    exclusions, _ = build_v7_3_exclusions(
        [adjustment, execution],
        v7_2_exclusions={},
        boundary_payload=_boundary(10, 50),
        action_horizon=30,
    )

    assert exclusions[1_005] == CROSS_PHASE_REASON
    assert 1_060 not in exclusions
