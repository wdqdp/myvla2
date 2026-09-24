"""Offline adjustment boundaries for the V7.4.2 new-environment data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from tactile_vla.vla.v7_2_boundary_filter import MotionRun, _motion_runs


BOUNDARY_SCHEMA = "tactile_vla_v7_4_2_adjustment_boundaries_v1"
AUDIT_SCHEMA = "tactile_vla_v7_4_2_adjustment_boundary_audit_v1"

DETECTOR_CONFIG: dict[str, Any] = {
    "signal": "lerobot_action",
    "gripper_step_threshold": 0.00005,
    "gripper_merge_gap_frames": 3,
    "minimum_gripper_active_frames": 5,
    "minimum_gripper_change": 0.005,
    "arm_step_threshold_rad": 0.0005,
    # The new collection deliberately removes the idle gaps.  Within the
    # open-to-close window, short controller pauses are one translation.
    "arm_merge_gap_frames": 20,
    "minimum_arm_active_frames": 10,
    "minimum_arm_run_range_rad": 0.02,
    "event_policy": {
        "arm_adjustment_start": (
            "first sustained arm motion at/after the first gripper-open maximum"
        ),
        "arm_adjustment_stop": (
            "last sustained arm-motion frame before the first subsequent gripper close"
        ),
    },
}


def _gripper_runs(actions: np.ndarray) -> tuple[list[MotionRun], list[MotionRun]]:
    delta = np.diff(actions[:, 6], prepend=actions[:1, 6])
    threshold = float(DETECTOR_CONFIG["gripper_step_threshold"])
    kwargs = {
        "merge_gap_frames": int(DETECTOR_CONFIG["gripper_merge_gap_frames"]),
        "minimum_active_frames": int(DETECTOR_CONFIG["minimum_gripper_active_frames"]),
    }
    opening = [
        run
        for run in _motion_runs(actions[:, 6], delta > threshold, **kwargs)
        if run.signed_change >= float(DETECTOR_CONFIG["minimum_gripper_change"])
    ]
    closing = [
        run
        for run in _motion_runs(actions[:, 6], delta < -threshold, **kwargs)
        if run.signed_change <= -float(DETECTOR_CONFIG["minimum_gripper_change"])
    ]
    return opening, closing


def detect_adjustment_boundaries(
    *,
    frame_indices: Sequence[int],
    timestamps: Sequence[float],
    actions: np.ndarray,
) -> dict[str, Any]:
    """Detect the first open-translate-close sequence in one attempt2."""

    frames = np.asarray(frame_indices, dtype=np.int64)
    times = np.asarray(timestamps, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions [T, 7], got {actions.shape}")
    if len(frames) != len(times) or len(frames) != len(actions):
        raise ValueError("frame/timestamp/action length mismatch")
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("attempt2 frame indices must be contiguous from zero")
    if not np.isfinite(actions).all() or not np.isfinite(times).all():
        raise ValueError("attempt2 contains non-finite action/timestamp values")

    opening_runs, closing_runs = _gripper_runs(actions)
    if not opening_runs:
        raise ValueError("No sustained gripper-opening run")
    opening = opening_runs[0]
    closing = next((run for run in closing_runs if run.start > opening.end), None)
    if closing is None:
        raise ValueError("No sustained gripper-closing run after the first opening")
    if closing.start - opening.end < 2:
        raise ValueError("Open-to-close window is too short")

    # Restrict arm detection to the semantic adjustment window.  This avoids
    # merging the pre-grasp approach with the adjustment when their gap is
    # intentionally very short in the new collection.
    window_start = opening.end
    window_stop = closing.start
    arm_actions = actions[window_start : window_stop + 1, :6]
    arm_delta = np.diff(arm_actions, axis=0, prepend=arm_actions[:1])
    arm_step = np.max(np.abs(arm_delta), axis=1)
    arm_runs = [
        run
        for run in _motion_runs(
            arm_actions,
            arm_step > float(DETECTOR_CONFIG["arm_step_threshold_rad"]),
            merge_gap_frames=int(DETECTOR_CONFIG["arm_merge_gap_frames"]),
            minimum_active_frames=int(DETECTOR_CONFIG["minimum_arm_active_frames"]),
        )
        if run.value_range >= float(DETECTOR_CONFIG["minimum_arm_run_range_rad"])
    ]
    if not arm_runs:
        raise ValueError("No sustained arm translation in the open-to-close window")

    # The first qualified run is the adjustment translation.  With the larger
    # merge gap it also absorbs short controller pauses and fine positioning.
    arm_run = arm_runs[0]
    start = window_start + arm_run.start
    uncapped_stop = window_start + arm_run.end
    stop = min(uncapped_stop, closing.start - 1)
    if not opening.end < start <= stop < closing.start:
        raise ValueError(
            "Invalid detected order: "
            f"open_end={opening.end}, arm_start={start}, arm_stop={stop}, "
            f"close_start={closing.start}"
        )

    return {
        "events": {
            "arm_adjustment_start": {
                "frame_index": int(start),
                "timestamp": float(times[start]),
            },
            "arm_adjustment_stop": {
                "frame_index": int(stop),
                "timestamp": float(times[stop]),
            },
        },
        "diagnostics": {
            "first_gripper_open": asdict(opening),
            "first_gripper_close": asdict(closing),
            "selected_arm_run_in_open_close_window": asdict(arm_run),
            "selected_arm_run_absolute_start": int(window_start + arm_run.start),
            "selected_arm_run_absolute_end": int(uncapped_stop),
            "pre_adjustment_idle_frames": int(start - opening.end - 1),
            "post_adjustment_idle_frames": int(closing.start - stop - 1),
            "arm_close_overlap_frames": int(max(0, uncapped_stop - closing.start + 1)),
            "stop_capped_before_gripper_close": bool(uncapped_stop >= closing.start),
            "opening_run_count": len(opening_runs),
            "closing_run_count": len(closing_runs),
            "arm_run_count_in_window": len(arm_runs),
        },
    }


__all__ = [
    "AUDIT_SCHEMA",
    "BOUNDARY_SCHEMA",
    "DETECTOR_CONFIG",
    "detect_adjustment_boundaries",
]
