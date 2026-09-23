"""Detect the four V8.2 small-grasp action boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from tactile_vla.vla.v7_2_boundary_filter import _motion_runs


V8_2_SMALL_GRASP_DETECTOR_CONFIG: dict[str, Any] = {
    "signal": "lerobot_action_with_piper_fk_z",
    "gripper_max_search_after_reexecution_frames": 5,
    "downward_step_threshold_m": 0.0001,
    "downward_merge_gap_frames": 5,
    "minimum_downward_active_frames": 5,
    "minimum_downward_drop_m": 0.005,
    "downward_end_before_reexecution_frames": 60,
    "downward_end_after_reexecution_tolerance_frames": 5,
    "gripper_step_threshold": 0.00005,
    "gripper_merge_gap_frames": 3,
    "minimum_gripper_active_frames": 5,
    "minimum_gripper_close_change": 0.005,
    "close_search_before_reexecution_frames": 5,
    "close_search_after_reexecution_frames": 60,
    "excluded_interval_semantics": "strictly_between_event_frames",
}


def detect_small_grasp_boundaries(
    actions: np.ndarray,
    end_effector_z: np.ndarray,
    *,
    rexecution_frame: int,
    config: Mapping[str, Any] = V8_2_SMALL_GRASP_DETECTOR_CONFIG,
) -> dict[str, Any]:
    """Locate gripper maximum, full downward motion, and regrasp start."""

    actions = np.asarray(actions, dtype=np.float64)
    z = np.asarray(end_effector_z, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected [T,7] actions, got {actions.shape}")
    if z.shape != (len(actions),) or not np.isfinite(actions).all() or not np.isfinite(z).all():
        raise ValueError("Invalid V8.2 action/FK signals")
    if not 0 < rexecution_frame < len(actions):
        raise ValueError(f"Invalid rexecution_frame={rexecution_frame} for T={len(actions)}")

    gripper = actions[:, 6]
    max_search_end = min(
        len(actions),
        rexecution_frame + int(config["gripper_max_search_after_reexecution_frames"]) + 1,
    )
    # np.argmax deliberately returns the first frame that reaches the maximum.
    gripper_motion_stop = int(np.argmax(gripper[:max_search_end]))

    delta_z = np.diff(z, prepend=z[:1])
    downward_runs = [
        run
        for run in _motion_runs(
            z,
            delta_z < -float(config["downward_step_threshold_m"]),
            merge_gap_frames=int(config["downward_merge_gap_frames"]),
            minimum_active_frames=int(config["minimum_downward_active_frames"]),
        )
        if run.start > gripper_motion_stop
        and run.end
        <= rexecution_frame
        + int(config["downward_end_after_reexecution_tolerance_frames"])
        and run.signed_change <= -float(config["minimum_downward_drop_m"])
    ]
    if not downward_runs:
        raise ValueError("No sustained downward end-effector motion after gripper maximum")
    arm_adjustment_start = min(run.start for run in downward_runs)
    arm_adjustment_stop = max(run.end for run in downward_runs)
    if arm_adjustment_stop < rexecution_frame - int(
        config["downward_end_before_reexecution_frames"]
    ):
        raise ValueError("Downward motion ends outside the rexecution search window")

    gripper_delta = np.diff(gripper, prepend=gripper[:1])
    closing_runs = [
        run
        for run in _motion_runs(
            gripper,
            gripper_delta < -float(config["gripper_step_threshold"]),
            merge_gap_frames=int(config["gripper_merge_gap_frames"]),
            minimum_active_frames=int(config["minimum_gripper_active_frames"]),
        )
        if run.signed_change <= -float(config["minimum_gripper_close_change"])
        and run.start > arm_adjustment_stop
        and rexecution_frame - int(config["close_search_before_reexecution_frames"])
        <= run.start
        <= rexecution_frame + int(config["close_search_after_reexecution_frames"])
    ]
    if not closing_runs:
        raise ValueError("No sustained gripper closing motion after downward motion")
    closing = min(closing_runs, key=lambda run: (run.start, run.end))
    gripper_close_start = closing.start

    events = {
        "gripper_motion_stop": gripper_motion_stop,
        "arm_adjustment_start": arm_adjustment_start,
        "arm_adjustment_stop": arm_adjustment_stop,
        "gripper_close_start": gripper_close_start,
    }
    if not (
        gripper_motion_stop
        < arm_adjustment_start
        <= arm_adjustment_stop
        < gripper_close_start
    ):
        raise ValueError(f"Invalid V8.2 small-grasp boundary order: {events}")

    return {
        "events": events,
        "signals": {
            "gripper_max": float(gripper[gripper_motion_stop]),
            "downward_drop_m": float(
                z[arm_adjustment_stop] - z[max(0, arm_adjustment_start - 1)]
            ),
            "downward_run_count": len(downward_runs),
        },
        "detected_runs": {
            "downward_end_effector": [asdict(run) for run in downward_runs],
            "gripper_closing": asdict(closing),
        },
    }


__all__ = ["V8_2_SMALL_GRASP_DETECTOR_CONFIG", "detect_small_grasp_boundaries"]
