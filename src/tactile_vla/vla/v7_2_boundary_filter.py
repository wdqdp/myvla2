"""Detect V7.2 idle gaps between consecutive manipulation motions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tactile_vla.vla.v4_data import V4Frame


V7_2_FILTER_SCHEMA = "tactile_vla_v7_2_boundary_filter_v1"
V7_2_AUDIT_SCHEMA = "tactile_vla_v7_2_boundary_audit_v1"

DEFAULT_DETECTOR_CONFIG: dict[str, Any] = {
    "signal": "lerobot_action",
    "arm_step_threshold_rad": 0.0005,
    "gripper_step_threshold": 0.00005,
    "arm_merge_gap_frames": 5,
    "gripper_merge_gap_frames": 3,
    "minimum_arm_active_frames": 10,
    "minimum_gripper_active_frames": 5,
    "minimum_arm_run_range_rad": 0.02,
    "minimum_gripper_run_change": 0.005,
    "move_arm_search_before_frames": 30,
    "move_arm_search_after_frames": 90,
    "move_gripper_search_before_frames": 120,
    "rexecution_arm_end_tolerance_frames": 5,
    "rexecution_close_search_before_frames": 5,
    "rexecution_close_search_after_frames": 150,
    "excluded_interval_semantics": "strictly_between_event_frames",
}


@dataclass(frozen=True)
class MotionRun:
    start: int
    end: int
    active_frames: int
    signed_change: float
    value_range: float


def _motion_runs(
    values: np.ndarray,
    active: np.ndarray,
    *,
    merge_gap_frames: int,
    minimum_active_frames: int,
) -> list[MotionRun]:
    """Group threshold crossings while tolerating short zero-command gaps."""

    hits = np.flatnonzero(active)
    if len(hits) == 0:
        return []
    groups: list[list[int]] = [[int(hits[0])]]
    for raw_frame in hits[1:]:
        frame = int(raw_frame)
        if frame - groups[-1][-1] <= merge_gap_frames + 1:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    result: list[MotionRun] = []
    for group in groups:
        if len(group) < minimum_active_frames:
            continue
        start, end = group[0], group[-1]
        segment = values[max(0, start - 1) : end + 1]
        if values.ndim == 1:
            signed_change = float(values[end] - values[max(0, start - 1)])
            value_range = float(np.max(segment) - np.min(segment))
        else:
            signed_change = 0.0
            value_range = float(np.max(np.ptp(segment, axis=0)))
        result.append(
            MotionRun(
                start=start,
                end=end,
                active_frames=len(group),
                signed_change=signed_change,
                value_range=value_range,
            )
        )
    return result


def detect_attempt_boundaries(
    actions: np.ndarray,
    *,
    move_start_frame: int,
    rexecution_frame: int,
    config: Mapping[str, Any] = DEFAULT_DETECTOR_CONFIG,
) -> dict[str, Any]:
    """Return the four ordered event frames for one attempt2 trajectory."""

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected [T, 7] actions, got {actions.shape}")
    delta = np.diff(actions, axis=0, prepend=actions[:1])
    arm_step = np.max(np.abs(delta[:, :6]), axis=1)
    gripper_step = delta[:, 6]
    arm_runs = [
        run
        for run in _motion_runs(
            actions[:, :6],
            arm_step > float(config["arm_step_threshold_rad"]),
            merge_gap_frames=int(config["arm_merge_gap_frames"]),
            minimum_active_frames=int(config["minimum_arm_active_frames"]),
        )
        if run.value_range >= float(config["minimum_arm_run_range_rad"])
    ]
    gripper_motion_runs = [
        run
        for run in _motion_runs(
            actions[:, 6],
            np.abs(gripper_step) > float(config["gripper_step_threshold"]),
            merge_gap_frames=int(config["gripper_merge_gap_frames"]),
            minimum_active_frames=int(config["minimum_gripper_active_frames"]),
        )
        if run.value_range >= float(config["minimum_gripper_run_change"])
    ]
    closing_runs = [
        run
        for run in _motion_runs(
            actions[:, 6],
            gripper_step < -float(config["gripper_step_threshold"]),
            merge_gap_frames=int(config["gripper_merge_gap_frames"]),
            minimum_active_frames=int(config["minimum_gripper_active_frames"]),
        )
        if run.signed_change <= -float(config["minimum_gripper_run_change"])
    ]

    arm_start_candidates = [
        run
        for run in arm_runs
        if move_start_frame - int(config["move_arm_search_before_frames"])
        <= run.start
        <= move_start_frame + int(config["move_arm_search_after_frames"])
        and run.start < rexecution_frame
        and any(
            move_start_frame - int(config["move_gripper_search_before_frames"])
            <= gripper_motion.end
            < run.start
            for gripper_motion in gripper_motion_runs
        )
    ]
    if not arm_start_candidates:
        raise ValueError("No sustained arm-start run near move_start_frame")
    adjustment_start_run = min(arm_start_candidates, key=lambda run: (run.start, run.end))

    gripper_motion_candidates = [
        run
        for run in gripper_motion_runs
        if move_start_frame - int(config["move_gripper_search_before_frames"])
        <= run.end
        < adjustment_start_run.start
    ]
    if not gripper_motion_candidates:
        raise ValueError("No sustained gripper-motion run before arm start")
    gripper_motion_run = max(
        gripper_motion_candidates, key=lambda run: (run.end, run.start)
    )

    arm_stop_candidates = [
        run
        for run in arm_runs
        if run.start >= adjustment_start_run.start
        and run.start <= rexecution_frame
        and run.end <= rexecution_frame + int(config["rexecution_arm_end_tolerance_frames"])
    ]
    if not arm_stop_candidates:
        raise ValueError("No sustained arm run ending near/before rexecution_frame")
    adjustment_stop_run = max(arm_stop_candidates, key=lambda run: (run.end, run.start))

    close_candidates = [
        run
        for run in closing_runs
        if rexecution_frame - int(config["rexecution_close_search_before_frames"])
        <= run.start
        <= rexecution_frame + int(config["rexecution_close_search_after_frames"])
        and run.start > adjustment_stop_run.end
    ]
    if not close_candidates:
        raise ValueError("No sustained gripper-closing run after arm stop")
    closing_run = min(close_candidates, key=lambda run: (run.start, run.end))

    events = {
        "gripper_motion_stop_frame": gripper_motion_run.end,
        "arm_adjustment_start_frame": adjustment_start_run.start,
        "arm_adjustment_stop_frame": adjustment_stop_run.end,
        "gripper_close_start_frame": closing_run.start,
    }
    if not (
        events["gripper_motion_stop_frame"] < events["arm_adjustment_start_frame"]
        and events["arm_adjustment_stop_frame"] < events["gripper_close_start_frame"]
    ):
        raise ValueError(f"Detected V7.2 events are not ordered: {events}")
    return {
        "events": events,
        "event_offsets_from_anchor": {
            "gripper_motion_stop_minus_move_start": gripper_motion_run.end
            - move_start_frame,
            "arm_adjustment_start_minus_move_start": adjustment_start_run.start - move_start_frame,
            "arm_adjustment_stop_minus_rexecution": adjustment_stop_run.end - rexecution_frame,
            "gripper_close_start_minus_rexecution": closing_run.start - rexecution_frame,
        },
        "detected_runs": {
            "gripper_motion_before_adjustment": asdict(gripper_motion_run),
            "arm_adjustment_start": asdict(adjustment_start_run),
            "arm_adjustment_stop": asdict(adjustment_stop_run),
            "gripper_closing": asdict(closing_run),
        },
    }


def build_attempt_filter_row(
    *,
    episode_id: int,
    attempt_frames: Sequence[V4Frame],
    actions: Sequence[np.ndarray],
    timestamps: Mapping[int, float],
    move_start_frame: int,
    rexecution_frame: int,
    candidate_global_indices: set[int],
    config: Mapping[str, Any] = DEFAULT_DETECTOR_CONFIG,
) -> dict[str, Any]:
    ordered = sorted(attempt_frames, key=lambda frame: frame.frame_index)
    attempt_actions = np.stack([actions[frame.global_index] for frame in ordered])
    detected = detect_attempt_boundaries(
        attempt_actions,
        move_start_frame=move_start_frame,
        rexecution_frame=rexecution_frame,
        config=config,
    )
    frame_lookup = {frame.frame_index: frame for frame in ordered}

    def event(name: str) -> dict[str, Any]:
        frame_index = int(detected["events"][name])
        frame = frame_lookup[frame_index]
        return {
            "frame_index": frame_index,
            "global_index": frame.global_index,
            "timestamp": float(timestamps[frame.global_index]),
        }

    events = {name.removesuffix("_frame"): event(name) for name in detected["events"]}
    interval_specs = (
        (
            "post_gripper_motion_pre_arm_idle",
            int(detected["events"]["gripper_motion_stop_frame"]),
            int(detected["events"]["arm_adjustment_start_frame"]),
        ),
        (
            "post_arm_pre_close_idle",
            int(detected["events"]["arm_adjustment_stop_frame"]),
            int(detected["events"]["gripper_close_start_frame"]),
        ),
    )
    intervals: list[dict[str, Any]] = []
    for reason, left_event, right_event in interval_specs:
        excluded_frames = list(range(left_event + 1, right_event))
        excluded_globals = [
            frame_lookup[index].global_index
            for index in excluded_frames
            if frame_lookup[index].global_index in candidate_global_indices
        ]
        intervals.append(
            {
                "reason": reason,
                "left_event_frame": left_event,
                "right_event_frame": right_event,
                "excluded_start_frame": excluded_frames[0] if excluded_frames else None,
                "excluded_end_frame": excluded_frames[-1] if excluded_frames else None,
                "excluded_frame_count": len(excluded_frames),
                "excluded_candidate_action_start_count": len(excluded_globals),
                "excluded_global_indices": excluded_globals,
            }
        )
    return {
        "episode_id": episode_id,
        "attempt_id": 2,
        "move_start_frame": move_start_frame,
        "rexecution_frame": rexecution_frame,
        "events": events,
        "event_offsets_from_anchor": detected["event_offsets_from_anchor"],
        "detected_runs": detected["detected_runs"],
        "excluded_intervals": intervals,
    }
