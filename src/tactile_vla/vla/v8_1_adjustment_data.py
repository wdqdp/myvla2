"""V8.1 Stage-A action data built from the refreshed black-box dataset.

V8.1 preserves the V7.4 policy byte-for-byte for the original task families.
The only task-specific change is ``small_grasp``: its two pre-adjustment
events both use ``move_start_timestamp`` and its two post-adjustment events
both use ``rexecution_timestamp``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.v4_data import SPLITS, V4Frame, file_identity, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v5_adjustment_data import phase_for_rexecution_frame
from tactile_vla.vla.v7_2_boundary_filter import DEFAULT_DETECTOR_CONFIG, V7_2_FILTER_SCHEMA
from tactile_vla.vla.v7_3_adjustment_data import (
    V7_3_FILTER_POLICY,
    build_v7_3_exclusions,
    compute_raw_h30_content_identity,
)
from tactile_vla.vla.v7_4_adjustment_data import (
    V7_4_TARGET_POLICY,
    apply_v7_4_target_policy,
)


ROTATION_PHASE_V8_1_ADJUSTMENT = "rotation_phase_v8_1_adjustment"
V8_1_EXPERIMENT_KIND = "phase_prompt_h30_v7_4_policy_refreshed_small_grasp_equal_boundaries"
V8_1_TIMING_SCHEMA = "tactile_vla_v8_1_attempt_timing_v1"
V8_1_BOUNDARY_SCHEMA = "tactile_vla_v8_1_boundary_filter_v1"
V8_1_ACTION_PHASE_SCHEMA = "tactile_vla_v8_1_adjustment_action_manifest_v1"
V8_1_TRAINING_INDEX_SCHEMA = "tactile_vla_v8_1_adjustment_training_index_v1"
V8_1_SUMMARY_SCHEMA = "tactile_vla_v8_1_adjustment_filter_summary_v1"
V8_1_HASH_SCHEMA = "tactile_vla_v8_1_adjustment_artifact_hashes_v1"
V8_1_H30_TARGET_SCHEMA = "tactile_vla_v8_1_h30_target_offsets_v1"

SMALL_GRASP_TASK = "small_grasp"
REFERENCE_TASKS = frozenset({"one_success", "moderate_lift", "slightly_lift"})
ATTEMPT2_REFERENCE_TASKS = frozenset({"moderate_lift", "slightly_lift"})
BOUNDARY_POLICY = {
    "reference_tasks": {
        "tasks": sorted(ATTEMPT2_REFERENCE_TASKS),
        "policy": "reuse_v7_4_audited_event_frames_exactly",
        "order": "gripper_motion_stop < arm_adjustment_start and arm_adjustment_stop < gripper_close_start",
    },
    "small_grasp": {
        "task": SMALL_GRASP_TASK,
        "gripper_motion_stop": "move_start_frame",
        "arm_adjustment_start": "move_start_frame",
        "arm_adjustment_stop": "rexecution_frame",
        "gripper_close_start": "rexecution_frame",
        "order": "gripper_motion_stop == arm_adjustment_start < arm_adjustment_stop == gripper_close_start",
    },
    "timestamp_to_frame": "first_ros_timestamp_greater_than_or_equal_to_annotation",
}


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _groups(frames: Sequence[V4Frame]) -> dict[tuple[int, int], list[V4Frame]]:
    grouped: dict[tuple[int, int], list[V4Frame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.attempt_key].append(frame)
    for key, values in grouped.items():
        values.sort(key=lambda frame: frame.frame_index)
        if [frame.frame_index for frame in values] != list(range(len(values))):
            raise ValueError(f"Non-contiguous frame indices for attempt {key}")
        timestamps = np.asarray([frame.ros_timestamp for frame in values], dtype=np.float64)
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Invalid ROS timestamp sequence for attempt {key}")
    return dict(grouped)


def _profile_rows(v4_index: Mapping[str, Any]) -> tuple[Path, dict[tuple[int, int], dict[str, Any]]]:
    identity = v4_index.get("source_files", {}).get("profile", {})
    profile_path = Path(str(identity.get("path", ""))).expanduser().resolve()
    if not profile_path.is_file() or sha256_file(profile_path) != identity.get("sha256"):
        raise ValueError("V8.1 V4 profile identity mismatch")
    profile = _load_object(profile_path)
    rows = {
        (int(row["episode_id"]), int(row["attempt_id"])): dict(row)
        for row in profile.get("attempts", [])
    }
    return profile_path, rows


def _timestamp_frame(values: Sequence[V4Frame], timestamp: float, *, context: str) -> tuple[int, float]:
    if not np.isfinite(timestamp):
        raise ValueError(f"{context} is not finite")
    timestamps = np.asarray([frame.ros_timestamp for frame in values], dtype=np.float64)
    frame_index = int(np.searchsorted(timestamps, timestamp, side="left"))
    if frame_index >= len(values):
        raise ValueError(f"{context} lies after the final ROS timestamp")
    return frame_index, float(timestamps[frame_index] - timestamp)


def build_attempt_timing_artifact(
    *, dataset_dir: Path, v4_index_file: Path, hdf5_dir: Path
) -> dict[str, Any]:
    """Recover native timing metadata without mutating the refreshed V4 index."""

    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    hdf5_dir = hdf5_dir.expanduser().resolve()
    v4_index = _load_object(v4_index_file)
    frames, _ = validate_v4_index_dataset(v4_index, dataset_dir)
    grouped = _groups(frames)
    profile_path, profiles = _profile_rows(v4_index)
    if set(grouped) != set(profiles):
        raise ValueError("V8.1 profile and LeRobot attempt sets differ")

    rows: list[dict[str, Any]] = []
    for key, attempt_frames in sorted(grouped.items()):
        episode_id, attempt_id = key
        profile = profiles[key]
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "attempt_id": attempt_id,
            "task": str(profile["task"]),
            "frame_count": len(attempt_frames),
            "hdf5_path": str(profile["hdf5_path"]),
            "move_start_timestamp": None,
            "move_start_frame_index": None,
            "move_start_frame_timestamp_offset_seconds": None,
            "rexecution_timestamp": None,
            "rexecution_frame_index": None,
            "rexecution_frame_timestamp_offset_seconds": None,
        }
        if attempt_id == 2:
            hdf5_path = hdf5_dir / str(profile["hdf5_path"])
            if not hdf5_path.is_file():
                raise FileNotFoundError(hdf5_path)
            with h5py.File(hdf5_path, "r") as stream:
                move_timestamp = float(stream["/meta/move_start_timestamp"][()])
                rexecution_timestamp = float(stream["/meta/rexecution_timestamp"][()])
                hdf5_frame_count = int(stream["/size"][()])
            if hdf5_frame_count != len(attempt_frames):
                raise ValueError(f"HDF5/LeRobot frame count mismatch for attempt {key}")
            move_frame, move_offset = _timestamp_frame(
                attempt_frames, move_timestamp, context=f"attempt {key} move_start_timestamp"
            )
            rexecution_frame, rexecution_offset = _timestamp_frame(
                attempt_frames,
                rexecution_timestamp,
                context=f"attempt {key} rexecution_timestamp",
            )
            if not 0 <= move_frame < rexecution_frame < len(attempt_frames):
                raise ValueError(f"Invalid move/reexecution order for attempt {key}")
            row.update(
                {
                    "move_start_timestamp": move_timestamp,
                    "move_start_frame_index": move_frame,
                    "move_start_frame_timestamp_offset_seconds": move_offset,
                    "rexecution_timestamp": rexecution_timestamp,
                    "rexecution_frame_index": rexecution_frame,
                    "rexecution_frame_timestamp_offset_seconds": rexecution_offset,
                }
            )
        rows.append(row)

    task_counts = Counter(str(row["task"]) for row in rows)
    attempt2_rows = [row for row in rows if int(row["attempt_id"]) == 2]
    payload: dict[str, Any] = {
        "schema_version": V8_1_TIMING_SCHEMA,
        "data_profile": ROTATION_PHASE_V8_1_ADJUSTMENT,
        "mapping_policy": BOUNDARY_POLICY["timestamp_to_frame"],
        "dataset_dir": str(dataset_dir),
        "hdf5_dir": str(hdf5_dir),
        "source_files": {
            "v4_training_index": file_identity(v4_index_file),
            "v4_profile": file_identity(profile_path),
        },
        "summary": {
            "attempt_count": len(rows),
            "attempt2_count": len(attempt2_rows),
            "task_attempt_counts": dict(sorted(task_counts.items())),
            "max_nonnegative_timestamp_offset_seconds": max(
                float(row["rexecution_frame_timestamp_offset_seconds"])
                for row in attempt2_rows
            ),
        },
        "attempts": rows,
    }
    payload["content_sha256"] = sha256_json(rows)
    payload["sha256"] = sha256_json(payload)
    return payload


def validate_attempt_timing_artifact(
    payload: Mapping[str, Any], *, v4_index_file: Path
) -> dict[tuple[int, int], dict[str, Any]]:
    if payload.get("schema_version") != V8_1_TIMING_SCHEMA:
        raise ValueError("V8.1 timing schema mismatch")
    stored_sha = payload.get("sha256")
    if stored_sha != sha256_json({key: value for key, value in payload.items() if key != "sha256"}):
        raise ValueError("V8.1 timing artifact hash mismatch")
    rows = payload.get("attempts")
    if not isinstance(rows, list) or payload.get("content_sha256") != sha256_json(rows):
        raise ValueError("V8.1 timing row identity mismatch")
    source = payload.get("source_files", {}).get("v4_training_index", {})
    if (
        Path(str(source.get("path", ""))).expanduser().resolve()
        != v4_index_file.expanduser().resolve()
        or source.get("sha256") != sha256_file(v4_index_file)
    ):
        raise ValueError("V8.1 timing artifact uses another V4 index")
    lookup = {(int(row["episode_id"]), int(row["attempt_id"])): dict(row) for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("V8.1 timing artifact contains duplicate attempts")
    return lookup


def _event(frame_lookup: Mapping[int, V4Frame], frame_index: int) -> dict[str, Any]:
    frame = frame_lookup.get(int(frame_index))
    if frame is None:
        raise ValueError(f"Boundary references missing frame {frame_index}")
    return {
        "frame_index": frame.frame_index,
        "global_index": frame.global_index,
        "timestamp": frame.ros_timestamp,
    }


def _interval(
    *, reason: str, left: int, right: int, frame_lookup: Mapping[int, V4Frame], candidates: set[int]
) -> dict[str, Any]:
    frames = list(range(left + 1, right))
    globals_ = [frame_lookup[index].global_index for index in frames if frame_lookup[index].global_index in candidates]
    return {
        "reason": reason,
        "left_event_frame": left,
        "right_event_frame": right,
        "excluded_start_frame": frames[0] if frames else None,
        "excluded_end_frame": frames[-1] if frames else None,
        "excluded_frame_count": len(frames),
        "excluded_candidate_action_start_count": len(globals_),
        "excluded_global_indices": globals_,
    }


def build_boundary_artifact(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    timing_payload: Mapping[str, Any],
    reference_v7_2_boundary_file: Path,
) -> dict[str, Any]:
    """Reuse old audited boundaries and add annotation-equal small-grasp rows."""

    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    reference_v7_2_boundary_file = reference_v7_2_boundary_file.expanduser().resolve()
    v4_index = _load_object(v4_index_file)
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    grouped = _groups(frames)
    timing = validate_attempt_timing_artifact(timing_payload, v4_index_file=v4_index_file)
    candidates = {
        int(index)
        for split in SPLITS
        for index in v4_index["splits"][split]["execution_indices"]
    }
    split_by_episode = {
        global_lookup[int(index)].episode_id: split
        for split in SPLITS
        for index in v4_index["splits"][split]["execution_indices"]
    }

    reference = _load_object(reference_v7_2_boundary_file)
    if reference.get("schema_version") != V7_2_FILTER_SCHEMA:
        raise ValueError("V8.1 reference boundary file is not V7.2")
    reference_rows = {
        (int(row["episode_id"]), int(row["attempt_id"])): row
        for row in reference.get("attempts", [])
    }
    expected_reference = {
        key for key, row in timing.items() if key[1] == 2 and row["task"] in ATTEMPT2_REFERENCE_TASKS
    }
    if set(reference_rows) != expected_reference:
        raise ValueError(
            "V8.1 old-task attempt set differs from the audited V7.4 source: "
            f"missing={sorted(expected_reference - set(reference_rows))[:10]}, "
            f"extra={sorted(set(reference_rows) - expected_reference)[:10]}"
        )

    rows: list[dict[str, Any]] = []
    for key, attempt_frames in sorted(grouped.items()):
        if key[1] != 2:
            continue
        timeline = timing[key]
        task = str(timeline["task"])
        frame_lookup = {frame.frame_index: frame for frame in attempt_frames}
        move_frame = int(timeline["move_start_frame_index"])
        rexecution_frame = int(timeline["rexecution_frame_index"])
        if task == SMALL_GRASP_TASK:
            indices = {
                "gripper_motion_stop": move_frame,
                "arm_adjustment_start": move_frame,
                "arm_adjustment_stop": rexecution_frame,
                "gripper_close_start": rexecution_frame,
            }
            boundary_source = "annotated_equal_boundary"
            detected_runs = None
        elif task in ATTEMPT2_REFERENCE_TASKS:
            old = reference_rows[key]
            if int(old["move_start_frame"]) != move_frame or int(old["rexecution_frame"]) != rexecution_frame:
                raise ValueError(f"Refreshed timing differs from V7.4 for attempt {key}")
            indices = {
                name: int(old["events"][name]["frame_index"])
                for name in (
                    "gripper_motion_stop",
                    "arm_adjustment_start",
                    "arm_adjustment_stop",
                    "gripper_close_start",
                )
            }
            if not (
                indices["gripper_motion_stop"] < indices["arm_adjustment_start"]
                and indices["arm_adjustment_stop"] < indices["gripper_close_start"]
            ):
                raise ValueError(f"Invalid inherited V7.4 boundary order for attempt {key}")
            boundary_source = "v7_4_audited_reference"
            detected_runs = old.get("detected_runs")
        else:
            raise ValueError(f"Unsupported V8.1 attempt2 task {task!r} for attempt {key}")

        if not (
            indices["gripper_motion_stop"] <= indices["arm_adjustment_start"]
            <= indices["arm_adjustment_stop"] <= indices["gripper_close_start"]
        ):
            raise ValueError(f"Invalid V8.1 boundary order for attempt {key}")
        events = {name: _event(frame_lookup, index) for name, index in indices.items()}
        intervals = [
            _interval(
                reason="post_gripper_motion_pre_arm_idle",
                left=indices["gripper_motion_stop"],
                right=indices["arm_adjustment_start"],
                frame_lookup=frame_lookup,
                candidates=candidates,
            ),
            _interval(
                reason="post_arm_pre_close_idle",
                left=indices["arm_adjustment_stop"],
                right=indices["gripper_close_start"],
                frame_lookup=frame_lookup,
                candidates=candidates,
            ),
        ]
        rows.append(
            {
                "episode_id": key[0],
                "attempt_id": key[1],
                "task": task,
                "split": split_by_episode[key[0]],
                "boundary_source": boundary_source,
                "move_start_frame": move_frame,
                "rexecution_frame": rexecution_frame,
                "events": events,
                "event_offsets_from_anchor": {
                    "gripper_motion_stop_minus_move_start": indices["gripper_motion_stop"] - move_frame,
                    "arm_adjustment_start_minus_move_start": indices["arm_adjustment_start"] - move_frame,
                    "arm_adjustment_stop_minus_rexecution": indices["arm_adjustment_stop"] - rexecution_frame,
                    "gripper_close_start_minus_rexecution": indices["gripper_close_start"] - rexecution_frame,
                },
                "detected_runs": detected_runs,
                "excluded_intervals": intervals,
            }
        )

    reason_counts: Counter[str] = Counter()
    excluded: set[int] = set()
    for row in rows:
        for interval in row["excluded_intervals"]:
            reason_counts[interval["reason"]] += interval["excluded_candidate_action_start_count"]
            overlap = excluded.intersection(interval["excluded_global_indices"])
            if overlap:
                raise ValueError(f"V8.1 boundary intervals overlap at {min(overlap)}")
            excluded.update(interval["excluded_global_indices"])
    small_rows = [row for row in rows if row["task"] == SMALL_GRASP_TASK]
    if any(interval["excluded_frame_count"] for row in small_rows for interval in row["excluded_intervals"]):
        raise AssertionError("small_grasp must have no idle exclusion interval")
    payload: dict[str, Any] = {
        "schema_version": V8_1_BOUNDARY_SCHEMA,
        "data_profile": ROTATION_PHASE_V8_1_ADJUSTMENT,
        "boundary_policy": BOUNDARY_POLICY,
        "reference_detector_config": DEFAULT_DETECTOR_CONFIG,
        "dataset_dir": str(dataset_dir),
        "source_files": {
            "v4_training_index": file_identity(v4_index_file),
            "attempt_timing": {
                "path": None,
                "sha256": timing_payload["sha256"],
                "content_sha256": timing_payload["content_sha256"],
            },
            "reference_v7_2_boundary": file_identity(reference_v7_2_boundary_file),
        },
        "summary": {
            "attempt2_count": len(rows),
            "reference_attempt_count": len(rows) - len(small_rows),
            "small_grasp_attempt_count": len(small_rows),
            "excluded_candidate_action_start_count": len(excluded),
            "excluded_counts_by_reason": dict(sorted(reason_counts.items())),
            "boundary_source_counts": dict(Counter(row["boundary_source"] for row in rows)),
        },
        "attempts": rows,
    }
    payload["content_sha256"] = sha256_json(rows)
    return payload


def validate_boundary_artifact(
    payload: Mapping[str, Any], *, v4_index_file: Path, timing_payload: Mapping[str, Any]
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, str]]:
    if payload.get("schema_version") != V8_1_BOUNDARY_SCHEMA or payload.get("boundary_policy") != BOUNDARY_POLICY:
        raise ValueError("V8.1 boundary schema/policy mismatch")
    if payload.get("content_sha256") != sha256_json(payload.get("attempts")):
        raise ValueError("V8.1 boundary content hash mismatch")
    source = payload.get("source_files", {})
    v4_source = source.get("v4_training_index", {})
    if v4_source.get("sha256") != sha256_file(v4_index_file):
        raise ValueError("V8.1 boundary file uses another V4 index")
    if source.get("attempt_timing", {}).get("sha256") != timing_payload.get("sha256"):
        raise ValueError("V8.1 boundary file uses another timing artifact")
    attempts: dict[tuple[int, int], dict[str, Any]] = {}
    exclusions: dict[int, str] = {}
    for raw in payload.get("attempts", []):
        row = dict(raw)
        key = (int(row["episode_id"]), int(row["attempt_id"]))
        if key in attempts:
            raise ValueError(f"Duplicate V8.1 boundary attempt {key}")
        attempts[key] = row
        events = row["events"]
        g = int(events["gripper_motion_stop"]["frame_index"])
        a = int(events["arm_adjustment_start"]["frame_index"])
        s = int(events["arm_adjustment_stop"]["frame_index"])
        c = int(events["gripper_close_start"]["frame_index"])
        if row["task"] == SMALL_GRASP_TASK:
            if not (g == a < s == c == int(row["rexecution_frame"])):
                raise ValueError(f"small_grasp equal-boundary rule failed for attempt {key}")
        elif not (g < a <= s < c):
            raise ValueError(f"V7.4 inherited boundary rule failed for attempt {key}")
        for interval in row["excluded_intervals"]:
            for raw_index in interval["excluded_global_indices"]:
                index = int(raw_index)
                if index in exclusions:
                    raise ValueError(f"V8.1 excludes action start {index} twice")
                exclusions[index] = str(interval["reason"])
    return attempts, exclusions


def _target_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    action_horizon: int,
    modifications: Mapping[str, Any],
    schema_version: str = V8_1_H30_TARGET_SCHEMA,
) -> dict[str, Any]:
    offset_rows = [
        {"global_index": int(row["global_index"]), "action_target_offsets": row["action_target_offsets"]}
        for row in rows
        if row.get("action_target_offsets") is not None
    ]
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "action_horizon": action_horizon,
        "offset_rows_sha256": sha256_json(offset_rows),
        "modified_rows": len(offset_rows),
        "modifications": dict(modifications),
    }
    payload["sha256"] = sha256_json(payload)
    return payload


def _assert_reference_equivalence(
    rows: Sequence[Mapping[str, Any]], *, reference_v7_4_index_file: Path
) -> dict[str, Any]:
    reference_index = _load_object(reference_v7_4_index_file)
    manifest_path = Path(reference_index["source_files"]["action_phase_manifest"]["path"])
    reference_rows = load_jsonl(manifest_path)
    expected = {
        (int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"])): row
        for row in reference_rows
    }
    actual = {
        (int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"])): row
        for row in rows
        if str(row["task"]) in REFERENCE_TASKS
    }
    if set(actual) != set(expected):
        raise ValueError("V8.1 original-task action candidate set differs from V7.4")
    fields = (
        "phase",
        "rexecution_frame",
        "raw_chunk_phase_pure",
        "trainable",
        "exclusion_reason",
        "action_target_offsets",
        "effective_h30_modified",
    )
    for key in sorted(expected):
        mismatches = {field: (expected[key].get(field), actual[key].get(field)) for field in fields if expected[key].get(field) != actual[key].get(field)}
        if mismatches:
            raise ValueError(f"V8.1 differs from V7.4 at {key}: {mismatches}")
    return {
        "reference_candidate_count": len(expected),
        "checked_fields": list(fields),
        "reference_manifest": file_identity(manifest_path),
        "equivalence_sha256": sha256_json(
            [{"key": list(key), **{field: actual[key].get(field) for field in fields}} for key in sorted(actual)]
        ),
    }


def build_v8_1_adjustment_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    v4_norm_stats_dir: Path,
    timing_file: Path,
    boundary_file: Path,
    reference_v7_4_index_file: Path,
    _data_profile: str = ROTATION_PHASE_V8_1_ADJUSTMENT,
    _experiment_kind: str = V8_1_EXPERIMENT_KIND,
    _action_schema: str = V8_1_ACTION_PHASE_SCHEMA,
    _training_schema: str = V8_1_TRAINING_INDEX_SCHEMA,
    _summary_schema: str = V8_1_SUMMARY_SCHEMA,
    _boundary_schema: str = V8_1_BOUNDARY_SCHEMA,
    _h30_target_schema: str = V8_1_H30_TARGET_SCHEMA,
    _boundary_policy: Mapping[str, Any] = BOUNDARY_POLICY,
    _boundary_validator: Any = validate_boundary_artifact,
    _exclusion_builder: Any = build_v7_3_exclusions,
    _filter_policy: Mapping[str, Any] = V7_3_FILTER_POLICY,
    _require_small_uncompressed: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    v4_norm_stats_dir = v4_norm_stats_dir.expanduser().resolve()
    timing_file = timing_file.expanduser().resolve()
    boundary_file = boundary_file.expanduser().resolve()
    reference_v7_4_index_file = reference_v7_4_index_file.expanduser().resolve()
    v4_index = _load_object(v4_index_file)
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    timing_payload = _load_object(timing_file)
    timing = validate_attempt_timing_artifact(timing_payload, v4_index_file=v4_index_file)
    boundary_payload = _load_object(boundary_file)
    boundaries, idle_exclusions = _boundary_validator(
        boundary_payload, v4_index_file=v4_index_file, timing_payload=timing_payload
    )
    expected_attempt2 = {key for key in timing if key[1] == 2}
    if set(boundaries) != expected_attempt2:
        raise ValueError("V8.1 boundary artifact does not cover every attempt2")

    base_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for raw_index in v4_index["splits"][split]["execution_indices"]:
            global_index = int(raw_index)
            frame = global_lookup[global_index]
            timeline = timing[frame.attempt_key]
            rexecution = timeline["rexecution_frame_index"]
            phase = phase_for_rexecution_frame(
                frame.attempt_id, frame.frame_index, None if rexecution is None else int(rexecution)
            )
            crosses = bool(
                phase == "adjustment"
                and rexecution is not None
                and frame.frame_index < int(rexecution) <= frame.frame_index + int(v4_index["action_horizon"]) - 1
            )
            base_rows.append(
                {
                    "schema_version": _action_schema,
                    "data_profile": _data_profile,
                    "prompt_profile": PHASE_PROMPT_PROFILE_V2,
                    "experiment_kind": _experiment_kind,
                    "split": split,
                    "global_index": global_index,
                    "episode_id": frame.episode_id,
                    "attempt_id": frame.attempt_id,
                    "frame_index": frame.frame_index,
                    "task": str(timeline["task"]),
                    "phase": phase,
                    "rexecution_frame": rexecution,
                    "raw_chunk_phase_pure": not crosses,
                    "effective_chunk_phase_pure": not crosses,
                    "chunk_phase_pure": not crosses,
                    "terminal_hold_from_offset": None,
                    "effective_h30_modified": False,
                    "chunk_end_frame": frame.frame_index + int(v4_index["action_horizon"]) - 1,
                    "action_horizon": int(v4_index["action_horizon"]),
                }
            )

    exclusions, stop_policies = _exclusion_builder(
        base_rows,
        v7_2_exclusions=idle_exclusions,
        boundary_payload=boundary_payload,
        action_horizon=int(v4_index["action_horizon"]),
    )
    filtered_rows = []
    for base in base_rows:
        row = dict(base)
        global_index = int(row["global_index"])
        row.update(
            {
                "trainable": global_index not in exclusions,
                "exclusion_reason": exclusions.get(global_index),
            }
        )
        filtered_rows.append(row)
    rows, modifications = apply_v7_4_target_policy(
        filtered_rows,
        boundary_payload=boundary_payload,
        action_horizon=int(v4_index["action_horizon"]),
    )
    for row in rows:
        row.update(
            {
                "schema_version": _action_schema,
                "data_profile": _data_profile,
                "prompt_profile": PHASE_PROMPT_PROFILE_V2,
                "experiment_kind": _experiment_kind,
                "terminal_hold_from_offset": None,
            }
        )
    small_rows = [row for row in rows if row["task"] == SMALL_GRASP_TASK]
    if _require_small_uncompressed and any(
        row.get("action_target_offsets") is not None for row in small_rows
    ):
        raise AssertionError("small_grasp must not apply V7.4 idle-gap compression")
    if any(bool(row["trainable"]) and row["phase"] == "adjustment" and not row["raw_chunk_phase_pure"] for row in rows):
        raise AssertionError("V8.1 retained a cross-reexecution adjustment H30")

    reference_equivalence = _assert_reference_equivalence(
        rows, reference_v7_4_index_file=reference_v7_4_index_file
    )
    split_entries: dict[str, Any] = {}
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        selected = [row for row in split_rows if bool(row["trainable"])]
        row_indices = [index for index, row in enumerate(rows) if row["split"] == split and bool(row["trainable"])]
        split_entries[split] = {
            "execution_indices": [int(row["global_index"]) for row in selected],
            "action_phase_manifest_row_indices": row_indices,
            "summary": {
                "candidate_count": len(split_rows),
                "action_count": len(selected),
                "excluded_count": len(split_rows) - len(selected),
                "phase_counts": dict(Counter(str(row["phase"]) for row in selected)),
                "task_counts": dict(Counter(str(row["task"]) for row in selected)),
                "task_phase_counts": dict(Counter(f"{row['task']}:{row['phase']}" for row in selected)),
                "exclusion_reason_counts": dict(Counter(str(row["exclusion_reason"]) for row in split_rows if not row["trainable"])),
                "raw_chunk_crossing": sum(not bool(row["raw_chunk_phase_pure"]) for row in selected),
                "terminal_hold_chunks": 0,
                "idle_compressed_h30_chunks": sum(row.get("action_target_offsets") is not None for row in selected),
            },
        }
    filtered_identity = action_indices_identity(split_entries)
    selected_rows = [row for row in rows if bool(row["trainable"])]
    raw_h30 = compute_raw_h30_content_identity(
        dataset_dir=dataset_dir,
        global_lookup=global_lookup,
        action_rows=selected_rows,
        action_indices_identity=filtered_identity,
        action_horizon=int(v4_index["action_horizon"]),
    )
    target_identity = _target_identity(
        selected_rows,
        action_horizon=int(v4_index["action_horizon"]),
        modifications=modifications,
        schema_version=_h30_target_schema,
    )

    norm_summary_path = v4_norm_stats_dir / "summary.json"
    norm_stats_path = v4_norm_stats_dir / "norm_stats.json"
    norm_summary = _load_object(norm_summary_path)
    norm_sha = sha256_file(norm_stats_path)
    if (
        norm_summary.get("norm_stats_sha256") != norm_sha
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity")
        != v4_index["action_indices_identity"]
        or int(norm_summary.get("num_frames", -1))
        != int(v4_index["action_indices_identity"]["train"]["count"])
    ):
        raise ValueError("V8.1 norm stats do not match the refreshed V4 action candidates")

    summary: dict[str, Any] = {
        "schema_version": _summary_schema,
        "data_profile": _data_profile,
        "boundary_policy": dict(_boundary_policy),
        "filter_policy": dict(_filter_policy),
        "target_policy": V7_4_TARGET_POLICY,
        "candidate_action_count": int(v4_index["action_indices_identity"]["all"]["count"]),
        "trainable_action_count": int(filtered_identity["all"]["count"]),
        "excluded_action_count": len(exclusions),
        "exclusion_reason_counts": dict(Counter(exclusions.values())),
        "h30_modifications": modifications,
        "reference_v7_4_equivalence": reference_equivalence,
        "attempt_filter_policies": stop_policies,
        "splits": {split: split_entries[split]["summary"] for split in SPLITS},
    }
    source_files = {
        "v4_training_index": file_identity(v4_index_file),
        "v4_norm_summary": file_identity(norm_summary_path),
        "v4_norm_stats": file_identity(norm_stats_path),
        "attempt_timing": file_identity(timing_file),
        "boundary_filter": file_identity(boundary_file),
        "reference_v7_4_training_index": file_identity(reference_v7_4_index_file),
    }
    index: dict[str, Any] = {
        "schema_version": _training_schema,
        "data_profile": _data_profile,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": _experiment_kind,
        "boundary_policy": dict(_boundary_policy),
        "filter_policy": dict(_filter_policy),
        "target_policy": V7_4_TARGET_POLICY,
        "data_config_hash": sha256_json(
            {
                "v4_training_data_hash": v4_index["training_data_hash"],
                "timing_sha256": timing_payload["sha256"],
                "boundary_content_sha256": boundary_payload["content_sha256"],
                "filter_policy": dict(_filter_policy),
                "target_policy": V7_4_TARGET_POLICY,
            }
        ),
        "selection_hash": v4_index["selection_hash"],
        "v4_profile_config_hash": v4_index["profile_config_hash"],
        "dataset_dir": str(dataset_dir),
        "action_horizon": int(v4_index["action_horizon"]),
        "splits": split_entries,
        "action_indices_identity": filtered_identity,
        "candidate_action_indices_identity": v4_index["action_indices_identity"],
        "native_reexecution_timing_identity": {
            "schema_version": V8_1_TIMING_SCHEMA,
            "attempt_count": timing_payload["summary"]["attempt_count"],
            "attempt2_count": timing_payload["summary"]["attempt2_count"],
            "content_sha256": timing_payload["content_sha256"],
            "sha256": timing_payload["sha256"],
        },
        "boundary_filter_identity": {
            "schema_version": _boundary_schema,
            "content_sha256": boundary_payload["content_sha256"],
            "summary": boundary_payload["summary"],
        },
        "h30_target_identity": target_identity,
        "raw_h30_target_identity": raw_h30,
        "v4_lerobot_identity": v4_index["lerobot_identity"],
        "v4_training_data_hash": v4_index["training_data_hash"],
        "v4_norm_stats_sha256": norm_sha,
        "norm_stats_policy": {
            "kind": "inherited",
            "source_data_profile": "rotation_v4",
            "source_action_indices_identity": v4_index["action_indices_identity"],
        },
        "h30_modifications": modifications,
        "reference_v7_4_equivalence": reference_equivalence,
        "summary": summary,
        "source_files": source_files,
        "action_phase_manifest_identity": {
            "count": len(rows),
            "content_sha256": sha256_json(rows),
        },
    }
    return rows, index, summary


def validate_v8_1_adjustment_training_index(
    payload: Mapping[str, Any], *, index_path: Path, dataset_dir: Path
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    expected = {
        "schema_version": V8_1_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V8_1_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V8_1_EXPERIMENT_KIND,
        "boundary_policy": BOUNDARY_POLICY,
        "filter_policy": V7_3_FILTER_POLICY,
        "target_policy": V7_4_TARGET_POLICY,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V8.1 index {key} mismatch")
    if payload.get("training_data_hash") != sha256_json(
        {key: value for key, value in payload.items() if key != "training_data_hash"}
    ):
        raise ValueError("V8.1 training_data_hash mismatch")
    sources = payload.get("source_files", {})
    required = {
        "v4_training_index",
        "v4_norm_summary",
        "v4_norm_stats",
        "attempt_timing",
        "boundary_filter",
        "reference_v7_4_training_index",
        "action_phase_manifest",
        "filter_summary",
    }
    if not required.issubset(sources):
        raise ValueError("V8.1 index lacks source files")
    paths: dict[str, Path] = {}
    for name in required:
        identity = sources[name]
        path = Path(str(identity["path"])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"V8.1 source identity mismatch: {name}")
        paths[name] = path
    validate_v4_index_dataset(_load_object(paths["v4_training_index"]), dataset_dir)
    rows = load_jsonl(paths["action_phase_manifest"])
    manifest_identity = payload["action_phase_manifest_identity"]
    if (
        len(rows) != int(manifest_identity["count"])
        or sha256_json(rows) != manifest_identity["content_sha256"]
        or _load_object(paths["filter_summary"]) != payload["summary"]
    ):
        raise ValueError("V8.1 action manifest/summary identity mismatch")
    lookup: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["global_index"])
        if index in lookup:
            raise ValueError(f"Duplicate V8.1 action global index {index}")
        if row.get("terminal_hold_from_offset") is not None:
            raise ValueError("V8.1 must not use terminal hold")
        offsets = row.get("action_target_offsets")
        if offsets is not None and (len(offsets) != 30 or any(a >= b for a, b in zip(offsets, offsets[1:], strict=False))):
            raise ValueError(f"Invalid V8.1 H30 target offsets at global index {index}")
        if row["task"] == SMALL_GRASP_TASK and offsets is not None:
            raise ValueError("small_grasp unexpectedly uses idle-gap compression")
        if bool(row["trainable"]):
            lookup[index] = row
    for split in SPLITS:
        expected_indices = [
            int(row["global_index"])
            for row in rows
            if row["split"] == split and bool(row["trainable"])
        ]
        if payload["splits"][split]["execution_indices"] != expected_indices:
            raise ValueError(f"V8.1 {split} execution indices mismatch")
    if not Path(index_path).is_file():
        raise FileNotFoundError(index_path)
    return rows, lookup


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": V8_1_HASH_SCHEMA,
        "files": {
            name: {"path": str(path.expanduser().resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }
    payload["sha256"] = sha256_json(payload)
    return payload


__all__ = [
    "BOUNDARY_POLICY",
    "ROTATION_PHASE_V8_1_ADJUSTMENT",
    "V8_1_EXPERIMENT_KIND",
    "V8_1_TRAINING_INDEX_SCHEMA",
    "artifact_hash_payload",
    "build_attempt_timing_artifact",
    "build_boundary_artifact",
    "build_v8_1_adjustment_artifacts",
    "validate_v8_1_adjustment_training_index",
]
