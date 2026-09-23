"""V8.2 Stage-A data with detected small-grasp motion boundaries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.v4_data import SPLITS, file_identity, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v5_adjustment_data import (
    DEFAULT_PIPER_URDF,
    _load_lerobot_actions,
    load_piper_fk_chain,
    piper_fk_positions,
)
from tactile_vla.vla.v7_2_boundary_filter import V7_2_FILTER_SCHEMA
from tactile_vla.vla.v7_3_adjustment_data import (
    CROSS_PHASE_REASON,
    STOP_H30_REASON,
    V7_3_FILTER_POLICY,
)
from tactile_vla.vla.v7_4_adjustment_data import (
    V7_4_TARGET_POLICY,
    compressed_h30_offsets,
)
from tactile_vla.vla.v8_1_adjustment_data import (
    ATTEMPT2_REFERENCE_TASKS,
    SMALL_GRASP_TASK,
    _event,
    _groups,
    _interval,
    _load_object,
    build_v8_1_adjustment_artifacts,
    validate_attempt_timing_artifact,
)
from tactile_vla.vla.v8_2_boundary_filter import (
    V8_2_SMALL_GRASP_DETECTOR_CONFIG,
    detect_small_grasp_boundaries,
)


ROTATION_PHASE_V8_2_ADJUSTMENT = "rotation_phase_v8_2_adjustment"
V8_2_EXPERIMENT_KIND = "phase_prompt_h30_v7_4_policy_detected_small_grasp_boundaries"
V8_2_BOUNDARY_SCHEMA = "tactile_vla_v8_2_boundary_filter_v1"
V8_2_ACTION_PHASE_SCHEMA = "tactile_vla_v8_2_adjustment_action_manifest_v1"
V8_2_TRAINING_INDEX_SCHEMA = "tactile_vla_v8_2_adjustment_training_index_v1"
V8_2_SUMMARY_SCHEMA = "tactile_vla_v8_2_adjustment_filter_summary_v1"
V8_2_HASH_SCHEMA = "tactile_vla_v8_2_adjustment_artifact_hashes_v1"
V8_2_H30_TARGET_SCHEMA = "tactile_vla_v8_2_h30_target_offsets_v1"
COMPRESSED_H30_LIMIT_REASON = "compressed_h30_exceeds_adjustment_stop_limit"
V8_2_FILTER_POLICY = {
    **V7_3_FILTER_POLICY,
    "small_grasp_too_short_for_post_stop_cap": (
        "exclude_all_raw_descent_starts; retain descent supervision through "
        "V7.4 pre-gap-compressed H30"
    ),
    "small_grasp_compressed_h30_limit": (
        "last source frame must be <= arm_adjustment_stop+10 and < rexecution_frame"
    ),
}

BOUNDARY_POLICY = {
    "reference_tasks": {
        "tasks": sorted(ATTEMPT2_REFERENCE_TASKS),
        "policy": "reuse_v7_4_audited_event_frames_exactly",
    },
    "small_grasp": {
        "gripper_motion_stop": "first_lerobot_action_gripper_maximum",
        "arm_adjustment_start": "first_sustained_piper_fk_z_descent_after_gripper_maximum",
        "arm_adjustment_stop": "last_sustained_piper_fk_z_descent_frame_near_reexecution",
        "gripper_close_start": "first_sustained_gripper_closing_run_after_descent",
        "order": "gripper_motion_stop < arm_adjustment_start <= arm_adjustment_stop < gripper_close_start",
    },
    "excluded_interval_semantics": "strictly_between_event_frames",
}


def build_boundary_artifact(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    timing_payload: Mapping[str, Any],
    reference_v7_2_boundary_file: Path,
    piper_urdf: Path = DEFAULT_PIPER_URDF,
) -> dict[str, Any]:
    """Reuse V7.4 boundaries and detect all four small-grasp events."""

    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    reference_v7_2_boundary_file = reference_v7_2_boundary_file.expanduser().resolve()
    piper_urdf = piper_urdf.expanduser().resolve()
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
    actions = _load_lerobot_actions(dataset_dir, len(global_lookup))
    fk_chain = load_piper_fk_chain(piper_urdf)

    reference = _load_object(reference_v7_2_boundary_file)
    if reference.get("schema_version") != V7_2_FILTER_SCHEMA:
        raise ValueError("V8.2 reference boundary file is not the audited V7.2 artifact")
    reference_rows = {
        (int(row["episode_id"]), int(row["attempt_id"])): row
        for row in reference.get("attempts", [])
    }
    expected_reference = {
        key
        for key, row in timing.items()
        if key[1] == 2 and row["task"] in ATTEMPT2_REFERENCE_TASKS
    }
    if set(reference_rows) != expected_reference:
        raise ValueError("V8.2 old-task attempt set differs from the V7.4 boundary source")

    rows: list[dict[str, Any]] = []
    for key, attempt_frames in sorted(grouped.items()):
        if key[1] != 2:
            continue
        timeline = timing[key]
        task = str(timeline["task"])
        frame_lookup = {frame.frame_index: frame for frame in attempt_frames}
        move_frame = int(timeline["move_start_frame_index"])
        rexecution_frame = int(timeline["rexecution_frame_index"])
        detection: dict[str, Any] | None = None
        if task == SMALL_GRASP_TASK:
            attempt_actions = np.asarray(
                [
                    actions[frame_lookup[index].global_index]
                    for index in range(len(attempt_frames))
                ],
                dtype=np.float64,
            )
            xyz = piper_fk_positions(attempt_actions[:, :6], fk_chain)
            detection = detect_small_grasp_boundaries(
                attempt_actions,
                xyz[:, 2],
                rexecution_frame=rexecution_frame,
            )
            indices = {name: int(value) for name, value in detection["events"].items()}
            boundary_source = "detected_small_grasp_action_fk"
            detected_runs = detection["detected_runs"]
        elif task in ATTEMPT2_REFERENCE_TASKS:
            old = reference_rows[key]
            if (
                int(old["move_start_frame"]) != move_frame
                or int(old["rexecution_frame"]) != rexecution_frame
            ):
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
            boundary_source = "v7_4_audited_reference"
            detected_runs = old.get("detected_runs")
        else:
            raise ValueError(f"Unsupported V8.2 attempt2 task {task!r} for attempt {key}")

        if not (
            indices["gripper_motion_stop"]
            < indices["arm_adjustment_start"]
            <= indices["arm_adjustment_stop"]
            < indices["gripper_close_start"]
        ):
            raise ValueError(f"Invalid V8.2 boundary order for attempt {key}: {indices}")
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
                    "gripper_motion_stop_minus_move_start": indices["gripper_motion_stop"]
                    - move_frame,
                    "arm_adjustment_start_minus_move_start": indices["arm_adjustment_start"]
                    - move_frame,
                    "arm_adjustment_stop_minus_reexecution": indices["arm_adjustment_stop"]
                    - rexecution_frame,
                    "gripper_close_start_minus_reexecution": indices["gripper_close_start"]
                    - rexecution_frame,
                },
                "detected_runs": detected_runs,
                "small_grasp_signals": None if detection is None else detection["signals"],
                "excluded_intervals": intervals,
            }
        )

    reason_counts: Counter[str] = Counter()
    excluded: set[int] = set()
    for row in rows:
        for interval in row["excluded_intervals"]:
            reason_counts[interval["reason"]] += interval[
                "excluded_candidate_action_start_count"
            ]
            overlap = excluded.intersection(interval["excluded_global_indices"])
            if overlap:
                raise ValueError(f"V8.2 boundary intervals overlap at {min(overlap)}")
            excluded.update(interval["excluded_global_indices"])
    small_rows = [row for row in rows if row["task"] == SMALL_GRASP_TASK]
    payload: dict[str, Any] = {
        "schema_version": V8_2_BOUNDARY_SCHEMA,
        "data_profile": ROTATION_PHASE_V8_2_ADJUSTMENT,
        "boundary_policy": BOUNDARY_POLICY,
        "small_grasp_detector_config": V8_2_SMALL_GRASP_DETECTOR_CONFIG,
        "dataset_dir": str(dataset_dir),
        "source_files": {
            "v4_training_index": file_identity(v4_index_file),
            "attempt_timing": {
                "path": None,
                "sha256": timing_payload["sha256"],
                "content_sha256": timing_payload["content_sha256"],
            },
            "reference_v7_2_boundary": file_identity(reference_v7_2_boundary_file),
            "piper_urdf": file_identity(piper_urdf),
        },
        "summary": {
            "attempt2_count": len(rows),
            "reference_attempt_count": len(rows) - len(small_rows),
            "small_grasp_attempt_count": len(small_rows),
            "small_grasp_detected_count": sum(
                row["boundary_source"] == "detected_small_grasp_action_fk" for row in rows
            ),
            "excluded_candidate_action_start_count": len(excluded),
            "excluded_counts_by_reason": dict(sorted(reason_counts.items())),
            "boundary_source_counts": dict(
                Counter(row["boundary_source"] for row in rows)
            ),
        },
        "attempts": rows,
    }
    payload["content_sha256"] = sha256_json(rows)
    return payload


def validate_boundary_artifact(
    payload: Mapping[str, Any],
    *,
    v4_index_file: Path,
    timing_payload: Mapping[str, Any],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, str]]:
    if (
        payload.get("schema_version") != V8_2_BOUNDARY_SCHEMA
        or payload.get("boundary_policy") != BOUNDARY_POLICY
        or payload.get("small_grasp_detector_config")
        != V8_2_SMALL_GRASP_DETECTOR_CONFIG
    ):
        raise ValueError("V8.2 boundary schema/policy mismatch")
    if payload.get("content_sha256") != sha256_json(payload.get("attempts")):
        raise ValueError("V8.2 boundary content hash mismatch")
    source = payload.get("source_files", {})
    if source.get("v4_training_index", {}).get("sha256") != sha256_file(v4_index_file):
        raise ValueError("V8.2 boundary file uses another V4 index")
    if source.get("attempt_timing", {}).get("sha256") != timing_payload.get("sha256"):
        raise ValueError("V8.2 boundary file uses another timing artifact")
    piper = source.get("piper_urdf", {})
    piper_path = Path(str(piper.get("path", ""))).expanduser().resolve()
    if not piper_path.is_file() or sha256_file(piper_path) != piper.get("sha256"):
        raise ValueError("V8.2 boundary Piper URDF identity mismatch")

    attempts: dict[tuple[int, int], dict[str, Any]] = {}
    exclusions: dict[int, str] = {}
    for raw in payload.get("attempts", []):
        row = dict(raw)
        key = (int(row["episode_id"]), int(row["attempt_id"]))
        if key in attempts:
            raise ValueError(f"Duplicate V8.2 boundary attempt {key}")
        attempts[key] = row
        events = row["events"]
        g = int(events["gripper_motion_stop"]["frame_index"])
        a = int(events["arm_adjustment_start"]["frame_index"])
        s = int(events["arm_adjustment_stop"]["frame_index"])
        c = int(events["gripper_close_start"]["frame_index"])
        if not g < a <= s < c:
            raise ValueError(f"V8.2 boundary order failed for attempt {key}")
        if row["task"] == SMALL_GRASP_TASK and row.get("small_grasp_signals") is None:
            raise ValueError(f"V8.2 small_grasp attempt lacks detector audit: {key}")
        for interval in row["excluded_intervals"]:
            for raw_index in interval["excluded_global_indices"]:
                index = int(raw_index)
                if index in exclusions:
                    raise ValueError(f"V8.2 excludes action start {index} twice")
                exclusions[index] = str(interval["reason"])
    return attempts, exclusions


def build_v8_2_exclusions(
    rows: list[dict[str, Any]],
    *,
    v7_2_exclusions: Mapping[int, str],
    boundary_payload: Mapping[str, Any],
    action_horizon: int,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Apply V7.3 start filtering, including very short small-grasp descents.

    A descent shorter than 20 frames cannot retain even its first raw H30
    while respecting V7.3's ten-frame post-stop cap.  Such raw starts are
    removed; V7.4's G-to-A gap compression still places the descent in the
    targets of starts immediately before G.
    """

    if action_horizon != int(V8_2_FILTER_POLICY["action_horizon"]):
        raise ValueError(f"V8.2 requires H30, got H{action_horizon}")
    exclusions = {int(index): str(reason) for index, reason in v7_2_exclusions.items()}
    rows_by_attempt: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["episode_id"]), int(row["attempt_id"]))
        rows_by_attempt.setdefault(key, []).append(row)

    max_post_stop = int(
        V8_2_FILTER_POLICY["max_post_stop_frames_for_retained_short_adjustment_h30"]
    )
    policies: list[dict[str, Any]] = []
    for attempt in boundary_payload["attempts"]:
        key = (int(attempt["episode_id"]), int(attempt["attempt_id"]))
        events = attempt["events"]
        arm_start = int(events["arm_adjustment_start"]["frame_index"])
        arm_stop = int(events["arm_adjustment_stop"]["frame_index"])
        if key[1] != 2 or arm_start > arm_stop:
            raise ValueError(f"Invalid V8.2 adjustment events for attempt {key}")

        full_filter_start = arm_stop - (action_horizon - 1)
        short_adjustment = arm_start >= full_filter_start
        last_retained_start = full_filter_start + max_post_stop
        if not short_adjustment:
            filter_start = full_filter_start
            mode = "full_pre_stop_h30"
        elif arm_start <= last_retained_start:
            filter_start = last_retained_start + 1
            mode = "short_adjustment_preserve_prefix"
        elif attempt["task"] == SMALL_GRASP_TASK:
            filter_start = full_filter_start
            mode = "short_small_grasp_supervised_by_pre_gap_compressed_h30"
        else:
            raise ValueError(
                "V7.4 cannot preserve an adjustment start while limiting post-stop "
                f"H30 frames to {max_post_stop}: episode={key[0]} "
                f"arm_start={arm_start} arm_stop={arm_stop}"
            )

        excluded_here = 0
        excluded_compressed_here = 0
        retained_motion_starts = 0
        max_retained_post_stop = 0
        for row in rows_by_attempt.get(key, []):
            if row["phase"] != "adjustment":
                continue
            frame = int(row["frame_index"])
            global_index = int(row["global_index"])
            if arm_start <= frame <= arm_stop and frame < filter_start:
                retained_motion_starts += global_index not in exclusions
                max_retained_post_stop = max(
                    max_retained_post_stop,
                    max(0, frame + action_horizon - 1 - arm_stop),
                )
            if filter_start <= frame <= arm_stop and global_index not in exclusions:
                exclusions[global_index] = STOP_H30_REASON
                excluded_here += 1

        if attempt["task"] == SMALL_GRASP_TASK:
            gripper_stop = int(events["gripper_motion_stop"]["frame_index"])
            rexecution = int(attempt["rexecution_frame"])
            max_last_target = min(arm_stop + max_post_stop, rexecution - 1)
            for row in rows_by_attempt.get(key, []):
                global_index = int(row["global_index"])
                if global_index in exclusions or row["phase"] != "adjustment":
                    continue
                start = int(row["frame_index"])
                offsets = compressed_h30_offsets(
                    start_frame=start,
                    gripper_motion_stop=gripper_stop,
                    arm_adjustment_start=arm_start,
                    action_horizon=action_horizon,
                )
                if offsets is not None and start + offsets[-1] > max_last_target:
                    exclusions[global_index] = COMPRESSED_H30_LIMIT_REASON
                    excluded_compressed_here += 1
        if mode == "short_adjustment_preserve_prefix":
            if retained_motion_starts == 0 or max_retained_post_stop > max_post_stop:
                raise AssertionError(f"V8.2 short-adjustment policy failed for attempt {key}")
        policies.append(
            {
                "episode_id": key[0],
                "attempt_id": key[1],
                "task": attempt["task"],
                "arm_adjustment_start": arm_start,
                "arm_adjustment_stop": arm_stop,
                "mode": mode,
                "filter_start": filter_start,
                "filter_end": arm_stop,
                "retained_motion_start_count": retained_motion_starts,
                "max_retained_post_stop_frames": max_retained_post_stop,
                "newly_excluded_start_count": excluded_here,
                "excluded_compressed_h30_start_count": excluded_compressed_here,
            }
        )

    for row in rows:
        global_index = int(row["global_index"])
        if (
            global_index not in exclusions
            and row["phase"] == "adjustment"
            and not bool(row["raw_chunk_phase_pure"])
        ):
            exclusions[global_index] = CROSS_PHASE_REASON
    return exclusions, policies


def build_v8_2_adjustment_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    v4_norm_stats_dir: Path,
    timing_file: Path,
    boundary_file: Path,
    reference_v7_4_index_file: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    return build_v8_1_adjustment_artifacts(
        dataset_dir=dataset_dir,
        v4_index_file=v4_index_file,
        v4_norm_stats_dir=v4_norm_stats_dir,
        timing_file=timing_file,
        boundary_file=boundary_file,
        reference_v7_4_index_file=reference_v7_4_index_file,
        _data_profile=ROTATION_PHASE_V8_2_ADJUSTMENT,
        _experiment_kind=V8_2_EXPERIMENT_KIND,
        _action_schema=V8_2_ACTION_PHASE_SCHEMA,
        _training_schema=V8_2_TRAINING_INDEX_SCHEMA,
        _summary_schema=V8_2_SUMMARY_SCHEMA,
        _boundary_schema=V8_2_BOUNDARY_SCHEMA,
        _h30_target_schema=V8_2_H30_TARGET_SCHEMA,
        _boundary_policy=BOUNDARY_POLICY,
        _boundary_validator=validate_boundary_artifact,
        _exclusion_builder=build_v8_2_exclusions,
        _filter_policy=V8_2_FILTER_POLICY,
        _require_small_uncompressed=False,
    )


def validate_v8_2_adjustment_training_index(
    payload: Mapping[str, Any], *, index_path: Path, dataset_dir: Path
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    expected = {
        "schema_version": V8_2_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V8_2_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V8_2_EXPERIMENT_KIND,
        "boundary_policy": BOUNDARY_POLICY,
        "filter_policy": V8_2_FILTER_POLICY,
        "target_policy": V7_4_TARGET_POLICY,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V8.2 index {key} mismatch")
    if payload.get("training_data_hash") != sha256_json(
        {key: value for key, value in payload.items() if key != "training_data_hash"}
    ):
        raise ValueError("V8.2 training_data_hash mismatch")
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
        raise ValueError("V8.2 index lacks source files")
    paths: dict[str, Path] = {}
    for name in required:
        identity = sources[name]
        path = Path(str(identity["path"])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"V8.2 source identity mismatch: {name}")
        paths[name] = path
    validate_v4_index_dataset(_load_object(paths["v4_training_index"]), dataset_dir)
    rows = load_jsonl(paths["action_phase_manifest"])
    manifest_identity = payload["action_phase_manifest_identity"]
    if (
        len(rows) != int(manifest_identity["count"])
        or sha256_json(rows) != manifest_identity["content_sha256"]
        or _load_object(paths["filter_summary"]) != payload["summary"]
    ):
        raise ValueError("V8.2 action manifest/summary identity mismatch")
    lookup: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    for row in rows:
        index = int(row["global_index"])
        if index in seen:
            raise ValueError(f"Duplicate V8.2 action global index {index}")
        seen.add(index)
        if row.get("terminal_hold_from_offset") is not None:
            raise ValueError("V8.2 must not use terminal hold")
        offsets = row.get("action_target_offsets")
        if offsets is not None and (
            len(offsets) != 30
            or any(a >= b for a, b in zip(offsets, offsets[1:], strict=False))
        ):
            raise ValueError(f"Invalid V8.2 H30 target offsets at global index {index}")
        if bool(row["trainable"]):
            lookup[index] = row
    for split in SPLITS:
        expected_indices = [
            int(row["global_index"])
            for row in rows
            if row["split"] == split and bool(row["trainable"])
        ]
        if payload["splits"][split]["execution_indices"] != expected_indices:
            raise ValueError(f"V8.2 {split} execution indices mismatch")
    if not Path(index_path).is_file():
        raise FileNotFoundError(index_path)
    return rows, lookup


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": V8_2_HASH_SCHEMA,
        "files": {
            name: {
                "path": str(path.expanduser().resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(paths.items())
        },
    }
    payload["sha256"] = sha256_json(payload)
    return payload


__all__ = [
    "BOUNDARY_POLICY",
    "ROTATION_PHASE_V8_2_ADJUSTMENT",
    "V8_2_EXPERIMENT_KIND",
    "V8_2_FILTER_POLICY",
    "V8_2_TRAINING_INDEX_SCHEMA",
    "artifact_hash_payload",
    "build_boundary_artifact",
    "build_v8_2_adjustment_artifacts",
    "validate_v8_2_adjustment_training_index",
]
