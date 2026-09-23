"""V8.3 short-small-grasp H30 targets with temporally resampled descent."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.v4_data import SPLITS, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v5_adjustment_data import _load_lerobot_actions
from tactile_vla.vla.v8_1_adjustment_data import SMALL_GRASP_TASK, _load_object
from tactile_vla.vla.v8_2_adjustment_data import (
    BOUNDARY_POLICY,
    V8_2_FILTER_POLICY,
    build_v8_2_adjustment_artifacts,
    validate_boundary_artifact,
)


ROTATION_PHASE_V8_3_ADJUSTMENT = "rotation_phase_v8_3_adjustment"
V8_3_EXPERIMENT_KIND = "phase_prompt_h30_short_small_grasp_descent_time_warp"
V8_3_ACTION_PHASE_SCHEMA = "tactile_vla_v8_3_adjustment_action_manifest_v1"
V8_3_TRAINING_INDEX_SCHEMA = "tactile_vla_v8_3_adjustment_training_index_v1"
V8_3_SUMMARY_SCHEMA = "tactile_vla_v8_3_adjustment_filter_summary_v1"
V8_3_HASH_SCHEMA = "tactile_vla_v8_3_adjustment_artifact_hashes_v1"
V8_3_TARGET_IDENTITY_SCHEMA = "tactile_vla_v8_3_h30_targets_v1"

DOWN_MODERATELY_PLAN = (
    "recovery_plan=move horizontally none moderately, move vertically down moderately."
)
V8_3_TARGET_POLICY = {
    "base_policy": "v8_2_v7_4_idle_gap_compression",
    "scope": "attempt2_small_grasp_adjustment_down_moderately_short_descent",
    "short_descent_condition": "arm_adjustment_stop-arm_adjustment_start+1 < 20",
    "focused_start_window": "gripper_motion_stop_minus_9_through_gripper_motion_stop",
    "focused_start_count": 10,
    "preserved_prefix": "start_through_gripper_motion_stop_inclusive",
    "descent_target": "linear_time_resample_recorded_7d_action_A_through_S",
    "resample_length": "30 - preserved_prefix_length",
    "resample_endpoint": "exact_arm_adjustment_stop_action",
    "post_stop_static": "disabled",
    "requires": "arm_adjustment_stop < rexecution_frame",
}


def _event_frames(boundary_payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, int]]:
    result: dict[tuple[int, int], dict[str, int]] = {}
    for attempt in boundary_payload["attempts"]:
        key = (int(attempt["episode_id"]), int(attempt["attempt_id"]))
        events = attempt["events"]
        result[key] = {
            "g": int(events["gripper_motion_stop"]["frame_index"]),
            "a": int(events["arm_adjustment_start"]["frame_index"]),
            "s": int(events["arm_adjustment_stop"]["frame_index"]),
            "r": int(attempt["rexecution_frame"]),
            "task": str(attempt["task"]),
        }
    return result


def _resample_actions(actions: np.ndarray, count: int) -> np.ndarray:
    """Linearly resample a recorded action path while retaining both endpoints."""

    source = np.asarray(actions, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 7 or len(source) < 2:
        raise ValueError(f"Expected a nontrivial [T,7] action path, got {source.shape}")
    if count < 2:
        raise ValueError(f"V8.3 descent resampling requires at least two outputs, got {count}")
    source_time = np.arange(len(source), dtype=np.float64)
    target_time = np.linspace(0.0, float(len(source) - 1), count, dtype=np.float64)
    result = np.stack(
        [np.interp(target_time, source_time, source[:, axis]) for axis in range(source.shape[1])],
        axis=1,
    ).astype(np.float32)
    result[0] = source[0]
    result[-1] = source[-1]
    return result


def apply_v8_3_short_descent_targets(
    rows: Sequence[Mapping[str, Any]],
    *,
    boundary_payload: Mapping[str, Any],
    actions: Sequence[np.ndarray],
    global_lookup: Mapping[int, Any],
    action_horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace the ten G-adjacent H30s for each short vertical descent."""

    if action_horizon != 30:
        raise ValueError(f"V8.3 requires H30, got H{action_horizon}")
    events_by_attempt = _event_frames(boundary_payload)
    transformed = [dict(row) for row in rows]
    by_identity = {
        (int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"])): row
        for row in transformed
    }
    modified_by_split: Counter[str] = Counter()
    modified_by_attempt: Counter[str] = Counter()
    reenabled_by_reason: Counter[str] = Counter()
    audited: list[dict[str, Any]] = []

    for key, event in sorted(events_by_attempt.items()):
        if event["task"] != SMALL_GRASP_TASK:
            continue
        g, a, s, r = (event[name] for name in ("g", "a", "s", "r"))
        descent_length = s - a + 1
        if descent_length >= 20 or s >= r:
            continue
        focused = list(range(g - 9, g + 1))
        if focused[0] < 0:
            raise ValueError(f"V8.3 focused start precedes attempt start for {key}")
        changed = 0
        for start in focused:
            row = by_identity.get((key[0], key[1], start))
            if row is None:
                raise ValueError(f"V8.3 lacks candidate row for {key} frame={start}")
            if row["phase"] != "adjustment" or row["task"] != SMALL_GRASP_TASK:
                raise ValueError(f"V8.3 focused row has invalid phase/task for {key} frame={start}")
            frame = global_lookup[int(row["global_index"])]
            if frame.input_recovery_plan != DOWN_MODERATELY_PLAN:
                raise ValueError(
                    "V8.3 short-descent target requires down-moderately recovery plan: "
                    f"attempt={key}, got={frame.input_recovery_plan!r}"
                )
            prefix_length = g - start + 1
            descent_count = action_horizon - prefix_length
            source_indices = [int(row["global_index"]) + offset for offset in range(prefix_length)]
            prefix = np.asarray([actions[index] for index in source_indices], dtype=np.float32)
            descent_indices = [
                int(row["global_index"]) + (a - start) + offset
                for offset in range(descent_length)
            ]
            descent = _resample_actions(
                np.asarray([actions[index] for index in descent_indices], dtype=np.float32),
                descent_count,
            )
            target = np.concatenate((prefix, descent), axis=0)
            if target.shape != (action_horizon, 7) or not np.isfinite(target).all():
                raise AssertionError(f"Invalid V8.3 target for {key} frame={start}")
            if row.get("exclusion_reason") is not None:
                reenabled_by_reason[str(row["exclusion_reason"])] += 1
            row.update(
                {
                    "trainable": True,
                    "exclusion_reason": None,
                    "action_target_offsets": None,
                    "action_target_values": target.tolist(),
                    "action_target_kind": "short_small_grasp_descent_time_warp",
                    "effective_h30_modified": True,
                    "effective_chunk_phase_pure": True,
                    "chunk_phase_pure": True,
                    "idle_gap_compression": {
                        "gripper_motion_stop_frame": g,
                        "arm_adjustment_start_frame": a,
                        "arm_adjustment_stop_frame": s,
                        "preserved_prefix_length": prefix_length,
                        "resampled_descent_length": descent_count,
                        "source_descent_length": descent_length,
                    },
                }
            )
            modified_by_split[str(row["split"])] += 1
            modified_by_attempt[f"{key[0]}:{key[1]}"] += 1
            changed += 1
        if changed != 10:
            raise AssertionError(f"V8.3 expected ten focused starts for {key}, got {changed}")
        audited.append(
            {
                "episode_id": key[0],
                "attempt_id": key[1],
                "gripper_motion_stop": g,
                "arm_adjustment_start": a,
                "arm_adjustment_stop": s,
                "rexecution_frame": r,
                "source_descent_length": descent_length,
                "focused_start_frames": focused,
            }
        )
    summary = {
        "modified_chunk_count": sum(modified_by_split.values()),
        "modified_chunks_by_split": dict(sorted(modified_by_split.items())),
        "modified_attempt_count": len(modified_by_attempt),
        "modified_chunks_by_attempt": dict(sorted(modified_by_attempt.items())),
        "reenabled_chunks_by_previous_exclusion_reason": dict(sorted(reenabled_by_reason.items())),
        "attempts": audited,
    }
    return transformed, summary


def _target_identity(rows: Sequence[Mapping[str, Any]], *, action_horizon: int) -> dict[str, Any]:
    targets = []
    for row in rows:
        if not bool(row["trainable"]):
            continue
        values = row.get("action_target_values")
        targets.append(
            {
                "global_index": int(row["global_index"]),
                "kind": row.get("action_target_kind", "raw_or_offset"),
                "offsets": row.get("action_target_offsets"),
                "values": values,
            }
        )
    payload = {
        "schema_version": V8_3_TARGET_IDENTITY_SCHEMA,
        "action_horizon": action_horizon,
        "target_rows_sha256": sha256_json(targets),
        "interpolated_target_count": sum(
            row.get("action_target_values") is not None and bool(row["trainable"])
            for row in rows
        ),
    }
    payload["sha256"] = sha256_json(payload)
    return payload


def build_v8_3_adjustment_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    v4_norm_stats_dir: Path,
    timing_file: Path,
    boundary_file: Path,
    reference_v7_4_index_file: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build V8.2 rows, then replace only the focused short-descent H30s."""

    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    rows, base_index, base_summary = build_v8_2_adjustment_artifacts(
        dataset_dir=dataset_dir,
        v4_index_file=v4_index_file,
        v4_norm_stats_dir=v4_norm_stats_dir,
        timing_file=timing_file,
        boundary_file=boundary_file,
        reference_v7_4_index_file=reference_v7_4_index_file,
    )
    v4_index = _load_object(v4_index_file)
    _, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    boundary_payload = _load_object(boundary_file.expanduser().resolve())
    validate_boundary_artifact(
        boundary_payload,
        v4_index_file=v4_index_file,
        timing_payload=_load_object(timing_file.expanduser().resolve()),
    )
    actions = _load_lerobot_actions(dataset_dir, len(global_lookup))
    rows, interpolation_summary = apply_v8_3_short_descent_targets(
        rows,
        boundary_payload=boundary_payload,
        actions=actions,
        global_lookup=global_lookup,
        action_horizon=int(v4_index["action_horizon"]),
    )
    for row in rows:
        row.update(
            {
                "schema_version": V8_3_ACTION_PHASE_SCHEMA,
                "data_profile": ROTATION_PHASE_V8_3_ADJUSTMENT,
                "prompt_profile": PHASE_PROMPT_PROFILE_V2,
                "experiment_kind": V8_3_EXPERIMENT_KIND,
                "terminal_hold_from_offset": None,
            }
        )

    split_entries: dict[str, Any] = {}
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        selected = [row for row in split_rows if bool(row["trainable"])]
        split_entries[split] = {
            "execution_indices": [int(row["global_index"]) for row in selected],
            "action_phase_manifest_row_indices": [
                index
                for index, row in enumerate(rows)
                if row["split"] == split and bool(row["trainable"])
            ],
            "summary": {
                "candidate_count": len(split_rows),
                "action_count": len(selected),
                "excluded_count": len(split_rows) - len(selected),
                "phase_counts": dict(Counter(str(row["phase"]) for row in selected)),
                "task_counts": dict(Counter(str(row["task"]) for row in selected)),
                "task_phase_counts": dict(
                    Counter(f"{row['task']}:{row['phase']}" for row in selected)
                ),
                "exclusion_reason_counts": dict(
                    Counter(
                        str(row["exclusion_reason"])
                        for row in split_rows
                        if not row["trainable"]
                    )
                ),
                "raw_chunk_crossing": sum(
                    not bool(row["raw_chunk_phase_pure"]) for row in selected
                ),
                "terminal_hold_chunks": 0,
                "idle_compressed_h30_chunks": sum(
                    row.get("action_target_offsets") is not None for row in selected
                ),
                "interpolated_descent_h30_chunks": sum(
                    row.get("action_target_values") is not None for row in selected
                ),
            },
        }
    selected_identity = action_indices_identity(split_entries)
    target_identity = _target_identity(rows, action_horizon=int(v4_index["action_horizon"]))
    summary = {
        **base_summary,
        "schema_version": V8_3_SUMMARY_SCHEMA,
        "data_profile": ROTATION_PHASE_V8_3_ADJUSTMENT,
        "target_policy": V8_3_TARGET_POLICY,
        "trainable_action_count": int(selected_identity["all"]["count"]),
        "excluded_action_count": int(v4_index["action_indices_identity"]["all"]["count"])
        - int(selected_identity["all"]["count"]),
        "h30_modifications": {
            "v8_2": base_summary["h30_modifications"],
            "v8_3_short_descent_time_warp": interpolation_summary,
        },
        "splits": {split: split_entries[split]["summary"] for split in SPLITS},
    }
    index = {
        **base_index,
        "schema_version": V8_3_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V8_3_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V8_3_EXPERIMENT_KIND,
        "filter_policy": V8_2_FILTER_POLICY,
        "target_policy": V8_3_TARGET_POLICY,
        "splits": split_entries,
        "action_indices_identity": selected_identity,
        "h30_target_identity": target_identity,
        "raw_h30_target_identity": {
            "schema_version": "tactile_vla_v8_3_raw_source_h30_v1",
            "source_v8_2_identity": base_index["raw_h30_target_identity"],
            "sha256": sha256_json(base_index["raw_h30_target_identity"]),
        },
        "h30_modifications": summary["h30_modifications"],
        "summary": summary,
        "action_phase_manifest_identity": {
            "count": len(rows),
            "content_sha256": sha256_json(rows),
        },
    }
    index["data_config_hash"] = sha256_json(
        {
            "v8_2_data_config_hash": base_index["data_config_hash"],
            "target_policy": V8_3_TARGET_POLICY,
            "target_identity": target_identity,
        }
    )
    return rows, index, summary


def validate_v8_3_adjustment_training_index(
    payload: Mapping[str, Any], *, index_path: Path, dataset_dir: Path
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    expected = {
        "schema_version": V8_3_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V8_3_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V8_3_EXPERIMENT_KIND,
        "boundary_policy": BOUNDARY_POLICY,
        "filter_policy": V8_2_FILTER_POLICY,
        "target_policy": V8_3_TARGET_POLICY,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V8.3 index {key} mismatch")
    if payload.get("training_data_hash") != sha256_json(
        {key: value for key, value in payload.items() if key != "training_data_hash"}
    ):
        raise ValueError("V8.3 training_data_hash mismatch")
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
        raise ValueError("V8.3 index lacks source files")
    paths: dict[str, Path] = {}
    for name in required:
        identity = sources[name]
        path = Path(str(identity["path"])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"V8.3 source identity mismatch: {name}")
        paths[name] = path
    validate_v4_index_dataset(_load_object(paths["v4_training_index"]), dataset_dir)
    rows = load_jsonl(paths["action_phase_manifest"])
    if (
        len(rows) != int(payload["action_phase_manifest_identity"]["count"])
        or sha256_json(rows) != payload["action_phase_manifest_identity"]["content_sha256"]
        or _load_object(paths["filter_summary"]) != payload["summary"]
    ):
        raise ValueError("V8.3 action manifest/summary identity mismatch")
    lookup: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["global_index"])
        if index in lookup:
            raise ValueError(f"Duplicate V8.3 action global index {index}")
        values = row.get("action_target_values")
        offsets = row.get("action_target_offsets")
        if values is not None:
            target = np.asarray(values, dtype=np.float32)
            if (
                target.shape != (30, 7)
                or not np.isfinite(target).all()
                or offsets is not None
                or row.get("action_target_kind") != "short_small_grasp_descent_time_warp"
            ):
                raise ValueError(f"Invalid V8.3 interpolated H30 at global index {index}")
        elif offsets is not None and (
            len(offsets) != 30 or any(a >= b for a, b in zip(offsets, offsets[1:], strict=False))
        ):
            raise ValueError(f"Invalid V8.3 H30 offsets at global index {index}")
        if row.get("terminal_hold_from_offset") is not None:
            raise ValueError("V8.3 must not use terminal hold")
        if bool(row["trainable"]):
            lookup[index] = row
    for split in SPLITS:
        expected_indices = [
            int(row["global_index"])
            for row in rows
            if row["split"] == split and bool(row["trainable"])
        ]
        if payload["splits"][split]["execution_indices"] != expected_indices:
            raise ValueError(f"V8.3 {split} execution indices mismatch")
    if not Path(index_path).is_file():
        raise FileNotFoundError(index_path)
    return rows, lookup


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    payload = {
        "schema_version": V8_3_HASH_SCHEMA,
        "files": {
            name: {"path": str(path.expanduser().resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }
    payload["sha256"] = sha256_json(payload)
    return payload


__all__ = [
    "ROTATION_PHASE_V8_3_ADJUSTMENT",
    "V8_3_EXPERIMENT_KIND",
    "V8_3_TRAINING_INDEX_SCHEMA",
    "apply_v8_3_short_descent_targets",
    "artifact_hash_payload",
    "build_v8_3_adjustment_artifacts",
    "validate_v8_3_adjustment_training_index",
]
