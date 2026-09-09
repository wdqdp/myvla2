"""V7.2 training artifacts filtered by four manually audited motion events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.v4_data import SPLITS, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v5_adjustment_data import V2_TERMINAL_HOLD_SCHEMA
from tactile_vla.vla.v5_adjustment_data import compute_h30_content_identities
from tactile_vla.vla.v5_adjustment_data import phase_for_rexecution_frame
from tactile_vla.vla.v7_adjustment_data import _file_identity, _validate_file_identity
from tactile_vla.vla.v7_adjustment_data import build_v7_adjustment_artifacts
from tactile_vla.vla.v7_adjustment_data import native_reexecution_timing_identity
from tactile_vla.vla.v7_2_boundary_filter import DEFAULT_DETECTOR_CONFIG
from tactile_vla.vla.v7_2_boundary_filter import V7_2_FILTER_SCHEMA


ROTATION_PHASE_V7_2_ADJUSTMENT = "rotation_phase_v7_2_adjustment"
V7_2_EXPERIMENT_KIND = "phase_prompt_h30_terminal_hold_native_reexecution_event_gap_filtered"
V7_2_ACTION_PHASE_SCHEMA = "tactile_vla_v7_2_adjustment_action_manifest_v1"
V7_2_TRAINING_INDEX_SCHEMA = "tactile_vla_v7_2_adjustment_training_index_v1"
V7_2_SUMMARY_SCHEMA = "tactile_vla_v7_2_adjustment_filter_summary_v1"
V7_2_HASH_SCHEMA = "tactile_vla_v7_2_adjustment_artifact_hashes_v1"

V7_2_EXCLUSION_REASONS = {
    "post_gripper_motion_pre_arm_idle",
    "post_arm_pre_close_idle",
}


def validate_boundary_filter(
    payload: Mapping[str, Any], *, v4_index_file: Path, candidate_indices: set[int]
) -> dict[int, str]:
    if payload.get("schema_version") != V7_2_FILTER_SCHEMA:
        raise ValueError("V7.2 boundary filter schema mismatch")
    if payload.get("data_profile") != ROTATION_PHASE_V7_2_ADJUSTMENT:
        raise ValueError("V7.2 boundary filter profile mismatch")
    if payload.get("detector_config") != DEFAULT_DETECTOR_CONFIG:
        raise ValueError("V7.2 boundary filter detector config mismatch")
    source = payload.get("source_v4_index", {})
    if (
        Path(str(source.get("path", ""))).expanduser().resolve()
        != v4_index_file.expanduser().resolve()
        or source.get("sha256") != sha256_file(v4_index_file)
    ):
        raise ValueError("V7.2 boundary filter was built from another V4 index")

    reasons: dict[int, str] = {}
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("V7.2 boundary filter has no attempts")
    for attempt in attempts:
        events = attempt.get("events", {})
        gripper_stop = int(events["gripper_motion_stop"]["frame_index"])
        arm_start = int(events["arm_adjustment_start"]["frame_index"])
        arm_stop = int(events["arm_adjustment_stop"]["frame_index"])
        close_start = int(events["gripper_close_start"]["frame_index"])
        if not (gripper_stop < arm_start and arm_stop < close_start):
            raise ValueError(
                f"V7.2 boundary order failed for episode {attempt.get('episode_id')}"
            )
        for interval in attempt.get("excluded_intervals", []):
            reason = str(interval.get("reason"))
            if reason not in V7_2_EXCLUSION_REASONS:
                raise ValueError(f"Unknown V7.2 exclusion reason {reason!r}")
            for raw_index in interval.get("excluded_global_indices", []):
                global_index = int(raw_index)
                if global_index not in candidate_indices:
                    raise ValueError(
                        f"V7.2 filter excludes non-candidate index {global_index}"
                    )
                if global_index in reasons:
                    raise ValueError(f"V7.2 filter excludes index {global_index} twice")
                reasons[global_index] = reason
    summary = payload.get("summary", {})
    if int(summary.get("detected_attempt_count", -1)) != len(attempts):
        raise ValueError("V7.2 filter attempt count mismatch")
    if int(summary.get("excluded_candidate_action_start_count", -1)) != len(reasons):
        raise ValueError("V7.2 filter excluded count mismatch")
    if Counter(reasons.values()) != Counter(summary.get("excluded_counts_by_reason", {})):
        raise ValueError("V7.2 filter reason counts mismatch")
    return reasons


def build_v7_2_adjustment_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    v4_norm_stats_dir: Path,
    boundary_filter_file: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    v4_norm_stats_dir = v4_norm_stats_dir.expanduser().resolve()
    boundary_filter_file = boundary_filter_file.expanduser().resolve()
    v4_index = json.loads(v4_index_file.read_text())
    _, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    candidate_indices = {
        int(index)
        for split in SPLITS
        for index in v4_index["splits"][split]["execution_indices"]
    }
    exclusion_by_index = validate_boundary_filter(
        json.loads(boundary_filter_file.read_text()),
        v4_index_file=v4_index_file,
        candidate_indices=candidate_indices,
    )
    v7_rows, v7_index = build_v7_adjustment_artifacts(
        dataset_dir=dataset_dir,
        v4_index_file=v4_index_file,
        v4_norm_stats_dir=v4_norm_stats_dir,
    )

    action_rows: list[dict[str, Any]] = []
    for base in v7_rows:
        global_index = int(base["global_index"])
        row = base.copy()
        row.update(
            {
                "schema_version": V7_2_ACTION_PHASE_SCHEMA,
                "data_profile": ROTATION_PHASE_V7_2_ADJUSTMENT,
                "experiment_kind": V7_2_EXPERIMENT_KIND,
                "trainable": global_index not in exclusion_by_index,
                "exclusion_reason": exclusion_by_index.get(global_index),
            }
        )
        action_rows.append(row)

    split_entries: dict[str, Any] = {}
    for split in SPLITS:
        row_indices = [
            position
            for position, row in enumerate(action_rows)
            if row["split"] == split and row["trainable"]
        ]
        selected = [action_rows[position] for position in row_indices]
        original_count = len(v4_index["splits"][split]["execution_indices"])
        split_entries[split] = {
            "execution_indices": [int(row["global_index"]) for row in selected],
            "action_phase_manifest_row_indices": row_indices,
            "summary": {
                "candidate_count": original_count,
                "action_count": len(selected),
                "excluded_count": original_count - len(selected),
                "phase_counts": dict(Counter(str(row["phase"]) for row in selected)),
                "exclusion_reason_counts": dict(
                    Counter(
                        str(row["exclusion_reason"])
                        for row in action_rows
                        if row["split"] == split and not row["trainable"]
                    )
                ),
                "raw_chunk_crossing": sum(
                    not bool(row["raw_chunk_phase_pure"]) for row in selected
                ),
                "terminal_hold_chunks": sum(
                    bool(row["effective_h30_modified"]) for row in selected
                ),
            },
        }

    filtered_identity = action_indices_identity(split_entries)
    candidate_identity = v4_index["action_indices_identity"]
    selected_rows = [row for row in action_rows if row["trainable"]]
    raw_h30, effective_h30, modifications = compute_h30_content_identities(
        dataset_dir=dataset_dir,
        global_lookup=global_lookup,
        action_rows=selected_rows,
        action_indices_identity=filtered_identity,
        action_horizon=int(v4_index["action_horizon"]),
    )
    summary = {
        "schema_version": V7_2_SUMMARY_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_2_ADJUSTMENT,
        "detector_config": DEFAULT_DETECTOR_CONFIG,
        "candidate_action_count": int(candidate_identity["all"]["count"]),
        "trainable_action_count": int(filtered_identity["all"]["count"]),
        "excluded_action_count": len(exclusion_by_index),
        "exclusion_reason_counts": dict(Counter(exclusion_by_index.values())),
        "splits": {split: split_entries[split]["summary"] for split in SPLITS},
    }
    filter_identity = {
        "schema_version": V7_2_FILTER_SCHEMA,
        "sha256": sha256_file(boundary_filter_file),
        "detector_config": DEFAULT_DETECTOR_CONFIG,
        "excluded_action_count": len(exclusion_by_index),
        "excluded_indices_sha256": sha256_json(sorted(exclusion_by_index)),
        "manual_audit": "passed",
    }
    index: dict[str, Any] = {
        "schema_version": V7_2_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_2_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V7_2_EXPERIMENT_KIND,
        "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
        "detector_config": DEFAULT_DETECTOR_CONFIG,
        "boundary_filter_identity": filter_identity,
        "data_config_hash": sha256_json(
            {
                "v4_training_data_hash": v4_index["training_data_hash"],
                "native_reexecution_timing_identity": v7_index[
                    "native_reexecution_timing_identity"
                ],
                "boundary_filter_identity": filter_identity,
                "experiment_kind": V7_2_EXPERIMENT_KIND,
                "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
            }
        ),
        "selection_hash": v4_index["selection_hash"],
        "v4_profile_config_hash": v4_index["profile_config_hash"],
        "dataset_dir": str(dataset_dir),
        "action_horizon": int(v4_index["action_horizon"]),
        "splits": split_entries,
        "action_indices_identity": filtered_identity,
        "candidate_action_indices_identity": candidate_identity,
        "native_reexecution_timing_identity": v7_index[
            "native_reexecution_timing_identity"
        ],
        "h30_target_identity": effective_h30,
        "raw_h30_target_identity": raw_h30,
        "v4_h30_source_identity": v7_index["v4_h30_source_identity"],
        "v4_lerobot_identity": v4_index["lerobot_identity"],
        "v4_training_data_hash": v4_index["training_data_hash"],
        "v4_norm_stats_sha256": v7_index["v4_norm_stats_sha256"],
        "norm_stats_policy": {
            "kind": "inherited",
            "source_data_profile": "rotation_v4",
            "source_action_indices_identity": candidate_identity,
            "note": "V7.2 filters action starts and intentionally reuses V4 normalization statistics",
        },
        "h30_modifications": modifications,
        "summary": summary,
        "source_files": {
            "v4_training_index": _file_identity(v4_index_file),
            "v4_norm_summary": _file_identity(v4_norm_stats_dir / "summary.json"),
            "v4_norm_stats": _file_identity(v4_norm_stats_dir / "norm_stats.json"),
            "boundary_filter": _file_identity(boundary_filter_file),
        },
        "action_phase_manifest_identity": {
            "count": len(action_rows),
            "content_sha256": sha256_json(action_rows),
        },
    }
    return action_rows, index, summary


def validate_v7_2_adjustment_training_index(
    payload: Mapping[str, Any],
    *,
    index_path: Path | None = None,
    dataset_dir: Path | None = None,
    revalidate_h30_targets: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    expected = {
        "schema_version": V7_2_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_2_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V7_2_EXPERIMENT_KIND,
        "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
        "detector_config": DEFAULT_DETECTOR_CONFIG,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V7.2 index {key} mismatch")
    if payload.get("training_data_hash") != sha256_json(
        {key: value for key, value in payload.items() if key != "training_data_hash"}
    ):
        raise ValueError("V7.2 training_data_hash mismatch")
    if index_path is not None and not Path(index_path).is_file():
        raise FileNotFoundError(index_path)
    sources = payload.get("source_files", {})
    required = {
        "v4_training_index",
        "v4_norm_summary",
        "v4_norm_stats",
        "boundary_filter",
        "action_phase_manifest",
        "filter_summary",
    }
    if not isinstance(sources, Mapping) or not required.issubset(sources):
        raise ValueError("V7.2 index lacks source identities")
    paths = {
        name: _validate_file_identity(sources[name], context=name) for name in required
    }
    v4_index = json.loads(paths["v4_training_index"].read_text())
    effective_dataset_dir = (
        dataset_dir or Path(str(payload["dataset_dir"]))
    ).expanduser().resolve()
    frames, global_lookup = validate_v4_index_dataset(v4_index, effective_dataset_dir)
    timing, boundaries = native_reexecution_timing_identity(v4_index, frames)
    if timing != payload.get("native_reexecution_timing_identity"):
        raise ValueError("V7.2 native timing identity mismatch")
    candidate_indices = {
        int(index)
        for split in SPLITS
        for index in v4_index["splits"][split]["execution_indices"]
    }
    exclusion_by_index = validate_boundary_filter(
        json.loads(paths["boundary_filter"].read_text()),
        v4_index_file=paths["v4_training_index"],
        candidate_indices=candidate_indices,
    )
    filter_identity = payload.get("boundary_filter_identity", {})
    if (
        filter_identity.get("sha256") != sha256_file(paths["boundary_filter"])
        or filter_identity.get("manual_audit") != "passed"
        or int(filter_identity.get("excluded_action_count", -1))
        != len(exclusion_by_index)
    ):
        raise ValueError("V7.2 boundary filter identity mismatch")
    if payload.get("candidate_action_indices_identity") != v4_index.get(
        "action_indices_identity"
    ):
        raise ValueError("V7.2 candidate action identity differs from V4")
    if payload.get("norm_stats_policy", {}).get("kind") != "inherited":
        raise ValueError("V7.2 must identify V4 norm stats as inherited")

    rows = load_jsonl(paths["action_phase_manifest"])
    manifest_identity = payload["action_phase_manifest_identity"]
    if (
        len(rows) != int(manifest_identity["count"])
        or sha256_json(rows) != manifest_identity["content_sha256"]
        or sha256_file(paths["action_phase_manifest"])
        != manifest_identity["file_sha256"]
    ):
        raise ValueError("V7.2 action manifest identity mismatch")
    lookup: dict[int, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        global_index = int(row["global_index"])
        frame = global_lookup[global_index]
        expected_reason = exclusion_by_index.get(global_index)
        if row.get("data_profile") != ROTATION_PHASE_V7_2_ADJUSTMENT:
            raise ValueError(f"Invalid V7.2 profile at {global_index}")
        if row.get("phase") != phase_for_rexecution_frame(
            frame.attempt_id, frame.frame_index, boundaries[frame.attempt_key]
        ):
            raise ValueError(f"Invalid V7.2 phase at {global_index}")
        if row.get("exclusion_reason") != expected_reason or bool(
            row.get("trainable")
        ) != (expected_reason is None):
            raise ValueError(f"Invalid V7.2 exclusion at {global_index}")
        if expected_reason is None:
            lookup[global_index] = row
            selected_rows.append(row)
    for split in SPLITS:
        expected_indices = [
            int(row["global_index"])
            for row in rows
            if row["split"] == split and row["trainable"]
        ]
        if payload["splits"][split]["execution_indices"] != expected_indices:
            raise ValueError(f"V7.2 {split} loader indices include excluded samples")
        original_episodes = {
            global_lookup[int(index)].episode_id
            for index in v4_index["splits"][split]["execution_indices"]
        }
        filtered_episodes = {
            global_lookup[index].episode_id for index in expected_indices
        }
        if filtered_episodes != original_episodes:
            raise ValueError(f"V7.2 filtering changed {split} episode coverage")
    if action_indices_identity(payload["splits"]) != payload.get(
        "action_indices_identity"
    ):
        raise ValueError("V7.2 filtered action identity mismatch")
    summary = json.loads(paths["filter_summary"].read_text())
    if summary != payload.get("summary"):
        raise ValueError("V7.2 filter summary mismatch")
    norm_summary = json.loads(paths["v4_norm_summary"].read_text())
    norm_sha = sha256_file(paths["v4_norm_stats"])
    if (
        norm_sha != payload.get("v4_norm_stats_sha256")
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity")
        != payload.get("candidate_action_indices_identity")
    ):
        raise ValueError("V7.2 inherited norm identity mismatch")
    if revalidate_h30_targets:
        raw, effective, modifications = compute_h30_content_identities(
            dataset_dir=effective_dataset_dir,
            global_lookup=global_lookup,
            action_rows=selected_rows,
            action_indices_identity=payload["action_indices_identity"],
            action_horizon=int(payload["action_horizon"]),
        )
        if (
            raw != payload.get("raw_h30_target_identity")
            or effective != payload.get("h30_target_identity")
            or modifications != payload.get("h30_modifications")
        ):
            raise ValueError("V7.2 persisted H30 identities mismatch")
    return rows, lookup


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    result = {
        "schema_version": V7_2_HASH_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_2_ADJUSTMENT,
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }
    result["sha256"] = sha256_json(result)
    return result
