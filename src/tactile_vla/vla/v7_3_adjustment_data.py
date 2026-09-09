"""V7.3 phase-pure H30 artifacts with adaptive adjustment-stop filtering."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.v4_data import SPLITS, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v5_adjustment_data import V2_H30_CONTENT_SCHEMA
from tactile_vla.vla.v5_adjustment_data import _load_lerobot_actions
from tactile_vla.vla.v5_adjustment_data import phase_for_rexecution_frame
from tactile_vla.vla.v7_adjustment_data import _file_identity, _validate_file_identity
from tactile_vla.vla.v7_adjustment_data import build_v7_adjustment_artifacts
from tactile_vla.vla.v7_adjustment_data import native_reexecution_timing_identity
from tactile_vla.vla.v7_2_adjustment_data import validate_boundary_filter
from tactile_vla.vla.v7_2_boundary_filter import DEFAULT_DETECTOR_CONFIG
from tactile_vla.vla.v7_2_boundary_filter import V7_2_FILTER_SCHEMA


ROTATION_PHASE_V7_3_ADJUSTMENT = "rotation_phase_v7_3_adjustment"
V7_3_EXPERIMENT_KIND = "phase_prompt_h30_native_reexecution_event_gap_adaptive_stop_filtered"
V7_3_ACTION_PHASE_SCHEMA = "tactile_vla_v7_3_adjustment_action_manifest_v1"
V7_3_TRAINING_INDEX_SCHEMA = "tactile_vla_v7_3_adjustment_training_index_v1"
V7_3_SUMMARY_SCHEMA = "tactile_vla_v7_3_adjustment_filter_summary_v1"
V7_3_HASH_SCHEMA = "tactile_vla_v7_3_adjustment_artifact_hashes_v1"

STOP_H30_REASON = "pre_arm_adjustment_stop_h30"
CROSS_PHASE_REASON = "crosses_reexecution_boundary"
V7_3_FILTER_POLICY = {
    "scope": "attempt2_adjustment_action_starts_only",
    "action_horizon": 30,
    "full_stop_window_start_offset": -29,
    "short_adjustment_condition": "arm_adjustment_start >= arm_adjustment_stop - 29",
    "max_post_stop_frames_for_retained_short_adjustment_h30": 10,
    "short_stop_window_start_offset": -18,
    "terminal_hold": "disabled",
    "residual_cross_reexecution": "exclude",
}


def compute_raw_h30_content_identity(
    *,
    dataset_dir: Path,
    global_lookup: Mapping[int, Any],
    action_rows: list[dict[str, Any]],
    action_indices_identity: Mapping[str, Any],
    action_horizon: int,
) -> dict[str, Any]:
    """Hash unmodified H30 targets; V7.3 has no effective/held target."""

    actions = _load_lerobot_actions(dataset_dir, len(global_lookup))
    digest = hashlib.sha256()
    digest.update(V2_H30_CONTENT_SCHEMA.encode())
    action_dim: int | None = None
    for row in action_rows:
        global_index = int(row["global_index"])
        frame = global_lookup[global_index]
        chunk_values: list[np.ndarray] = []
        for offset in range(action_horizon):
            target_index = global_index + offset
            target_frame = global_lookup.get(target_index)
            if target_frame is None or target_frame.attempt_key != frame.attempt_key:
                raise ValueError(f"H30 chunk crosses an attempt at global index {global_index}")
            if target_frame.frame_index != frame.frame_index + offset:
                raise ValueError(f"H30 frame identity is discontinuous at global index {global_index}")
            chunk_values.append(actions[target_index])
        chunk = np.ascontiguousarray(np.stack(chunk_values), dtype="<f4")
        action_dim = chunk.shape[1] if action_dim is None else action_dim
        if chunk.shape != (action_horizon, action_dim):
            raise ValueError(f"Inconsistent H30 shape at global index {global_index}")
        digest.update(global_index.to_bytes(8, "little", signed=False))
        digest.update(chunk.tobytes(order="C"))
    if action_dim is None:
        raise ValueError("Cannot hash an empty V7.3 H30 manifest")
    identity = {
        "schema_version": V2_H30_CONTENT_SCHEMA,
        "dtype": "float32_le",
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "chunk_count": len(action_rows),
        "execution_indices": action_indices_identity,
        "content_sha256": digest.hexdigest(),
    }
    identity["sha256"] = sha256_json(identity)
    return identity


def _attempt_events(boundary_payload: Mapping[str, Any]) -> dict[tuple[int, int], tuple[int, int]]:
    result: dict[tuple[int, int], tuple[int, int]] = {}
    for attempt in boundary_payload["attempts"]:
        key = (int(attempt["episode_id"]), int(attempt["attempt_id"]))
        events = attempt["events"]
        result[key] = (
            int(events["arm_adjustment_start"]["frame_index"]),
            int(events["arm_adjustment_stop"]["frame_index"]),
        )
    return result


def build_v7_3_exclusions(
    rows: list[dict[str, Any]],
    *,
    v7_2_exclusions: Mapping[int, str],
    boundary_payload: Mapping[str, Any],
    action_horizon: int,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Add adaptive pre-stop and residual cross-phase exclusions to V7.2."""

    if action_horizon != int(V7_3_FILTER_POLICY["action_horizon"]):
        raise ValueError(f"V7.3 requires H30, got H{action_horizon}")
    exclusions = {int(index): str(reason) for index, reason in v7_2_exclusions.items()}
    events_by_attempt = _attempt_events(boundary_payload)
    rows_by_attempt: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["episode_id"]), int(row["attempt_id"]))
        rows_by_attempt.setdefault(key, []).append(row)

    policies: list[dict[str, Any]] = []
    for key, (arm_start, arm_stop) in sorted(events_by_attempt.items()):
        if key[1] != 2 or arm_start > arm_stop:
            raise ValueError(f"Invalid V7.3 adjustment events for attempt {key}")
        full_filter_start = arm_stop - (action_horizon - 1)
        short_adjustment = arm_start >= full_filter_start
        max_post_stop = int(
            V7_3_FILTER_POLICY["max_post_stop_frames_for_retained_short_adjustment_h30"]
        )
        last_retained_start = arm_stop - (action_horizon - 1) + max_post_stop
        if short_adjustment:
            if arm_start > last_retained_start:
                raise ValueError(
                    "V7.3 cannot preserve an adjustment start while limiting post-stop "
                    f"H30 frames to {max_post_stop}: episode={key[0]} "
                    f"arm_start={arm_start} arm_stop={arm_stop}"
                )
            filter_start = last_retained_start + 1
            mode = "short_adjustment_preserve_prefix"
        else:
            filter_start = full_filter_start
            mode = "full_pre_stop_h30"

        excluded_here = 0
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
        if short_adjustment and retained_motion_starts == 0:
            raise ValueError(f"V7.3 removed every adjustment-motion start for episode {key[0]}")
        if short_adjustment and max_retained_post_stop > max_post_stop:
            raise AssertionError("V7.3 retained a short-adjustment H30 beyond its stop limit")
        policies.append(
            {
                "episode_id": key[0],
                "attempt_id": key[1],
                "arm_adjustment_start": arm_start,
                "arm_adjustment_stop": arm_stop,
                "mode": mode,
                "filter_start": filter_start,
                "filter_end": arm_stop,
                "retained_motion_start_count": retained_motion_starts,
                "max_retained_post_stop_frames": max_retained_post_stop,
                "newly_excluded_start_count": excluded_here,
            }
        )

    # Terminal hold is disabled. Any adjustment H30 that still crosses the
    # native re-execution boundary must therefore be removed, never rewritten.
    for row in rows:
        global_index = int(row["global_index"])
        if (
            global_index not in exclusions
            and row["phase"] == "adjustment"
            and not bool(row["raw_chunk_phase_pure"])
        ):
            exclusions[global_index] = CROSS_PHASE_REASON
    return exclusions, policies


def build_v7_3_adjustment_artifacts(
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
    boundary_payload = json.loads(boundary_filter_file.read_text())
    v7_2_exclusions = validate_boundary_filter(
        boundary_payload,
        v4_index_file=v4_index_file,
        candidate_indices=candidate_indices,
    )
    v7_rows, v7_index = build_v7_adjustment_artifacts(
        dataset_dir=dataset_dir,
        v4_index_file=v4_index_file,
        v4_norm_stats_dir=v4_norm_stats_dir,
    )
    exclusions, attempt_policies = build_v7_3_exclusions(
        v7_rows,
        v7_2_exclusions=v7_2_exclusions,
        boundary_payload=boundary_payload,
        action_horizon=int(v4_index["action_horizon"]),
    )

    action_rows: list[dict[str, Any]] = []
    for base in v7_rows:
        global_index = int(base["global_index"])
        row = base.copy()
        row.update(
            {
                "schema_version": V7_3_ACTION_PHASE_SCHEMA,
                "data_profile": ROTATION_PHASE_V7_3_ADJUSTMENT,
                "experiment_kind": V7_3_EXPERIMENT_KIND,
                "trainable": global_index not in exclusions,
                "exclusion_reason": exclusions.get(global_index),
                "terminal_hold_from_offset": None,
                "effective_h30_modified": False,
                "effective_chunk_phase_pure": bool(base["raw_chunk_phase_pure"]),
                "chunk_phase_pure": bool(base["raw_chunk_phase_pure"]),
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
        split_rows = [row for row in action_rows if row["split"] == split]
        split_entries[split] = {
            "execution_indices": [int(row["global_index"]) for row in selected],
            "action_phase_manifest_row_indices": row_indices,
            "summary": {
                "candidate_count": len(v4_index["splits"][split]["execution_indices"]),
                "action_count": len(selected),
                "excluded_count": sum(not bool(row["trainable"]) for row in split_rows),
                "phase_counts": dict(Counter(str(row["phase"]) for row in selected)),
                "exclusion_reason_counts": dict(
                    Counter(str(row["exclusion_reason"]) for row in split_rows if not row["trainable"])
                ),
                "raw_chunk_crossing": sum(not bool(row["raw_chunk_phase_pure"]) for row in selected),
                "terminal_hold_chunks": 0,
            },
        }

    filtered_identity = action_indices_identity(split_entries)
    candidate_identity = v4_index["action_indices_identity"]
    selected_rows = [row for row in action_rows if row["trainable"]]
    raw_h30 = compute_raw_h30_content_identity(
        dataset_dir=dataset_dir,
        global_lookup=global_lookup,
        action_rows=selected_rows,
        action_indices_identity=filtered_identity,
        action_horizon=int(v4_index["action_horizon"]),
    )
    modifications = {
        "modified_chunk_count": 0,
        "modified_action_step_count": 0,
    }
    if any(not bool(row["raw_chunk_phase_pure"]) for row in selected_rows):
        raise AssertionError("V7.3 retained a cross-phase H30")

    filter_identity = {
        "schema_version": V7_2_FILTER_SCHEMA,
        "sha256": sha256_file(boundary_filter_file),
        "detector_config": DEFAULT_DETECTOR_CONFIG,
        "excluded_action_count": len(v7_2_exclusions),
        "excluded_indices_sha256": sha256_json(sorted(v7_2_exclusions)),
        "manual_audit": "passed",
    }
    summary = {
        "schema_version": V7_3_SUMMARY_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_3_ADJUSTMENT,
        "filter_policy": V7_3_FILTER_POLICY,
        "candidate_action_count": int(candidate_identity["all"]["count"]),
        "trainable_action_count": int(filtered_identity["all"]["count"]),
        "excluded_action_count": len(exclusions),
        "exclusion_reason_counts": dict(Counter(exclusions.values())),
        "attempt_filter_policies": attempt_policies,
        "splits": {split: split_entries[split]["summary"] for split in SPLITS},
    }
    index: dict[str, Any] = {
        "schema_version": V7_3_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_3_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V7_3_EXPERIMENT_KIND,
        "detector_config": DEFAULT_DETECTOR_CONFIG,
        "filter_policy": V7_3_FILTER_POLICY,
        "boundary_filter_identity": filter_identity,
        "data_config_hash": sha256_json(
            {
                "v4_training_data_hash": v4_index["training_data_hash"],
                "native_reexecution_timing_identity": v7_index["native_reexecution_timing_identity"],
                "boundary_filter_identity": filter_identity,
                "filter_policy": V7_3_FILTER_POLICY,
                "experiment_kind": V7_3_EXPERIMENT_KIND,
            }
        ),
        "selection_hash": v4_index["selection_hash"],
        "v4_profile_config_hash": v4_index["profile_config_hash"],
        "dataset_dir": str(dataset_dir),
        "action_horizon": int(v4_index["action_horizon"]),
        "splits": split_entries,
        "action_indices_identity": filtered_identity,
        "candidate_action_indices_identity": candidate_identity,
        "native_reexecution_timing_identity": v7_index["native_reexecution_timing_identity"],
        "h30_target_identity": raw_h30,
        "v4_h30_source_identity": v7_index["v4_h30_source_identity"],
        "v4_lerobot_identity": v4_index["lerobot_identity"],
        "v4_training_data_hash": v4_index["training_data_hash"],
        "v4_norm_stats_sha256": v7_index["v4_norm_stats_sha256"],
        "norm_stats_policy": {
            "kind": "inherited",
            "source_data_profile": "rotation_v4",
            "source_action_indices_identity": candidate_identity,
            "note": "V7.3 filters action starts and intentionally reuses V4 normalization statistics",
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


def validate_v7_3_adjustment_training_index(
    payload: Mapping[str, Any],
    *,
    index_path: Path | None = None,
    dataset_dir: Path | None = None,
    revalidate_h30_targets: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    expected = {
        "schema_version": V7_3_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_3_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V7_3_EXPERIMENT_KIND,
        "detector_config": DEFAULT_DETECTOR_CONFIG,
        "filter_policy": V7_3_FILTER_POLICY,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V7.3 index {key} mismatch")
    if "terminal_hold_schema" in payload:
        raise ValueError("V7.3 index must not enable terminal hold")
    if payload.get("training_data_hash") != sha256_json(
        {key: value for key, value in payload.items() if key != "training_data_hash"}
    ):
        raise ValueError("V7.3 training_data_hash mismatch")
    if index_path is not None and not Path(index_path).is_file():
        raise FileNotFoundError(index_path)
    required = {
        "v4_training_index", "v4_norm_summary", "v4_norm_stats",
        "boundary_filter", "action_phase_manifest", "filter_summary",
    }
    sources = payload.get("source_files", {})
    if not isinstance(sources, Mapping) or not required.issubset(sources):
        raise ValueError("V7.3 index lacks source identities")
    paths = {name: _validate_file_identity(sources[name], context=name) for name in required}
    v4_index = json.loads(paths["v4_training_index"].read_text())
    effective_dataset_dir = (dataset_dir or Path(str(payload["dataset_dir"]))).expanduser().resolve()
    frames, global_lookup = validate_v4_index_dataset(v4_index, effective_dataset_dir)
    timing, boundaries = native_reexecution_timing_identity(v4_index, frames)
    if timing != payload.get("native_reexecution_timing_identity"):
        raise ValueError("V7.3 native timing identity mismatch")
    candidate_indices = {
        int(index) for split in SPLITS for index in v4_index["splits"][split]["execution_indices"]
    }
    boundary_payload = json.loads(paths["boundary_filter"].read_text())
    v7_2_exclusions = validate_boundary_filter(
        boundary_payload,
        v4_index_file=paths["v4_training_index"],
        candidate_indices=candidate_indices,
    )
    filter_identity = payload.get("boundary_filter_identity", {})
    if (
        filter_identity.get("sha256") != sha256_file(paths["boundary_filter"])
        or filter_identity.get("manual_audit") != "passed"
        or int(filter_identity.get("excluded_action_count", -1)) != len(v7_2_exclusions)
    ):
        raise ValueError("V7.3 boundary filter identity mismatch")
    if payload.get("candidate_action_indices_identity") != v4_index.get("action_indices_identity"):
        raise ValueError("V7.3 candidate action identity differs from V4")

    rows = load_jsonl(paths["action_phase_manifest"])
    manifest_identity = payload["action_phase_manifest_identity"]
    if (
        len(rows) != int(manifest_identity["count"])
        or sha256_json(rows) != manifest_identity["content_sha256"]
        or sha256_file(paths["action_phase_manifest"]) != manifest_identity["file_sha256"]
    ):
        raise ValueError("V7.3 action manifest identity mismatch")
    expected_exclusions, policies = build_v7_3_exclusions(
        rows,
        v7_2_exclusions=v7_2_exclusions,
        boundary_payload=boundary_payload,
        action_horizon=int(payload["action_horizon"]),
    )
    if policies != payload.get("summary", {}).get("attempt_filter_policies"):
        raise ValueError("V7.3 adaptive stop policies mismatch")
    lookup: dict[int, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        global_index = int(row["global_index"])
        frame = global_lookup[global_index]
        expected_reason = expected_exclusions.get(global_index)
        expected_phase = phase_for_rexecution_frame(
            frame.attempt_id, frame.frame_index, boundaries[frame.attempt_key]
        )
        if row.get("data_profile") != ROTATION_PHASE_V7_3_ADJUSTMENT or row.get("phase") != expected_phase:
            raise ValueError(f"Invalid V7.3 profile/phase at {global_index}")
        if row.get("exclusion_reason") != expected_reason or bool(row.get("trainable")) != (expected_reason is None):
            raise ValueError(f"Invalid V7.3 exclusion at {global_index}")
        if row.get("terminal_hold_from_offset") is not None or bool(row.get("effective_h30_modified")):
            raise ValueError(f"V7.3 terminal hold found at {global_index}")
        if expected_reason is None:
            if not bool(row.get("raw_chunk_phase_pure")):
                raise ValueError(f"V7.3 retained cross-phase H30 at {global_index}")
            lookup[global_index] = row
            selected_rows.append(row)
    for split in SPLITS:
        expected_indices = [
            int(row["global_index"]) for row in rows if row["split"] == split and row["trainable"]
        ]
        if payload["splits"][split]["execution_indices"] != expected_indices:
            raise ValueError(f"V7.3 {split} loader indices include excluded samples")
        original_episodes = {
            global_lookup[int(index)].episode_id for index in v4_index["splits"][split]["execution_indices"]
        }
        filtered_episodes = {global_lookup[index].episode_id for index in expected_indices}
        if filtered_episodes != original_episodes:
            raise ValueError(f"V7.3 filtering changed {split} episode coverage")
    if action_indices_identity(payload["splits"]) != payload.get("action_indices_identity"):
        raise ValueError("V7.3 filtered action identity mismatch")
    if json.loads(paths["filter_summary"].read_text()) != payload.get("summary"):
        raise ValueError("V7.3 filter summary mismatch")
    norm_summary = json.loads(paths["v4_norm_summary"].read_text())
    if (
        sha256_file(paths["v4_norm_stats"]) != payload.get("v4_norm_stats_sha256")
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity")
        != payload.get("candidate_action_indices_identity")
    ):
        raise ValueError("V7.3 inherited norm identity mismatch")
    if revalidate_h30_targets:
        raw = compute_raw_h30_content_identity(
            dataset_dir=effective_dataset_dir,
            global_lookup=global_lookup,
            action_rows=selected_rows,
            action_indices_identity=payload["action_indices_identity"],
            action_horizon=int(payload["action_horizon"]),
        )
        if (
            raw != payload.get("h30_target_identity")
            or payload.get("h30_modifications")
            != {"modified_chunk_count": 0, "modified_action_step_count": 0}
        ):
            raise ValueError("V7.3 persisted H30 identity mismatch")
    return rows, lookup


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    result = {
        "schema_version": V7_3_HASH_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_3_ADJUSTMENT,
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }
    result["sha256"] = sha256_json(result)
    return result
