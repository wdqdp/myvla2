"""V7.4 H30 targets with the pre-adjustment idle gap compressed out."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
from typing import Any

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.v4_data import SPLITS, load_jsonl
from tactile_vla.vla.v7_adjustment_data import _file_identity, _validate_file_identity
from tactile_vla.vla.v7_3_adjustment_data import V7_3_TRAINING_INDEX_SCHEMA
from tactile_vla.vla.v7_3_adjustment_data import validate_v7_3_adjustment_training_index


ROTATION_PHASE_V7_4_ADJUSTMENT = "rotation_phase_v7_4_adjustment"
V7_4_EXPERIMENT_KIND = "phase_prompt_h30_v7_3_pre_adjustment_idle_compressed"
V7_4_ACTION_PHASE_SCHEMA = "tactile_vla_v7_4_adjustment_action_manifest_v1"
V7_4_TRAINING_INDEX_SCHEMA = "tactile_vla_v7_4_adjustment_training_index_v1"
V7_4_SUMMARY_SCHEMA = "tactile_vla_v7_4_adjustment_filter_summary_v1"
V7_4_HASH_SCHEMA = "tactile_vla_v7_4_adjustment_artifact_hashes_v1"
V7_4_TARGET_IDENTITY_SCHEMA = "tactile_vla_v7_4_h30_target_offsets_v1"

V7_4_TARGET_POLICY = {
    "scope": "attempt2_trainable_starts_in_gripper_stop_minus_29_through_gripper_stop",
    "action_horizon": 30,
    "preserved_prefix": "start_through_gripper_motion_stop_inclusive",
    "skipped_interval": "strictly_between_gripper_motion_stop_and_arm_adjustment_start",
    "continuation": "arm_adjustment_start_inclusive_until_h30_is_full",
    "terminal_hold": "disabled",
}


def compressed_h30_offsets(
    *,
    start_frame: int,
    gripper_motion_stop: int,
    arm_adjustment_start: int,
    action_horizon: int,
) -> list[int] | None:
    """Return source offsets for H30 after removing the inter-motion idle gap."""

    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if gripper_motion_stop >= arm_adjustment_start:
        raise ValueError("gripper_motion_stop must precede arm_adjustment_start")
    if not gripper_motion_stop - (action_horizon - 1) <= start_frame <= gripper_motion_stop:
        return None

    prefix_length = gripper_motion_stop - start_frame + 1
    if prefix_length >= action_horizon:
        return None
    continuation_length = action_horizon - prefix_length
    continuation_offset = arm_adjustment_start - start_frame
    return [
        *range(prefix_length),
        *range(continuation_offset, continuation_offset + continuation_length),
    ]


def _attempt_events(boundary_payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, int]]:
    events_by_attempt: dict[tuple[int, int], dict[str, int]] = {}
    for attempt in boundary_payload["attempts"]:
        key = (int(attempt["episode_id"]), int(attempt["attempt_id"]))
        events = attempt["events"]
        events_by_attempt[key] = {
            "gripper_motion_stop": int(events["gripper_motion_stop"]["frame_index"]),
            "arm_adjustment_start": int(events["arm_adjustment_start"]["frame_index"]),
            "rexecution_frame": int(attempt["rexecution_frame"]),
        }
    return events_by_attempt


def apply_v7_4_target_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    boundary_payload: Mapping[str, Any],
    action_horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate V7.3 rows with per-sample action source offsets."""

    if action_horizon != int(V7_4_TARGET_POLICY["action_horizon"]):
        raise ValueError(f"V7.4 requires H30, got H{action_horizon}")
    events_by_attempt = _attempt_events(boundary_payload)
    transformed: list[dict[str, Any]] = []
    modified_by_split: Counter[str] = Counter()
    modified_by_attempt: Counter[str] = Counter()
    modified_step_count = 0
    skipped_idle_step_count = 0
    max_source_offset = action_horizon - 1

    for base in rows:
        row = dict(base)
        row.update(
            {
                "schema_version": V7_4_ACTION_PHASE_SCHEMA,
                "data_profile": ROTATION_PHASE_V7_4_ADJUSTMENT,
                "experiment_kind": V7_4_EXPERIMENT_KIND,
                "action_target_offsets": None,
                "idle_gap_compression": None,
            }
        )
        key = (int(row["episode_id"]), int(row["attempt_id"]))
        events = events_by_attempt.get(key)
        if bool(row["trainable"]) and row["phase"] == "adjustment" and events is not None:
            start = int(row["frame_index"])
            offsets = compressed_h30_offsets(
                start_frame=start,
                gripper_motion_stop=events["gripper_motion_stop"],
                arm_adjustment_start=events["arm_adjustment_start"],
                action_horizon=action_horizon,
            )
            if offsets is not None:
                last_target_frame = start + offsets[-1]
                if last_target_frame >= events["rexecution_frame"]:
                    raise ValueError(
                        "Compressed V7.4 H30 crosses rexecution_frame: "
                        f"attempt={key}, start={start}, last_target={last_target_frame}, "
                        f"rexecution={events['rexecution_frame']}"
                    )
                gripper_offset = events["gripper_motion_stop"] - start
                continuation_steps = action_horizon - (gripper_offset + 1)
                skipped_steps = events["arm_adjustment_start"] - events["gripper_motion_stop"] - 1
                row["action_target_offsets"] = offsets
                row["idle_gap_compression"] = {
                    "gripper_motion_stop_frame": events["gripper_motion_stop"],
                    "arm_adjustment_start_frame": events["arm_adjustment_start"],
                    "preserved_prefix_end_offset": gripper_offset,
                    "continuation_start_offset": events["arm_adjustment_start"] - start,
                    "continuation_step_count": continuation_steps,
                    "skipped_idle_frame_count": skipped_steps,
                }
                row["effective_h30_modified"] = True
                modified_by_split[str(row["split"])] += 1
                modified_by_attempt[f"{key[0]}:{key[1]}"] += 1
                modified_step_count += continuation_steps
                skipped_idle_step_count += skipped_steps
                max_source_offset = max(max_source_offset, offsets[-1])
        transformed.append(row)

    modifications = {
        "modified_chunk_count": sum(modified_by_split.values()),
        "modified_action_step_count": modified_step_count,
        "modified_chunks_by_split": dict(sorted(modified_by_split.items())),
        "modified_attempt_count": len(modified_by_attempt),
        "modified_chunks_by_attempt": dict(sorted(modified_by_attempt.items())),
        "summed_skipped_idle_frame_count": skipped_idle_step_count,
        "max_action_source_offset": max_source_offset,
    }
    return transformed, modifications


def _target_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_training_data_hash: str,
    action_horizon: int,
    modifications: Mapping[str, Any],
) -> dict[str, Any]:
    offset_rows = [
        {
            "global_index": int(row["global_index"]),
            "action_target_offsets": row["action_target_offsets"],
        }
        for row in rows
        if row.get("action_target_offsets") is not None
    ]
    identity = {
        "schema_version": V7_4_TARGET_IDENTITY_SCHEMA,
        "source_v7_3_training_data_hash": source_training_data_hash,
        "action_horizon": action_horizon,
        "offset_rows_sha256": sha256_json(offset_rows),
        "modifications": dict(modifications),
    }
    identity["sha256"] = sha256_json(identity)
    return identity


def build_v7_4_adjustment_artifacts(
    *,
    v7_3_index_file: Path,
    dataset_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    v7_3_index_file = v7_3_index_file.expanduser().resolve()
    v7_3_index = json.loads(v7_3_index_file.read_text())
    if v7_3_index.get("schema_version") != V7_3_TRAINING_INDEX_SCHEMA:
        raise ValueError("V7.4 source must be a V7.3 training index")
    source_rows, _ = validate_v7_3_adjustment_training_index(
        v7_3_index,
        index_path=v7_3_index_file,
        dataset_dir=dataset_dir,
    )
    boundary_path = Path(v7_3_index["source_files"]["boundary_filter"]["path"])
    boundary_payload = json.loads(boundary_path.read_text())
    action_horizon = int(v7_3_index["action_horizon"])
    rows, modifications = apply_v7_4_target_policy(
        source_rows,
        boundary_payload=boundary_payload,
        action_horizon=action_horizon,
    )

    summary = copy.deepcopy(v7_3_index["summary"])
    summary.update(
        {
            "schema_version": V7_4_SUMMARY_SCHEMA,
            "data_profile": ROTATION_PHASE_V7_4_ADJUSTMENT,
            "target_policy": V7_4_TARGET_POLICY,
            "h30_modifications": modifications,
        }
    )
    for split in SPLITS:
        summary["splits"][split]["idle_compressed_h30_chunks"] = int(
            modifications["modified_chunks_by_split"].get(split, 0)
        )

    index = copy.deepcopy(v7_3_index)
    index.pop("training_data_hash", None)
    index.update(
        {
            "schema_version": V7_4_TRAINING_INDEX_SCHEMA,
            "data_profile": ROTATION_PHASE_V7_4_ADJUSTMENT,
            "prompt_profile": PHASE_PROMPT_PROFILE_V2,
            "experiment_kind": V7_4_EXPERIMENT_KIND,
            "target_policy": V7_4_TARGET_POLICY,
            "data_config_hash": sha256_json(
                {
                    "source_v7_3_training_data_hash": v7_3_index["training_data_hash"],
                    "target_policy": V7_4_TARGET_POLICY,
                    "experiment_kind": V7_4_EXPERIMENT_KIND,
                }
            ),
            "h30_target_identity": _target_identity(
                rows,
                source_training_data_hash=v7_3_index["training_data_hash"],
                action_horizon=action_horizon,
                modifications=modifications,
            ),
            "h30_modifications": modifications,
            "summary": summary,
            "source_files": {
                "v7_3_training_index": _file_identity(v7_3_index_file),
            },
            "action_phase_manifest_identity": {
                "count": len(rows),
                "content_sha256": sha256_json(rows),
            },
        }
    )
    index["norm_stats_policy"]["note"] = (
        "V7.4 changes only H30 temporal target selection and reuses V4 normalization statistics"
    )
    for split in SPLITS:
        index["splits"][split]["summary"] = summary["splits"][split]
    return rows, index, summary


def validate_v7_4_adjustment_training_index(
    payload: Mapping[str, Any],
    *,
    index_path: Path | None = None,
    dataset_dir: Path | None = None,
    revalidate_h30_targets: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    del revalidate_h30_targets  # Offset identity and source dataset identity are always validated.
    expected = {
        "schema_version": V7_4_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_4_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V7_4_EXPERIMENT_KIND,
        "target_policy": V7_4_TARGET_POLICY,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V7.4 index {key} mismatch")
    if payload.get("training_data_hash") != sha256_json(
        {key: value for key, value in payload.items() if key != "training_data_hash"}
    ):
        raise ValueError("V7.4 training_data_hash mismatch")
    if index_path is not None and not Path(index_path).is_file():
        raise FileNotFoundError(index_path)

    sources = payload.get("source_files", {})
    required = {"v7_3_training_index", "action_phase_manifest", "filter_summary"}
    if not isinstance(sources, Mapping) or not required.issubset(sources):
        raise ValueError("V7.4 index lacks source identities")
    paths = {name: _validate_file_identity(sources[name], context=name) for name in required}
    v7_3_index = json.loads(paths["v7_3_training_index"].read_text())
    source_rows, _ = validate_v7_3_adjustment_training_index(
        v7_3_index,
        index_path=paths["v7_3_training_index"],
        dataset_dir=dataset_dir,
    )
    boundary_path = Path(v7_3_index["source_files"]["boundary_filter"]["path"])
    expected_rows, modifications = apply_v7_4_target_policy(
        source_rows,
        boundary_payload=json.loads(boundary_path.read_text()),
        action_horizon=int(payload["action_horizon"]),
    )
    rows = load_jsonl(paths["action_phase_manifest"])
    manifest_identity = payload["action_phase_manifest_identity"]
    if (
        rows != expected_rows
        or len(rows) != int(manifest_identity["count"])
        or sha256_json(rows) != manifest_identity["content_sha256"]
        or sha256_file(paths["action_phase_manifest"]) != manifest_identity["file_sha256"]
    ):
        raise ValueError("V7.4 action manifest mismatch")
    if json.loads(paths["filter_summary"].read_text()) != payload.get("summary"):
        raise ValueError("V7.4 summary mismatch")
    expected_target_identity = _target_identity(
        rows,
        source_training_data_hash=v7_3_index["training_data_hash"],
        action_horizon=int(payload["action_horizon"]),
        modifications=modifications,
    )
    if payload.get("h30_target_identity") != expected_target_identity:
        raise ValueError("V7.4 H30 target identity mismatch")
    if payload.get("h30_modifications") != modifications:
        raise ValueError("V7.4 H30 modification summary mismatch")
    for split in SPLITS:
        expected_indices = [
            int(row["global_index"])
            for row in rows
            if row["split"] == split and row["trainable"]
        ]
        if payload["splits"][split]["execution_indices"] != expected_indices:
            raise ValueError(f"V7.4 {split} indices mismatch")
    return rows, {
        int(row["global_index"]): row for row in rows if bool(row["trainable"])
    }


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    payload = {
        "schema_version": V7_4_HASH_SCHEMA,
        "files": {
            name: {"path": str(path.expanduser().resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }
    payload["sha256"] = sha256_json(payload)
    return payload
