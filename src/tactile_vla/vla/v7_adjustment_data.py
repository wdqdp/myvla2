"""V7 two-phase labels sourced from native V4 rexecution timing."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2, V2_ACTION_PHASES
from tactile_vla.vla.v4_data import SPLITS, V4Frame, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v5_adjustment_data import V2_TERMINAL_HOLD_SCHEMA
from tactile_vla.vla.v5_adjustment_data import compute_h30_content_identities
from tactile_vla.vla.v5_adjustment_data import phase_for_rexecution_frame


ROTATION_PHASE_V7_ADJUSTMENT = "rotation_phase_v7_adjustment"
V7_EXPERIMENT_KIND = "phase_prompt_h30_terminal_hold_native_reexecution"
V7_ACTION_PHASE_SCHEMA = "tactile_vla_v7_adjustment_action_manifest_v1"
V7_TRAINING_INDEX_SCHEMA = "tactile_vla_v7_adjustment_training_index_v1"
V7_NATIVE_TIMING_SCHEMA = "tactile_vla_v7_native_reexecution_timing_v1"


def _file_identity(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_file_identity(identity: Mapping[str, Any], *, context: str) -> Path:
    path = Path(str(identity.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(identity.get("sha256", ""))
    actual = sha256_file(path)
    if len(expected) != 64 or actual != expected:
        raise ValueError(f"V7 {context} hash mismatch: expected={expected}, actual={actual}")
    return path


def _attempt_key(episode_id: int, attempt_id: int) -> str:
    return f"episode{episode_id}/attempt{attempt_id}"


def _frames_by_attempt(frames: Sequence[V4Frame]) -> dict[tuple[int, int], list[V4Frame]]:
    grouped: dict[tuple[int, int], list[V4Frame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.attempt_key].append(frame)
    for key, values in grouped.items():
        values.sort(key=lambda frame: frame.frame_index)
        if [frame.frame_index for frame in values] != list(range(len(values))):
            raise ValueError(f"V7 V4 attempt {key} does not have contiguous frame indices")
    return dict(grouped)


def native_reexecution_timing_identity(
    v4_index: Mapping[str, Any],
    frames: Sequence[V4Frame],
) -> tuple[dict[str, Any], dict[tuple[int, int], int | None]]:
    """Validate and hash native rexecution fields without deriving new boundaries."""

    timing = v4_index.get("attempt_timing")
    if not isinstance(timing, Mapping):
        raise ValueError("V7 requires v4_index['attempt_timing']")
    grouped = _frames_by_attempt(frames)
    expected_keys = {_attempt_key(*key) for key in grouped}
    actual_keys = {str(key) for key in timing}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"V7 attempt_timing attempt set mismatch: missing={missing[:10]}, extra={extra[:10]}")

    rows: list[dict[str, Any]] = []
    boundaries: dict[tuple[int, int], int | None] = {}
    for (episode_id, attempt_id), attempt_frames in sorted(grouped.items()):
        key = _attempt_key(episode_id, attempt_id)
        row = timing[key]
        if not isinstance(row, Mapping):
            raise ValueError(f"V7 attempt_timing[{key!r}] must be an object")
        frame_value = row.get("rexecution_frame_index")
        timestamp_value = row.get("rexecution_timestamp")
        if attempt_id == 1:
            if frame_value is not None or timestamp_value is not None:
                raise ValueError(f"V7 attempt1 {key} must not define rexecution timing")
            rexecution_frame = None
            rexecution_timestamp = None
        elif attempt_id == 2:
            if isinstance(frame_value, bool) or not isinstance(frame_value, int):
                raise ValueError(f"V7 attempt2 {key} lacks integer rexecution_frame_index")
            rexecution_frame = int(frame_value)
            if not 0 < rexecution_frame < len(attempt_frames):
                raise ValueError(
                    f"V7 attempt2 {key} rexecution_frame_index={rexecution_frame} "
                    f"is outside (0, {len(attempt_frames)})"
                )
            if isinstance(timestamp_value, bool) or not isinstance(timestamp_value, (int, float)):
                raise ValueError(f"V7 attempt2 {key} lacks numeric rexecution_timestamp")
            rexecution_timestamp = float(timestamp_value)
            if not math.isfinite(rexecution_timestamp):
                raise ValueError(f"V7 attempt2 {key} has non-finite rexecution_timestamp")
        else:
            raise ValueError(f"V7 only supports attempt1/attempt2, got {key}")
        boundaries[(episode_id, attempt_id)] = rexecution_frame
        rows.append(
            {
                "attempt_key": key,
                "episode_id": episode_id,
                "attempt_id": attempt_id,
                "frame_count": len(attempt_frames),
                "rexecution_frame_index": rexecution_frame,
                "rexecution_timestamp": rexecution_timestamp,
            }
        )

    identity: dict[str, Any] = {
        "schema_version": V7_NATIVE_TIMING_SCHEMA,
        "attempt_count": len(rows),
        "attempt2_count": sum(row["attempt_id"] == 2 for row in rows),
        "content_sha256": sha256_json(rows),
    }
    identity["sha256"] = sha256_json(identity)
    return identity, boundaries


def _validate_h30_identity(identity: Any, *, context: str) -> None:
    if not isinstance(identity, Mapping):
        raise ValueError(f"V7 lacks {context}")
    stored = str(identity.get("sha256", ""))
    actual = sha256_json({key: value for key, value in identity.items() if key != "sha256"})
    if stored != actual or len(str(identity.get("content_sha256", ""))) != 64:
        raise ValueError(f"V7 {context} identity is invalid")


def build_v7_adjustment_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    v4_norm_stats_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the minimal V7 action manifest and training index payload."""

    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    v4_norm_stats_dir = v4_norm_stats_dir.expanduser().resolve()
    v4_index = json.loads(v4_index_file.read_text())
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    timing_identity, rexecution_by_attempt = native_reexecution_timing_identity(v4_index, frames)

    horizon = int(v4_index["action_horizon"])
    action_rows: list[dict[str, Any]] = []
    split_entries: dict[str, Any] = {}
    all_indices: list[int] = []
    for split in SPLITS:
        indices = [int(value) for value in v4_index["splits"][split]["execution_indices"]]
        row_indices: list[int] = []
        for global_index in indices:
            frame = global_lookup[global_index]
            rexecution = rexecution_by_attempt[frame.attempt_key]
            phase = phase_for_rexecution_frame(frame.attempt_id, frame.frame_index, rexecution)
            crosses = bool(
                phase == "adjustment"
                and rexecution is not None
                and frame.frame_index < rexecution <= frame.frame_index + horizon - 1
            )
            hold_offset = rexecution - frame.frame_index if crosses and rexecution is not None else None
            row_indices.append(len(action_rows))
            action_rows.append(
                {
                    "schema_version": V7_ACTION_PHASE_SCHEMA,
                    "data_profile": ROTATION_PHASE_V7_ADJUSTMENT,
                    "prompt_profile": PHASE_PROMPT_PROFILE_V2,
                    "experiment_kind": V7_EXPERIMENT_KIND,
                    "split": split,
                    "global_index": global_index,
                    "episode_id": frame.episode_id,
                    "attempt_id": frame.attempt_id,
                    "frame_index": frame.frame_index,
                    "phase": phase,
                    "rexecution_frame": rexecution,
                    "raw_chunk_phase_pure": not crosses,
                    "effective_chunk_phase_pure": True,
                    "chunk_phase_pure": not crosses,
                    "terminal_hold_from_offset": hold_offset,
                    "effective_h30_modified": crosses,
                    "chunk_end_frame": frame.frame_index + horizon - 1,
                    "action_horizon": horizon,
                }
            )
        selected = [action_rows[index] for index in row_indices]
        split_entries[split] = {
            "execution_indices": indices,
            "action_phase_manifest_row_indices": row_indices,
            "summary": {
                "action_count": len(indices),
                "phase_counts": dict(Counter(str(row["phase"]) for row in selected)),
                "raw_chunk_crossing": sum(not bool(row["raw_chunk_phase_pure"]) for row in selected),
                "terminal_hold_chunks": sum(bool(row["effective_h30_modified"]) for row in selected),
            },
        }
        all_indices.extend(indices)

    action_identity = v4_index["action_indices_identity"]
    if action_identity["all"] != {
        "count": len(all_indices),
        "sha256": sha256_json(all_indices),
    }:
        raise ValueError("V7 action starts differ from V4")
    raw_h30, effective_h30, modifications = compute_h30_content_identities(
        dataset_dir=dataset_dir,
        global_lookup=global_lookup,
        action_rows=action_rows,
        action_indices_identity=action_identity,
        action_horizon=horizon,
    )

    norm_summary_path = v4_norm_stats_dir / "summary.json"
    norm_stats_path = v4_norm_stats_dir / "norm_stats.json"
    norm_summary = json.loads(norm_summary_path.read_text())
    norm_sha = sha256_file(norm_stats_path)
    if (
        norm_summary.get("norm_stats_sha256") != norm_sha
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity") != action_identity
        or int(norm_summary.get("num_frames", -1)) != int(action_identity["train"]["count"])
    ):
        raise ValueError("V7 V4 norm stats identity does not match its action starts")

    source_files = v4_index.get("source_files", {})
    v4_h30_source_identity = {
        "action_horizon": horizon,
        "execution_indices": action_identity,
        "lerobot_identity": v4_index["lerobot_identity"],
        "lerobot_parquet_sha256": {
            name: value for name, value in sorted(source_files["lerobot_parquet"].items())
        },
    }
    v4_h30_source_identity["sha256"] = sha256_json(v4_h30_source_identity)
    index: dict[str, Any] = {
        "schema_version": V7_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V7_EXPERIMENT_KIND,
        "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
        "data_config_hash": sha256_json(
            {
                "v4_training_data_hash": v4_index["training_data_hash"],
                "native_reexecution_timing_identity": timing_identity,
                "prompt_profile": PHASE_PROMPT_PROFILE_V2,
                "experiment_kind": V7_EXPERIMENT_KIND,
                "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
            }
        ),
        "selection_hash": v4_index["selection_hash"],
        "v4_profile_config_hash": v4_index["profile_config_hash"],
        "dataset_dir": str(dataset_dir),
        "action_horizon": horizon,
        "splits": split_entries,
        "action_indices_identity": action_identity,
        "native_reexecution_timing_identity": timing_identity,
        "h30_target_identity": effective_h30,
        "v4_h30_target_identity": raw_h30,
        "v4_h30_source_identity": v4_h30_source_identity,
        "v4_lerobot_identity": v4_index["lerobot_identity"],
        "v4_training_data_hash": v4_index["training_data_hash"],
        "v4_norm_stats_sha256": norm_sha,
        "h30_modifications": modifications,
        "summary": {
            "episode_count": len({frame.episode_id for frame in frames}),
            "attempt_count": len(rexecution_by_attempt),
            "attempt2_count": sum(attempt_id == 2 for _, attempt_id in rexecution_by_attempt),
            "phase_counts": dict(Counter(str(row["phase"]) for row in action_rows)),
            "splits": {split: split_entries[split]["summary"] for split in SPLITS},
        },
        "source_files": {
            "v4_training_index": _file_identity(v4_index_file),
            "v4_norm_summary": _file_identity(norm_summary_path),
            "v4_norm_stats": _file_identity(norm_stats_path),
        },
        "action_phase_manifest_identity": {
            "count": len(action_rows),
            "content_sha256": sha256_json(action_rows),
        },
    }
    return action_rows, index


def validate_v7_adjustment_training_index(
    payload: Mapping[str, Any],
    *,
    index_path: Path | None = None,
    dataset_dir: Path | None = None,
    revalidate_h30_targets: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    expected_header = {
        "schema_version": V7_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V7_EXPERIMENT_KIND,
        "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
    }
    for key, expected in expected_header.items():
        if payload.get(key) != expected:
            raise ValueError(f"V7 index {key}={payload.get(key)!r}, expected={expected!r}")
    stored_hash = str(payload.get("training_data_hash", ""))
    actual_hash = sha256_json({key: value for key, value in payload.items() if key != "training_data_hash"})
    if not stored_hash or stored_hash != actual_hash:
        raise ValueError("V7 training_data_hash does not match the index payload")
    if index_path is not None and not Path(index_path).is_file():
        raise FileNotFoundError(index_path)

    sources = payload.get("source_files")
    required = {"v4_training_index", "v4_norm_summary", "v4_norm_stats", "action_phase_manifest"}
    if not isinstance(sources, Mapping) or not required.issubset(sources):
        raise ValueError(f"V7 index lacks source identities: {sorted(required - set(sources or {}))}")
    paths = {name: _validate_file_identity(sources[name], context=name) for name in required}
    v4_index = json.loads(paths["v4_training_index"].read_text())
    effective_dataset_dir = (dataset_dir or Path(str(payload["dataset_dir"]))).expanduser().resolve()
    frames, global_lookup = validate_v4_index_dataset(v4_index, effective_dataset_dir)
    timing_identity, rexecution_by_attempt = native_reexecution_timing_identity(v4_index, frames)
    if payload.get("native_reexecution_timing_identity") != timing_identity:
        raise ValueError("V7 native rexecution timing identity differs from V4")
    if payload.get("v4_training_data_hash") != v4_index.get("training_data_hash"):
        raise ValueError("V7 references a different V4 index")
    if payload.get("v4_lerobot_identity") != v4_index.get("lerobot_identity"):
        raise ValueError("V7/V4 LeRobot identities differ")
    if payload.get("action_indices_identity") != v4_index.get("action_indices_identity"):
        raise ValueError("V7/V4 action indices differ")
    for split in SPLITS:
        if payload["splits"][split]["execution_indices"] != v4_index["splits"][split]["execution_indices"]:
            raise ValueError(f"V7/V4 {split} action starts differ")

    raw_h30 = payload.get("v4_h30_target_identity")
    effective_h30 = payload.get("h30_target_identity")
    _validate_h30_identity(raw_h30, context="raw V4 H30")
    _validate_h30_identity(effective_h30, context="effective H30")
    if raw_h30["content_sha256"] == effective_h30["content_sha256"]:
        raise ValueError("V7 raw/effective H30 hashes must differ")

    action_rows = load_jsonl(paths["action_phase_manifest"])
    manifest_identity = payload["action_phase_manifest_identity"]
    if (
        len(action_rows) != int(manifest_identity["count"])
        or sha256_json(action_rows) != manifest_identity["content_sha256"]
        or sha256_file(paths["action_phase_manifest"]) != manifest_identity["file_sha256"]
    ):
        raise ValueError("V7 action manifest identity mismatch")

    action_lookup: dict[int, dict[str, Any]] = {}
    all_indices: list[int] = []
    modified_chunks = 0
    modified_steps = 0
    horizon = int(payload["action_horizon"])
    for split in SPLITS:
        row_indices = payload["splits"][split]["action_phase_manifest_row_indices"]
        indices = payload["splits"][split]["execution_indices"]
        if len(row_indices) != len(indices):
            raise ValueError(f"V7 {split} manifest/action lengths differ")
        for row_index, global_index in zip(row_indices, indices, strict=True):
            row = action_rows[int(row_index)]
            frame = global_lookup[int(global_index)]
            rexecution = rexecution_by_attempt[frame.attempt_key]
            expected_phase = phase_for_rexecution_frame(frame.attempt_id, frame.frame_index, rexecution)
            if (
                row.get("schema_version") != V7_ACTION_PHASE_SCHEMA
                or row.get("data_profile") != ROTATION_PHASE_V7_ADJUSTMENT
                or row.get("split") != split
                or int(row.get("global_index", -1)) != int(global_index)
                or (int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"])) != frame.key
                or row.get("phase") != expected_phase
                or row.get("phase") not in V2_ACTION_PHASES
                or row.get("rexecution_frame") != rexecution
            ):
                raise ValueError(f"V7 action manifest row {row_index} differs from V4/native timing")
            hold = row.get("terminal_hold_from_offset")
            crosses = bool(
                expected_phase == "adjustment"
                and rexecution is not None
                and frame.frame_index < rexecution <= frame.frame_index + horizon - 1
            )
            expected_hold = rexecution - frame.frame_index if crosses and rexecution is not None else None
            if hold != expected_hold or bool(row.get("effective_h30_modified")) != crosses:
                raise ValueError(f"V7 action manifest row {row_index} has invalid terminal hold")
            if row.get("effective_chunk_phase_pure") is not True:
                raise ValueError("V7 effective H30 must be phase-pure")
            if hold is not None:
                modified_chunks += 1
                modified_steps += horizon - int(hold)
            if int(global_index) in action_lookup:
                raise ValueError(f"Duplicate V7 global index {global_index}")
            action_lookup[int(global_index)] = row
            all_indices.append(int(global_index))
    if payload["action_indices_identity"]["all"] != {
        "count": len(all_indices),
        "sha256": sha256_json(all_indices),
    }:
        raise ValueError("V7 action identity is invalid")
    if payload.get("h30_modifications") != {
        "modified_chunk_count": modified_chunks,
        "modified_action_step_count": modified_steps,
    }:
        raise ValueError("V7 terminal-hold totals are invalid")
    if (
        int(effective_h30.get("modified_chunk_count", -1)) != modified_chunks
        or int(effective_h30.get("modified_action_step_count", -1)) != modified_steps
    ):
        raise ValueError("V7 effective H30 modification totals are invalid")

    norm_summary = json.loads(paths["v4_norm_summary"].read_text())
    norm_sha = sha256_file(paths["v4_norm_stats"])
    if (
        norm_sha != payload.get("v4_norm_stats_sha256")
        or norm_summary.get("norm_stats_sha256") != norm_sha
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity")
        != payload.get("action_indices_identity")
    ):
        raise ValueError("V7 did not preserve V4 norm stats")
    if revalidate_h30_targets:
        actual_raw, actual_effective, _ = compute_h30_content_identities(
            dataset_dir=effective_dataset_dir,
            global_lookup=global_lookup,
            action_rows=action_rows,
            action_indices_identity=payload["action_indices_identity"],
            action_horizon=horizon,
        )
        if actual_raw != raw_h30 or actual_effective != effective_h30:
            raise ValueError("V7 persisted H30 hashes do not match LeRobot/manifest")
    return action_rows, action_lookup
