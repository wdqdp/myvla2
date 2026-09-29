"""Book Stage A phase labels from the V4 native rexecution event."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.v4_data import SPLITS, V4Frame, load_jsonl, validate_v4_index_dataset


DATA_PROFILE = "book_stage_a_v1"
EXPERIMENT_KIND = "phase_prompt_h30_book_native_reexecution_raw_actions"
INDEX_SCHEMA = "tactile_vla_book_stage_a_v1_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_book_stage_a_v1_action_manifest_v1"
SUMMARY_SCHEMA = "tactile_vla_book_stage_a_v1_filter_summary_v1"
POLICY = {
    "action_horizon": 30,
    "adjustment_last_frame": "rexecution_frame_index_minus_1",
    "execution_first_frame": "rexecution_frame_index",
    "attempt1_phase": "execution",
    "adjustment_cross_phase_h30": "exclude",
    "action_targets": "raw_contiguous_no_idle_compression_no_terminal_hold",
}


def _file_identity(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def native_reexecution_events(
    *, v4_index: Mapping[str, Any], frames: Sequence[V4Frame]
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    """Read R from the hashed V4 action manifest; never infer it from motion."""
    source = v4_index["source_files"]["action_frame_manifest"]
    manifest_path = Path(source["path"])
    if sha256_file(manifest_path) != source["sha256"]:
        raise ValueError("Book V4 action manifest SHA256 mismatch")
    rows = load_jsonl(manifest_path)
    expected = {
        int(index): split
        for split in SPLITS
        for index in v4_index["splits"][split]["execution_indices"]
    }
    frame_by_key = {frame.key: frame for frame in frames}
    frame_by_attempt: dict[tuple[int, int], list[V4Frame]] = {}
    for frame in frames:
        frame_by_attempt.setdefault(frame.attempt_key, []).append(frame)
    if len(rows) != len(expected):
        raise ValueError("Book V4 action manifest count differs from candidate index")
    seen: set[int] = set()
    events: dict[tuple[int, int], tuple[int | None, float | None]] = {}
    for row in rows:
        key = (int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"]))
        frame = frame_by_key.get(key)
        if frame is None or frame.global_index in seen:
            raise ValueError(f"Book V4 action manifest has missing/duplicate frame: {key}")
        seen.add(frame.global_index)
        if expected.get(frame.global_index) != row["split"]:
            raise ValueError(f"Book V4 action manifest split mismatch: {key}")
        if abs(frame.ros_timestamp - float(row["ros_timestamp"])) > 1e-4:
            raise ValueError(f"Book V4 action manifest timestamp mismatch: {key}")
        if int(row["action_horizon"]) != POLICY["action_horizon"]:
            raise ValueError(f"Book V4 action manifest horizon mismatch: {key}")
        attempt = frame.attempt_key
        current = (row["rexecution_frame_index"], row["rexecution_timestamp"])
        if attempt in events and events[attempt] != current:
            raise ValueError(f"Book rexecution timing is inconsistent within {attempt}")
        events[attempt] = current
    if seen != set(expected) or set(events) != set(frame_by_attempt):
        raise ValueError("Book V4 action manifest does not cover all candidates/attempts")
    boundaries: dict[tuple[int, int], int] = {}
    timing_rows: list[dict[str, Any]] = []
    for attempt, attempt_frames in sorted(frame_by_attempt.items()):
        frame_value, timestamp_value = events[attempt]
        if attempt[1] == 1:
            if frame_value is not None or timestamp_value is not None:
                raise ValueError(f"Book attempt1 must not have rexecution timing: {attempt}")
            boundary = None
        elif attempt[1] == 2:
            if isinstance(frame_value, bool) or not isinstance(frame_value, int):
                raise ValueError(f"Book attempt2 lacks rexecution frame: {attempt}")
            boundary = int(frame_value)
            if not 0 < boundary < len(attempt_frames):
                raise ValueError(f"Book rexecution frame is outside attempt: {attempt}")
            if isinstance(timestamp_value, bool) or not isinstance(timestamp_value, (int, float)):
                raise ValueError(f"Book attempt2 lacks rexecution timestamp: {attempt}")
            if not math.isfinite(float(timestamp_value)):
                raise ValueError(f"Book rexecution timestamp is non-finite: {attempt}")
            boundaries[attempt] = boundary
        else:
            raise ValueError(f"Book only supports attempt1/attempt2: {attempt}")
        timing_rows.append({
            "episode_id": attempt[0], "attempt_id": attempt[1],
            "frame_count": len(attempt_frames),
            "rexecution_frame_index": boundary,
            "rexecution_timestamp": timestamp_value,
        })
    return boundaries, {
        "attempt_count": len(timing_rows),
        "attempt2_count": len(boundaries),
        "content_sha256": sha256_json(timing_rows),
        "source_manifest_sha256": source["sha256"],
    }


def phase_and_trainable(
    *, attempt_id: int, frame_index: int, rexecution_frame: int | None, horizon: int
) -> tuple[str, bool]:
    if attempt_id == 1:
        return "execution", True
    if attempt_id != 2 or rexecution_frame is None:
        raise ValueError("Book attempt2 requires native rexecution frame")
    if frame_index < rexecution_frame:
        return "adjustment", frame_index + horizon <= rexecution_frame
    return "execution", True


def build_artifacts(
    *, dataset_dir: Path, v4_index_file: Path, norm_stats_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    norm_stats_dir = norm_stats_dir.expanduser().resolve()
    v4_index = json.loads(v4_index_file.read_text())
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    if int(v4_index["action_horizon"]) != POLICY["action_horizon"]:
        raise ValueError("Book Stage A requires H30")
    events, timing_identity = native_reexecution_events(v4_index=v4_index, frames=frames)
    norm_file = norm_stats_dir / "norm_stats.json"
    norm_sha = sha256_file(norm_file)
    norm_summary = json.loads((norm_stats_dir / "summary.json").read_text())
    if (
        norm_summary.get("norm_stats_sha256") != norm_sha
        or norm_summary.get("artifact_identity", {}).get("index_sha256") != sha256_file(v4_index_file)
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity")
        != v4_index["action_indices_identity"]
    ):
        raise ValueError("Book norm stats do not match the source V4 index")
    rows: list[dict[str, Any]] = []
    splits: dict[str, Any] = {}
    split_summary: dict[str, Any] = {}
    for split in SPLITS:
        candidate = [int(value) for value in v4_index["splits"][split]["execution_indices"]]
        retained: list[int] = []
        counts: Counter[str] = Counter()
        for global_index in candidate:
            frame = global_lookup[global_index]
            boundary = events.get(frame.attempt_key)
            phase, trainable = phase_and_trainable(
                attempt_id=frame.attempt_id, frame_index=frame.frame_index,
                rexecution_frame=boundary, horizon=POLICY["action_horizon"],
            )
            counts[f"{phase}_candidates"] += 1
            if trainable:
                retained.append(global_index)
                counts[f"{phase}_trainable"] += 1
            else:
                counts["cross_phase_h30_excluded"] += 1
            rows.append({
                "schema_version": MANIFEST_SCHEMA, "data_profile": DATA_PROFILE,
                "prompt_profile": PHASE_PROMPT_PROFILE_V2, "experiment_kind": EXPERIMENT_KIND,
                "split": split, "global_index": global_index,
                "episode_id": frame.episode_id, "attempt_id": frame.attempt_id,
                "frame_index": frame.frame_index, "phase": phase,
                "rexecution_frame": boundary, "trainable": trainable,
                "chunk_phase_pure": trainable, "raw_chunk_phase_pure": trainable,
                "exclusion_reason": None if trainable else "crosses_rexecution_frame",
                "action_horizon": POLICY["action_horizon"], "action_target_offsets": None,
            })
        splits[split] = {"execution_indices": retained}
        split_summary[split] = {
            "candidate_count": len(candidate), "trainable_count": len(retained),
            **dict(sorted(counts.items())),
        }
    summary = {
        "schema_version": SUMMARY_SCHEMA, "data_profile": DATA_PROFILE,
        "policy": POLICY, "splits": split_summary, "attempt2_count": len(events),
    }
    index = {
        "schema_version": INDEX_SCHEMA, "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2, "experiment_kind": EXPERIMENT_KIND,
        "target_policy": POLICY, "dataset_dir": str(dataset_dir),
        "action_horizon": POLICY["action_horizon"],
        "data_config_hash": sha256_json({
            "v4_training_data_hash": v4_index["training_data_hash"],
            "native_reexecution_timing_identity": timing_identity, "policy": POLICY,
        }),
        "selection_hash": v4_index["selection_hash"],
        "v4_profile_config_hash": v4_index["profile_config_hash"],
        "v4_training_data_hash": v4_index["training_data_hash"],
        "v4_lerobot_identity": v4_index["lerobot_identity"],
        "v4_norm_stats_sha256": norm_sha,
        "candidate_action_indices_identity": v4_index["action_indices_identity"],
        "native_reexecution_timing_identity": timing_identity,
        "action_phase_manifest_identity": {"count": len(rows), "content_sha256": sha256_json(rows)},
        "splits": splits, "action_indices_identity": action_indices_identity(splits),
        "summary": summary,
        "source_files": {
            "v4_training_index": _file_identity(v4_index_file),
            "v4_action_frame_manifest": _file_identity(Path(v4_index["source_files"]["action_frame_manifest"]["path"])),
            "norm_stats": _file_identity(norm_file),
        },
    }
    return rows, index, summary


def validate_training_index(
    payload: Mapping[str, Any], *, index_path: Path, dataset_dir: Path
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if (
        payload.get("schema_version") != INDEX_SCHEMA
        or payload.get("data_profile") != DATA_PROFILE
        or payload.get("prompt_profile") != PHASE_PROMPT_PROFILE_V2
        or payload.get("experiment_kind") != EXPERIMENT_KIND
        or payload.get("target_policy") != POLICY
    ):
        raise ValueError("Book Stage A index schema/profile/policy mismatch")
    if payload.get("training_data_hash") != sha256_json(
        {key: value for key, value in payload.items() if key != "training_data_hash"}
    ):
        raise ValueError("Book Stage A training_data_hash mismatch")
    for name, source in payload["source_files"].items():
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError(f"Book Stage A {name} SHA256 mismatch")
    if payload["dataset_dir"] != str(dataset_dir.expanduser().resolve()):
        raise ValueError("Book Stage A dataset directory mismatch")
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    sources = payload["source_files"]
    expected_rows, expected_index, _ = build_artifacts(
        dataset_dir=dataset_dir,
        v4_index_file=Path(sources["v4_training_index"]["path"]),
        norm_stats_dir=Path(sources["norm_stats"]["path"]).parent,
    )
    rows = load_jsonl(Path(sources["action_phase_manifest"]["path"]))
    identity = payload["action_phase_manifest_identity"]
    if (
        rows != expected_rows or identity["count"] != len(rows)
        or identity["content_sha256"] != sha256_json(rows)
        or identity["file_sha256"] != sources["action_phase_manifest"]["sha256"]
    ):
        raise ValueError("Book Stage A action manifest mismatch")
    for key, value in expected_index.items():
        if key in {"source_files", "action_phase_manifest_identity"}:
            continue
        if payload.get(key) != value:
            raise ValueError(f"Book Stage A index {key} mismatch")
    if {name: sources[name] for name in expected_index["source_files"]} != expected_index["source_files"]:
        raise ValueError("Book Stage A source identity mismatch")
    return rows, {int(row["global_index"]): row for row in rows if row["trainable"]}
