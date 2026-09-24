"""V7.4.2 phase labels for the new environment and raw H30 actions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
from typing import Any

from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.v4_data import SPLITS, V4Frame, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v7_4_2_adjustment_boundaries import BOUNDARY_SCHEMA


DATA_PROFILE = "rotation_phase_v7_4_2_adjustment"
EXPERIMENT_KIND = "phase_prompt_h30_offline_arm_adjustment_stop_raw_actions"
INDEX_SCHEMA = "tactile_vla_v7_4_2_adjustment_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_v7_4_2_adjustment_action_manifest_v1"
SUMMARY_SCHEMA = "tactile_vla_v7_4_2_adjustment_filter_summary_v1"
POLICY = {
    "action_horizon": 30,
    "adjustment_last_frame": "arm_adjustment_stop_inclusive",
    "execution_first_frame": "arm_adjustment_stop_plus_1",
    "attempt1_phase": "execution",
    "adjustment_cross_phase_h30": "exclude",
    "action_targets": "raw_contiguous_no_idle_compression_no_terminal_hold",
}


def _file_identity(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_boundaries(
    boundary: Mapping[str, Any], *, v4_index_file: Path, frames: Sequence[V4Frame]
) -> dict[tuple[int, int], tuple[int, int]]:
    """Bind each offline event to its exact V4 frame and source index."""
    if boundary.get("schema_version") != BOUNDARY_SCHEMA or boundary.get("data_profile") != DATA_PROFILE:
        raise ValueError("V7.4.2 boundary schema/profile mismatch")
    source = boundary.get("source_v4_index", {})
    if (
        source.get("path") != str(v4_index_file.expanduser().resolve())
        or source.get("sha256") != sha256_file(v4_index_file)
    ):
        raise ValueError("V7.4.2 boundaries belong to a different V4 index")
    hash_payload = copy.deepcopy(
        {key: value for key, value in boundary.items() if key != "content_sha256"}
    )
    # The boundary writer hashed Counter's integer keys before JSON turned
    # them into strings. Restore those keys for its original canonical hash.
    for name in ("pre_adjustment_idle_frames", "post_adjustment_idle_frames"):
        hash_payload["summary"][name]["counts"] = {
            int(key): value for key, value in boundary["summary"][name]["counts"].items()
        }
    if boundary.get("content_sha256") != sha256_json(hash_payload):
        raise ValueError("V7.4.2 boundary content hash mismatch")
    frame_lookup = {(f.episode_id, f.attempt_id, f.frame_index): f for f in frames}
    expected = {(f.episode_id, f.attempt_id) for f in frames if f.attempt_id == 2}
    events: dict[tuple[int, int], tuple[int, int]] = {}
    for attempt in boundary["attempts"]:
        key = (int(attempt["episode_id"]), int(attempt["attempt_id"]))
        if key not in expected or key in events:
            raise ValueError(f"Unexpected or duplicate V7.4.2 boundary attempt: {key}")
        names = set(attempt["events"])
        if names != {"arm_adjustment_start", "arm_adjustment_stop"}:
            raise ValueError(f"V7.4.2 boundary must contain exactly two arm events: {key}")
        indices: list[int] = []
        for name in ("arm_adjustment_start", "arm_adjustment_stop"):
            event = attempt["events"][name]
            frame_index = int(event["frame_index"])
            frame = frame_lookup.get((*key, frame_index))
            if frame is None or frame.global_index != int(event["global_index"]):
                raise ValueError(f"V7.4.2 {name} frame identity mismatch: {key}")
            if abs(frame.ros_timestamp - float(event["timestamp"])) > 1e-4:
                raise ValueError(f"V7.4.2 {name} timestamp mismatch: {key}")
            indices.append(frame_index)
        if not indices[0] <= indices[1]:
            raise ValueError(f"V7.4.2 arm boundaries are reversed: {key}")
        events[key] = indices[0], indices[1]
    if set(events) != expected:
        raise ValueError(f"V7.4.2 boundaries do not cover all attempt2: {expected - set(events)}")
    return events


def phase_and_trainable(
    *, attempt_id: int, frame_index: int, stop: int | None, horizon: int
) -> tuple[str, bool]:
    if attempt_id == 1:
        return "execution", True
    if attempt_id != 2 or stop is None:
        raise ValueError("V7.4.2 attempt2 requires arm_adjustment_stop")
    if frame_index <= stop:
        return "adjustment", frame_index + horizon - 1 <= stop
    return "execution", True


def make_manifest(
    *, v4_index: Mapping[str, Any], frames: Sequence[V4Frame],
    events: Mapping[tuple[int, int], tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    horizon = int(v4_index["action_horizon"])
    if horizon != POLICY["action_horizon"]:
        raise ValueError("V7.4.2 requires H30")
    global_lookup = {frame.global_index: frame for frame in frames}
    rows: list[dict[str, Any]] = []
    split_entries: dict[str, Any] = {}
    split_summary: dict[str, Any] = {}
    for split in SPLITS:
        candidate = [int(value) for value in v4_index["splits"][split]["execution_indices"]]
        retained: list[int] = []
        counts: Counter[str] = Counter()
        for global_index in candidate:
            frame = global_lookup[global_index]
            event = events.get(frame.attempt_key)
            stop = event[1] if event is not None else None
            phase, trainable = phase_and_trainable(
                attempt_id=frame.attempt_id,
                frame_index=frame.frame_index,
                stop=stop,
                horizon=horizon,
            )
            if trainable:
                retained.append(global_index)
            else:
                counts["cross_phase_h30_excluded"] += 1
            counts[f"{phase}_candidates"] += 1
            if trainable:
                counts[f"{phase}_trainable"] += 1
            rows.append({
                "schema_version": MANIFEST_SCHEMA,
                "data_profile": DATA_PROFILE,
                "prompt_profile": PHASE_PROMPT_PROFILE_V2,
                "experiment_kind": EXPERIMENT_KIND,
                "split": split,
                "global_index": global_index,
                "episode_id": frame.episode_id,
                "attempt_id": frame.attempt_id,
                "frame_index": frame.frame_index,
                "phase": phase,
                "arm_adjustment_start_frame": event[0] if event is not None else None,
                "arm_adjustment_stop_frame": stop,
                "trainable": trainable,
                "chunk_phase_pure": trainable,
                "raw_chunk_phase_pure": trainable,
                "exclusion_reason": None if trainable else "crosses_arm_adjustment_stop",
                "action_horizon": horizon,
                "action_target_offsets": None,
            })
        split_entries[split] = {"execution_indices": retained}
        split_summary[split] = {
            "candidate_count": len(candidate),
            "trainable_count": len(retained),
            **dict(sorted(counts.items())),
        }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "data_profile": DATA_PROFILE,
        "policy": POLICY,
        "splits": split_summary,
        "attempt2_count": len(events),
    }
    return rows, split_entries, summary


def build_artifacts(
    *, dataset_dir: Path, v4_index_file: Path, boundary_file: Path, norm_stats_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    boundary_file = boundary_file.expanduser().resolve()
    norm_stats_dir = norm_stats_dir.expanduser().resolve()
    v4_index = json.loads(v4_index_file.read_text())
    frames, _ = validate_v4_index_dataset(v4_index, dataset_dir)
    boundary = json.loads(boundary_file.read_text())
    events = validate_boundaries(boundary, v4_index_file=v4_index_file, frames=frames)
    rows, splits, summary = make_manifest(v4_index=v4_index, frames=frames, events=events)
    norm_file = norm_stats_dir / "norm_stats.json"
    norm_summary = json.loads((norm_stats_dir / "summary.json").read_text())
    norm_sha = sha256_file(norm_file)
    if (
        norm_summary.get("norm_stats_sha256") != norm_sha
        or norm_summary.get("artifact_identity", {}).get("index_sha256") != sha256_file(v4_index_file)
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity")
        != v4_index["action_indices_identity"]
    ):
        raise ValueError("V7.4.2 norm stats do not match the new V4 index")
    identity = action_indices_identity(splits)
    index: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": EXPERIMENT_KIND,
        "target_policy": POLICY,
        "dataset_dir": str(dataset_dir),
        "action_horizon": POLICY["action_horizon"],
        "data_config_hash": sha256_json({
            "v4_training_data_hash": v4_index["training_data_hash"],
            "boundary_sha256": sha256_file(boundary_file),
            "policy": POLICY,
        }),
        "selection_hash": v4_index["selection_hash"],
        "v4_profile_config_hash": v4_index["profile_config_hash"],
        "v4_training_data_hash": v4_index["training_data_hash"],
        "v4_lerobot_identity": v4_index["lerobot_identity"],
        "v4_norm_stats_sha256": norm_sha,
        "candidate_action_indices_identity": v4_index["action_indices_identity"],
        "boundary_identity": _file_identity(boundary_file),
        "action_phase_manifest_identity": {
            "count": len(rows), "content_sha256": sha256_json(rows),
        },
        "splits": splits,
        "action_indices_identity": identity,
        "summary": summary,
        "source_files": {
            "v4_training_index": _file_identity(v4_index_file),
            "adjustment_boundaries": _file_identity(boundary_file),
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
        raise ValueError("V7.4.2 index schema/profile/policy mismatch")
    if payload.get("training_data_hash") != sha256_json(
        {key: value for key, value in payload.items() if key != "training_data_hash"}
    ):
        raise ValueError("V7.4.2 training_data_hash mismatch")
    sources = payload["source_files"]
    for name in ("v4_training_index", "adjustment_boundaries", "norm_stats", "action_phase_manifest"):
        source = sources[name]
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError(f"V7.4.2 {name} SHA256 mismatch")
    if payload["dataset_dir"] != str(dataset_dir.expanduser().resolve()):
        raise ValueError("V7.4.2 dataset directory mismatch")
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    v4_file = Path(sources["v4_training_index"]["path"])
    boundary_file = Path(sources["adjustment_boundaries"]["path"])
    norm_dir = Path(sources["norm_stats"]["path"]).parent
    expected_rows, expected_index, _ = build_artifacts(
        dataset_dir=dataset_dir, v4_index_file=v4_file,
        boundary_file=boundary_file, norm_stats_dir=norm_dir,
    )
    rows = load_jsonl(Path(sources["action_phase_manifest"]["path"]))
    identity = payload["action_phase_manifest_identity"]
    if (
        rows != expected_rows
        or identity["count"] != len(rows)
        or identity["content_sha256"] != sha256_json(rows)
        or identity["file_sha256"] != sources["action_phase_manifest"]["sha256"]
    ):
        raise ValueError("V7.4.2 action manifest mismatch")
    for key, value in expected_index.items():
        if key in {"source_files", "action_phase_manifest_identity"}:
            continue
        if payload.get(key) != value:
            raise ValueError(f"V7.4.2 index {key} mismatch")
    return rows, {int(row["global_index"]): row for row in rows if row["trainable"]}
