"""Sponge V10.1 Stage A: native R phases, failure-safe raw H30 targets."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from tactile_vla.vla import book_stage_a_data as common
from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.v4_data import SPLITS, load_jsonl

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/sponge")
VERSION_TAG = "sponge_v10_1"
DATA_PROFILE = common.DATA_PROFILE  # Reuse the proven phase_v2/H30 action transforms.
EXPERIMENT_KIND = common.EXPERIMENT_KIND
RUN_NAME = "pi05_delta_tac_sponge_stage_a_v10_1_no_history"
DEFAULT_INDEX = ROOT / "outputs/sponge_stage_a_v10_1/stage_a_training_index.json"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action" / RUN_NAME / "15000"
FAILURE_ACTION_POLICY = {
    "schema_version": "sponge_v10_1_failure_safe_action_v1",
    "C": "native_V4_failure_boundary_frame_index_captioner_first_left_plus_3",
    "failed_attempt_keep": "s+29<C",
    "failed_attempt_exclude": "cross_C_or_start_at_or_after_C",
    "raw_frames": "unchanged_keep_for_H100_and_other_tasks",
    "norm_stats": "reuse_unchanged_source_V4_stats",
}
POLICY = common.POLICY | {"failure_action_filter": FAILURE_ACTION_POLICY}


def failure_boundary(meta):
    """B is the first captioner left frame; C is the existing effective boundary."""
    if (
        meta["result"] != "failure"
        or meta.get("failure_boundary_source") != "captioner_fz_bias"
        or meta.get("failure_boundary_status") != "resolved"
        or meta.get("fz_bias_direction") != "left"
        or meta.get("rotation_direction") != "none"
        or meta.get("grasp_position") != "appropriate"
    ):
        raise ValueError("V10.1 requires a resolved pure fz_bias-left failure")
    b, c = int(meta["captioner_left_start_frame"]), int(meta["failure_boundary_frame_index"])
    if b < 0 or c != b + 3 or c != int(meta["failure_window_start"]) or c + 14 >= int(meta["frame_count"]):
        raise ValueError("V10.1 requires C=B+3 and a complete [C,C+14] window")
    return b, c


def failure_safe_chunk(start, boundary, horizon=30):
    return int(start) + int(horizon) - 1 < int(boundary)


def build_artifacts(*, dataset_dir: Path, v4_index_file: Path, norm_stats_dir: Path):
    rows, index, _ = common.build_artifacts(
        dataset_dir=dataset_dir, v4_index_file=v4_index_file, norm_stats_dir=norm_stats_dir
    )
    v4 = json.loads(v4_index_file.read_text())
    profile_path = Path(v4["source_files"]["profile"]["path"])
    profile = json.loads(profile_path.read_text())
    metadata = {(m["episode_id"], m["attempt_id"]): m for m in profile["attempts"]}
    counts = {split: Counter() for split in SPLITS}
    for row in rows:
        m = metadata[row["episode_id"], row["attempt_id"]]
        row["experiment_version"] = VERSION_TAG
        row["failure_boundary_frame"] = None
        if m["result"] == "failure":
            _, c = failure_boundary(m)
            row["failure_boundary_frame"] = c
            if not failure_safe_chunk(row["frame_index"], c):
                row.update(
                    trainable=False,
                    chunk_phase_pure=False,
                    raw_chunk_phase_pure=False,
                    exclusion_reason="failure_at_or_after_C" if row["frame_index"] >= c else "crosses_failure_C",
                )
        counts[row["split"]]["candidate_count"] += 1
        counts[row["split"]][row["phase"] + "_candidates"] += 1
        if row["trainable"]:
            counts[row["split"]]["trainable_count"] += 1
            counts[row["split"]][row["phase"] + "_trainable"] += 1
        else:
            counts[row["split"]][row["exclusion_reason"]] += 1
    splits = {
        s: {"execution_indices": [r["global_index"] for r in rows if r["split"] == s and r["trainable"]]}
        for s in SPLITS
    }
    scope = {
        "schema_version": "sponge_v10_1_source_scope_v1",
        "dataset_dir": str(dataset_dir.resolve()),
        "v4_index_file": str(v4_index_file.resolve()),
        "selection": "explicit_selected_dataset_and_hashed_V4_sources_no_recursive_archive_scan",
    }
    summary = {
        "schema_version": "sponge_v10_1_stage_a_summary_v1",
        "experiment_version": VERSION_TAG,
        "policy": POLICY,
        "source_scope": scope,
        "splits": {s: dict(counts[s]) for s in SPLITS},
        "attempt2_count": sum(m["attempt_id"] == 2 for m in profile["attempts"]),
    }
    index.update(
        experiment_version=VERSION_TAG,
        target_policy=POLICY,
        source_scope=scope,
        splits=splits,
        action_indices_identity=action_indices_identity(splits),
        summary=summary,
        data_config_hash=sha256_json({"source_data_config_hash": index["data_config_hash"], "policy": POLICY}),
        action_phase_manifest_identity={"count": len(rows), "content_sha256": sha256_json(rows)},
    )
    index["source_files"]["profile"] = {"path": str(profile_path), "sha256": sha256_file(profile_path)}
    return rows, index, summary


def validate_training_index(payload, *, index_path: Path, dataset_dir: Path):
    if payload.get("experiment_version") != VERSION_TAG or payload.get("target_policy") != POLICY:
        raise ValueError("V10.1 needs its own failure-safe Stage A index")
    if not index_path.is_file() or payload.get("training_data_hash") != sha256_json(
        {k: v for k, v in payload.items() if k != "training_data_hash"}
    ):
        raise ValueError("V10.1 Stage A index missing or hash mismatch")
    sources = payload["source_files"]
    for name, source in sources.items():
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError(f"V10.1 Stage A {name} hash mismatch")
    expected_rows, expected, _ = build_artifacts(
        dataset_dir=dataset_dir,
        v4_index_file=Path(sources["v4_training_index"]["path"]),
        norm_stats_dir=Path(sources["norm_stats"]["path"]).parent,
    )
    rows = load_jsonl(Path(sources["action_phase_manifest"]["path"]))
    if rows != expected_rows:
        raise ValueError("V10.1 Stage A action rows differ from source-derived C/R rules")
    for key, value in expected.items():
        if key not in {"source_files", "action_phase_manifest_identity"} and payload.get(key) != value:
            raise ValueError(f"V10.1 Stage A {key} mismatch")
    if {k: sources[k] for k in expected["source_files"]} != expected["source_files"]:
        raise ValueError("V10.1 Stage A source identity mismatch")
    identity = payload["action_phase_manifest_identity"]
    if identity != expected["action_phase_manifest_identity"] | {
        "file_sha256": sources["action_phase_manifest"]["sha256"]
    }:
        raise ValueError("V10.1 Stage A manifest identity mismatch")
    return rows, {r["global_index"]: r for r in rows if r["trainable"]}
