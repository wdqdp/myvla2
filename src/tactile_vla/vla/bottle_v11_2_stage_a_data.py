"""Bottle V11.2: compress only the verified full-open-to-rotation idle gap."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tactile_vla.vla import bottle_v11_1_stage_a_data as previous
from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.v4_data import SPLITS, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v7_2_boundary_filter import _motion_runs

ROOT = previous.ROOT
DATA_PROFILE = previous.DATA_PROFILE
EXPERIMENT_KIND = "phase_prompt_h30_bottle_open_rotation_idle_compressed"
VERSION_TAG = "bottle_v11_2"
RUN_NAME = "pi05_delta_tac_bottle_stage_a_v11_2_no_history"
DEFAULT_INDEX = ROOT / "outputs/bottle_stage_a_v11_2/stage_a_training_index.json"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action" / RUN_NAME / "15000"
FAILURE_ACTION_POLICY = previous.FAILURE_ACTION_POLICY
DEFAULT_SETTLE_FRAMES = 5
DETECTOR_POLICY = {
    "schema_version": "bottle_v11_2_open_rotation_idle_v1",
    "gripper_step_threshold": 0.00005,
    "gripper_merge_gap_frames": 3,
    "minimum_gripper_active_frames": 5,
    "minimum_gripper_change": 0.005,
    "full_open_tolerance": 0.00005,
    "wrist_step_threshold_rad": 0.0005,
    "wrist_merge_gap_frames": 5,
    "minimum_wrist_active_frames": 10,
    "minimum_wrist_range_rad": 0.02,
    "idle_arm_range_limit_rad": 0.01,
    "idle_gripper_range_limit": 0.0002,
    "settling_anchor": "later_of_master_full_open_and_puppet_full_open",
    "interval": "half_open_[full_open_plus_1_plus_settle_frames,rotation_start)",
    "scope": "attempt2_before_native_R_only",
    "initial_wait_and_execution": "unchanged",
}


def target_policy(settle_frames=DEFAULT_SETTLE_FRAMES):
    if isinstance(settle_frames, bool) or not isinstance(settle_frames, int) or settle_frames < 0:
        raise ValueError("settle_frames must be a nonnegative integer")
    return previous.POLICY | {
        "action_targets": "H30_skip_verified_open_rotation_idle_no_padding_no_cross_R",
        "idle_compression": DETECTOR_POLICY | {"settle_frames": settle_frames},
        "idle_start_samples": "exclude_only_removed_interval",
    }


def _open_rotation_events(signal, r):
    delta = np.diff(signal[:r], axis=0, prepend=signal[:1])
    opening = [
        run
        for run in _motion_runs(
            signal[:r, 6],
            delta[:, 6] > DETECTOR_POLICY["gripper_step_threshold"],
            merge_gap_frames=3,
            minimum_active_frames=5,
        )
        if run.signed_change >= DETECTOR_POLICY["minimum_gripper_change"]
    ]
    if not opening:
        raise ValueError("V11.2 missing sustained gripper opening")
    rotations = [
        run
        for run in _motion_runs(
            signal[:r, 5],
            np.abs(delta[:, 5]) > DETECTOR_POLICY["wrist_step_threshold_rad"],
            merge_gap_frames=5,
            minimum_active_frames=10,
        )
        if run.value_range >= DETECTOR_POLICY["minimum_wrist_range_rad"] and run.start > opening[0].start
    ]
    if not rotations:
        raise ValueError("V11.2 missing sustained wrist rotation after opening")
    rotation_start = rotations[0].start
    segment = signal[opening[0].start : rotation_start, 6]
    maximum = float(segment.max())
    # A single opening may contain multiple runs; do not use the first partial run's end.
    full_open = opening[0].start + int(np.flatnonzero(segment >= maximum - DETECTOR_POLICY["full_open_tolerance"])[0])
    return opening[0].start, full_open, rotation_start


def detect_idle_gap(actions, states, timestamps, *, reexecution_frame, settle_frames=DEFAULT_SETTLE_FRAMES):
    target_policy(settle_frames)
    actions, states, timestamps = map(np.asarray, (actions, states, timestamps))
    r = int(reexecution_frame)
    if actions.ndim != 2 or actions.shape[1] != 7 or states.shape != actions.shape:
        raise ValueError("V11.2 requires matching [T,7] master actions and puppet states")
    if timestamps.shape != (len(actions),) or not 0 < r < len(actions):
        raise ValueError("V11.2 invalid timestamps/native R")
    if not all(np.isfinite(x).all() for x in (actions, states, timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("V11.2 requires finite signals and increasing timestamps")
    opening, master_full, rotation = _open_rotation_events(actions, r)
    _, puppet_full, puppet_rotation = _open_rotation_events(states, r)
    anchor = max(master_full, puppet_full)
    if anchor >= rotation:
        raise ValueError("V11.2 cannot establish full-open settling before commanded rotation")
    remove_start = min(rotation, anchor + 1 + settle_frames)
    segment = actions[anchor:rotation]
    if (
        float(np.max(np.ptp(segment[:, :6], axis=0))) > DETECTOR_POLICY["idle_arm_range_limit_rad"]
        or float(np.ptp(segment[:, 6])) > DETECTOR_POLICY["idle_gripper_range_limit"]
    ):
        raise ValueError("V11.2 open-to-rotation gap contains non-idle commands; refusing compression")
    return {
        "opening_start": opening,
        "master_full_open": master_full,
        "puppet_full_open": puppet_full,
        "rotation_start": rotation,
        "puppet_rotation_start": puppet_rotation,
        "rexecution_frame": r,
        "settling_anchor": anchor,
        "settle_frames": settle_frames,
        "removed_start": remove_start,
        "removed_stop": rotation,
        "removed_frame_count": rotation - remove_start,
        "original_idle_frames": rotation - master_full - 1,
        "original_idle_seconds": float(timestamps[rotation] - timestamps[master_full]),
        "full_open_timestamp": float(timestamps[master_full]),
        "rotation_start_timestamp": float(timestamps[rotation]),
        "removed_start_timestamp": float(timestamps[remove_start]),
    }


def compressed_h30_offsets(start_frame, gap, *, action_horizon=30):
    """Offsets in the original timeline; None means excluded, never padded."""
    start, r = int(start_frame), gap["rexecution_frame"]
    left, right = gap["removed_start"], gap["removed_stop"]
    if not 0 <= start < r or left <= start < right:
        return None
    frames = [i for i in range(start, min(r, start + action_horizon + right - left)) if not left <= i < right][
        :action_horizon
    ]
    if len(frames) != action_horizon:
        return None
    return [i - start for i in frames]


def build_artifacts(*, dataset_dir, v4_index_file, norm_stats_dir, settle_frames=DEFAULT_SETTLE_FRAMES):
    policy = target_policy(settle_frames)
    rows, index, _ = previous.build_artifacts(
        dataset_dir=Path(dataset_dir), v4_index_file=Path(v4_index_file), norm_stats_dir=Path(norm_stats_dir)
    )
    profile = json.loads(Path(index["source_files"]["profile"]["path"]).read_text())
    meta = {(m["episode_id"], m["attempt_id"]): m for m in profile["attempts"]}
    gaps, parquet_sources = {}, {}
    for path in sorted((Path(dataset_dir) / "data").rglob("*.parquet")):
        table = pq.read_table(
            path, columns=["episode_id", "attempt_id", "action", "observation.state", "ros_timestamp"]
        )
        if table["attempt_id"][0].as_py() != 2:
            continue
        key = int(table["episode_id"][0].as_py()), 2
        if key in gaps:
            raise ValueError(f"V11.2 duplicate attempt parquet: {key}")
        gap = detect_idle_gap(
            np.asarray(table["action"].to_pylist()),
            np.asarray(table["observation.state"].to_pylist()),
            np.asarray(table["ros_timestamp"].to_pylist()).reshape(-1),
            reexecution_frame=meta[key]["rexecution_frame_index"],
            settle_frames=settle_frames,
        )
        gaps[key] = gap | {"episode_id": key[0], "attempt_id": 2, "split": meta[key]["split"]}
        parquet_sources[str(path.resolve())] = sha256_file(path)
    if set(gaps) != {key for key in meta if key[1] == 2}:
        raise ValueError("V11.2 gap audit does not cover every attempt2")
    counts = {split: Counter() for split in SPLITS}
    for row in rows:
        row.update(
            experiment_version=VERSION_TAG,
            experiment_kind=EXPERIMENT_KIND,
            effective_h30_modified=False,
            idle_gap_compression=None,
        )
        key = row["episode_id"], row["attempt_id"]
        if row["phase"] == "adjustment" and row["trainable"]:
            gap = gaps[key]
            offsets = compressed_h30_offsets(row["frame_index"], gap)
            if offsets is None:
                reason = (
                    "inside_removed_open_rotation_idle"
                    if gap["removed_start"] <= row["frame_index"] < gap["removed_stop"]
                    else "compressed_h30_crosses_R"
                )
                row.update(trainable=False, chunk_phase_pure=False, exclusion_reason=reason)
            elif offsets != list(range(30)):
                row.update(
                    action_target_offsets=offsets,
                    effective_h30_modified=True,
                    idle_gap_compression={
                        k: gap[k] for k in ("removed_start", "removed_stop", "removed_frame_count", "settle_frames")
                    },
                )
        c = counts[row["split"]]
        c["candidate_count"] += 1
        c[row["phase"] + "_candidates"] += 1
        if row["trainable"]:
            c["trainable_count"] += 1
            c[row["phase"] + "_trainable"] += 1
            c["compressed_h30_count"] += int(row["effective_h30_modified"])
        else:
            c[row["exclusion_reason"]] += 1
    splits = {
        s: {"execution_indices": [r["global_index"] for r in rows if r["split"] == s and r["trainable"]]}
        for s in SPLITS
    }
    audit = {
        "schema_version": DETECTOR_POLICY["schema_version"],
        "policy": policy["idle_compression"],
        "attempts": [gaps[key] for key in sorted(gaps)],
    }
    scope = index["source_scope"] | {"schema_version": "bottle_v11_2_source_scope_v1"}
    summary = {
        "experiment_version": VERSION_TAG,
        "policy": policy,
        "source_scope": scope,
        "splits": {s: dict(c) for s, c in counts.items()},
        "attempt2_count": len(gaps),
        "removed_idle_frames": sum(g["removed_frame_count"] for g in gaps.values()),
        "token_validation": previous.action_token_summary(
            rows,
            validate_v4_index_dataset(json.loads(Path(v4_index_file).read_text()), Path(dataset_dir))[0],
            dataset_dir=Path(dataset_dir),
            norm_stats_dir=Path(norm_stats_dir),
        ),
    }
    index.update(
        experiment_version=VERSION_TAG,
        target_policy=policy,
        splits=splits,
        summary=summary,
        source_scope=scope,
        action_indices_identity=action_indices_identity(splits),
        idle_gap_audit=audit,
        idle_parquet_source_hashes=parquet_sources,
        data_config_hash=sha256_json(
            {"source_data_config_hash": index["data_config_hash"], "policy": policy, "audit": audit}
        ),
        action_phase_manifest_identity={"count": len(rows), "content_sha256": sha256_json(rows)},
    )
    return rows, index, summary


def validate_training_index(payload, *, index_path, dataset_dir):
    if payload.get("experiment_version") != VERSION_TAG or not Path(index_path).is_file():
        raise ValueError("V11.2 requires its own Stage A index")
    if payload.get("training_data_hash") != sha256_json(
        {k: v for k, v in payload.items() if k != "training_data_hash"}
    ):
        raise ValueError("V11.2 Stage A index hash mismatch")
    settle = payload["target_policy"]["idle_compression"]["settle_frames"]
    if payload["target_policy"] != target_policy(settle):
        raise ValueError("V11.2 idle compression policy mismatch")
    for name, source in payload["source_files"].items():
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError(f"V11.2 source changed: {name}")
    expected_rows, expected, _ = build_artifacts(
        dataset_dir=Path(dataset_dir),
        v4_index_file=Path(payload["source_files"]["v4_training_index"]["path"]),
        norm_stats_dir=Path(payload["source_files"]["norm_stats"]["path"]).parent,
        settle_frames=settle,
    )
    rows = load_jsonl(Path(payload["source_files"]["action_phase_manifest"]["path"]))
    if rows != expected_rows:
        raise ValueError("V11.2 action targets differ from source-derived compressed H30")
    for key, value in expected.items():
        if key not in {"source_files", "action_phase_manifest_identity"} and payload.get(key) != value:
            raise ValueError(f"V11.2 source-derived {key} mismatch")
    if {k: payload["source_files"][k] for k in expected["source_files"]} != expected["source_files"]:
        raise ValueError("V11.2 Stage A source identity mismatch")
    expected_identity = expected["action_phase_manifest_identity"] | {
        "file_sha256": payload["source_files"]["action_phase_manifest"]["sha256"]
    }
    if payload["action_phase_manifest_identity"] != expected_identity:
        raise ValueError("V11.2 action manifest identity mismatch")
    return rows, {r["global_index"]: r for r in rows if r["trainable"]}
