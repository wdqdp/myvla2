"""V7.1 action-start filtering around move/reexecution boundaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.v4_data import SPLITS, V4Frame, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v5_adjustment_data import V2_TERMINAL_HOLD_SCHEMA
from tactile_vla.vla.v5_adjustment_data import compute_h30_content_identities
from tactile_vla.vla.v5_adjustment_data import phase_for_rexecution_frame
from tactile_vla.vla.v7_adjustment_data import _file_identity
from tactile_vla.vla.v7_adjustment_data import _validate_file_identity
from tactile_vla.vla.v7_adjustment_data import native_reexecution_timing_identity


ROTATION_PHASE_V7_1_ADJUSTMENT = "rotation_phase_v7_1_adjustment"
V7_1_EXPERIMENT_KIND = "phase_prompt_h30_terminal_hold_native_reexecution_static_filtered"
V7_1_ACTION_PHASE_SCHEMA = "tactile_vla_v7_1_adjustment_action_manifest_v1"
V7_1_TRAINING_INDEX_SCHEMA = "tactile_vla_v7_1_adjustment_training_index_v1"
V7_1_SUMMARY_SCHEMA = "tactile_vla_v7_1_adjustment_filter_summary_v1"
V7_1_HASH_SCHEMA = "tactile_vla_v7_1_adjustment_artifact_hashes_v1"

DEFAULT_FILTER_CONFIG: dict[str, Any] = {
    "arm_h30_motion_threshold_rad": 0.02,
    "gripper_h30_motion_threshold": 0.002,
    "minimum_static_run_frames": 10,
    "open_gripper_action_min": 0.09,
    "open_gripper_qpos_min": 0.09,
    "motion_cross_check": "action_target_and_puppet_qpos",
    "gripper_threshold_basis": {
        "population": "H30 windows with commanded gripper range <= 0.00011",
        "window_count": 159656,
        "puppet_qpos_h30_range_q995": 0.0011999979615211487,
        "selected_threshold": 0.002,
    },
}

EXCLUSION_REASONS = (
    "pre_move_static_wait",
    "pre_rexecution_static_wait",
    "post_reexecution_static_wait",
)


def _load_lerobot_signals(
    dataset_dir: Path, frame_count: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    import pyarrow.parquet as pq

    actions: list[np.ndarray | None] = [None] * frame_count
    qpos: list[np.ndarray | None] = [None] * frame_count
    for path in sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(path, columns=["index", "action", "observation.state"]).to_pydict()
        for raw_index, raw_action, raw_qpos in zip(
            table["index"], table["action"], table["observation.state"], strict=True
        ):
            index = int(raw_index)
            if not 0 <= index < frame_count or actions[index] is not None:
                raise ValueError(f"Invalid or duplicate LeRobot signal index {index}")
            action = np.asarray(raw_action, dtype=np.float32)
            state = np.asarray(raw_qpos, dtype=np.float32)
            if action.shape != (7,) or state.shape != (7,) or not (
                np.isfinite(action).all() and np.isfinite(state).all()
            ):
                raise ValueError(f"Invalid V7.1 action/qpos at global index {index}")
            actions[index], qpos[index] = action, state
    if any(value is None for value in actions) or any(value is None for value in qpos):
        raise ValueError("V7.1 LeRobot signal scan does not cover all global indices")
    return (
        [value for value in actions if value is not None],
        [value for value in qpos if value is not None],
    )


def _motion_metrics(
    *,
    frame: V4Frame,
    global_lookup: Mapping[int, V4Frame],
    actions: Sequence[np.ndarray],
    qpos: Sequence[np.ndarray],
    horizon: int,
) -> dict[str, float]:
    action_chunk: list[np.ndarray] = []
    qpos_chunk: list[np.ndarray] = []
    for offset in range(horizon):
        target = global_lookup.get(frame.global_index + offset)
        if target is None or target.attempt_key != frame.attempt_key:
            raise ValueError(f"V7.1 H30 crosses attempt at global index {frame.global_index}")
        action_chunk.append(actions[target.global_index])
        qpos_chunk.append(qpos[target.global_index])
    action_array, qpos_array = np.stack(action_chunk), np.stack(qpos_chunk)
    return {
        "arm_h30_motion": float(np.abs(action_array[:, :6] - action_array[0, :6]).max()),
        "arm_qpos_h30_motion": float(np.abs(qpos_array[:, :6] - qpos_array[0, :6]).max()),
        "gripper_h30_motion": float(np.abs(action_array[:, 6] - action_array[0, 6]).max()),
        "gripper_qpos_h30_motion": float(np.abs(qpos_array[:, 6] - qpos_array[0, 6]).max()),
        "gripper_action_start": float(action_array[0, 6]),
        "gripper_qpos_start": float(qpos_array[0, 6]),
    }


def contiguous_true_runs(
    frame_indices: Sequence[int], flags: Sequence[bool]
) -> list[tuple[int, int]]:
    """Return inclusive frame-index runs; gaps break a run."""

    if len(frame_indices) != len(flags):
        raise ValueError("frame_indices/flags length mismatch")
    runs: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for frame, enabled in zip(frame_indices, flags, strict=True):
        frame = int(frame)
        if enabled and (start is None or previous is None or frame != previous + 1):
            if start is not None and previous is not None:
                runs.append((start, previous))
            start = frame
        elif not enabled and start is not None and previous is not None:
            runs.append((start, previous))
            start = None
        previous = frame
    if start is not None and previous is not None:
        runs.append((start, previous))
    return runs


def select_boundary_exclusions(
    rows: Sequence[Mapping[str, Any]],
    *,
    move_start_frame: int | None,
    rexecution_frame: int,
    horizon: int,
    minimum_run: int,
) -> dict[int, str]:
    """Select only the final pre-move and rexecution-adjacent static H30 runs."""

    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    frames = [int(row["frame_index"]) for row in ordered]
    static = [bool(row["h30_static"]) for row in ordered]
    by_frame = {int(row["frame_index"]): row for row in ordered}
    runs = [run for run in contiguous_true_runs(frames, static) if run[1] - run[0] + 1 >= minimum_run]
    selected: dict[int, str] = {}

    if move_start_frame is not None:
        candidates = []
        for start, end in runs:
            if start >= move_start_frame:
                continue
            pre_move_end = min(end, move_start_frame - 1)
            if pre_move_end - start + 1 < minimum_run:
                continue
            run_rows = [by_frame[frame] for frame in range(start, pre_move_end + 1) if frame in by_frame]
            if len(run_rows) == pre_move_end - start + 1 and all(bool(row["gripper_stably_open"]) for row in run_rows):
                candidates.append((start, pre_move_end))
        if candidates:
            start, end = candidates[-1]
            selected.update({frame: "pre_move_static_wait" for frame in range(start, end + 1)})

    # An H30 static action-start run is adjacent to the boundary when at least
    # one of its original target windows covers rexecution_frame.
    candidates = [
        run
        for run in runs
        if run[0] <= rexecution_frame <= run[1] + horizon - 1
    ]
    if candidates:
        start, end = max(candidates, key=lambda run: (run[1], run[0]))
        for frame in range(start, end + 1):
            selected[frame] = (
                "pre_rexecution_static_wait"
                if frame < rexecution_frame
                else "post_reexecution_static_wait"
            )
    return selected


def build_v7_1_adjustment_artifacts(
    *, dataset_dir: Path, v4_index_file: Path, v4_norm_stats_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index_file = v4_index_file.expanduser().resolve()
    v4_norm_stats_dir = v4_norm_stats_dir.expanduser().resolve()
    v4_index = json.loads(v4_index_file.read_text())
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    timing_identity, rexecution_by_attempt = native_reexecution_timing_identity(v4_index, frames)
    actions, qpos = _load_lerobot_signals(dataset_dir, len(global_lookup))
    horizon = int(v4_index["action_horizon"])
    candidate_by_split = {
        split: [int(value) for value in v4_index["splits"][split]["execution_indices"]]
        for split in SPLITS
    }
    split_by_index = {
        global_index: split
        for split, indices in candidate_by_split.items()
        for global_index in indices
    }
    rows_by_attempt: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    action_rows: list[dict[str, Any]] = []
    arm_threshold = float(DEFAULT_FILTER_CONFIG["arm_h30_motion_threshold_rad"])
    gripper_threshold = float(DEFAULT_FILTER_CONFIG["gripper_h30_motion_threshold"])
    for global_index in [index for split in SPLITS for index in candidate_by_split[split]]:
        frame = global_lookup[global_index]
        rexecution = rexecution_by_attempt[frame.attempt_key]
        attempt_timing = v4_index["attempt_timing"][
            f"episode{frame.episode_id}/attempt{frame.attempt_id}"
        ]
        phase = phase_for_rexecution_frame(frame.attempt_id, frame.frame_index, rexecution)
        crosses = bool(
            phase == "adjustment" and rexecution is not None
            and frame.frame_index < rexecution <= frame.frame_index + horizon - 1
        )
        metrics = _motion_metrics(
            frame=frame, global_lookup=global_lookup, actions=actions, qpos=qpos, horizon=horizon
        )
        arm_static = metrics["arm_h30_motion"] <= arm_threshold and metrics["arm_qpos_h30_motion"] <= arm_threshold
        gripper_static = metrics["gripper_h30_motion"] <= gripper_threshold and metrics["gripper_qpos_h30_motion"] <= gripper_threshold
        row = {
            "schema_version": V7_1_ACTION_PHASE_SCHEMA,
            "data_profile": ROTATION_PHASE_V7_1_ADJUSTMENT,
            "prompt_profile": PHASE_PROMPT_PROFILE_V2,
            "experiment_kind": V7_1_EXPERIMENT_KIND,
            "split": split_by_index[global_index],
            "global_index": global_index,
            "episode_id": frame.episode_id,
            "attempt_id": frame.attempt_id,
            "frame_index": frame.frame_index,
            "phase": phase,
            "move_start_frame": attempt_timing.get("move_start_frame_index"),
            "rexecution_frame": rexecution,
            "trainable": True,
            "exclusion_reason": None,
            **metrics,
            "arm_h30_static": arm_static,
            "gripper_h30_static": gripper_static,
            "h30_static": arm_static and gripper_static,
            "gripper_stably_open": (
                gripper_static
                and metrics["gripper_action_start"] >= float(DEFAULT_FILTER_CONFIG["open_gripper_action_min"])
                and metrics["gripper_qpos_start"] >= float(DEFAULT_FILTER_CONFIG["open_gripper_qpos_min"])
            ),
            "raw_chunk_phase_pure": not crosses,
            "effective_chunk_phase_pure": True,
            "chunk_phase_pure": not crosses,
            "terminal_hold_from_offset": rexecution - frame.frame_index if crosses else None,
            "effective_h30_modified": crosses,
            "chunk_end_frame": frame.frame_index + horizon - 1,
            "action_horizon": horizon,
        }
        action_rows.append(row)
        rows_by_attempt[frame.attempt_key].append(row)

    per_attempt: list[dict[str, Any]] = []
    excluded_counter: Counter[str] = Counter()
    for (episode_id, attempt_id), attempt_rows in sorted(rows_by_attempt.items()):
        timing = v4_index["attempt_timing"][f"episode{episode_id}/attempt{attempt_id}"]
        exclusions: dict[int, str] = {}
        if attempt_id == 2:
            exclusions = select_boundary_exclusions(
                attempt_rows,
                move_start_frame=timing.get("move_start_frame_index"),
                rexecution_frame=int(timing["rexecution_frame_index"]),
                horizon=horizon,
                minimum_run=int(DEFAULT_FILTER_CONFIG["minimum_static_run_frames"]),
            )
        intervals: list[dict[str, Any]] = []
        current: tuple[int, int, str] | None = None
        for row in sorted(attempt_rows, key=lambda value: int(value["frame_index"])):
            frame_index = int(row["frame_index"])
            reason = exclusions.get(frame_index)
            if reason is not None:
                row["trainable"] = False
                row["exclusion_reason"] = reason
                excluded_counter[reason] += 1
                if current is None or current[2] != reason or frame_index != current[1] + 1:
                    if current is not None:
                        intervals.append({"start_frame": current[0], "end_frame": current[1], "reason": current[2]})
                    current = (frame_index, frame_index, reason)
                else:
                    current = (current[0], frame_index, reason)
        if current is not None:
            intervals.append({"start_frame": current[0], "end_frame": current[1], "reason": current[2]})
        per_attempt.append({
            "episode_id": episode_id,
            "attempt_id": attempt_id,
            "split": attempt_rows[0]["split"],
            "move_start_frame": timing.get("move_start_frame_index"),
            "rexecution_frame": timing.get("rexecution_frame_index"),
            "candidate_count": len(attempt_rows),
            "trainable_count": sum(bool(row["trainable"]) for row in attempt_rows),
            "excluded_count": len(exclusions),
            "exclusion_intervals": intervals,
        })

    split_entries: dict[str, Any] = {}
    for split in SPLITS:
        row_indices = [index for index, row in enumerate(action_rows) if row["split"] == split and row["trainable"]]
        indices = [int(action_rows[index]["global_index"]) for index in row_indices]
        selected = [action_rows[index] for index in row_indices]
        split_entries[split] = {
            "execution_indices": indices,
            "action_phase_manifest_row_indices": row_indices,
            "summary": {
                "candidate_count": len(candidate_by_split[split]),
                "action_count": len(indices),
                "excluded_count": len(candidate_by_split[split]) - len(indices),
                "phase_counts": dict(Counter(str(row["phase"]) for row in selected)),
                "exclusion_reason_counts": dict(Counter(
                    str(row["exclusion_reason"]) for row in action_rows
                    if row["split"] == split and not row["trainable"]
                )),
                "raw_chunk_crossing": sum(not bool(row["raw_chunk_phase_pure"]) for row in selected),
                "terminal_hold_chunks": sum(bool(row["effective_h30_modified"]) for row in selected),
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
        action_horizon=horizon,
    )
    norm_summary_path = v4_norm_stats_dir / "summary.json"
    norm_stats_path = v4_norm_stats_dir / "norm_stats.json"
    norm_summary = json.loads(norm_summary_path.read_text())
    norm_sha = sha256_file(norm_stats_path)
    if (
        norm_summary.get("norm_stats_sha256") != norm_sha
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity") != candidate_identity
    ):
        raise ValueError("V7.1 inherited V4 norm stats identity mismatch")
    source_files = v4_index.get("source_files", {})
    v4_h30_source_identity = {
        "action_horizon": horizon,
        "execution_indices": candidate_identity,
        "lerobot_identity": v4_index["lerobot_identity"],
        "lerobot_parquet_sha256": {name: value for name, value in sorted(source_files["lerobot_parquet"].items())},
    }
    v4_h30_source_identity["sha256"] = sha256_json(v4_h30_source_identity)
    summary = {
        "schema_version": V7_1_SUMMARY_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_1_ADJUSTMENT,
        "filter_config": DEFAULT_FILTER_CONFIG,
        "candidate_action_count": int(candidate_identity["all"]["count"]),
        "trainable_action_count": int(filtered_identity["all"]["count"]),
        "excluded_action_count": sum(excluded_counter.values()),
        "exclusion_reason_counts": dict(excluded_counter),
        "splits": {split: split_entries[split]["summary"] for split in SPLITS},
        "attempts": per_attempt,
    }
    index: dict[str, Any] = {
        "schema_version": V7_1_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_1_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V7_1_EXPERIMENT_KIND,
        "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
        "filter_config": DEFAULT_FILTER_CONFIG,
        "data_config_hash": sha256_json({
            "v4_training_data_hash": v4_index["training_data_hash"],
            "native_reexecution_timing_identity": timing_identity,
            "filter_config": DEFAULT_FILTER_CONFIG,
            "experiment_kind": V7_1_EXPERIMENT_KIND,
            "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
        }),
        "selection_hash": v4_index["selection_hash"],
        "v4_profile_config_hash": v4_index["profile_config_hash"],
        "dataset_dir": str(dataset_dir),
        "action_horizon": horizon,
        "splits": split_entries,
        "action_indices_identity": filtered_identity,
        "candidate_action_indices_identity": candidate_identity,
        "native_reexecution_timing_identity": timing_identity,
        "h30_target_identity": effective_h30,
        "raw_h30_target_identity": raw_h30,
        "v4_h30_source_identity": v4_h30_source_identity,
        "v4_lerobot_identity": v4_index["lerobot_identity"],
        "v4_training_data_hash": v4_index["training_data_hash"],
        "v4_norm_stats_sha256": norm_sha,
        "norm_stats_policy": {
            "kind": "inherited",
            "source_data_profile": "rotation_v4",
            "source_action_indices_identity": candidate_identity,
            "note": "V7.1 filters action starts but intentionally reuses V4 normalization statistics",
        },
        "h30_modifications": modifications,
        "summary": {key: value for key, value in summary.items() if key != "attempts"},
        "source_files": {
            "v4_training_index": _file_identity(v4_index_file),
            "v4_norm_summary": _file_identity(norm_summary_path),
            "v4_norm_stats": _file_identity(norm_stats_path),
        },
        "action_phase_manifest_identity": {"count": len(action_rows), "content_sha256": sha256_json(action_rows)},
    }
    return action_rows, index, summary


def validate_v7_1_adjustment_training_index(
    payload: Mapping[str, Any], *, index_path: Path | None = None, dataset_dir: Path | None = None,
    revalidate_h30_targets: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    expected = {
        "schema_version": V7_1_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_1_ADJUSTMENT,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V7_1_EXPERIMENT_KIND,
        "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
        "filter_config": DEFAULT_FILTER_CONFIG,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V7.1 index {key} mismatch")
    if payload.get("training_data_hash") != sha256_json({key: value for key, value in payload.items() if key != "training_data_hash"}):
        raise ValueError("V7.1 training_data_hash mismatch")
    if index_path is not None and not Path(index_path).is_file():
        raise FileNotFoundError(index_path)
    sources = payload.get("source_files", {})
    required = {"v4_training_index", "v4_norm_summary", "v4_norm_stats", "action_phase_manifest", "filter_summary"}
    if not isinstance(sources, Mapping) or not required.issubset(sources):
        raise ValueError("V7.1 index lacks source identities")
    paths = {name: _validate_file_identity(sources[name], context=name) for name in required}
    v4_index = json.loads(paths["v4_training_index"].read_text())
    effective_dataset_dir = (dataset_dir or Path(str(payload["dataset_dir"]))).expanduser().resolve()
    frames, global_lookup = validate_v4_index_dataset(v4_index, effective_dataset_dir)
    timing, boundaries = native_reexecution_timing_identity(v4_index, frames)
    if timing != payload.get("native_reexecution_timing_identity"):
        raise ValueError("V7.1 native timing identity mismatch")
    if payload.get("candidate_action_indices_identity") != v4_index.get("action_indices_identity"):
        raise ValueError("V7.1 candidate action identity differs from V4")
    if payload.get("norm_stats_policy", {}).get("kind") != "inherited":
        raise ValueError("V7.1 must identify V4 norm stats as inherited")
    rows = load_jsonl(paths["action_phase_manifest"])
    manifest_identity = payload["action_phase_manifest_identity"]
    if len(rows) != int(manifest_identity["count"]) or sha256_json(rows) != manifest_identity["content_sha256"] or sha256_file(paths["action_phase_manifest"]) != manifest_identity["file_sha256"]:
        raise ValueError("V7.1 action manifest identity mismatch")
    lookup: dict[int, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        global_index = int(row["global_index"])
        frame = global_lookup[global_index]
        rexecution = boundaries[frame.attempt_key]
        if row.get("data_profile") != ROTATION_PHASE_V7_1_ADJUSTMENT or row.get("phase") != phase_for_rexecution_frame(frame.attempt_id, frame.frame_index, rexecution):
            raise ValueError(f"Invalid V7.1 manifest row at {global_index}")
        reason = row.get("exclusion_reason")
        if bool(row.get("trainable")) == (reason is not None):
            raise ValueError(f"Invalid V7.1 trainable/exclusion fields at {global_index}")
        if reason is not None:
            if reason not in EXCLUSION_REASONS or not bool(row.get("h30_static")):
                raise ValueError(f"Invalid V7.1 exclusion at {global_index}")
            reason_counts[str(reason)] += 1
        else:
            lookup[global_index] = row
            selected_rows.append(row)
    for split in SPLITS:
        expected_indices = [int(row["global_index"]) for row in rows if row["split"] == split and row["trainable"]]
        if payload["splits"][split]["execution_indices"] != expected_indices:
            raise ValueError(f"V7.1 {split} loader indices include excluded samples")
        original_episodes = {global_lookup[index].episode_id for index in v4_index["splits"][split]["execution_indices"]}
        filtered_episodes = {global_lookup[index].episode_id for index in expected_indices}
        if filtered_episodes != original_episodes:
            raise ValueError(f"V7.1 filtering changed {split} episode coverage")
    if action_indices_identity(payload["splits"]) != payload.get("action_indices_identity"):
        raise ValueError("V7.1 filtered action identity mismatch")
    summary = json.loads(paths["filter_summary"].read_text())
    if summary.get("exclusion_reason_counts") != dict(reason_counts):
        raise ValueError("V7.1 summary exclusion counts mismatch")
    norm_summary = json.loads(paths["v4_norm_summary"].read_text())
    norm_sha = sha256_file(paths["v4_norm_stats"])
    if norm_sha != payload.get("v4_norm_stats_sha256") or norm_summary.get("artifact_identity", {}).get("action_indices_identity") != payload.get("candidate_action_indices_identity"):
        raise ValueError("V7.1 inherited norm identity mismatch")
    if revalidate_h30_targets:
        raw, effective, modifications = compute_h30_content_identities(
            dataset_dir=effective_dataset_dir, global_lookup=global_lookup, action_rows=selected_rows,
            action_indices_identity=payload["action_indices_identity"], action_horizon=int(payload["action_horizon"]),
        )
        if raw != payload.get("raw_h30_target_identity") or effective != payload.get("h30_target_identity") or modifications != payload.get("h30_modifications"):
            raise ValueError("V7.1 persisted H30 identities mismatch")
    return rows, lookup


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    result = {
        "schema_version": V7_1_HASH_SCHEMA,
        "data_profile": ROTATION_PHASE_V7_1_ADJUSTMENT,
        "files": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in sorted(paths.items())},
    }
    result["sha256"] = sha256_json(result)
    return result
