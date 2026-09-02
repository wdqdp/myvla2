"""V5.3 adjustment-end manifest construction and strict validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.artifacts import canonical_json_bytes
from tactile_vla.vla.v4_data import SPLITS, V4Frame, file_identity, load_jsonl
from tactile_vla.vla.v5_adjustment_data import validate_v5_adjustment_training_index
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_PROMPT_PROFILE
from tactile_vla.vla.v5_3_phase_change import QPOS_HISTORY_FRAMES
from tactile_vla.vla.v5_3_phase_change import QPOS_SAMPLE_OFFSETS
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v5_3_phase_change import build_adjustment_end_prompt
from tactile_vla.vla.v5_3_phase_change import helper_identity
from tactile_vla.vla.v5_3_phase_change import normalize_state_qpos
from tactile_vla.vla.v5_3_phase_change import pi05_phase_change_token_length
from tactile_vla.vla.v5_3_phase_change import runtime_reachable_endpoint


DATA_PROFILE = "rotation_phase_v5_adjustment_end_v2"
EXPERIMENT_KIND = "adjustment_end_qpos_h30_text"
MANIFEST_SCHEMA = "tactile_vla_v5_3_adjustment_end_manifest_v2"
TRAINING_INDEX_SCHEMA = "tactile_vla_v5_3_adjustment_end_training_index_v2"
SUMMARY_SCHEMA = "tactile_vla_v5_3_adjustment_end_summary_v2"
ARTIFACT_HASH_SCHEMA = "tactile_vla_v5_3_artifact_hashes_v1"
ADJUSTMENT_END_START_OFFSET = -10
ADJUSTMENT_END_END_OFFSET = 5
LABEL_POLICY = {
    "boundary": "rexecution_frame",
    "positive_start_offset_inclusive": ADJUSTMENT_END_START_OFFSET,
    "positive_end_offset_inclusive": ADJUSTMENT_END_END_OFFSET,
    "valid_end_offset_inclusive": ADJUSTMENT_END_END_OFFSET,
}
EXPECTED_ATTEMPT2_COUNTS = {"train": 328, "val": 44, "test": 36}
EXPECTED_POSITIVE_COUNTS = {"train": 5248, "val": 704, "test": 576}
EXPECTED_SAMPLE_COUNTS = {"train": 74480, "val": 9684, "test": 8190}
EXPECTED_MISSING_ATTEMPT1_COUNTS = {"train": 60, "val": 13, "test": 7}
EXPECTED_HISTORY_UNAVAILABLE_COUNTS = {"train": 1800, "val": 390, "test": 210}
EXPECTED_ATTEMPT2_COUNT = 408


def is_adjustment_end_positive(frame_index: int, rexecution_frame: int) -> bool:
    relative = int(frame_index) - int(rexecution_frame)
    return ADJUSTMENT_END_START_OFFSET <= relative <= ADJUSTMENT_END_END_OFFSET


def is_adjustment_end_valid(frame_index: int, rexecution_frame: int) -> bool:
    return int(frame_index) <= int(rexecution_frame) + ADJUSTMENT_END_END_OFFSET


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_state_quantiles(norm_stats_file: Path) -> StateQuantileStats:
    payload = _load_object(norm_stats_file)
    try:
        state = payload["norm_stats"]["state"]
        return StateQuantileStats(q01=state["q01"], q99=state["q99"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{norm_stats_file}: missing norm_stats.state.q01/q99") from exc


def scan_selected_qpos(
    *,
    dataset_dir: Path,
    selected_episode_ids: set[int],
) -> dict[int, np.ndarray]:
    """Read raw current qpos by immutable LeRobot global index."""

    import pyarrow.parquet as pq

    columns = ("index", "episode_id", "attempt_id", "frame_index", "observation.state")
    result: dict[int, np.ndarray] = {}
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files under {dataset_dir / 'data'}")
    for path in parquet_files:
        schema = set(pq.read_schema(path).names)
        missing = sorted(set(columns) - schema)
        if missing:
            raise ValueError(f"{path}: missing V5.3 qpos columns {missing}")
        data = pq.read_table(path, columns=list(columns)).to_pydict()
        for offset, episode_value in enumerate(data["episode_id"]):
            episode_id = int(episode_value)
            if episode_id not in selected_episode_ids:
                continue
            global_index = int(data["index"][offset])
            qpos = np.asarray(data["observation.state"][offset], dtype=np.float64)
            if qpos.shape != (7,) or not np.isfinite(qpos).all():
                raise ValueError(f"global_index={global_index}: state must be finite [7], got {qpos.shape}")
            if global_index in result:
                raise ValueError(f"Duplicate LeRobot global index {global_index}")
            result[global_index] = qpos
    return result


def _token_summary(lengths: list[int]) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=np.int32)
    if values.size == 0:
        raise ValueError("No V5.3 prompt token lengths were produced")
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "p50": int(np.percentile(values, 50)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
        "max": int(values.max()),
        "over_limit_count": int(np.count_nonzero(values > PHASE_CHANGE_MAX_TOKEN_LEN)),
    }


def _frame_groups(frames: list[V4Frame], selected_episode_ids: set[int]) -> dict[tuple[int, int], list[V4Frame]]:
    groups: dict[tuple[int, int], list[V4Frame]] = defaultdict(list)
    for frame in frames:
        if frame.episode_id in selected_episode_ids:
            groups[frame.attempt_key].append(frame)
    for key, rows in groups.items():
        rows.sort(key=lambda item: item.frame_index)
        if [row.frame_index for row in rows] != list(range(len(rows))):
            raise ValueError(f"Non-contiguous frame indices for episode/attempt {key}")
        if any(left.global_index >= right.global_index for left, right in zip(rows, rows[1:])):
            raise ValueError(f"Non-increasing global indices for episode/attempt {key}")
    return groups


def build_adjustment_end_artifacts(
    *,
    dataset_dir: Path,
    v5_2_index_file: Path,
    norm_stats_file: Path,
    captioner_checkpoint: Path,
    caption_summary_file: Path,
    backbone_checkpoint: Path,
    backbone_config_file: Path,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build all in-memory V5.3 data artifacts from immutable V5.2 labels."""

    v5_2_index = _load_object(v5_2_index_file)
    _, _ = validate_v5_adjustment_training_index(
        v5_2_index,
        index_path=v5_2_index_file,
        dataset_dir=dataset_dir,
    )
    source_files = v5_2_index["source_files"]
    boundary_file = Path(str(source_files["phase_boundaries"]["path"]))
    v4_index_file = Path(str(source_files["v4_training_index"]["path"]))
    v4_index = _load_object(v4_index_file)
    # The strict V5.2 validator already validates V4 rows and returns no frame
    # list, so obtain it from the V4 validator explicitly for history joins.
    from tactile_vla.vla.v4_data import validate_v4_index_dataset

    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    boundary_rows = load_jsonl(boundary_file)
    attempt2_boundaries = [row for row in boundary_rows if int(row["attempt_id"]) == 2]
    if len(attempt2_boundaries) != EXPECTED_ATTEMPT2_COUNT:
        raise ValueError(f"Expected 408 attempt2 boundaries, got {len(attempt2_boundaries)}")
    selected_episode_ids = {int(row["episode_id"]) for row in attempt2_boundaries}
    if len(selected_episode_ids) != EXPECTED_ATTEMPT2_COUNT:
        raise ValueError("Each V5.3 attempt2 boundary must belong to a unique physical episode")

    groups = _frame_groups(frames, selected_episode_ids)
    qpos_lookup = scan_selected_qpos(dataset_dir=dataset_dir, selected_episode_ids=selected_episode_ids)
    stats = load_state_quantiles(norm_stats_file)
    prompt_helper = helper_identity()

    manifest: list[dict[str, Any]] = []
    split_manifest_rows: dict[str, list[int]] = {split: [] for split in SPLITS}
    split_global_indices: dict[str, list[int]] = {split: [] for split in SPLITS}
    token_lengths: list[int] = []
    split_attempts: Counter[str] = Counter()
    split_missing_attempt1: Counter[str] = Counter()
    split_history_unavailable: Counter[str] = Counter()
    seen_boundaries: set[tuple[int, int]] = set()

    for boundary in sorted(attempt2_boundaries, key=lambda row: int(row["first_global_index"])):
        episode_id = int(boundary["episode_id"])
        key = (episode_id, 2)
        if key in seen_boundaries:
            raise ValueError(f"Duplicate V5.2 attempt2 boundary {key}")
        seen_boundaries.add(key)
        split = str(boundary["split"])
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split!r} for {key}")
        split_attempts[split] += 1
        attempt1 = groups.get((episode_id, 1), [])
        attempt2 = groups.get(key, [])
        missing_attempt1 = len(attempt1) == 0
        if missing_attempt1:
            split_missing_attempt1[split] += 1
        if len(attempt2) != int(boundary["frame_count"]):
            raise ValueError(f"episode {episode_id} attempt2 frame count differs from V5.2 boundary")
        rexecution = int(boundary["rexecution_frame"])
        if not (
            rexecution >= -ADJUSTMENT_END_START_OFFSET
            and rexecution + ADJUSTMENT_END_END_OFFSET < len(attempt2)
        ):
            raise ValueError(f"episode {episode_id}: invalid rexecution_frame={rexecution}")
        if attempt2[rexecution].key != (episode_id, 2, rexecution):
            raise ValueError(f"episode {episode_id}: rexecution frame identity is not unique")

        physical_timeline = attempt1 + attempt2
        attempt2_start = len(attempt1)
        for frame in attempt2:
            timeline_position = attempt2_start + frame.frame_index
            history_available = timeline_position >= QPOS_HISTORY_FRAMES
            adjustment_end = is_adjustment_end_positive(frame.frame_index, rexecution)
            valid = is_adjustment_end_valid(frame.frame_index, rexecution)
            training_sample_valid = valid and history_available
            if not history_available:
                if frame.frame_index >= QPOS_HISTORY_FRAMES:
                    raise ValueError(
                        f"episode {episode_id} frame {frame.frame_index}: unexpected qpos history gap"
                    )
                split_history_unavailable[split] += 1
                unavailable_reason = (
                    "physical_episode_has_no_attempt1"
                    if missing_attempt1
                    else "insufficient_same_episode_history"
                )
                row = {
                    "schema_version": MANIFEST_SCHEMA,
                    "data_profile": DATA_PROFILE,
                    "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
                    "experiment_kind": EXPERIMENT_KIND,
                    "episode_id": episode_id,
                    "attempt_id": 2,
                    "frame_index": frame.frame_index,
                    "current_global_index": frame.global_index,
                    "rexecution_frame": rexecution,
                    "split": split,
                    "history_global_indices": [],
                    "history_attempt_ids": [],
                    "history_crosses_attempt": False,
                    "history_crosses_episode": False,
                    "history_available": False,
                    "history_unavailable_reason": unavailable_reason,
                    "runtime_reachable_endpoint": runtime_reachable_endpoint(frame.frame_index),
                    "qpos_h30_sample_offsets": list(QPOS_SAMPLE_OFFSETS),
                    "qpos_h10_discrete": [],
                    "adjustment_end": adjustment_end,
                    "adjustment_end_valid": valid,
                    "classification_sample_valid": False,
                    "phase_change_token_len": None,
                    "prompt": None,
                }
                manifest.append(row)
                continue
            history_frames = physical_timeline[
                timeline_position - QPOS_HISTORY_FRAMES : timeline_position
            ]
            if len(history_frames) != QPOS_HISTORY_FRAMES:
                raise ValueError(f"episode {episode_id} frame {frame.frame_index}: incomplete qpos_h30")
            history_globals = [item.global_index for item in history_frames]
            if any(item.episode_id != episode_id for item in history_frames):
                raise ValueError("qpos_h30 crosses physical episode")
            if any(left >= right for left, right in zip(history_globals, history_globals[1:])):
                raise ValueError("qpos_h30 global indices are not strictly increasing")
            if history_globals[-1] >= frame.global_index:
                raise ValueError("qpos_h30 contains current/future frame")
            try:
                history_qpos = np.stack([qpos_lookup[index] for index in history_globals])
                current_qpos = qpos_lookup[frame.global_index]
            except KeyError as exc:
                raise ValueError(f"Missing LeRobot qpos for global index {exc.args[0]}") from exc
            prompt, discrete = build_adjustment_end_prompt(
                instruction=frame.instruction,
                tactile_caption=frame.tactile_caption,
                recovery_plan=frame.input_recovery_plan,
                qpos_h30=history_qpos,
                stats=stats,
            )
            normalized_current = normalize_state_qpos(current_qpos, stats)
            token_length = pi05_phase_change_token_length(
                tokenizer=tokenizer,
                prompt=prompt,
                normalized_current_qpos=normalized_current,
            )
            if token_length > PHASE_CHANGE_MAX_TOKEN_LEN:
                raise ValueError(
                    f"V5.3 phase-change prompt truncation at global_index={frame.global_index}: "
                    f"tokens={token_length}, limit={PHASE_CHANGE_MAX_TOKEN_LEN}"
                )
            history_attempt_ids = [item.attempt_id for item in history_frames]
            row = {
                "schema_version": MANIFEST_SCHEMA,
                "data_profile": DATA_PROFILE,
                "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
                "experiment_kind": EXPERIMENT_KIND,
                "episode_id": episode_id,
                "attempt_id": 2,
                "frame_index": frame.frame_index,
                "current_global_index": frame.global_index,
                "rexecution_frame": rexecution,
                "split": split,
                "history_global_indices": history_globals,
                "history_attempt_ids": history_attempt_ids,
                "history_crosses_attempt": len(set(history_attempt_ids)) > 1,
                "history_crosses_episode": False,
                "history_available": True,
                "history_unavailable_reason": None,
                "runtime_reachable_endpoint": runtime_reachable_endpoint(frame.frame_index),
                "qpos_h30_sample_offsets": list(QPOS_SAMPLE_OFFSETS),
                "qpos_h10_discrete": discrete.tolist(),
                "adjustment_end": adjustment_end,
                "adjustment_end_valid": valid,
                "classification_sample_valid": training_sample_valid,
                "phase_change_token_len": token_length,
                "prompt": prompt,
            }
            row_index = len(manifest)
            manifest.append(row)
            token_lengths.append(token_length)
            if training_sample_valid:
                split_manifest_rows[split].append(row_index)
                split_global_indices[split].append(frame.global_index)

    if dict(split_attempts) != EXPECTED_ATTEMPT2_COUNTS:
        raise ValueError(f"V5.3 attempt2 split changed: {dict(split_attempts)}")
    if dict(split_missing_attempt1) != EXPECTED_MISSING_ATTEMPT1_COUNTS:
        raise ValueError(f"V5.3 missing-attempt1 split changed: {dict(split_missing_attempt1)}")
    if dict(split_history_unavailable) != EXPECTED_HISTORY_UNAVAILABLE_COUNTS:
        raise ValueError(
            f"V5.3 history-unavailable split changed: {dict(split_history_unavailable)}"
        )

    splits: dict[str, Any] = {}
    for split in SPLITS:
        indices = split_manifest_rows[split]
        selected = [manifest[index] for index in indices]
        positive = sum(bool(row["adjustment_end"]) for row in selected)
        negative = len(selected) - positive
        if len(selected) != EXPECTED_SAMPLE_COUNTS[split]:
            raise ValueError(f"V5.3 {split} sample count changed: {len(selected)}")
        if positive != EXPECTED_POSITIVE_COUNTS[split]:
            raise ValueError(f"V5.3 {split} positive count changed: {positive}")
        splits[split] = {
            "manifest_row_indices": indices,
            "global_indices": split_global_indices[split],
            "sample_count": len(selected),
            "positive_count": positive,
            "negative_count": negative,
            "attempt2_count": split_attempts[split],
            "missing_attempt1_count": split_missing_attempt1[split],
            "history_unavailable_count": split_history_unavailable[split],
        }

    caption_summary = _load_object(caption_summary_file)
    caption_checkpoint_identity = file_identity(captioner_checkpoint)
    caption_source = {
        "field": "tactile_caption",
        "window_size": 30,
        "checkpoint": caption_checkpoint_identity,
        "annotation_summary": file_identity(caption_summary_file),
    }
    if int(caption_summary.get("window_size", -1)) != 30:
        raise ValueError("V4 tactile caption source does not use window_size=30")
    if Path(str(caption_summary.get("checkpoint", ""))).resolve() != captioner_checkpoint.resolve():
        raise ValueError("V4 tactile caption summary references a different checkpoint")
    if int(caption_summary.get("annotated_frames", -1)) != int(v5_2_index["v4_lerobot_identity"]["frame_count"]):
        raise ValueError("V4 tactile caption annotation does not cover the immutable LeRobot dataset")

    source_identity = {
        "v5_2_training_index": file_identity(v5_2_index_file),
        "v5_2_phase_boundaries": file_identity(boundary_file),
        "v5_2_action_phase_manifest": file_identity(Path(str(source_files["action_phase_manifest"]["path"]))),
        "v4_training_index": file_identity(v4_index_file),
        "v4_norm_stats": file_identity(norm_stats_file),
        "caption_source": caption_source,
        "backbone_checkpoint": {"path": str(backbone_checkpoint.resolve()), "step": 15000},
        "backbone_config": file_identity(backbone_config_file),
    }
    index: dict[str, Any] = {
        "schema_version": TRAINING_INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "dataset_dir": str(dataset_dir.resolve()),
        "selection_hash": v5_2_index["selection_hash"],
        "attempt2_count": EXPECTED_ATTEMPT2_COUNT,
        "missing_attempt1_count": sum(EXPECTED_MISSING_ATTEMPT1_COUNTS.values()),
        "history_unavailable_count": sum(EXPECTED_HISTORY_UNAVAILABLE_COUNTS.values()),
        "label_policy": LABEL_POLICY,
        "splits": splits,
        "manifest_identity": {
            "count": len(manifest),
            "content_sha256": sha256_json(manifest),
        },
        "prompt_helper": prompt_helper,
        "prompt_token_lengths": _token_summary(token_lengths),
        "state_norm": {
            "method": "q01_q99_pi05",
            "norm_stats_sha256": sha256_file(norm_stats_file),
            "q01": stats.q01.tolist(),
            "q99": stats.q99.tolist(),
        },
        "caption_source": caption_source,
        "source_files": source_identity,
    }
    index["training_data_hash"] = sha256_json(index)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "attempt2_count": EXPECTED_ATTEMPT2_COUNT,
        "missing_attempt1_count": sum(EXPECTED_MISSING_ATTEMPT1_COUNTS.values()),
        "history_unavailable_count": sum(EXPECTED_HISTORY_UNAVAILABLE_COUNTS.values()),
        "label_policy": LABEL_POLICY,
        "manifest_frame_count": len(manifest),
        "valid_sample_count": sum(value["sample_count"] for value in splits.values()),
        "positive_count": sum(value["positive_count"] for value in splits.values()),
        "negative_count": sum(value["negative_count"] for value in splits.values()),
        "split_summary": {
            split: {
                key: value
                for key, value in splits[split].items()
                if key not in {"manifest_row_indices", "global_indices"}
            }
            for split in SPLITS
        },
        "prompt_token_lengths": index["prompt_token_lengths"],
        "training_data_hash": index["training_data_hash"],
    }
    return manifest, index, summary


def validate_adjustment_end_artifacts(
    *,
    index: Mapping[str, Any],
    manifest: list[dict[str, Any]],
) -> None:
    expected = {
        "schema_version": TRAINING_INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "attempt2_count": EXPECTED_ATTEMPT2_COUNT,
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise ValueError(f"V5.3 index {key}={index.get(key)!r}, expected {value!r}")
    if int(index.get("missing_attempt1_count", -1)) != sum(
        EXPECTED_MISSING_ATTEMPT1_COUNTS.values()
    ):
        raise ValueError("V5.3 missing-attempt1 count mismatch")
    if int(index.get("history_unavailable_count", -1)) != sum(
        EXPECTED_HISTORY_UNAVAILABLE_COUNTS.values()
    ):
        raise ValueError("V5.3 history-unavailable count mismatch")
    if index.get("label_policy") != LABEL_POLICY:
        raise ValueError("V5.3 label policy mismatch")
    actual_hash = sha256_json({key: value for key, value in index.items() if key != "training_data_hash"})
    if index.get("training_data_hash") != actual_hash:
        raise ValueError("V5.3 training_data_hash mismatch")
    identity = index.get("manifest_identity", {})
    if (
        int(identity.get("count", -1)) != len(manifest)
        or identity.get("content_sha256") != sha256_json(manifest)
    ):
        raise ValueError("V5.3 manifest content identity mismatch")
    if index.get("prompt_helper") != helper_identity():
        raise ValueError("V5.3 prompt helper identity mismatch")
    for split in SPLITS:
        split_payload = index["splits"][split]
        if int(split_payload.get("missing_attempt1_count", -1)) != EXPECTED_MISSING_ATTEMPT1_COUNTS[split]:
            raise ValueError(f"V5.3 {split} missing-attempt1 count mismatch")
        if int(split_payload.get("history_unavailable_count", -1)) != EXPECTED_HISTORY_UNAVAILABLE_COUNTS[split]:
            raise ValueError(f"V5.3 {split} history-unavailable count mismatch")
        rows = [manifest[int(value)] for value in split_payload["manifest_row_indices"]]
        if len(rows) != EXPECTED_SAMPLE_COUNTS[split]:
            raise ValueError(f"V5.3 {split} sample count mismatch")
        globals_ = [int(row["current_global_index"]) for row in rows]
        if globals_ != [int(value) for value in split_payload["global_indices"]]:
            raise ValueError(f"V5.3 {split} global/manifest order mismatch")
        if any(
            not row["adjustment_end_valid"]
            or not row["classification_sample_valid"]
            or not row["history_available"]
            or row["split"] != split
            for row in rows
        ):
            raise ValueError(f"V5.3 {split} index contains invalid sample")
        positives = sum(bool(row["adjustment_end"]) for row in rows)
        if positives != EXPECTED_POSITIVE_COUNTS[split]:
            raise ValueError(f"V5.3 {split} positive count mismatch")
        if len({int(row["episode_id"]) for row in rows}) != EXPECTED_ATTEMPT2_COUNTS[split]:
            raise ValueError(f"V5.3 {split} attempt count mismatch")
    for row in manifest:
        if row.get("schema_version") != MANIFEST_SCHEMA or int(row["attempt_id"]) != 2:
            raise ValueError("V5.3 manifest schema/attempt mismatch")
        frame = int(row["frame_index"])
        rexecution = int(row["rexecution_frame"])
        if bool(row["adjustment_end"]) != is_adjustment_end_positive(frame, rexecution):
            raise ValueError("V5.3 adjustment_end label mismatch")
        if bool(row["adjustment_end_valid"]) != is_adjustment_end_valid(frame, rexecution):
            raise ValueError("V5.3 adjustment_end_valid mismatch")
        history_available = bool(row.get("history_available"))
        expected_sample_valid = is_adjustment_end_valid(frame, rexecution) and history_available
        if bool(row.get("classification_sample_valid")) != expected_sample_valid:
            raise ValueError("V5.3 classification_sample_valid mismatch")
        history = [int(value) for value in row["history_global_indices"]]
        if history_available:
            if len(history) != 30 or any(left >= right for left, right in pairwise(history)):
                raise ValueError("V5.3 manifest has invalid qpos history identity")
            if int(row["phase_change_token_len"]) > PHASE_CHANGE_MAX_TOKEN_LEN:
                raise ValueError("V5.3 manifest contains truncated prompt")
            if row.get("prompt") is None or len(row.get("qpos_h10_discrete", [])) != 10:
                raise ValueError("V5.3 history-available row lacks prompt/qpos_h10")
        else:
            if not (
                frame < 30
                and not history
                and row.get("prompt") is None
                and row.get("phase_change_token_len") is None
                and row.get("history_unavailable_reason")
                in {
                    "physical_episode_has_no_attempt1",
                    "insufficient_same_episode_history",
                }
            ):
                raise ValueError("V5.3 history-unavailable audit row is invalid")
        if bool(row["runtime_reachable_endpoint"]) != runtime_reachable_endpoint(frame):
            raise ValueError("V5.3 runtime endpoint flag mismatch")
    unavailable_by_split = Counter(
        str(row["split"]) for row in manifest if not bool(row.get("history_available"))
    )
    if dict(unavailable_by_split) != EXPECTED_HISTORY_UNAVAILABLE_COUNTS:
        raise ValueError("V5.3 manifest history-unavailable split counts mismatch")


def load_indexed_manifest_rows(
    *,
    index: Mapping[str, Any],
    manifest_path: Path,
) -> dict[int, dict[str, Any]]:
    """Stream-validate the large manifest and retain only training-index rows."""

    expected = {
        "schema_version": TRAINING_INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "attempt2_count": EXPECTED_ATTEMPT2_COUNT,
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise ValueError(f"V5.3 index {key}={index.get(key)!r}, expected {value!r}")
    if index.get("label_policy") != LABEL_POLICY:
        raise ValueError("V5.3 index label policy mismatch")
    actual_index_hash = sha256_json(
        {key: value for key, value in index.items() if key != "training_data_hash"}
    )
    if index.get("training_data_hash") != actual_index_hash:
        raise ValueError("V5.3 training_data_hash mismatch")

    wanted: dict[int, tuple[str, int]] = {}
    for split in SPLITS:
        payload = index["splits"][split]
        row_indices = [int(value) for value in payload["manifest_row_indices"]]
        global_indices = [int(value) for value in payload["global_indices"]]
        if len(row_indices) != len(global_indices):
            raise ValueError(f"V5.3 {split} row/global index lengths differ")
        for row_index, global_index in zip(row_indices, global_indices, strict=True):
            if row_index in wanted:
                raise ValueError(f"V5.3 duplicate indexed manifest row {row_index}")
            wanted[row_index] = (split, global_index)

    import hashlib

    file_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    content_digest.update(b"[")
    selected: dict[int, dict[str, Any]] = {}
    row_count = 0
    with manifest_path.open("rb") as stream:
        for raw_line in stream:
            file_digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"V5.3 manifest row {row_count} is not an object")
            if row_count:
                content_digest.update(b",")
            content_digest.update(canonical_json_bytes(row))
            if row_count in wanted:
                split, global_index = wanted[row_count]
                if (
                    row.get("schema_version") != MANIFEST_SCHEMA
                    or row.get("split") != split
                    or int(row.get("current_global_index", -1)) != global_index
                    or not bool(row.get("adjustment_end_valid"))
                    or not bool(row.get("classification_sample_valid"))
                    or not bool(row.get("history_available"))
                    or len(row.get("history_global_indices", [])) != QPOS_HISTORY_FRAMES
                    or len(row.get("qpos_h10_discrete", [])) != len(QPOS_SAMPLE_OFFSETS)
                    or row.get("prompt") is None
                    or bool(row.get("adjustment_end"))
                    != is_adjustment_end_positive(
                        int(row.get("frame_index", -1)),
                        int(row.get("rexecution_frame", -1)),
                    )
                    or bool(row.get("adjustment_end_valid"))
                    != is_adjustment_end_valid(
                        int(row.get("frame_index", -1)),
                        int(row.get("rexecution_frame", -1)),
                    )
                ):
                    raise ValueError(f"V5.3 indexed manifest row {row_count} is invalid")
                selected[row_count] = row
            row_count += 1
    content_digest.update(b"]")
    identity = index["manifest_identity"]
    if row_count != int(identity["count"]):
        raise ValueError("V5.3 manifest row count mismatch")
    if content_digest.hexdigest() != identity["content_sha256"]:
        raise ValueError("V5.3 manifest canonical content SHA mismatch")
    expected_file_sha = str(identity.get("file_sha256", ""))
    if len(expected_file_sha) != 64 or file_digest.hexdigest() != expected_file_sha:
        raise ValueError("V5.3 manifest file SHA mismatch")
    if set(selected) != set(wanted):
        missing = sorted(set(wanted) - set(selected))
        raise ValueError(f"V5.3 manifest lacks indexed rows: {missing[:10]}")
    return selected


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_HASH_SCHEMA,
        "artifacts": {name: file_identity(path) for name, path in sorted(paths.items())},
    }


class AdjustmentEndManifestDataset:
    """Read only samples selected by the immutable V5.3 training index."""

    def __init__(
        self,
        *,
        manifest: Mapping[int, dict[str, Any]] | list[dict[str, Any]],
        manifest_row_indices: list[int],
        global_indices: list[int],
        lerobot_dataset: Any,
        state_history_len: int = 60,
    ) -> None:
        self.manifest = manifest
        self.row_indices = [int(value) for value in manifest_row_indices]
        self.global_indices = [int(value) for value in global_indices]
        self._dataset = lerobot_dataset
        self.state_history_len = int(state_history_len)
        if len(self.row_indices) != len(self.global_indices):
            raise ValueError("V5.3 manifest/global index lengths differ")
        for row_index, global_index in zip(self.row_indices, self.global_indices, strict=True):
            row = self.manifest[row_index]
            if (
                int(row["current_global_index"]) != global_index
                or not row["adjustment_end_valid"]
                or not row["classification_sample_valid"]
            ):
                raise ValueError("V5.3 dataset index points to an invalid manifest row")

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, dataset_index: int) -> dict[str, Any]:
        row = self.manifest[self.row_indices[dataset_index]]
        global_index = self.global_indices[dataset_index]
        item = self._dataset[global_index]
        identity = (
            int(item["index"]),
            int(item["episode_id"]),
            int(item["attempt_id"]),
            int(item["frame_index"]),
        )
        expected = (
            global_index,
            int(row["episode_id"]),
            int(row["attempt_id"]),
            int(row["frame_index"]),
        )
        if identity != expected:
            raise ValueError(f"V5.3 manifest/LeRobot identity mismatch: {expected} != {identity}")
        state = np.asarray(item["observation.state"], dtype=np.float32)
        history_is_pad = np.asarray(item["observation.state_is_pad"], dtype=np.bool_)
        if state.shape != (self.state_history_len, 7):
            raise ValueError(f"V5.3 continuous state history has shape {state.shape}")
        if history_is_pad.shape != (self.state_history_len,):
            raise ValueError(f"V5.3 continuous history mask has shape {history_is_pad.shape}")
        return {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": state[-1],
            "observation/state_history": state,
            "observation/state_history_mask": np.logical_not(history_is_pad),
            "prompt": str(row["prompt"]),
            "adjustment_end_label": int(bool(row["adjustment_end"])),
            "global_index": identity[0],
            "episode_id": identity[1],
            "attempt_id": identity[2],
            "frame_index": identity[3],
            "rexecution_frame": int(row["rexecution_frame"]),
        }


class DeterministicOneToThreeBatchSampler:
    """Fixed 2-positive/6-negative batches with deterministic negative rotation."""

    def __init__(
        self,
        *,
        labels: list[bool],
        num_batches: int,
        seed: int,
        start_batch: int = 0,
    ) -> None:
        self.positive = np.flatnonzero(np.asarray(labels, dtype=np.bool_)).astype(np.int64)
        self.negative = np.flatnonzero(np.logical_not(np.asarray(labels, dtype=np.bool_))).astype(np.int64)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        if self.positive.size == 0 or self.positive.size % 2:
            raise ValueError("Positive pool must be non-empty and divisible by two")
        if self.negative.size < self.positive.size * 3:
            raise ValueError("Negative pool is too small for one 1:3 epoch without replacement")
        if self.num_batches <= 0 or not 0 <= self.start_batch <= self.num_batches:
            raise ValueError("Invalid sampler batch range")

    def __len__(self) -> int:
        return self.num_batches - self.start_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        negative_permutation = rng.permutation(self.negative)
        negative_cursor = 0
        emitted = 0
        epoch = 0
        batches_per_epoch = self.positive.size // 2
        while emitted < self.num_batches:
            positive_permutation = np.random.default_rng(self.seed + 10_000 + epoch).permutation(
                self.positive
            )
            for batch_in_epoch in range(batches_per_epoch):
                if emitted >= self.num_batches:
                    break
                if negative_cursor + 6 > negative_permutation.size:
                    negative_permutation = rng.permutation(self.negative)
                    negative_cursor = 0
                batch = np.concatenate(
                    (
                        positive_permutation[batch_in_epoch * 2 : batch_in_epoch * 2 + 2],
                        negative_permutation[negative_cursor : negative_cursor + 6],
                    )
                )
                negative_cursor += 6
                batch = rng.permutation(batch).tolist()
                if emitted >= self.start_batch:
                    yield [int(value) for value in batch]
                emitted += 1
            epoch += 1


class TransformedAdjustmentEndDataset:
    """Apply the normal Pi0.5 action-prefix transform while retaining labels."""

    def __init__(self, dataset: AdjustmentEndManifestDataset, transform: Any) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.dataset[index]
        label = np.asarray(raw["adjustment_end_label"], dtype=np.int32)
        identity = {
            key: np.asarray(raw[key], dtype=np.int64)
            for key in (
                "global_index",
                "episode_id",
                "attempt_id",
                "frame_index",
                "rexecution_frame",
            )
        }
        transformed = self.transform(raw)
        transformed["adjustment_end_label"] = label
        transformed.update(identity)
        return transformed
