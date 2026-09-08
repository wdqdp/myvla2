"""V7 adjustment-end data built from native rexecution timing."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from tactile_vla.vla.artifacts import canonical_json_bytes, sha256_file, sha256_json
from tactile_vla.vla.v4_data import SPLITS, V4Frame, file_identity, validate_v4_index_dataset
from tactile_vla.vla.v5_3_adjustment_end_data import (
    DeterministicOneToThreeBatchSampler as DeterministicOneToThreeBatchSampler,
)
from tactile_vla.vla.v5_3_adjustment_end_data import (
    TransformedAdjustmentEndDataset as TransformedAdjustmentEndDataset,
)
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles
from tactile_vla.vla.v5_3_adjustment_end_data import scan_selected_qpos
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_PROMPT_PROFILE
from tactile_vla.vla.v5_3_phase_change import QPOS_HISTORY_FRAMES
from tactile_vla.vla.v5_3_phase_change import QPOS_SAMPLE_OFFSETS
from tactile_vla.vla.v5_3_phase_change import build_adjustment_end_prompt
from tactile_vla.vla.v5_3_phase_change import helper_identity
from tactile_vla.vla.v5_3_phase_change import normalize_state_qpos
from tactile_vla.vla.v5_3_phase_change import pi05_phase_change_token_length
from tactile_vla.vla.v5_3_phase_change import runtime_reachable_endpoint
from tactile_vla.vla.v7_adjustment_data import native_reexecution_timing_identity
from tactile_vla.vla.v7_adjustment_data import validate_v7_adjustment_training_index


DATA_PROFILE = "rotation_phase_v7_adjustment_end_r10_r0"
EXPERIMENT_KIND = "adjustment_end_action_multitask_v7_no_history"
MANIFEST_SCHEMA = "tactile_vla_v7_adjustment_end_manifest_v1"
TRAINING_INDEX_SCHEMA = "tactile_vla_v7_adjustment_end_training_index_v1"
SUMMARY_SCHEMA = "tactile_vla_v7_adjustment_end_summary_v1"
ARTIFACT_HASH_SCHEMA = "tactile_vla_v7_adjustment_end_artifact_hashes_v1"
ADJUSTMENT_END_START_OFFSET = -10
ADJUSTMENT_END_END_OFFSET = 0
LABEL_POLICY = {
    "boundary": "native_reexecution_frame_index",
    "positive_start_offset_inclusive": ADJUSTMENT_END_START_OFFSET,
    "positive_end_offset_inclusive": ADJUSTMENT_END_END_OFFSET,
    "valid_end_offset_inclusive": ADJUSTMENT_END_END_OFFSET,
}
EXPECTED_ATTEMPT2_COUNTS = {"train": 144, "val": 18, "test": 18}
EXPECTED_POSITIVE_COUNTS = {"train": 1584, "val": 198, "test": 198}
EXPECTED_SAMPLE_COUNTS = {"train": 37803, "val": 4872, "test": 4472}


def is_adjustment_end_positive(frame_index: int, rexecution_frame: int) -> bool:
    relative = int(frame_index) - int(rexecution_frame)
    return ADJUSTMENT_END_START_OFFSET <= relative <= ADJUSTMENT_END_END_OFFSET


def is_adjustment_end_valid(frame_index: int, rexecution_frame: int) -> bool:
    return int(frame_index) <= int(rexecution_frame)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _token_summary(lengths: list[int]) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=np.int32)
    if values.size == 0:
        raise ValueError("No V7 adjustment-end prompts were produced")
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "p50": int(np.percentile(values, 50)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
        "max": int(values.max()),
        "over_limit_count": int(np.count_nonzero(values > PHASE_CHANGE_MAX_TOKEN_LEN)),
    }


def _frames_by_attempt(frames: Sequence[V4Frame]) -> dict[tuple[int, int], list[V4Frame]]:
    groups: dict[tuple[int, int], list[V4Frame]] = defaultdict(list)
    for frame in frames:
        groups[frame.attempt_key].append(frame)
    for key, rows in groups.items():
        rows.sort(key=lambda row: row.frame_index)
        if [row.frame_index for row in rows] != list(range(len(rows))):
            raise ValueError(f"V7 adjustment_end attempt {key} is not contiguous")
    return dict(groups)


def _episode_splits(v4_index: Mapping[str, Any], lookup: Mapping[int, V4Frame]) -> dict[int, str]:
    result: dict[int, str] = {}
    for split in SPLITS:
        for value in v4_index["splits"][split]["execution_indices"]:
            episode_id = lookup[int(value)].episode_id
            previous = result.setdefault(episode_id, split)
            if previous != split:
                raise ValueError(f"episode {episode_id} crosses V4 splits")
    return result


def _caption_source(summary_path: Path, frame_count: int) -> dict[str, Any]:
    summary = _load_object(summary_path)
    if int(summary.get("window_size", -1)) != 30:
        raise ValueError("V7 tactile captions must use window_size=30")
    if int(summary.get("annotated_frames", -1)) != frame_count:
        raise ValueError("Caption annotation does not cover the V7 LeRobot dataset")
    declared = Path(str(summary.get("checkpoint", ""))).expanduser()
    checkpoint: dict[str, Any] = {
        "path": str(declared),
        "sha256": None,
        "identity_status": "unverified_missing",
    }
    if declared.is_file():
        checkpoint.update({"sha256": sha256_file(declared), "identity_status": "verified"})
    return {
        "field": "tactile_caption",
        "window_size": 30,
        "checkpoint": checkpoint,
        "annotation_summary": file_identity(summary_path),
    }


def build_adjustment_end_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    action_index_file: Path,
    norm_stats_file: Path,
    caption_summary_file: Path,
    stage_a_checkpoint: Path,
    stage_a_config_file: Path,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index = _load_object(v4_index_file)
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    action_index = _load_object(action_index_file)
    validate_v7_adjustment_training_index(
        action_index,
        index_path=action_index_file,
        dataset_dir=dataset_dir,
    )
    if action_index["selection_hash"] != v4_index["selection_hash"]:
        raise ValueError("V7 action and V4 indices use different selections")
    timing_identity, boundaries = native_reexecution_timing_identity(v4_index, frames)
    groups = _frames_by_attempt(frames)
    episode_splits = _episode_splits(v4_index, global_lookup)
    attempt2_keys = sorted(key for key, value in boundaries.items() if key[1] == 2 and value is not None)
    selected_episodes = {episode_id for episode_id, _ in attempt2_keys}
    if len(attempt2_keys) != 180 or len(selected_episodes) != 180:
        raise ValueError(f"Expected 180 V7 attempt2 episodes, got {len(attempt2_keys)}")

    qpos_lookup = scan_selected_qpos(dataset_dir=dataset_dir, selected_episode_ids=selected_episodes)
    stats = load_state_quantiles(norm_stats_file)
    prompt_helper = helper_identity()
    manifest: list[dict[str, Any]] = []
    split_rows: dict[str, list[int]] = {split: [] for split in SPLITS}
    split_globals: dict[str, list[int]] = {split: [] for split in SPLITS}
    split_attempts: Counter[str] = Counter()
    token_lengths: list[int] = []

    for episode_id, attempt_id in attempt2_keys:
        split = episode_splits[episode_id]
        split_attempts[split] += 1
        rexecution = int(boundaries[(episode_id, attempt_id)])
        attempt1 = groups.get((episode_id, 1), [])
        attempt2 = groups[(episode_id, 2)]
        timeline = attempt1 + attempt2
        attempt2_start = len(attempt1)
        if attempt2_start < QPOS_HISTORY_FRAMES:
            raise ValueError(f"episode {episode_id} lacks physical qpos history before attempt2")
        if rexecution < -ADJUSTMENT_END_START_OFFSET or rexecution >= len(attempt2):
            raise ValueError(f"episode {episode_id} has invalid native rexecution frame {rexecution}")
        for frame in attempt2[: rexecution + 1]:
            timeline_position = attempt2_start + frame.frame_index
            history_frames = timeline[timeline_position - QPOS_HISTORY_FRAMES : timeline_position]
            if len(history_frames) != QPOS_HISTORY_FRAMES:
                raise ValueError(f"episode {episode_id} frame {frame.frame_index} lacks qpos_h30")
            history_globals = [item.global_index for item in history_frames]
            if any(left >= right for left, right in pairwise(history_globals)):
                raise ValueError("qpos_h30 global indices are not increasing")
            history_qpos = np.stack([qpos_lookup[index] for index in history_globals])
            current_qpos = qpos_lookup[frame.global_index]
            prompt, discrete = build_adjustment_end_prompt(
                instruction=frame.instruction,
                tactile_caption=frame.tactile_caption,
                recovery_plan=frame.input_recovery_plan,
                qpos_h30=history_qpos,
                stats=stats,
            )
            token_length = pi05_phase_change_token_length(
                tokenizer=tokenizer,
                prompt=prompt,
                normalized_current_qpos=normalize_state_qpos(current_qpos, stats),
            )
            if token_length > PHASE_CHANGE_MAX_TOKEN_LEN:
                raise ValueError(
                    f"V7 adjustment_end prompt truncation at global_index={frame.global_index}: "
                    f"tokens={token_length}, limit={PHASE_CHANGE_MAX_TOKEN_LEN}"
                )
            positive = is_adjustment_end_positive(frame.frame_index, rexecution)
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
                "history_attempt_ids": [item.attempt_id for item in history_frames],
                "history_crosses_attempt": len({item.attempt_id for item in history_frames}) > 1,
                "history_crosses_episode": False,
                "history_available": True,
                "runtime_reachable_endpoint": runtime_reachable_endpoint(frame.frame_index),
                "qpos_h30_sample_offsets": list(QPOS_SAMPLE_OFFSETS),
                "qpos_h10_discrete": discrete.tolist(),
                "adjustment_end": positive,
                "adjustment_end_valid": True,
                "classification_sample_valid": True,
                "phase_change_token_len": token_length,
                "prompt": prompt,
            }
            row_index = len(manifest)
            manifest.append(row)
            split_rows[split].append(row_index)
            split_globals[split].append(frame.global_index)
            token_lengths.append(token_length)

    if dict(split_attempts) != EXPECTED_ATTEMPT2_COUNTS:
        raise ValueError(f"V7 attempt2 split changed: {dict(split_attempts)}")
    splits: dict[str, Any] = {}
    for split in SPLITS:
        selected = [manifest[index] for index in split_rows[split]]
        positive = sum(bool(row["adjustment_end"]) for row in selected)
        if len(selected) != EXPECTED_SAMPLE_COUNTS[split] or positive != EXPECTED_POSITIVE_COUNTS[split]:
            raise ValueError(
                f"V7 {split} counts changed: samples={len(selected)}, positive={positive}"
            )
        splits[split] = {
            "manifest_row_indices": split_rows[split],
            "global_indices": split_globals[split],
            "sample_count": len(selected),
            "positive_count": positive,
            "negative_count": len(selected) - positive,
            "attempt2_count": split_attempts[split],
        }

    caption_source = _caption_source(caption_summary_file, int(v4_index["lerobot_identity"]["frame_count"]))
    source_files = {
        "v4_training_index": file_identity(v4_index_file),
        "v7_action_index": file_identity(action_index_file),
        "v4_norm_stats": file_identity(norm_stats_file),
        "caption_summary": file_identity(caption_summary_file),
        "backbone_config": file_identity(stage_a_config_file),
        "stage_a_params_metadata": file_identity(stage_a_checkpoint / "params" / "_METADATA"),
    }
    index: dict[str, Any] = {
        "schema_version": TRAINING_INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "dataset_dir": str(dataset_dir),
        "selection_hash": v4_index["selection_hash"],
        "attempt2_count": 180,
        "label_policy": LABEL_POLICY,
        "native_reexecution_timing_identity": timing_identity,
        "action_training_data_hash": action_index["training_data_hash"],
        "splits": splits,
        "manifest_identity": {"count": len(manifest), "content_sha256": sha256_json(manifest)},
        "prompt_helper": prompt_helper,
        "prompt_token_lengths": _token_summary(token_lengths),
        "state_norm": {
            "method": "q01_q99_pi05",
            "norm_stats_sha256": sha256_file(norm_stats_file),
            "q01": stats.q01.tolist(),
            "q99": stats.q99.tolist(),
        },
        "caption_source": caption_source,
        "stage_a_checkpoint": {"path": str(stage_a_checkpoint.resolve()), "step": 15000},
        "source_files": source_files,
    }
    index["training_data_hash"] = sha256_json(index)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "attempt2_count": 180,
        "label_policy": LABEL_POLICY,
        "valid_sample_count": len(manifest),
        "positive_count": sum(EXPECTED_POSITIVE_COUNTS.values()),
        "negative_count": len(manifest) - sum(EXPECTED_POSITIVE_COUNTS.values()),
        "split_summary": {
            split: {key: value for key, value in splits[split].items() if key not in {"manifest_row_indices", "global_indices"}}
            for split in SPLITS
        },
        "prompt_token_lengths": index["prompt_token_lengths"],
        "caption_source": caption_source,
        "training_data_hash": index["training_data_hash"],
    }
    return manifest, index, summary


def validate_adjustment_end_artifacts(*, index: Mapping[str, Any], manifest: list[dict[str, Any]]) -> None:
    expected = {
        "schema_version": TRAINING_INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "attempt2_count": 180,
        "label_policy": LABEL_POLICY,
    }
    mismatch = {key: (index.get(key), value) for key, value in expected.items() if index.get(key) != value}
    if mismatch:
        raise ValueError(f"V7 adjustment_end index mismatch: {mismatch}")
    actual_hash = sha256_json({key: value for key, value in index.items() if key != "training_data_hash"})
    if index.get("training_data_hash") != actual_hash:
        raise ValueError("V7 adjustment_end training_data_hash mismatch")
    identity = index.get("manifest_identity", {})
    if int(identity.get("count", -1)) != len(manifest) or identity.get("content_sha256") != sha256_json(manifest):
        raise ValueError("V7 adjustment_end manifest identity mismatch")
    if index.get("prompt_helper") != helper_identity():
        raise ValueError("V7 adjustment_end prompt helper mismatch")
    for split in SPLITS:
        payload = index["splits"][split]
        rows = [manifest[int(value)] for value in payload["manifest_row_indices"]]
        if len(rows) != EXPECTED_SAMPLE_COUNTS[split]:
            raise ValueError(f"V7 {split} sample count mismatch")
        if sum(bool(row["adjustment_end"]) for row in rows) != EXPECTED_POSITIVE_COUNTS[split]:
            raise ValueError(f"V7 {split} positive count mismatch")
        if [int(row["current_global_index"]) for row in rows] != [int(value) for value in payload["global_indices"]]:
            raise ValueError(f"V7 {split} manifest/global order mismatch")
    for row in manifest:
        frame, rexecution = int(row["frame_index"]), int(row["rexecution_frame"])
        if row.get("schema_version") != MANIFEST_SCHEMA or int(row["attempt_id"]) != 2:
            raise ValueError("V7 adjustment_end manifest schema mismatch")
        if not is_adjustment_end_valid(frame, rexecution) or not row["classification_sample_valid"]:
            raise ValueError("V7 adjustment_end manifest contains an invalid sample")
        if bool(row["adjustment_end"]) != is_adjustment_end_positive(frame, rexecution):
            raise ValueError("V7 adjustment_end label mismatch")
        history = [int(value) for value in row["history_global_indices"]]
        if len(history) != 30 or any(left >= right for left, right in pairwise(history)):
            raise ValueError("V7 adjustment_end qpos_h30 identity is invalid")
        if row.get("prompt") is None or len(row.get("qpos_h10_discrete", [])) != 10:
            raise ValueError("V7 adjustment_end prompt is incomplete")


def load_indexed_manifest_rows(*, index: Mapping[str, Any], manifest_path: Path) -> dict[int, dict[str, Any]]:
    expected_hash = sha256_json({key: value for key, value in index.items() if key != "training_data_hash"})
    if index.get("training_data_hash") != expected_hash or index.get("label_policy") != LABEL_POLICY:
        raise ValueError("V7 adjustment_end index identity mismatch")
    wanted: dict[int, tuple[str, int]] = {}
    for split in SPLITS:
        rows = index["splits"][split]["manifest_row_indices"]
        globals_ = index["splits"][split]["global_indices"]
        for row_index, global_index in zip(rows, globals_, strict=True):
            wanted[int(row_index)] = (split, int(global_index))
    file_digest, content_digest = hashlib.sha256(), hashlib.sha256()
    content_digest.update(b"[")
    selected: dict[int, dict[str, Any]] = {}
    row_count = 0
    with manifest_path.open("rb") as stream:
        for raw in stream:
            file_digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row_count:
                content_digest.update(b",")
            content_digest.update(canonical_json_bytes(row))
            if row_count in wanted:
                split, global_index = wanted[row_count]
                if row.get("split") != split or int(row.get("current_global_index", -1)) != global_index:
                    raise ValueError(f"V7 indexed manifest row {row_count} identity mismatch")
                selected[row_count] = row
            row_count += 1
    content_digest.update(b"]")
    identity = index["manifest_identity"]
    if row_count != int(identity["count"]) or content_digest.hexdigest() != identity["content_sha256"]:
        raise ValueError("V7 adjustment_end manifest content mismatch")
    if file_digest.hexdigest() != str(identity.get("file_sha256", "")):
        raise ValueError("V7 adjustment_end manifest file SHA mismatch")
    if set(selected) != set(wanted):
        raise ValueError("V7 adjustment_end manifest lacks indexed rows")
    return selected


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_HASH_SCHEMA,
        "artifacts": {name: file_identity(path) for name, path in sorted(paths.items())},
    }


class AdjustmentEndManifestDataset:
    """V7 classifier dataset: current state only; qpos_h30 lives in prompt text."""

    def __init__(
        self,
        *,
        manifest: Mapping[int, dict[str, Any]] | list[dict[str, Any]],
        manifest_row_indices: list[int],
        global_indices: list[int],
        lerobot_dataset: Any,
        state_history_len: int = 0,
    ) -> None:
        if int(state_history_len) != 0:
            raise ValueError("V7 adjustment_end forbids continuous state history")
        self.manifest = manifest
        self.row_indices = [int(value) for value in manifest_row_indices]
        self.global_indices = [int(value) for value in global_indices]
        self._dataset = lerobot_dataset
        if len(self.row_indices) != len(self.global_indices):
            raise ValueError("V7 manifest/global index lengths differ")

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, dataset_index: int) -> dict[str, Any]:
        row = self.manifest[self.row_indices[dataset_index]]
        global_index = self.global_indices[dataset_index]
        item = self._dataset[global_index]
        identity = (int(item["index"]), int(item["episode_id"]), int(item["attempt_id"]), int(item["frame_index"]))
        expected = (global_index, int(row["episode_id"]), 2, int(row["frame_index"]))
        if identity != expected:
            raise ValueError(f"V7 manifest/LeRobot identity mismatch: {expected} != {identity}")
        state = np.asarray(item["observation.state"], dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f"V7 current state has shape {state.shape}, expected [7]")
        return {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": state,
            "prompt": str(row["prompt"]),
            "adjustment_end_label": int(bool(row["adjustment_end"])),
            "global_index": identity[0],
            "episode_id": identity[1],
            "attempt_id": identity[2],
            "frame_index": identity[3],
            "rexecution_frame": int(row["rexecution_frame"]),
        }
