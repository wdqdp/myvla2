"""V7.5 adjustment-end data from audited motion boundaries and H100 qpos."""

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
from tactile_vla.vla.v4_data import SPLITS, file_identity, validate_v4_index_dataset
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles, scan_selected_qpos
from tactile_vla.vla.v7_adjustment_end_data import _caption_source, _episode_splits, _frames_by_attempt
from tactile_vla.vla.v7_4_adjustment_data import validate_v7_4_adjustment_training_index
from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_PROMPT_PROFILE
from tactile_vla.vla.v7_5_phase_change import QPOS_HISTORY_FRAMES, QPOS_SAMPLED_FRAMES
from tactile_vla.vla.v7_5_phase_change import build_adjustment_end_prompt, helper_identity
from tactile_vla.vla.v7_5_phase_change import normalize_state_qpos, pi05_phase_change_token_length


DATA_PROFILE = "rotation_phase_v7_5_adjustment_end_h100"
EXPERIMENT_KIND = "adjustment_end_action_multitask_v7_5_h100_idle_811"
MANIFEST_SCHEMA = "tactile_vla_v7_5_adjustment_end_manifest_v1"
TRAINING_INDEX_SCHEMA = "tactile_vla_v7_5_adjustment_end_training_index_v1"
SUMMARY_SCHEMA = "tactile_vla_v7_5_adjustment_end_summary_v1"
ARTIFACT_HASH_SCHEMA = "tactile_vla_v7_5_adjustment_end_artifact_hashes_v1"
HISTORY_SEED = 42
EXPECTED_ATTEMPT2_COUNTS = {"train": 144, "val": 18, "test": 18}
EXPECTED_POSITIVE_COUNTS = {"train": 3024, "val": 378, "test": 378}
EXPECTED_NEGATIVE_COUNTS = {"train": 7386, "val": 1017, "test": 809}

LABEL_POLICY = {
    "boundary": "audited_arm_adjustment_stop",
    "sample_start": "arm_adjustment_start_inclusive",
    "sample_end_offset_inclusive": 10,
    "negative_end_offset_inclusive": -11,
    "positive_start_offset_inclusive": -10,
    "positive_end_offset_inclusive": 10,
}
HISTORY_POLICY = {
    "window": "inclusive_[p-99,p]",
    "window_frames": 100,
    "sampled_qpos_points": 11,
    "idle_interval": "strictly_between_gripper_motion_stop_and_arm_adjustment_start",
    "idle_keep_k_ratio": {"0": 8, "1": 1, "2": 1},
    "eligibility": "overlap_length_l > k*10",
    "ratio_scope": "per_split_among_samples_with_l_gt_0",
    "ratio_rounding": "largest_remainder_k0_then_k1_then_k2",
    "sampling": "retain_k_random_idle_frames_then_uniformly_sample_non_idle_timeline",
    "seed": HISTORY_SEED,
}


def adjustment_end_label(frame_index: int, arm_adjustment_stop: int) -> bool:
    relative = int(frame_index) - int(arm_adjustment_stop)
    return -10 <= relative <= 10


def valid_classification_frame(frame_index: int, arm_adjustment_start: int, arm_adjustment_stop: int) -> bool:
    return int(arm_adjustment_start) <= int(frame_index) <= int(arm_adjustment_stop) + 10


def idle_overlap_indices(
    *, frame_index: int, gripper_motion_stop: int, arm_adjustment_start: int
) -> list[int]:
    history_start = int(frame_index) - (QPOS_HISTORY_FRAMES - 1)
    lower = max(history_start, int(gripper_motion_stop) + 1)
    upper = min(int(frame_index), int(arm_adjustment_start) - 1)
    return list(range(lower, upper + 1)) if lower <= upper else []


def _largest_remainder_811(size: int) -> dict[int, int]:
    floors = {0: size * 8 // 10, 1: size // 10, 2: size // 10}
    remainder = size - sum(floors.values())
    fractions = {0: (size * 8) % 10, 1: size % 10, 2: size % 10}
    for key in sorted(fractions, key=lambda value: (-fractions[value], value))[:remainder]:
        floors[key] += 1
    return floors


def assign_idle_keep_counts(
    candidates: Sequence[tuple[int, int]], *, seed: int
) -> tuple[dict[int, int], dict[str, Any]]:
    """Assign k with an 8:1:1 target while respecting l > k*10."""

    pooled = [(int(identifier), int(overlap)) for identifier, overlap in candidates if int(overlap) > 0]
    targets = _largest_remainder_811(len(pooled))
    rng = np.random.default_rng(seed)
    priority = {identifier: float(rng.random()) for identifier, _ in pooled}
    eligible_two = sorted(
        (identifier for identifier, overlap in pooled if overlap > 20), key=priority.__getitem__
    )
    if len(eligible_two) < targets[2]:
        raise ValueError("Not enough l>20 histories to satisfy the requested k=2 ratio")
    selected_two = set(eligible_two[: targets[2]])
    eligible_one = sorted(
        (
            identifier
            for identifier, overlap in pooled
            if overlap > 10 and identifier not in selected_two
        ),
        key=priority.__getitem__,
    )
    if len(eligible_one) < targets[1]:
        raise ValueError("Not enough remaining l>10 histories to satisfy the requested k=1 ratio")
    selected_one = set(eligible_one[: targets[1]])
    assignments = {
        identifier: 2 if identifier in selected_two else 1 if identifier in selected_one else 0
        for identifier, _ in pooled
    }
    actual = Counter(assignments.values())
    if any(overlap <= assignments[identifier] * 10 for identifier, overlap in pooled):
        raise AssertionError("Idle keep assignment violated l > k*10")
    return assignments, {
        "pool_size": len(pooled),
        "target_counts": {str(key): value for key, value in targets.items()},
        "actual_counts": {str(key): int(actual[key]) for key in (0, 1, 2)},
    }


def sample_history_frame_indices(
    *,
    frame_index: int,
    gripper_motion_stop: int,
    arm_adjustment_start: int,
    idle_keep_count: int,
    seed: int,
) -> tuple[list[int], list[int], int]:
    history = list(range(int(frame_index) - 99, int(frame_index) + 1))
    idle = idle_overlap_indices(
        frame_index=frame_index,
        gripper_motion_stop=gripper_motion_stop,
        arm_adjustment_start=arm_adjustment_start,
    )
    k = int(idle_keep_count)
    if k not in {0, 1, 2} or len(idle) <= k * 10:
        raise ValueError(f"Invalid idle keep k={k} for overlap l={len(idle)}")
    rng = np.random.default_rng(seed)
    retained_idle = sorted(int(value) for value in rng.choice(idle, size=k, replace=False)) if k else []
    idle_set = set(idle)
    non_idle = [value for value in history if value not in idle_set]
    needed = QPOS_SAMPLED_FRAMES - k
    if len(non_idle) < needed:
        raise ValueError("Not enough non-idle H100 frames for 11-point sampling")
    positions = np.rint(np.linspace(0, len(non_idle) - 1, needed)).astype(np.int64)
    sampled = sorted([non_idle[int(position)] for position in positions] + retained_idle)
    if len(sampled) != QPOS_SAMPLED_FRAMES or len(set(sampled)) != QPOS_SAMPLED_FRAMES:
        raise AssertionError("V7.5 history sampler did not produce 11 unique frames")
    if sampled[-1] != int(frame_index):
        raise AssertionError("V7.5 history sampler must retain current frame p")
    return sampled, retained_idle, len(idle)


def _row_seed(episode_id: int, frame_index: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{episode_id}:2:{frame_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _token_summary(lengths: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=np.int32)
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "p50": int(np.percentile(values, 50)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
        "max": int(values.max()),
        "over_limit_count": int(np.count_nonzero(values > PHASE_CHANGE_MAX_TOKEN_LEN)),
    }


def _boundary_events(payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, int]]:
    result = {}
    for attempt in payload["attempts"]:
        key = (int(attempt["episode_id"]), int(attempt["attempt_id"]))
        events = attempt["events"]
        result[key] = {
            "gripper_motion_stop": int(events["gripper_motion_stop"]["frame_index"]),
            "arm_adjustment_start": int(events["arm_adjustment_start"]["frame_index"]),
            "arm_adjustment_stop": int(events["arm_adjustment_stop"]["frame_index"]),
            "gripper_close_start": int(events["gripper_close_start"]["frame_index"]),
            "rexecution_frame": int(attempt["rexecution_frame"]),
        }
    return result


def build_adjustment_end_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    action_index_file: Path,
    boundary_filter_file: Path,
    norm_stats_file: Path,
    caption_summary_file: Path,
    stage_a_checkpoint: Path,
    stage_a_config_file: Path,
    tokenizer: Any,
    seed: int = HISTORY_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index = _load_object(v4_index_file)
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    action_index = _load_object(action_index_file)
    validate_v7_4_adjustment_training_index(
        action_index, index_path=action_index_file, dataset_dir=dataset_dir
    )
    boundary_payload = _load_object(boundary_filter_file)
    events = _boundary_events(boundary_payload)
    groups = _frames_by_attempt(frames)
    episode_splits = _episode_splits(v4_index, global_lookup)
    attempt2_keys = sorted(key for key in events if key[1] == 2)
    if len(attempt2_keys) != 180:
        raise ValueError(f"Expected 180 audited attempt2 boundaries, got {len(attempt2_keys)}")

    candidates: list[dict[str, Any]] = []
    split_candidate_ids: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for episode_id, attempt_id in attempt2_keys:
        split = episode_splits[episode_id]
        event = events[(episode_id, attempt_id)]
        if not (
            event["gripper_motion_stop"] < event["arm_adjustment_start"]
            <= event["arm_adjustment_stop"] < event["gripper_close_start"]
            and event["arm_adjustment_stop"] + 10 < event["gripper_close_start"]
        ):
            raise ValueError(f"Invalid V7.5 boundary order for episode {episode_id}")
        attempt = groups[(episode_id, attempt_id)]
        for frame_index in range(event["arm_adjustment_start"], event["arm_adjustment_stop"] + 11):
            if frame_index - 99 < 0 or frame_index >= len(attempt):
                raise ValueError(f"Episode {episode_id} frame {frame_index} lacks an in-attempt H100")
            identifier = len(candidates)
            overlap = len(
                idle_overlap_indices(
                    frame_index=frame_index,
                    gripper_motion_stop=event["gripper_motion_stop"],
                    arm_adjustment_start=event["arm_adjustment_start"],
                )
            )
            candidates.append(
                {
                    "identifier": identifier,
                    "episode_id": episode_id,
                    "attempt_id": attempt_id,
                    "frame_index": frame_index,
                    "split": split,
                    "events": event,
                    "idle_overlap": overlap,
                }
            )
            split_candidate_ids[split].append((identifier, overlap))

    assignments: dict[int, int] = {}
    ratio_summary = {}
    for split_offset, split in enumerate(SPLITS):
        assigned, details = assign_idle_keep_counts(
            split_candidate_ids[split], seed=seed + split_offset * 10_000
        )
        assignments.update(assigned)
        ratio_summary[split] = details

    qpos_lookup = scan_selected_qpos(
        dataset_dir=dataset_dir, selected_episode_ids={key[0] for key in attempt2_keys}
    )
    stats = load_state_quantiles(norm_stats_file)
    manifest: list[dict[str, Any]] = []
    split_rows: dict[str, list[int]] = {split: [] for split in SPLITS}
    split_globals: dict[str, list[int]] = {split: [] for split in SPLITS}
    token_lengths: list[int] = []
    for candidate in candidates:
        episode_id = int(candidate["episode_id"])
        frame_index = int(candidate["frame_index"])
        split = str(candidate["split"])
        event = candidate["events"]
        attempt = groups[(episode_id, 2)]
        k = assignments.get(int(candidate["identifier"]), 0)
        sampled_frames, retained_idle, overlap = sample_history_frame_indices(
            frame_index=frame_index,
            gripper_motion_stop=event["gripper_motion_stop"],
            arm_adjustment_start=event["arm_adjustment_start"],
            idle_keep_count=k,
            seed=_row_seed(episode_id, frame_index, seed),
        ) if candidate["idle_overlap"] > 0 else (
            [
                int(value)
                for value in np.rint(
                    np.linspace(frame_index - 99, frame_index, 11)
                ).astype(np.int64)
            ],
            [],
            0,
        )
        sampled_globals = [attempt[value].global_index for value in sampled_frames]
        sampled_qpos = np.stack([qpos_lookup[value] for value in sampled_globals])
        frame = attempt[frame_index]
        current_qpos = qpos_lookup[frame.global_index]
        prompt, discrete = build_adjustment_end_prompt(
            instruction=frame.instruction,
            tactile_caption=frame.tactile_caption,
            recovery_plan=frame.input_recovery_plan,
            sampled_qpos=sampled_qpos,
            stats=stats,
        )
        token_length = pi05_phase_change_token_length(
            tokenizer=tokenizer,
            prompt=prompt,
            normalized_current_qpos=normalize_state_qpos(current_qpos, stats),
        )
        if token_length > PHASE_CHANGE_MAX_TOKEN_LEN:
            raise ValueError(f"V7.5 prompt truncation at global_index={frame.global_index}")
        positive = adjustment_end_label(frame_index, event["arm_adjustment_stop"])
        row = {
            "schema_version": MANIFEST_SCHEMA,
            "data_profile": DATA_PROFILE,
            "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
            "experiment_kind": EXPERIMENT_KIND,
            "episode_id": episode_id,
            "attempt_id": 2,
            "frame_index": frame_index,
            "current_global_index": frame.global_index,
            "split": split,
            "gripper_motion_stop_frame": event["gripper_motion_stop"],
            "arm_adjustment_start_frame": event["arm_adjustment_start"],
            "arm_adjustment_stop_frame": event["arm_adjustment_stop"],
            "gripper_close_start_frame": event["gripper_close_start"],
            "rexecution_frame": event["rexecution_frame"],
            "history_window_start_frame": frame_index - 99,
            "history_window_end_frame": frame_index,
            "idle_overlap_length": overlap,
            "idle_keep_count": k,
            "retained_idle_frame_indices": retained_idle,
            "sampled_history_frame_indices": sampled_frames,
            "history_global_indices": sampled_globals,
            "qpos_h100_11_discrete": discrete.tolist(),
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

    split_attempts = Counter(episode_splits[key[0]] for key in attempt2_keys)
    splits = {}
    for split in SPLITS:
        selected = [manifest[index] for index in split_rows[split]]
        positive = sum(bool(row["adjustment_end"]) for row in selected)
        negative = len(selected) - positive
        expected = (
            EXPECTED_ATTEMPT2_COUNTS[split], EXPECTED_POSITIVE_COUNTS[split], EXPECTED_NEGATIVE_COUNTS[split]
        )
        if (split_attempts[split], positive, negative) != expected:
            raise ValueError(f"V7.5 {split} counts changed: {(split_attempts[split], positive, negative)}")
        splits[split] = {
            "manifest_row_indices": split_rows[split],
            "global_indices": split_globals[split],
            "sample_count": len(selected),
            "positive_count": positive,
            "negative_count": negative,
            "attempt2_count": split_attempts[split],
            "idle_keep_ratio": ratio_summary[split],
        }

    caption_source = _caption_source(caption_summary_file, int(v4_index["lerobot_identity"]["frame_count"]))
    source_files = {
        "v4_training_index": file_identity(v4_index_file),
        "v7_4_action_index": file_identity(action_index_file),
        "v7_2_boundary_filter": file_identity(boundary_filter_file),
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
        "history_policy": HISTORY_POLICY,
        "action_training_data_hash": action_index["training_data_hash"],
        "splits": splits,
        "manifest_identity": {"count": len(manifest), "content_sha256": sha256_json(manifest)},
        "prompt_helper": helper_identity(),
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
        "label_policy": LABEL_POLICY,
        "history_policy": HISTORY_POLICY,
        "valid_sample_count": len(manifest),
        "positive_count": sum(EXPECTED_POSITIVE_COUNTS.values()),
        "negative_count": sum(EXPECTED_NEGATIVE_COUNTS.values()),
        "split_summary": {
            split: {key: value for key, value in splits[split].items() if key not in {"manifest_row_indices", "global_indices"}}
            for split in SPLITS
        },
        "prompt_token_lengths": index["prompt_token_lengths"],
        "training_data_hash": index["training_data_hash"],
    }
    return manifest, index, summary


def validate_adjustment_end_artifacts(*, index: Mapping[str, Any], manifest: list[dict[str, Any]]) -> None:
    expected = {
        "schema_version": TRAINING_INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "label_policy": LABEL_POLICY,
        "history_policy": HISTORY_POLICY,
    }
    if any(index.get(key) != value for key, value in expected.items()):
        raise ValueError("V7.5 adjustment_end index header mismatch")
    if index.get("training_data_hash") != sha256_json({key: value for key, value in index.items() if key != "training_data_hash"}):
        raise ValueError("V7.5 adjustment_end training_data_hash mismatch")
    identity = index["manifest_identity"]
    if int(identity["count"]) != len(manifest) or identity["content_sha256"] != sha256_json(manifest):
        raise ValueError("V7.5 manifest identity mismatch")
    for row in manifest:
        frame = int(row["frame_index"])
        start = int(row["arm_adjustment_start_frame"])
        stop = int(row["arm_adjustment_stop_frame"])
        if not valid_classification_frame(frame, start, stop):
            raise ValueError("V7.5 manifest frame lies outside the classification range")
        if bool(row["adjustment_end"]) != adjustment_end_label(frame, stop):
            raise ValueError("V7.5 label mismatch")
        sampled = [int(value) for value in row["sampled_history_frame_indices"]]
        if len(sampled) != 11 or any(left >= right for left, right in pairwise(sampled)) or sampled[-1] != frame:
            raise ValueError("V7.5 sampled history is invalid")
        k, overlap = int(row["idle_keep_count"]), int(row["idle_overlap_length"])
        if overlap > 0 and overlap <= k * 10:
            raise ValueError("V7.5 idle retention violates l > k*10")
        if len(row["retained_idle_frame_indices"]) != k or len(row["qpos_h100_11_discrete"]) != 11:
            raise ValueError("V7.5 prompt history audit fields are incomplete")
        if int(row["phase_change_token_len"]) > PHASE_CHANGE_MAX_TOKEN_LEN:
            raise ValueError("V7.5 prompt exceeds token limit")


def load_indexed_manifest_rows(*, index: Mapping[str, Any], manifest_path: Path) -> dict[int, dict[str, Any]]:
    expected_hash = sha256_json({key: value for key, value in index.items() if key != "training_data_hash"})
    if index.get("training_data_hash") != expected_hash or index.get("label_policy") != LABEL_POLICY:
        raise ValueError("V7.5 adjustment_end index identity mismatch")
    wanted = {
        int(row_index): (split, int(global_index))
        for split in SPLITS
        for row_index, global_index in zip(
            index["splits"][split]["manifest_row_indices"],
            index["splits"][split]["global_indices"], strict=True,
        )
    }
    file_digest, content_digest = hashlib.sha256(), hashlib.sha256()
    content_digest.update(b"[")
    selected = {}
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
                if row["split"] != split or int(row["current_global_index"]) != global_index:
                    raise ValueError(f"V7.5 indexed manifest row {row_count} identity mismatch")
                selected[row_count] = row
            row_count += 1
    content_digest.update(b"]")
    identity = index["manifest_identity"]
    if row_count != int(identity["count"]) or content_digest.hexdigest() != identity["content_sha256"]:
        raise ValueError("V7.5 manifest content mismatch")
    if file_digest.hexdigest() != str(identity.get("file_sha256", "")):
        raise ValueError("V7.5 manifest file SHA mismatch")
    if set(selected) != set(wanted):
        raise ValueError("V7.5 manifest lacks indexed rows")
    return selected


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_HASH_SCHEMA,
        "artifacts": {name: file_identity(path) for name, path in sorted(paths.items())},
    }


class DeterministicOneToThreeBatchSampler:
    """Deterministic 2-positive/6-negative batches with pool cycling.

    V7.5 has fewer than three distinct negatives per positive.  Negative rows
    are still sampled without replacement until their pool is exhausted, then
    deterministically reshuffled and reused.
    """

    def __init__(self, *, labels: list[bool], num_batches: int, seed: int, start_batch: int = 0):
        values = np.asarray(labels, dtype=np.bool_)
        self.positive = np.flatnonzero(values).astype(np.int64)
        self.negative = np.flatnonzero(np.logical_not(values)).astype(np.int64)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        if self.positive.size == 0 or self.positive.size % 2:
            raise ValueError("Positive pool must be non-empty and divisible by two")
        if self.negative.size < 6:
            raise ValueError("Negative pool must contain at least six rows")
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
            positives = np.random.default_rng(self.seed + 10_000 + epoch).permutation(self.positive)
            for batch_in_epoch in range(batches_per_epoch):
                if emitted >= self.num_batches:
                    break
                if negative_cursor + 6 > negative_permutation.size:
                    negative_permutation = rng.permutation(self.negative)
                    negative_cursor = 0
                batch = np.concatenate(
                    (
                        positives[batch_in_epoch * 2 : batch_in_epoch * 2 + 2],
                        negative_permutation[negative_cursor : negative_cursor + 6],
                    )
                )
                negative_cursor += 6
                if emitted >= self.start_batch:
                    yield [int(value) for value in rng.permutation(batch)]
                emitted += 1
            epoch += 1


class AdjustmentEndManifestDataset:
    def __init__(self, *, manifest, manifest_row_indices, global_indices, lerobot_dataset, state_history_len=0):
        if int(state_history_len) != 0:
            raise ValueError("V7.5 forbids continuous state history")
        self.manifest = manifest
        self.row_indices = [int(value) for value in manifest_row_indices]
        self.global_indices = [int(value) for value in global_indices]
        self._dataset = lerobot_dataset

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, dataset_index: int) -> dict[str, Any]:
        row = self.manifest[self.row_indices[dataset_index]]
        global_index = self.global_indices[dataset_index]
        item = self._dataset[global_index]
        identity = tuple(int(item[key]) for key in ("index", "episode_id", "attempt_id", "frame_index"))
        expected = (global_index, int(row["episode_id"]), 2, int(row["frame_index"]))
        if identity != expected:
            raise ValueError(f"V7.5 manifest/LeRobot identity mismatch: {expected} != {identity}")
        state = np.asarray(item["observation.state"], dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f"V7.5 current state has shape {state.shape}, expected [7]")
        return {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": state,
            "prompt": str(row["prompt"]),
            "adjustment_end_label": int(bool(row["adjustment_end"])),
            "global_index": identity[0], "episode_id": identity[1], "attempt_id": identity[2],
            "frame_index": identity[3], "rexecution_frame": int(row["rexecution_frame"]),
            "arm_adjustment_stop_frame": int(row["arm_adjustment_stop_frame"]),
        }


class TransformedAdjustmentEndDataset:
    def __init__(self, dataset, transform):
        self.dataset, self.transform = dataset, transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.dataset[index]
        identity = {
            key: np.asarray(raw[key], dtype=np.int64)
            for key in ("global_index", "episode_id", "attempt_id", "frame_index", "rexecution_frame", "arm_adjustment_stop_frame")
        }
        transformed = self.transform(raw)
        transformed["adjustment_end_label"] = np.asarray(raw["adjustment_end_label"], dtype=np.int32)
        transformed.update(identity)
        return transformed
