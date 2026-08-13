"""Index building utilities for the local tactile VLA LeRobot dataset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class FrameRecord:
    global_index: int
    lerobot_episode_index: int
    original_episode_id: int
    attempt_id: int
    frame_index: int
    case_id: str
    instruction: str
    rotation_state_name: str
    tactile_caption: str
    input_recovery_plan: str
    failure_recovery_memory: str
    failure_reason: str
    recovery_plan: str
    need_recovery: bool
    action_chunk_valid: bool
    reasoning_has_sample: bool
    reasoning_failed_attempt_id: int
    reasoning_failed_tactile_caption: str
    reasoning_failure_reason: str
    reasoning_failure_recovery_memory: str
    reasoning_recovery_plan: str
    stage_a_eligible: bool = True
    execution_eligible: bool = True


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def _as_bool(values: Iterable[Any]) -> list[bool]:
    return [bool(value) for value in values]


def _string_values(values: Iterable[Any]) -> list[str]:
    return ["" if value is None else str(value) for value in values]


def scan_lerobot_frames(dataset_dir: str | Path) -> list[FrameRecord]:
    dataset_dir = Path(dataset_dir)
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")

    columns = [
        "index",
        "episode_index",
        "episode_id",
        "attempt_id",
        "frame_index",
        "case_id",
        "instruction",
        "rotation_state_name",
        "tactile_caption",
        "input_recovery_plan",
        "failure_recovery_memory",
        "failure_reason",
        "recovery_plan",
        "need_recovery",
        "action_chunk_valid",
        "reasoning_has_sample",
        "reasoning_failed_attempt_id",
        "reasoning_failed_tactile_caption",
        "reasoning_failure_reason",
        "reasoning_failure_recovery_memory",
        "reasoning_recovery_plan",
    ]

    records: list[FrameRecord] = []
    for parquet_file in parquet_files:
        available_columns = set(pq.read_schema(parquet_file).names)
        selected_columns = list(columns)
        has_stage_a_eligible = "stage_a_eligible" in available_columns
        if has_stage_a_eligible:
            selected_columns.append("stage_a_eligible")
        has_execution_eligible = "execution_eligible" in available_columns
        if has_execution_eligible:
            selected_columns.append("execution_eligible")
        data = pq.read_table(parquet_file, columns=selected_columns).to_pydict()
        num_rows = len(data["index"])
        string_columns = [
            "case_id",
            "instruction",
            "rotation_state_name",
            "tactile_caption",
            "input_recovery_plan",
            "failure_recovery_memory",
            "failure_reason",
            "recovery_plan",
            "reasoning_failed_tactile_caption",
            "reasoning_failure_reason",
            "reasoning_failure_recovery_memory",
            "reasoning_recovery_plan",
        ]
        strings = {key: _string_values(data[key]) for key in string_columns}
        need_recovery = _as_bool(data["need_recovery"])
        action_chunk_valid = _as_bool(data["action_chunk_valid"])
        execution_eligible = (
            _as_bool(data["execution_eligible"])
            if has_execution_eligible
            else [True] * num_rows
        )
        stage_a_eligible = (
            _as_bool(data["stage_a_eligible"])
            if has_stage_a_eligible
            else [True] * num_rows
        )
        reasoning_has_sample = _as_bool(data["reasoning_has_sample"])
        for idx in range(num_rows):
            records.append(
                FrameRecord(
                    global_index=int(data["index"][idx]),
                    lerobot_episode_index=int(data["episode_index"][idx]),
                    original_episode_id=int(data["episode_id"][idx]),
                    attempt_id=int(data["attempt_id"][idx]),
                    frame_index=int(data["frame_index"][idx]),
                    case_id=strings["case_id"][idx],
                    instruction=strings["instruction"][idx],
                    rotation_state_name=strings["rotation_state_name"][idx],
                    tactile_caption=strings["tactile_caption"][idx],
                    input_recovery_plan=strings["input_recovery_plan"][idx],
                    failure_recovery_memory=strings["failure_recovery_memory"][idx],
                    failure_reason=strings["failure_reason"][idx],
                    recovery_plan=strings["recovery_plan"][idx],
                    need_recovery=need_recovery[idx],
                    action_chunk_valid=action_chunk_valid[idx],
                    reasoning_has_sample=reasoning_has_sample[idx],
                    reasoning_failed_attempt_id=int(data["reasoning_failed_attempt_id"][idx]),
                    reasoning_failed_tactile_caption=strings["reasoning_failed_tactile_caption"][idx],
                    reasoning_failure_reason=strings["reasoning_failure_reason"][idx],
                    reasoning_failure_recovery_memory=strings["reasoning_failure_recovery_memory"][idx],
                    reasoning_recovery_plan=strings["reasoning_recovery_plan"][idx],
                    stage_a_eligible=stage_a_eligible[idx],
                    execution_eligible=execution_eligible[idx],
                )
            )
    records.sort(key=lambda record: record.global_index)
    return records


def build_splits(records: Iterable[FrameRecord], config: SplitConfig) -> dict[str, list[int]]:
    original_episode_ids = sorted({record.original_episode_id for record in records})
    rng = random.Random(config.seed)
    rng.shuffle(original_episode_ids)

    train_count, val_count, _ = _target_split_counts(len(original_episode_ids), config)
    train_ids = sorted(original_episode_ids[:train_count])
    val_ids = sorted(original_episode_ids[train_count : train_count + val_count])
    test_ids = sorted(original_episode_ids[train_count + val_count :])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _target_split_counts(total: int, config: SplitConfig) -> tuple[int, int, int]:
    train_count = int(round(total * config.train_ratio))
    val_count = int(round(total * config.val_ratio))
    return train_count, val_count, total - train_count - val_count


def _split_episode_ids(payload: dict[str, Any]) -> dict[str, list[int]]:
    try:
        stored_splits = payload["original_episode_ids"]
        return {name: [int(value) for value in stored_splits[name]] for name in ("train", "val", "test")}
    except (KeyError, TypeError) as exc:
        raise ValueError("Split file must contain train/val/test under original_episode_ids") from exc


def validate_split_coverage(records: Iterable[FrameRecord], splits: dict[str, list[int]]) -> None:
    dataset_ids = {record.original_episode_id for record in records}
    assigned: set[int] = set()
    for name in ("train", "val", "test"):
        split_ids = splits.get(name)
        if split_ids is None:
            raise ValueError(f"Split is missing {name!r}")
        duplicate_ids = assigned.intersection(split_ids)
        if duplicate_ids:
            raise ValueError(f"Original episode IDs occur in multiple splits: {sorted(duplicate_ids)}")
        assigned.update(split_ids)

    missing_ids = dataset_ids - assigned
    unknown_ids = assigned - dataset_ids
    if missing_ids or unknown_ids:
        raise ValueError(
            "Split does not exactly cover the dataset: "
            f"missing episode IDs={sorted(missing_ids)}, unknown episode IDs={sorted(unknown_ids)}"
        )


def extend_splits(
    records: Iterable[FrameRecord],
    base_splits: dict[str, list[int]],
    config: SplitConfig,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Preserve a base split and assign only newly added original episodes."""
    records = list(records)
    dataset_ids = {record.original_episode_id for record in records}
    base_ids: set[int] = set()
    for name in ("train", "val", "test"):
        if name not in base_splits:
            raise ValueError(f"Base split is missing {name!r}")
        duplicate_ids = base_ids.intersection(base_splits[name])
        if duplicate_ids:
            raise ValueError(f"Original episode IDs occur in multiple base splits: {sorted(duplicate_ids)}")
        base_ids.update(base_splits[name])

    unknown_base_ids = base_ids - dataset_ids
    if unknown_base_ids:
        raise ValueError(f"Base split contains episode IDs absent from the expanded dataset: {sorted(unknown_base_ids)}")

    new_ids = sorted(dataset_ids - base_ids)
    target_counts = dict(zip(("train", "val", "test"), _target_split_counts(len(dataset_ids), config), strict=True))
    added_counts = {name: target_counts[name] - len(base_splits[name]) for name in target_counts}
    negative_counts = {name: count for name, count in added_counts.items() if count < 0}
    if negative_counts:
        raise ValueError(
            "Base split is larger than the requested final split allocation; "
            f"negative added counts={negative_counts}"
        )
    if sum(added_counts.values()) != len(new_ids):
        raise ValueError(
            f"Cannot assign {len(new_ids)} new episodes to requested added counts {added_counts}; "
            "check split ratios and the base split"
        )

    rng = random.Random(config.seed)
    rng.shuffle(new_ids)
    additions: dict[str, list[int]] = {}
    offset = 0
    for name in ("train", "val", "test"):
        end = offset + added_counts[name]
        additions[name] = sorted(new_ids[offset:end])
        offset = end

    splits = {name: sorted([*base_splits[name], *additions[name]]) for name in ("train", "val", "test")}
    validate_split_coverage(records, splits)
    return splits, additions


def load_or_create_splits(
    records: list[FrameRecord],
    split_path: str | Path,
    config: SplitConfig,
    *,
    overwrite: bool = False,
    base_split_path: str | Path | None = None,
) -> dict[str, list[int]]:
    split_path = Path(split_path)
    if split_path.exists() and not overwrite:
        payload = json.loads(split_path.read_text())
        splits = _split_episode_ids(payload)
        validate_split_coverage(records, splits)
        return splits

    additions = None
    if base_split_path is not None:
        base_split_path = Path(base_split_path)
        if not base_split_path.is_file():
            raise FileNotFoundError(f"Base split file does not exist: {base_split_path}")
        base_splits = _split_episode_ids(json.loads(base_split_path.read_text()))
        splits, additions = extend_splits(records, base_splits, config)
    else:
        splits = build_splits(records, config)
        validate_split_coverage(records, splits)

    payload = {
        "split_unit": "original_episode_id",
        "config": asdict(config),
        "original_episode_ids": splits,
        "counts": {name: len(values) for name, values in splits.items()},
    }
    if base_split_path is not None:
        payload["base_split_file"] = str(base_split_path)
        payload["added_original_episode_ids"] = additions
        payload["added_counts"] = {name: len(values) for name, values in additions.items()}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return splits


def records_for_split(records: Iterable[FrameRecord], split_episode_ids: Iterable[int]) -> list[FrameRecord]:
    split_episode_ids = set(split_episode_ids)
    return [record for record in records if record.original_episode_id in split_episode_ids]


def execution_indices(
    records: Iterable[FrameRecord],
    *,
    action_chunk_valid_only: bool = True,
    action_horizon: int | None = None,
) -> list[int]:
    records = list(records)
    if action_horizon is not None:
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        episode_frame_counts: dict[int, int] = {}
        for record in records:
            episode_frame_counts[record.lerobot_episode_index] = max(
                episode_frame_counts.get(record.lerobot_episode_index, 0),
                record.frame_index + 1,
            )
        return [
            record.global_index
            for record in records
            if record.stage_a_eligible
            and record.execution_eligible
            and record.frame_index + action_horizon <= episode_frame_counts[record.lerobot_episode_index]
        ]

    selected = []
    for record in records:
        if not record.stage_a_eligible or not record.execution_eligible:
            continue
        if action_chunk_valid_only and not record.action_chunk_valid:
            continue
        selected.append(record.global_index)
    return selected


def status_indices(
    records: Iterable[FrameRecord],
    *,
    negative_ratio: float | None,
    seed: int,
) -> list[int]:
    records = list(records)
    positive = [record.global_index for record in records if record.need_recovery]
    negative_records = [record for record in records if not record.need_recovery]
    if negative_ratio is not None and positive:
        first_positive_frame_by_attempt: dict[tuple[int, int], int] = {}
        for record in records:
            if not record.need_recovery:
                continue
            key = (record.lerobot_episode_index, record.attempt_id)
            first_positive_frame_by_attempt[key] = min(
                first_positive_frame_by_attempt.get(key, record.frame_index),
                record.frame_index,
            )

        hard_negative = []
        easy_negative = []
        for record in negative_records:
            first_positive = first_positive_frame_by_attempt.get((record.lerobot_episode_index, record.attempt_id))
            if first_positive is not None and 0 < first_positive - record.frame_index <= 60:
                hard_negative.append(record.global_index)
            else:
                easy_negative.append(record.global_index)

        rng = random.Random(seed)
        keep = min(len(negative_records), int(round(len(positive) * negative_ratio)))
        rng.shuffle(hard_negative)
        selected_negative = hard_negative[:keep]
        if len(selected_negative) < keep:
            remaining = [index for index in easy_negative if index not in set(selected_negative)]
            selected_negative.extend(rng.sample(remaining, min(len(remaining), keep - len(selected_negative))))
        negative = selected_negative
    else:
        negative = [record.global_index for record in negative_records]
    return sorted(positive + negative)


def stratified_status_indices(
    records: Iterable[FrameRecord],
    *,
    negative_ratio: float,
    seed: int,
) -> list[int]:
    """Return deterministic class-stratified evaluation rows.

    The ordering is deliberately interleaved so a caller-side evaluation cap
    cannot consume only the long pre-failure negative prefix.
    """

    if negative_ratio <= 0:
        raise ValueError(f"negative_ratio must be positive, got {negative_ratio}")
    records = list(records)
    positive = [record.global_index for record in records if record.need_recovery]
    negative = [record.global_index for record in records if not record.need_recovery]
    if not positive or not negative:
        raise ValueError(
            "Stratified need-recovery evaluation requires both classes: "
            f"positive={len(positive)}, negative={len(negative)}"
        )
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    keep_negative = min(len(negative), int(round(len(positive) * negative_ratio)))
    negative = negative[:keep_negative]

    result: list[int] = []
    negative_offset = 0
    per_positive = max(1, int(round(negative_ratio)))
    for index in positive:
        result.append(index)
        result.extend(negative[negative_offset : negative_offset + per_positive])
        negative_offset += per_positive
    result.extend(negative[negative_offset:])
    return result


def failure_reason_indices(
    records: Iterable[FrameRecord],
    *,
    window_frames: int,
    training: bool,
) -> list[int]:
    """Select the fixed post-failure window used by the V3 diagnosis task.

    The first frame whose ``need_recovery`` flag is true is the failure trigger.
    Training uses exactly ``window_frames`` consecutive frames starting at that
    trigger.  Evaluation uses only the final frame in the same window so that a
    single attempt contributes a single, stable prediction.
    """

    if window_frames <= 0:
        raise ValueError(f"window_frames must be positive, got {window_frames}")

    by_attempt: dict[tuple[int, int], list[FrameRecord]] = {}
    for record in records:
        by_attempt.setdefault((record.lerobot_episode_index, record.attempt_id), []).append(record)

    selected: list[int] = []
    for attempt_key, attempt_records in sorted(by_attempt.items()):
        attempt_records.sort(key=lambda record: record.frame_index)
        failure_records = [
            record
            for record in attempt_records
            if record.need_recovery and bool(record.failure_reason.strip())
        ]
        if not failure_records:
            continue

        first = failure_records[0]
        by_frame = {record.frame_index: record for record in attempt_records}
        required_frames = range(first.frame_index, first.frame_index + window_frames)
        missing = [frame_index for frame_index in required_frames if frame_index not in by_frame]
        if missing:
            raise ValueError(
                "V3 failure window is incomplete for "
                f"lerobot_episode_index={attempt_key[0]}, attempt_id={attempt_key[1]}: "
                f"start_frame={first.frame_index}, window_frames={window_frames}, missing={missing}"
            )
        window = [by_frame[frame_index] for frame_index in required_frames]
        invalid = [record.frame_index for record in window if not record.failure_reason.strip()]
        if invalid:
            raise ValueError(
                "V3 failure window contains frames without failure_reason for "
                f"lerobot_episode_index={attempt_key[0]}, attempt_id={attempt_key[1]}: {invalid}"
            )
        if training:
            selected.extend(record.global_index for record in window)
        else:
            selected.append(window[-1].global_index)
    return sorted(selected)


def reasoning_index_pairs(records: Iterable[FrameRecord], *, augment_after_frames: int = 0) -> list[dict[str, int]]:
    records = sorted(records, key=lambda record: (record.lerobot_episode_index, record.frame_index, record.global_index))
    by_attempt: dict[tuple[int, int], list[FrameRecord]] = {}
    for record in records:
        by_attempt.setdefault((record.lerobot_episode_index, record.attempt_id), []).append(record)

    pairs: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for source in records:
        if not source.reasoning_has_sample:
            continue
        attempt_records = by_attempt[(source.lerobot_episode_index, source.attempt_id)]
        end_frame = source.frame_index + max(0, augment_after_frames)
        for target in attempt_records:
            if target.frame_index < source.frame_index or target.frame_index > end_frame:
                continue
            key = (target.global_index, source.global_index)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"index": target.global_index, "source_index": source.global_index})
    pairs.sort(key=lambda pair: pair["index"])
    return pairs


def reasoning_indices(records: Iterable[FrameRecord], *, augment_after_frames: int = 0) -> list[int]:
    return [pair["index"] for pair in reasoning_index_pairs(records, augment_after_frames=augment_after_frames)]


def summarize_records(records: Iterable[FrameRecord]) -> dict[str, Any]:
    records = list(records)
    return {
        "frames": len(records),
        "original_episodes": len({record.original_episode_id for record in records}),
        "lerobot_episodes": len({record.lerobot_episode_index for record in records}),
        "attempts": dict(Counter(record.attempt_id for record in records)),
        "case_id": dict(Counter(record.case_id for record in records)),
        "need_recovery": dict(Counter(str(record.need_recovery).lower() for record in records)),
        "rotation_state_name": dict(Counter(record.rotation_state_name for record in records)),
        "reasoning_samples": sum(record.reasoning_has_sample for record in records),
        "action_chunk_valid": dict(Counter(str(record.action_chunk_valid).lower() for record in records)),
        "execution_eligible": dict(Counter(str(record.execution_eligible).lower() for record in records)),
        "stage_a_eligible": dict(Counter(str(record.stage_a_eligible).lower() for record in records)),
    }


def index_payload(
    records: list[FrameRecord],
    splits: dict[str, list[int]],
    *,
    seed: int,
    negative_ratio: float,
    reasoning_augment_after_frames: int = 10,
    action_horizon: int = 30,
) -> dict:
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    payload: dict[str, Any] = {
        "seed": seed,
        "status_negative_ratio": negative_ratio,
        "reasoning_augment_after_frames": reasoning_augment_after_frames,
        "action_horizon": action_horizon,
        "splits": {},
    }
    for split_name, episode_ids in splits.items():
        split_records = records_for_split(records, episode_ids)
        reasoning_pairs = reasoning_index_pairs(
            split_records,
            augment_after_frames=reasoning_augment_after_frames,
        )
        train_like = split_name == "train"
        split_execution_indices = execution_indices(split_records, action_horizon=action_horizon)
        summary = summarize_records(split_records)
        summary["execution_action_horizon"] = action_horizon
        summary["execution_indices"] = len(split_execution_indices)
        payload["splits"][split_name] = {
            "summary": summary,
            "execution_indices": split_execution_indices,
            "status_indices": status_indices(
                split_records,
                negative_ratio=negative_ratio if train_like else None,
                seed=seed,
            ),
            "reasoning_indices": [pair["index"] for pair in reasoning_pairs],
            "reasoning_source_indices": [pair["source_index"] for pair in reasoning_pairs],
        }
    return payload


def v3_index_payload(
    records: list[FrameRecord],
    splits: dict[str, list[int]],
    *,
    seed: int,
    negative_ratio: float = 3.0,
    reasoning_window_frames: int = 15,
    action_horizon: int = 30,
) -> dict[str, Any]:
    """Build the four V3 Stage-B frame streams without legacy reasoning augmentation."""

    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if reasoning_window_frames <= 0:
        raise ValueError(
            f"reasoning_window_frames must be positive, got {reasoning_window_frames}"
        )

    payload: dict[str, Any] = {
        "schema_version": "tactile_vla_v3_stage_b_index_v1",
        "seed": seed,
        "status_negative_ratio": negative_ratio,
        "reasoning_window_frames": reasoning_window_frames,
        "action_horizon": action_horizon,
        "splits": {},
    }
    for split_name, episode_ids in splits.items():
        split_records = records_for_split(records, episode_ids)
        training = split_name == "train"
        execution = execution_indices(split_records, action_horizon=action_horizon)
        failure = failure_reason_indices(
            split_records,
            window_frames=reasoning_window_frames,
            training=training,
        )
        summary = summarize_records(split_records)
        summary.update(
            {
                "execution_action_horizon": action_horizon,
                "execution_indices": len(execution),
                "failure_reason_indices": len(failure),
            }
        )
        payload["splits"][split_name] = {
            "summary": summary,
            "execution_indices": execution,
            "status_indices": status_indices(
                split_records,
                negative_ratio=negative_ratio,
                seed=seed,
            )
            if training
            else stratified_status_indices(
                split_records,
                negative_ratio=negative_ratio,
                seed=seed,
            ),
            "failure_reason_indices": failure,
        }
    return payload


def validate_index_action_horizon(
    payload: dict[str, Any],
    expected_action_horizon: int,
    *,
    index_path: str | Path | None = None,
) -> None:
    """Fail fast when a training run is paired with indices for another horizon."""
    actual_action_horizon = int(payload.get("action_horizon", 30))
    if actual_action_horizon == expected_action_horizon:
        return
    location = f" at {index_path}" if index_path is not None else ""
    raise ValueError(
        f"Index action_horizon={actual_action_horizon}{location}, but the run requested "
        f"action_horizon={expected_action_horizon}. Regenerate or select the matching index file."
    )
