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
        data = pq.read_table(parquet_file, columns=columns).to_pydict()
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
                )
            )
    records.sort(key=lambda record: record.global_index)
    return records


def build_splits(records: Iterable[FrameRecord], config: SplitConfig) -> dict[str, list[int]]:
    original_episode_ids = sorted({record.original_episode_id for record in records})
    rng = random.Random(config.seed)
    rng.shuffle(original_episode_ids)

    total = len(original_episode_ids)
    train_count = int(round(total * config.train_ratio))
    val_count = int(round(total * config.val_ratio))
    train_ids = sorted(original_episode_ids[:train_count])
    val_ids = sorted(original_episode_ids[train_count : train_count + val_count])
    test_ids = sorted(original_episode_ids[train_count + val_count :])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def load_or_create_splits(
    records: list[FrameRecord],
    split_path: str | Path,
    config: SplitConfig,
    *,
    overwrite: bool = False,
) -> dict[str, list[int]]:
    split_path = Path(split_path)
    if split_path.exists() and not overwrite:
        payload = json.loads(split_path.read_text())
        return {name: [int(value) for value in values] for name, values in payload["original_episode_ids"].items()}

    splits = build_splits(records, config)
    payload = {
        "split_unit": "original_episode_id",
        "config": asdict(config),
        "original_episode_ids": splits,
        "counts": {name: len(values) for name, values in splits.items()},
    }
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
            if record.frame_index + action_horizon <= episode_frame_counts[record.lerobot_episode_index]
        ]

    selected = []
    for record in records:
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
