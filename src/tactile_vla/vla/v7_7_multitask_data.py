"""Data primitives shared by the V7.7 builder and trainer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE


DATA_PROFILE = "rotation_phase_v7_7_five_task_h100"
INDEX_SCHEMA = "tactile_vla_v7_7_multitask_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_v7_7_multitask_manifest_v1"
TASK_CYCLE = ("action", "adjustment", "need", "failure", "plan")
NEGATIVE_SOURCES = (
    "pre_failure_hard_negative",
    "one_success_easy_negative",
    "successful_recovery_easy_negative",
)


def stable_priority(seed: int, *values: Any) -> int:
    raw = ":".join(map(str, (seed, *values))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def allocate_balanced_without_replacement(
    capacities: Mapping[str, int], total: int, *, seed: int = 42,
) -> dict[str, int]:
    """Allocate a fixed total with minimum practical deviation from 1:1:1."""
    if set(capacities) != set(NEGATIVE_SOURCES):
        raise ValueError("need negative capacities have unexpected sources")
    if total < 0 or sum(int(v) for v in capacities.values()) < total:
        raise ValueError("not enough unique need-recovery negatives")
    result = {name: 0 for name in NEGATIVE_SOURCES}
    order = sorted(NEGATIVE_SOURCES, key=lambda name: (stable_priority(seed, name), name))
    while sum(result.values()) < total:
        available = [name for name in order if result[name] < int(capacities[name])]
        if not available:
            raise AssertionError("negative allocation exhausted unexpectedly")
        minimum = min(result[name] for name in available)
        candidates = [name for name in available if result[name] == minimum]
        result[candidates[0]] += 1
    return result


def deterministic_uniform_select(
    rows: Sequence[Mapping[str, Any]], count: int, *, seed: int, source: str,
) -> list[dict[str, Any]]:
    if not 0 <= count <= len(rows):
        raise ValueError(f"cannot select {count} from {len(rows)} {source} rows")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"]),
        ),
    )
    if count == 0:
        return []
    # Uniform coverage of the full ordered time population; the stable offset
    # prevents all three sources from sharing an identical rounding phase.
    edges = np.linspace(0, len(ordered), count + 1, dtype=np.int64)
    selected = []
    for index in range(count):
        left, right = int(edges[index]), int(edges[index + 1])
        width = max(1, right - left)
        offset = stable_priority(seed, source, index) % width
        selected.append(ordered[min(left + offset, len(ordered) - 1)])
    identities = {
        (int(row["global_index"]), int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"]))
        for row in selected
    }
    if len(identities) != len(selected):
        raise AssertionError("uniform selection produced duplicate frames")
    return selected


def select_need_rows(
    positives: Sequence[Mapping[str, Any]],
    negatives: Mapping[str, Sequence[Mapping[str, Any]]], *, seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positive_rows = [dict(row) for row in positives]
    target = 3 * len(positive_rows)
    capacities = {name: len(negatives[name]) for name in NEGATIVE_SOURCES}
    allocation = allocate_balanced_without_replacement(capacities, target, seed=seed)
    selected = []
    for source in NEGATIVE_SOURCES:
        selected.extend(deterministic_uniform_select(
            negatives[source], allocation[source], seed=seed, source=source
        ))
    rows = positive_rows + selected
    rows.sort(key=lambda row: stable_priority(
        seed, "need-shuffle", row["global_index"], row.get("source", "failure_active")
    ))
    return rows, {
        "positive_count": len(positive_rows),
        "negative_target": target,
        "candidate_counts": capacities,
        "selected_counts": allocation,
        "negative_to_positive_ratio": "3:1",
        "negative_source_target": "as_close_as_possible_to_1:1:1",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class V77ManifestDataset:
    """Read a pre-built V7.7 manifest without rebuilding prompt state online."""

    def __init__(self, *, rows, row_indices, global_indices, task: str, lerobot_dataset):
        if task not in {"adjustment", "need", "failure", "plan"}:
            raise ValueError(task)
        self.rows = rows
        self.row_indices = [int(v) for v in row_indices]
        self.global_indices = [int(v) for v in global_indices]
        self.task = task
        self.dataset = lerobot_dataset
        if len(self.row_indices) != len(self.global_indices):
            raise ValueError("manifest row/global index lengths differ")

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, index):
        row_index, global_index = self.row_indices[index], self.global_indices[index]
        row, item = self.rows[row_index], self.dataset[global_index]
        identity = tuple(int(item[key]) for key in ("index", "episode_id", "attempt_id", "frame_index"))
        expected = (global_index, int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"]))
        if identity != expected:
            raise ValueError(f"V7.7 manifest identity mismatch: {identity} != {expected}")
        result = {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": np.asarray(item["observation.state"], dtype=np.float32),
            "prompt": str(row["prompt"]),
            "global_index": np.int64(global_index),
            "episode_id": np.int64(identity[1]),
            "attempt_id": np.int64(identity[2]),
            "frame_index": np.int64(identity[3]),
            "manifest_row_index": np.int64(row_index),
        }
        if self.task == "adjustment":
            result["adjustment_end_label"] = np.int32(bool(row["adjustment_end"]))
        elif self.task == "need":
            result["need_recovery_label"] = np.int32(bool(row["need_recovery"]))
        elif self.task == "failure":
            result["target_text"] = str(row["target_failure_reason"])
        else:
            result["target_text"] = str(row["target_recovery_plan"])
        return result


class TransformedV77Dataset:
    def __init__(self, dataset, transform):
        self.dataset, self.transform = dataset, transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raw = self.dataset[index]
        metadata = {
            key: raw[key] for key in (
                "global_index", "episode_id", "attempt_id", "frame_index", "manifest_row_index"
            )
        }
        labels = {key: raw[key] for key in ("adjustment_end_label", "need_recovery_label") if key in raw}
        output = self.transform(raw)
        output.update(metadata)
        output.update(labels)
        return output


def validate_index(index: Mapping[str, Any]) -> None:
    if index.get("schema_version") != INDEX_SCHEMA or index.get("data_profile") != DATA_PROFILE:
        raise ValueError("V7.7 index header mismatch")
    if index.get("prompt_profile") != PROMPT_PROFILE or tuple(index.get("task_cycle", ())) != TASK_CYCLE:
        raise ValueError("V7.7 prompt/task protocol mismatch")
    for split in ("train", "val", "test"):
        streams = index["splits"][split]
        for task in ("adjustment", "need", "failure", "plan"):
            if len(streams[task]["manifest_row_indices"]) != len(streams[task]["global_indices"]):
                raise ValueError(f"{split}/{task} manifest identity lengths differ")


__all__ = [
    "DATA_PROFILE", "INDEX_SCHEMA", "MANIFEST_SCHEMA", "NEGATIVE_SOURCES", "PROMPT_PROFILE",
    "TASK_CYCLE", "TransformedV77Dataset", "V77ManifestDataset",
    "allocate_balanced_without_replacement", "deterministic_uniform_select", "load_jsonl",
    "select_need_rows", "stable_priority", "validate_index",
]
