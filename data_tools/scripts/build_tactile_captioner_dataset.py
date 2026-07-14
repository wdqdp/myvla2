#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a compact tactile-captioner dataset from per-attempt HDF5 files.

The output stores each attempt sequence once in HDF5 shards and stores training
windows as lightweight indices. This avoids materializing overlapping windows.
"""

import argparse
import collections
import dataclasses
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import tqdm


GRID_H = 35
GRID_W = 20
MESH_MOTION_C = 12
FORCE_C = 6
FORCE_RESULTANT_C = 12

LABEL_MAP = {
    "none": 0,
    "clockwise": 1,
    "counterclockwise": 2,
}

ID_TO_LABEL = {
    0: "none",
    1: "clockwise",
    2: "counterclockwise",
}

REQUIRED_DATASETS = (
    "timestamp",
    "tactile/mesh_motion",
    "tactile/force_concat",
    "label/rotation_state_id",
)


@dataclasses.dataclass
class AttemptRecord:
    attempt_index: int
    source_hdf5: str
    source_relpath: str
    episode_id: int
    attempt_id: int
    case_id: str
    num_frames: int
    shard_index: int
    shard: str
    group: str
    labels: np.ndarray
    split: str = ""

    @property
    def num_windows(self) -> int:
        return max(0, self.num_frames)

    def to_meta(self, window_size: int) -> Dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "source_hdf5": self.source_hdf5,
            "source_relpath": self.source_relpath,
            "episode_id": self.episode_id,
            "attempt_id": self.attempt_id,
            "case_id": self.case_id,
            "split": self.split,
            "num_frames": self.num_frames,
            "num_windows": max(0, self.num_frames - window_size + 1),
            "shard_index": self.shard_index,
            "shard": self.shard,
            "group": self.group,
            "label_counts": label_counts_dict(self.labels),
            "window_label_counts": label_counts_dict(self.labels[window_size - 1 :]),
        }


class ChannelStats:
    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.count = 0
        self.sum = np.zeros((channels,), dtype=np.float64)
        self.sq_sum = np.zeros((channels,), dtype=np.float64)

    def update_grid(self, values: np.ndarray) -> None:
        if values.ndim != 4 or values.shape[-1] != self.channels:
            raise ValueError(f"Expected NHWC grid with {self.channels} channels, got {values.shape}")
        values64 = values.astype(np.float64, copy=False)
        self.sum += values64.sum(axis=(0, 1, 2))
        self.sq_sum += (values64 * values64).sum(axis=(0, 1, 2))
        self.count += int(values.shape[0] * values.shape[1] * values.shape[2])

    def update_vector(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[-1] != self.channels:
            raise ValueError(f"Expected NC vector with {self.channels} channels, got {values.shape}")
        values64 = values.astype(np.float64, copy=False)
        self.sum += values64.sum(axis=0)
        self.sq_sum += (values64 * values64).sum(axis=0)
        self.count += int(values.shape[0])

    def finalize(self) -> Dict[str, Any]:
        if self.count == 0:
            mean = np.zeros((self.channels,), dtype=np.float64)
            std = np.ones((self.channels,), dtype=np.float64)
        else:
            mean = self.sum / self.count
            var = np.maximum(self.sq_sum / self.count - mean * mean, 1e-12)
            std = np.sqrt(var)
        return {
            "count": int(self.count),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def decode_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_value(value[()])
        if value.size == 1:
            return decode_value(value.reshape(-1)[0])
    if value is None:
        return ""
    return str(value)


def parse_number_from_name(name: str, prefix: str, default: int) -> int:
    match = re.match(rf"{re.escape(prefix)}(\d+)$", name)
    return int(match.group(1)) if match else default


def read_meta_scalar(handle: h5py.File, key: str, default: Any) -> Any:
    dataset_name = f"meta/{key}"
    if dataset_name not in handle:
        return default
    return handle[dataset_name][()]


def read_meta_int(handle: h5py.File, key: str, default: int) -> int:
    try:
        return int(read_meta_scalar(handle, key, default))
    except (TypeError, ValueError):
        return default


def read_meta_string(handle: h5py.File, key: str, default: str) -> str:
    return decode_value(read_meta_scalar(handle, key, default))


def iter_hdf5_files(input_dir: Path) -> Iterable[Path]:
    if (input_dir / "data.hdf5").is_file():
        yield input_dir / "data.hdf5"
        return

    episode_dirs = sorted(
        [path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith("episode")],
        key=lambda path: natural_key(path.name),
    )
    for episode_dir in episode_dirs:
        attempt_dirs = sorted(
            [path for path in episode_dir.iterdir() if path.is_dir() and path.name.startswith("attempt")],
            key=lambda path: natural_key(path.name),
        )
        for attempt_dir in attempt_dirs:
            hdf5_path = attempt_dir / "data.hdf5"
            if hdf5_path.is_file():
                yield hdf5_path


def label_counts_dict(labels: np.ndarray) -> Dict[str, int]:
    counts = collections.Counter(int(value) for value in labels.tolist())
    return {str(label_id): int(counts.get(label_id, 0)) for label_id in sorted(ID_TO_LABEL)}


def require_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    if name not in handle:
        raise KeyError(f"Missing dataset: {name}")
    return handle[name]


def validate_attempt(
    hdf5_path: Path,
    input_dir: Path,
    attempt_index: int,
    shard_size: int,
    window_size: int,
    include_force_resultant: bool,
) -> AttemptRecord:
    with h5py.File(hdf5_path, "r") as handle:
        for dataset_name in REQUIRED_DATASETS:
            require_dataset(handle, dataset_name)

        timestamps = handle["timestamp"]
        labels = handle["label/rotation_state_id"][()].astype(np.int64)
        num_frames = int(timestamps.shape[0])
        if num_frames < window_size:
            raise ValueError(f"{hdf5_path} has {num_frames} frames, shorter than windowSize={window_size}")

        mesh_shape = tuple(handle["tactile/mesh_motion"].shape)
        force_shape = tuple(handle["tactile/force_concat"].shape)
        if mesh_shape != (num_frames, GRID_H, GRID_W, MESH_MOTION_C):
            raise ValueError(
                f"{hdf5_path} tactile/mesh_motion shape {mesh_shape} != "
                f"{(num_frames, GRID_H, GRID_W, MESH_MOTION_C)}"
            )
        if force_shape != (num_frames, GRID_H, GRID_W, FORCE_C):
            raise ValueError(
                f"{hdf5_path} tactile/force_concat shape {force_shape} != "
                f"{(num_frames, GRID_H, GRID_W, FORCE_C)}"
            )
        if labels.shape != (num_frames,):
            raise ValueError(f"{hdf5_path} label/rotation_state_id shape {labels.shape} != {(num_frames,)}")
        label_values = set(int(value) for value in np.unique(labels).tolist())
        if not label_values.issubset(set(ID_TO_LABEL)):
            raise ValueError(f"{hdf5_path} has invalid rotation labels: {sorted(label_values)}")

        if include_force_resultant:
            for name in ("left", "right"):
                dataset_name = f"tactile/force_resultant/{name}"
                dataset = require_dataset(handle, dataset_name)
                if tuple(dataset.shape) != (num_frames, FORCE_C):
                    raise ValueError(f"{hdf5_path} {dataset_name} shape {dataset.shape} != {(num_frames, FORCE_C)}")

        default_episode_id = parse_number_from_name(hdf5_path.parent.parent.name, "episode", -1)
        default_attempt_id = parse_number_from_name(hdf5_path.parent.name, "attempt", -1)
        episode_id = read_meta_int(handle, "episode_id", default_episode_id)
        attempt_id = read_meta_int(handle, "attempt_id", default_attempt_id)
        case_id = read_meta_string(handle, "case_id", "")

    shard_index = attempt_index // shard_size
    return AttemptRecord(
        attempt_index=attempt_index,
        source_hdf5=str(hdf5_path),
        source_relpath=hdf5_path.relative_to(input_dir).as_posix() if hdf5_path.is_relative_to(input_dir) else str(hdf5_path),
        episode_id=episode_id,
        attempt_id=attempt_id,
        case_id=case_id,
        num_frames=num_frames,
        shard_index=shard_index,
        shard=f"shard_{shard_index:03d}.hdf5",
        group=f"/attempts/{attempt_index:06d}",
        labels=labels,
    )


def scan_attempts(args: argparse.Namespace) -> List[AttemptRecord]:
    input_dir = Path(args.inputDir)
    hdf5_files = list(iter_hdf5_files(input_dir))
    if args.maxAttempts > 0:
        hdf5_files = hdf5_files[: args.maxAttempts]
    if not hdf5_files:
        raise ValueError(f"No data.hdf5 files found under {input_dir}")

    records: List[AttemptRecord] = []
    for attempt_index, hdf5_path in enumerate(tqdm.tqdm(hdf5_files, desc="Scanning HDF5 attempts")):
        records.append(
            validate_attempt(
                hdf5_path=hdf5_path,
                input_dir=input_dir,
                attempt_index=attempt_index,
                shard_size=args.shardSize,
                window_size=args.windowSize,
                include_force_resultant=args.includeForceResultant,
            )
        )
    return records


def assign_splits(records: Sequence[AttemptRecord], val_ratio: float, test_ratio: float, seed: int) -> None:
    episodes_by_case: Dict[str, Dict[int, List[AttemptRecord]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for record in records:
        episodes_by_case[record.case_id][record.episode_id].append(record)

    rng = np.random.default_rng(seed)
    episode_to_split: Dict[int, str] = {}
    for case_id, episodes in sorted(episodes_by_case.items()):
        episode_ids = np.asarray(sorted(episodes), dtype=np.int64)
        rng.shuffle(episode_ids)
        count = len(episode_ids)
        test_count = int(round(count * test_ratio))
        val_count = int(round(count * val_ratio))
        if count > 2 and test_ratio > 0:
            test_count = max(1, test_count)
        if count > 2 and val_ratio > 0:
            val_count = max(1, val_count)
        if test_count + val_count >= count:
            overflow = test_count + val_count - count + 1
            val_count = max(0, val_count - overflow)
        test_ids = set(int(value) for value in episode_ids[:test_count])
        val_ids = set(int(value) for value in episode_ids[test_count : test_count + val_count])
        train_ids = set(int(value) for value in episode_ids[test_count + val_count :])
        for episode_id in test_ids:
            episode_to_split[episode_id] = "test"
        for episode_id in val_ids:
            episode_to_split[episode_id] = "val"
        for episode_id in train_ids:
            episode_to_split[episode_id] = "train"

    for record in records:
        record.split = episode_to_split[record.episode_id]


def dataset_kwargs(args: argparse.Namespace, shape: Tuple[int, ...]) -> Dict[str, Any]:
    if args.compression == "none":
        return {}
    first_dim = int(shape[0])
    if len(shape) == 4:
        chunks = (min(first_dim, max(args.windowSize * 2, 32)), shape[1], shape[2], shape[3])
    elif len(shape) == 2:
        chunks = (min(first_dim, 512), shape[1])
    else:
        chunks = (min(first_dim, 2048),)

    kwargs: Dict[str, Any] = {
        "compression": args.compression,
        "chunks": chunks,
        "shuffle": True,
    }
    if args.compression == "gzip":
        kwargs["compression_opts"] = args.compressionLevel
    return kwargs


def check_finite(name: str, values: np.ndarray, source_path: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{source_path} {name} contains non-finite values")


def write_array_dataset(group: h5py.Group, name: str, values: np.ndarray, args: argparse.Namespace) -> None:
    group.create_dataset(name, data=values, **dataset_kwargs(args, tuple(values.shape)))


def write_shards_and_stats(records: Sequence[AttemptRecord], args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.outputDir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    mesh_stats = ChannelStats(MESH_MOTION_C)
    force_stats = ChannelStats(FORCE_C)
    force_resultant_stats = ChannelStats(FORCE_RESULTANT_C)

    records_by_shard: Dict[int, List[AttemptRecord]] = collections.defaultdict(list)
    for record in records:
        records_by_shard[record.shard_index].append(record)

    for shard_index in tqdm.tqdm(sorted(records_by_shard), desc="Writing tactile shards"):
        shard_path = shards_dir / f"shard_{shard_index:03d}.hdf5"
        with h5py.File(shard_path, "w") as shard:
            attempts_group = shard.require_group("attempts")
            shard.attrs["shard_index"] = shard_index
            shard.attrs["window_size"] = args.windowSize

            for record in records_by_shard[shard_index]:
                with h5py.File(record.source_hdf5, "r") as source:
                    mesh_motion = source["tactile/mesh_motion"][()].astype(np.float32, copy=False)
                    force = source["tactile/force_concat"][()].astype(np.float32, copy=False)
                    labels = source["label/rotation_state_id"][()].astype(np.int64, copy=False)
                    timestamps = source["timestamp"][()].astype(np.float64, copy=False)

                    if args.checkFinite:
                        check_finite("tactile/mesh_motion", mesh_motion, record.source_hdf5)
                        check_finite("tactile/force_concat", force, record.source_hdf5)
                        check_finite("timestamp", timestamps, record.source_hdf5)

                    group = attempts_group.create_group(f"{record.attempt_index:06d}")
                    group.attrs["attempt_index"] = record.attempt_index
                    group.attrs["episode_id"] = record.episode_id
                    group.attrs["attempt_id"] = record.attempt_id
                    group.attrs["case_id"] = record.case_id
                    group.attrs["split"] = record.split
                    group.attrs["source_hdf5"] = record.source_hdf5
                    group.attrs["source_relpath"] = record.source_relpath
                    group.attrs["num_frames"] = record.num_frames

                    write_array_dataset(group, "mesh_motion", mesh_motion, args)
                    write_array_dataset(group, "force", force, args)
                    write_array_dataset(group, "rotation_label", labels, args)
                    write_array_dataset(group, "timestamp", timestamps, args)

                    if args.includeForceResultant:
                        force_resultant = np.concatenate(
                            [
                                source["tactile/force_resultant/left"][()].astype(np.float32, copy=False),
                                source["tactile/force_resultant/right"][()].astype(np.float32, copy=False),
                            ],
                            axis=-1,
                        )
                        if args.checkFinite:
                            check_finite("tactile/force_resultant", force_resultant, record.source_hdf5)
                        write_array_dataset(group, "force_resultant", force_resultant, args)

                    if record.split == "train":
                        mesh_stats.update_grid(mesh_motion)
                        force_stats.update_grid(force)
                        if args.includeForceResultant:
                            force_resultant_stats.update_vector(force_resultant)

    stats = {
        "source": "train_split_frames",
        "mesh_motion": mesh_stats.finalize(),
        "force": force_stats.finalize(),
    }
    if args.includeForceResultant:
        stats["force_resultant"] = force_resultant_stats.finalize()
    return stats


def build_index_for_split(
    records: Sequence[AttemptRecord],
    split: str,
    window_size: int,
    case_id_to_index: Dict[str, int],
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, List[np.ndarray]] = {
        "attempt_index": [],
        "shard_index": [],
        "end_frame_index": [],
        "label": [],
        "episode_id": [],
        "attempt_id": [],
        "case_id_index": [],
    }

    for record in records:
        if record.split != split:
            continue
        labels = record.labels[window_size - 1 :]
        end_frame_index = np.arange(window_size - 1, record.num_frames, dtype=np.int64)
        count = len(end_frame_index)
        arrays["attempt_index"].append(np.full((count,), record.attempt_index, dtype=np.int64))
        arrays["shard_index"].append(np.full((count,), record.shard_index, dtype=np.int64))
        arrays["end_frame_index"].append(end_frame_index)
        arrays["label"].append(labels.astype(np.int64, copy=False))
        arrays["episode_id"].append(np.full((count,), record.episode_id, dtype=np.int64))
        arrays["attempt_id"].append(np.full((count,), record.attempt_id, dtype=np.int64))
        arrays["case_id_index"].append(np.full((count,), case_id_to_index[record.case_id], dtype=np.int64))

    result: Dict[str, np.ndarray] = {}
    for key, parts in arrays.items():
        if parts:
            result[key] = np.concatenate(parts, axis=0)
        else:
            result[key] = np.asarray([], dtype=np.int64)
    return result


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def subset_index(index: Dict[str, np.ndarray], selected: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: values[selected] for key, values in index.items()}


def make_balanced_train_index(
    train_index: Dict[str, np.ndarray],
    records: Sequence[AttemptRecord],
    none_multiplier: float,
    time_bins: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    labels = train_index["label"]
    if len(labels) == 0:
        return train_index

    positive_indices = np.flatnonzero(labels != 0)
    none_indices = np.flatnonzero(labels == 0)
    if len(positive_indices) == 0 or len(none_indices) == 0:
        return train_index

    none_keep = int(round(none_multiplier * len(positive_indices)))
    none_keep = max(0, min(none_keep, len(none_indices)))
    if none_keep == len(none_indices):
        selected = np.arange(len(labels), dtype=np.int64)
        return subset_index(train_index, selected)

    max_attempt_index = max(record.attempt_index for record in records)
    num_frames_by_attempt = np.zeros((max_attempt_index + 1,), dtype=np.int64)
    for record in records:
        num_frames_by_attempt[record.attempt_index] = record.num_frames

    attempt_index = train_index["attempt_index"][none_indices]
    end_frame_index = train_index["end_frame_index"][none_indices]
    case_id_index = train_index["case_id_index"][none_indices]
    attempt_id = train_index["attempt_id"][none_indices]
    denom = np.maximum(num_frames_by_attempt[attempt_index], 1)
    time_bin = np.floor(end_frame_index / denom * time_bins).astype(np.int64)
    time_bin = np.clip(time_bin, 0, time_bins - 1)

    strata: Dict[Tuple[int, int, int], List[int]] = collections.defaultdict(list)
    for local_index, key in enumerate(zip(case_id_index.tolist(), attempt_id.tolist(), time_bin.tolist())):
        strata[key].append(int(none_indices[local_index]))

    quotas: Dict[Tuple[int, int, int], int] = {}
    fractional: List[Tuple[float, Tuple[int, int, int]]] = []
    total_floor = 0
    for key, values in strata.items():
        exact = len(values) * none_keep / len(none_indices)
        floor_value = int(math.floor(exact))
        quotas[key] = floor_value
        total_floor += floor_value
        fractional.append((exact - floor_value, key))
    fractional.sort(reverse=True)
    for _, key in fractional[: none_keep - total_floor]:
        quotas[key] += 1

    rng = np.random.default_rng(seed)
    selected_none: List[np.ndarray] = []
    for key, values in strata.items():
        quota = quotas[key]
        if quota <= 0:
            continue
        values_array = np.asarray(values, dtype=np.int64)
        selected_none.append(rng.choice(values_array, size=quota, replace=False))

    selected_parts = [positive_indices]
    if selected_none:
        selected_parts.append(np.concatenate(selected_none, axis=0))
    selected = np.concatenate(selected_parts, axis=0).astype(np.int64)
    rng.shuffle(selected)
    return subset_index(train_index, selected)


def index_summary(index: Dict[str, np.ndarray]) -> Dict[str, Any]:
    labels = index["label"]
    counts = label_counts_dict(labels.astype(np.int64))
    return {
        "num_windows": int(len(labels)),
        "label_counts": counts,
        "num_attempts": int(len(set(index["attempt_index"].tolist()))) if len(labels) else 0,
        "num_episodes": int(len(set(index["episode_id"].tolist()))) if len(labels) else 0,
    }


def write_splits(records: Sequence[AttemptRecord], output_dir: Path, window_size: int) -> None:
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        attempts = [record.to_meta(window_size) for record in records if record.split == split]
        data = {
            "split": split,
            "num_attempts": len(attempts),
            "num_episodes": len(set(item["episode_id"] for item in attempts)),
            "num_frames": int(sum(item["num_frames"] for item in attempts)),
            "num_windows": int(sum(item["num_windows"] for item in attempts)),
            "attempts": attempts,
        }
        with (splits_dir / f"{split}.json").open("w") as file:
            json.dump(data, file, indent=2)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2)


def prepare_output_dir(output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"outputDir already exists, pass --overwrite true to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build(args: argparse.Namespace) -> None:
    input_dir = Path(args.inputDir)
    output_dir = Path(args.outputDir)
    if not input_dir.exists():
        raise ValueError(f"inputDir does not exist: {input_dir}")
    if args.windowSize <= 0:
        raise ValueError("windowSize must be positive")
    if args.shardSize <= 0:
        raise ValueError("shardSize must be positive")
    if args.valRatio < 0 or args.testRatio < 0 or args.valRatio + args.testRatio >= 1:
        raise ValueError("valRatio and testRatio must be non-negative and sum to less than 1")

    records = scan_attempts(args)
    assign_splits(records, args.valRatio, args.testRatio, args.seed)

    case_ids = sorted(set(record.case_id for record in records))
    case_id_to_index = {case_id: index for index, case_id in enumerate(case_ids)}
    split_indices = {
        split: build_index_for_split(records, split, args.windowSize, case_id_to_index)
        for split in ("train", "val", "test")
    }
    train_balanced = make_balanced_train_index(
        split_indices["train"],
        records=records,
        none_multiplier=args.balanceNoneMultiplier,
        time_bins=args.balanceTimeBins,
        seed=args.seed,
    )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window_size": args.windowSize,
        "num_attempts": len(records),
        "num_episodes": len(set(record.episode_id for record in records)),
        "num_frames": int(sum(record.num_frames for record in records)),
        "num_windows": int(sum(max(0, record.num_frames - args.windowSize + 1) for record in records)),
        "splits": {split: index_summary(index) for split, index in split_indices.items()},
        "train_balanced": index_summary(train_balanced),
    }

    if args.dryRun:
        print(json.dumps(summary, indent=2))
        return

    prepare_output_dir(output_dir, args.overwrite, args.dryRun)
    stats = write_shards_and_stats(records, args)

    write_splits(records, output_dir, args.windowSize)
    indices_dir = output_dir / "indices"
    for split, index in split_indices.items():
        save_npz(indices_dir / f"{split}.npz", index)
    save_npz(indices_dir / "train_balanced.npz", train_balanced)

    meta = {
        "dataset_format": "tactile_captioner_shards_v1",
        "source_input_dir": str(input_dir),
        "window_size": args.windowSize,
        "label_map": LABEL_MAP,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
        "case_ids": case_ids,
        "mesh_motion_shape": [GRID_H, GRID_W, MESH_MOTION_C],
        "force_shape": [GRID_H, GRID_W, FORCE_C],
        "force_resultant_shape": [FORCE_RESULTANT_C] if args.includeForceResultant else None,
        "include_force_resultant": args.includeForceResultant,
        "shard_size": args.shardSize,
        "compression": args.compression,
        "compression_level": args.compressionLevel if args.compression == "gzip" else None,
        "attempts": [record.to_meta(args.windowSize) for record in records],
    }
    write_json(output_dir / "meta.json", meta)
    write_json(output_dir / "stats.json", stats)
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    print(f"Wrote tactile captioner dataset to {output_dir}")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputDir", type=str, default="/home/test/qxh/workspace/tac_ws/data")
    parser.add_argument("--outputDir", type=str, default="/home/test/qxh/workspace/tac_ws/tactile_captioner_data")
    parser.add_argument("--windowSize", type=int, default=30)
    parser.add_argument("--shardSize", type=int, default=64)
    parser.add_argument("--includeForceResultant", type=str2bool, default=True)
    parser.add_argument("--compression", type=str, choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--compressionLevel", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valRatio", type=float, default=0.1)
    parser.add_argument("--testRatio", type=float, default=0.1)
    parser.add_argument("--balanceNoneMultiplier", type=float, default=2.0)
    parser.add_argument("--balanceTimeBins", type=int, default=3)
    parser.add_argument("--checkFinite", type=str2bool, default=True)
    parser.add_argument("--maxAttempts", type=int, default=-1)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--dryRun", type=str2bool, default=False)
    return parser.parse_args()


def main() -> None:
    build(get_arguments())


if __name__ == "__main__":
    main()
