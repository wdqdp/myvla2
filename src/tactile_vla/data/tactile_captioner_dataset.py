"""Load six-head masked tactile shards and legacy five-head shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tactile_vla.common.labels import LABEL_SCHEMA_VERSION as LEGACY_SCHEMA_VERSION
from tactile_vla.common.labels_v4 import DATASET_FORMAT
from tactile_vla.common.labels_v4 import INVALID_LABEL_ID
from tactile_vla.common.labels_v4 import LABEL_SCHEMA_VERSION
from tactile_vla.common.labels_v4 import schema_for_version
from tactile_vla.common.labels_v4 import validate_label_maps


class TactileCaptionerDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str = "train",
        balanced: bool = False,
        normalize: bool = True,
        include_metadata: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.balanced = balanced
        self.normalize = normalize
        self.include_metadata = include_metadata
        self._files: dict[str, h5py.File] = {}

        with (self.dataset_dir / "meta.json").open("r", encoding="utf-8") as file:
            self.meta = json.load(file)
        with (self.dataset_dir / "norm_stats.json").open("r", encoding="utf-8") as file:
            self.stats = json.load(file)
        self._validate_meta()

        index_name = "train_balanced.npz" if split == "train" and balanced else f"{split}.npz"
        index_path = self.dataset_dir / "indices" / index_name
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index file: {index_path}")
        with np.load(index_path) as loaded:
            self.index = {key: loaded[key] for key in loaded.files}
        self._validate_index(index_path)

        self.sequences = {int(record["sequence_index"]): record for record in self.meta["sequences"]}
        if len(self.sequences) != len(self.meta["sequences"]):
            raise ValueError("meta.json contains duplicate sequence_index values")

        self.mesh_mean = self._stat_tensor("mesh_motion", "mean", expected_channels=12)
        self.mesh_std = self._stat_tensor("mesh_motion", "std", expected_channels=12)
        self.force_mean = self._stat_tensor("force", "mean", expected_channels=6)
        self.force_std = self._stat_tensor("force", "std", expected_channels=6)
        self._validate_windows()

    def _validate_meta(self) -> None:
        self.masked = self.meta.get("dataset_format") == DATASET_FORMAT
        if not self.masked and self.meta.get("dataset_format") != "tactile_captioner_shards_v2":
            raise ValueError(f"Unsupported tactile dataset format: {self.meta.get('dataset_format')!r}")
        expected_schema = LABEL_SCHEMA_VERSION if self.masked else LEGACY_SCHEMA_VERSION
        if self.meta.get("label_schema_version") != expected_schema:
            raise ValueError(f"Dataset format requires label schema {expected_schema!r}")
        self.label_fields, self.label_maps = schema_for_version(expected_schema)
        validate_label_maps(self.meta.get("label_maps", {}), schema_version=expected_schema)
        if self.masked and self.meta.get("label_policy") != "episode_constant":
            raise ValueError("Masked tactile data must use label_policy='episode_constant'")
        if not isinstance(self.meta.get("sequences"), list):
            raise ValueError("meta.json is missing the sequences list")
        if int(self.meta.get("window_size", 0)) <= 0:
            raise ValueError("meta.json window_size must be positive")

    def _validate_index(self, index_path: Path) -> None:
        required = {"sequence_index", "end_frame_index"}
        required.update(f"{field}_label" for field in self.label_fields)
        if self.masked:
            required.update(f"{field}_mask" for field in self.label_fields)
        missing = required - set(self.index)
        if missing:
            raise ValueError(f"{index_path} is missing index arrays: {sorted(missing)}")
        length = len(self.index["end_frame_index"])
        for key in required:
            if self.index[key].shape != (length,):
                raise ValueError(f"{index_path} {key} shape {self.index[key].shape} != {(length,)}")
        active_count = np.zeros(length, dtype=np.int64)
        for field in self.label_fields:
            values = self.index[f"{field}_label"]
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{index_path} {field} labels must be integers")
            mask = self.index[f"{field}_mask"] if self.masked else np.ones(length, dtype=np.bool_)
            if mask.dtype != np.bool_:
                raise ValueError(f"{index_path} {field} mask must be boolean")
            active_count += mask
            valid_ids = set(self.label_maps[field].values())
            found = set(int(value) for value in np.unique(values[mask]).tolist())
            if not found.issubset(valid_ids):
                raise ValueError(f"{index_path} has invalid supervised {field} ids: {sorted(found - valid_ids)}")
            if self.masked and np.any(values[~mask] != INVALID_LABEL_ID):
                raise ValueError(f"{index_path} inactive {field} labels must be -1")
        if self.masked and np.any(active_count != 1):
            raise ValueError(f"{index_path} requires exactly one supervised head per window")

    def _validate_windows(self) -> None:
        sequence_ids = self.index["sequence_index"]
        ends = self.index["end_frame_index"]
        for sequence_id in np.unique(sequence_ids):
            record = self.sequences.get(int(sequence_id))
            if record is None:
                raise ValueError(f"Unknown sequence_index={sequence_id}")
            if self.meta.get("split_policy") != "window_random" and record.get("split", self.split) != self.split:
                raise ValueError(f"Sequence {sequence_id} belongs to a different split")
            selected = sequence_ids == sequence_id
            sequence_ends = ends[selected]
            if np.any(sequence_ends < self.window_size - 1):
                raise ValueError(f"Sequence {sequence_id} has incomplete windows")
            if "num_frames" in record and np.any(sequence_ends >= int(record["num_frames"])):
                raise ValueError(f"Sequence {sequence_id} window exceeds frame count")
            if self.masked:
                field = record.get("label_task")
                target = record.get("target_label")
                if field not in self.label_maps or target not in self.label_maps[field]:
                    raise ValueError(f"Sequence {sequence_id} has invalid episode target")
                if not self.index[f"{field}_mask"][selected].all():
                    raise ValueError(f"Sequence {sequence_id} mask differs from episode target")
                if np.any(self.index[f"{field}_label"][selected] != self.label_maps[field][target]):
                    raise ValueError(f"Sequence {sequence_id} labels differ from episode target")

    @property
    def window_size(self) -> int:
        return int(self.meta["window_size"])

    @property
    def head_num_classes(self) -> dict[str, int]:
        return {field: len(self.label_maps[field]) for field in self.label_fields}

    def __len__(self) -> int:
        return int(len(self.index["end_frame_index"]))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def _stat_tensor(self, name: str, field: str, *, expected_channels: int) -> torch.Tensor:
        values = self.stats[name][field]
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.numel() != expected_channels:
            raise ValueError(
                f"norm_stats.json {name}.{field} has {tensor.numel()} values, expected {expected_channels}"
            )
        return tensor

    def _open_shard(self, shard_name: str) -> h5py.File:
        handle = self._files.get(shard_name)
        if handle is None:
            handle = h5py.File(self.dataset_dir / "shards" / shard_name, "r")
            self._files[shard_name] = handle
        return handle

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _normalize_grid(self, values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (values - mean.view(1, -1, 1, 1)) / std.clamp_min(1e-6).view(1, -1, 1, 1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index = int(self.index["sequence_index"][index])
        end_frame_index = int(self.index["end_frame_index"][index])
        try:
            record = self.sequences[sequence_index]
        except KeyError as exc:
            raise IndexError(f"Unknown sequence_index={sequence_index} for sample {index}") from exc
        start = end_frame_index - self.window_size + 1
        stop = end_frame_index + 1
        if start < 0:
            raise IndexError(f"Invalid tactile window start {start} for sample {index}")

        shard = self._open_shard(record["shard"])
        group = shard[record["group"]]
        mesh_motion = torch.from_numpy(group["mesh_motion"][start:stop]).float().permute(0, 3, 1, 2).contiguous()
        force = torch.from_numpy(group["force"][start:stop]).float().permute(0, 3, 1, 2).contiguous()
        if mesh_motion.shape[0] != self.window_size or force.shape[0] != self.window_size:
            raise IndexError(f"Sample {index} does not contain a complete window ending at {end_frame_index}")
        if self.normalize:
            mesh_motion = self._normalize_grid(mesh_motion, self.mesh_mean, self.mesh_std)
            force = self._normalize_grid(force, self.force_mean, self.force_std)

        item: dict[str, Any] = {
            "mesh_motion": mesh_motion,
            "force": force,
            "labels": {
                field: torch.tensor(int(self.index[f"{field}_label"][index]), dtype=torch.long)
                for field in self.label_fields
            },
        }
        item["masks"] = {
            field: torch.tensor(bool(self.index[f"{field}_mask"][index]) if self.masked else True, dtype=torch.bool)
            for field in self.label_fields
        }
        if self.include_metadata:
            item.update(
                {
                    "episode_id": int(record["episode_id"]),
                    "sequence_index": sequence_index,
                    "target_class": record.get("target_class", ""),
                    "end_frame_index": end_frame_index,
                    "end_timestamp": float(group["timestamp"][end_frame_index]),
                }
            )
        return item

    def normalization_stats(self) -> dict[str, list[float]]:
        return {
            "mesh_motion_mean": self.mesh_mean.tolist(),
            "mesh_motion_std": self.mesh_std.tolist(),
            "force_mean": self.force_mean.tolist(),
            "force_std": self.force_std.tolist(),
        }


def label_counts(dataset: TactileCaptionerDataset) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for field in dataset.label_fields:
        values = np.asarray(dataset.index[f"{field}_label"]).astype(np.int64)
        if dataset.masked:
            values = values[dataset.index[f"{field}_mask"]]
        result[field] = {
            label_id: int((values == label_id).sum())
            for label_id in range(len(dataset.label_maps[field]))
        }
    return result
