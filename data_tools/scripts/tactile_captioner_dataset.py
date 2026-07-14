#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch dataset for tactile-captioner shard datasets.

The dataset expects output produced by build_tactile_captioner_dataset.py.
"""

import json
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TactileCaptionerDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        balanced: bool = False,
        normalize: bool = True,
        include_force_resultant: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.balanced = balanced
        self.normalize = normalize
        self.include_force_resultant = include_force_resultant
        self._files: Dict[str, h5py.File] = {}

        with (self.dataset_dir / "meta.json").open("r") as file:
            self.meta = json.load(file)
        with (self.dataset_dir / "stats.json").open("r") as file:
            self.stats = json.load(file)

        index_name = "train_balanced.npz" if split == "train" and balanced else f"{split}.npz"
        index_path = self.dataset_dir / "indices" / index_name
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index file: {index_path}")

        loaded = np.load(index_path)
        self.index = {key: loaded[key] for key in loaded.files}
        self.attempts = {
            int(record["attempt_index"]): record
            for record in self.meta["attempts"]
        }
        self.case_ids = self.meta.get("case_ids", [])

        self.mesh_mean = self._stat_tensor("mesh_motion", "mean", expected_channels=12)
        self.mesh_std = self._stat_tensor("mesh_motion", "std", expected_channels=12)
        self.force_mean = self._stat_tensor("force", "mean", expected_channels=6)
        self.force_std = self._stat_tensor("force", "std", expected_channels=6)
        if "force_resultant" in self.stats:
            self.force_resultant_mean = self._stat_tensor("force_resultant", "mean", expected_channels=12)
            self.force_resultant_std = self._stat_tensor("force_resultant", "std", expected_channels=12)
        else:
            self.force_resultant_mean = None
            self.force_resultant_std = None

    def __len__(self) -> int:
        return int(len(self.index["label"]))

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def _stat_tensor(self, name: str, field: str, expected_channels: int) -> torch.Tensor:
        values = self.stats[name][field]
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.numel() != expected_channels:
            raise ValueError(f"stats.json {name}.{field} has {tensor.numel()} values, expected {expected_channels}")
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
        # values: [T, C, H, W]
        return (values - mean.view(1, -1, 1, 1)) / std.clamp_min(1e-6).view(1, -1, 1, 1)

    def _normalize_vector(self, values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # values: [T, C]
        return (values - mean.view(1, -1)) / std.clamp_min(1e-6).view(1, -1)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        attempt_index = int(self.index["attempt_index"][index])
        end_frame_index = int(self.index["end_frame_index"][index])
        label = int(self.index["label"][index])
        record = self.attempts[attempt_index]
        window_size = int(self.meta["window_size"])
        start = end_frame_index - window_size + 1
        stop = end_frame_index + 1
        if start < 0:
            raise IndexError(f"Invalid tactile window start {start} for sample {index}")

        shard = self._open_shard(record["shard"])
        group = shard[record["group"]]

        mesh_motion = torch.from_numpy(group["mesh_motion"][start:stop]).float().permute(0, 3, 1, 2).contiguous()
        force = torch.from_numpy(group["force"][start:stop]).float().permute(0, 3, 1, 2).contiguous()
        if self.normalize:
            mesh_motion = self._normalize_grid(mesh_motion, self.mesh_mean, self.mesh_std)
            force = self._normalize_grid(force, self.force_mean, self.force_std)

        item: Dict[str, Any] = {
            "mesh_motion": mesh_motion,
            "force": force,
            "label": torch.tensor(label, dtype=torch.long),
            "episode_id": int(record["episode_id"]),
            "attempt_id": int(record["attempt_id"]),
            "case_id": record["case_id"],
            "attempt_index": attempt_index,
            "end_frame_index": end_frame_index,
            "end_timestamp": float(group["timestamp"][end_frame_index]),
        }

        if self.include_force_resultant and "force_resultant" in group:
            force_resultant = torch.from_numpy(group["force_resultant"][start:stop]).float().contiguous()
            if self.normalize and self.force_resultant_mean is not None and self.force_resultant_std is not None:
                force_resultant = self._normalize_vector(
                    force_resultant,
                    self.force_resultant_mean,
                    self.force_resultant_std,
                )
            item["force_resultant"] = force_resultant

        return item


def label_counts(dataset: TactileCaptionerDataset) -> Dict[int, int]:
    labels = dataset.index["label"].astype(np.int64)
    return {
        int(label_id): int((labels == label_id).sum())
        for label_id in sorted(np.unique(labels).tolist())
    }
