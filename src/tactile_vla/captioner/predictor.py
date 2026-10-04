"""Predict captions from six-head V4 or legacy five-head V3 checkpoints."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tactile_vla.captioner.training import load_model_from_checkpoint
from tactile_vla.common.labels_v4 import label_id_to_name
from tactile_vla.common.labels_v4 import labels_to_caption
from tactile_vla.common.labels_v4 import validate_label_maps


@dataclasses.dataclass(frozen=True)
class CaptionerPrediction:
    label_ids: dict[str, int]
    label_names: dict[str, str]
    caption: str
    probabilities: dict[str, list[float]]


class TactileCaptionerPredictor:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        self.model, self.checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
        self.device = next(self.model.parameters()).device
        self.label_fields = tuple(self.model.head_num_classes)
        self.schema_version = self.checkpoint["label_schema_version"]
        validate_label_maps(self.checkpoint.get("label_maps", {}), schema_version=self.schema_version)
        self.window_size = int(self.checkpoint["dataset_meta"]["window_size"])
        normalization = self.checkpoint["normalization"]
        self.mesh_mean = self._stat_tensor(normalization["mesh_motion_mean"], channels=12)
        self.mesh_std = self._stat_tensor(normalization["mesh_motion_std"], channels=12)
        self.force_mean = self._stat_tensor(normalization["force_mean"], channels=6)
        self.force_std = self._stat_tensor(normalization["force_std"], channels=6)

    def _stat_tensor(self, values: Any, *, channels: int) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.numel() != channels:
            raise ValueError(f"Expected {channels} normalization values, got {tensor.numel()}")
        return tensor

    def _prepare_grid(self, values: np.ndarray | torch.Tensor, *, channels: int) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim == 4 and tensor.shape[-1] == channels:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.ndim == 4 and tensor.shape[1] == channels:
            pass
        else:
            raise ValueError(f"Expected [T,H,W,{channels}] or [T,{channels},H,W], got {tuple(tensor.shape)}")
        if tensor.shape[0] != self.window_size:
            raise ValueError(
                f"Captioner checkpoint expects window_size={self.window_size}, got {tensor.shape[0]} tactile frames"
            )
        return tensor.unsqueeze(0).contiguous()

    def _prepare_grid_batch(self, values: np.ndarray | torch.Tensor, *, channels: int) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim == 5 and tensor.shape[-1] == channels:
            tensor = tensor.permute(0, 1, 4, 2, 3)
        elif tensor.ndim == 5 and tensor.shape[2] == channels:
            pass
        else:
            raise ValueError(
                f"Expected [B,T,H,W,{channels}] or [B,T,{channels},H,W], got {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != self.window_size:
            raise ValueError(
                f"Captioner checkpoint expects window_size={self.window_size}, "
                f"got {tensor.shape[1]} tactile frames"
            )
        return tensor.contiguous()

    def _normalize(self, mesh: torch.Tensor, force_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mesh = (mesh - self.mesh_mean.view(1, 1, -1, 1, 1)) / self.mesh_std.clamp_min(1e-6).view(
            1, 1, -1, 1, 1
        )
        force_grid = (force_grid - self.force_mean.view(1, 1, -1, 1, 1)) / self.force_std.clamp_min(
            1e-6
        ).view(1, 1, -1, 1, 1)
        return mesh, force_grid

    def _decode_predictions(
        self,
        probabilities: dict[str, torch.Tensor],
    ) -> list[CaptionerPrediction]:
        batch_size = int(probabilities[self.label_fields[0]].shape[0])
        predictions: list[CaptionerPrediction] = []
        for batch_index in range(batch_size):
            label_ids = {
                field: int(torch.argmax(probabilities[field][batch_index]).item())
                for field in self.label_fields
            }
            label_names = {
                field: label_id_to_name(field, label_ids[field])
                for field in self.label_fields
            }
            predictions.append(
                CaptionerPrediction(
                    label_ids=label_ids,
                    label_names=label_names,
                    caption=labels_to_caption(label_names, schema_version=self.schema_version),
                    probabilities={
                        field: [
                            float(value)
                            for value in probabilities[field][batch_index].detach().cpu().tolist()
                        ]
                        for field in self.label_fields
                    },
                )
            )
        return predictions

    def predict(self, mesh_motion: np.ndarray | torch.Tensor, force: np.ndarray | torch.Tensor) -> CaptionerPrediction:
        mesh = self._prepare_grid(mesh_motion, channels=12)
        force_grid = self._prepare_grid(force, channels=6)
        mesh, force_grid = self._normalize(mesh, force_grid)
        with torch.no_grad():
            logits = self.model(mesh, force_grid)
            probabilities = {field: torch.softmax(logits[field], dim=-1) for field in self.label_fields}
        return self._decode_predictions(probabilities)[0]

    def predict_batch(
        self,
        mesh_motion: np.ndarray | torch.Tensor,
        force: np.ndarray | torch.Tensor,
    ) -> list[CaptionerPrediction]:
        """Predict a batch of complete trailing windows.

        Inputs use ``[B,T,H,W,C]`` (or channel-first ``[B,T,C,H,W]``),
        where ``T`` must equal the checkpoint's window size.
        """
        mesh = self._prepare_grid_batch(mesh_motion, channels=12)
        force_grid = self._prepare_grid_batch(force, channels=6)
        if mesh.shape[:2] != force_grid.shape[:2] or mesh.shape[-2:] != force_grid.shape[-2:]:
            raise ValueError(
                f"mesh_motion and force batch/window/grid shapes differ: {mesh.shape} vs {force_grid.shape}"
            )
        if mesh.shape[0] == 0:
            return []
        mesh, force_grid = self._normalize(mesh, force_grid)
        with torch.no_grad():
            logits = self.model(mesh, force_grid)
            probabilities = {field: torch.softmax(logits[field], dim=-1) for field in self.label_fields}
        return self._decode_predictions(probabilities)
