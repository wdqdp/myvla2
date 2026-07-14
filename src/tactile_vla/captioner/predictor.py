"""Runtime predictor wrapper for the tactile captioner."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tactile_vla.captioner.training import load_model_from_checkpoint
from tactile_vla.common.labels import label_id_to_name
from tactile_vla.common.labels import label_to_caption


@dataclasses.dataclass(frozen=True)
class CaptionerPrediction:
    label_id: int
    label_name: str
    caption: str
    probabilities: list[float]


class TactileCaptionerPredictor:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        self.model, self.checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
        self.device = next(self.model.parameters()).device
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
        return tensor.unsqueeze(0).contiguous()

    def predict(self, mesh_motion: np.ndarray | torch.Tensor, force: np.ndarray | torch.Tensor) -> CaptionerPrediction:
        mesh = self._prepare_grid(mesh_motion, channels=12)
        force_grid = self._prepare_grid(force, channels=6)
        mesh = (mesh - self.mesh_mean.view(1, 1, -1, 1, 1)) / self.mesh_std.clamp_min(1e-6).view(1, 1, -1, 1, 1)
        force_grid = (force_grid - self.force_mean.view(1, 1, -1, 1, 1)) / self.force_std.clamp_min(1e-6).view(
            1, 1, -1, 1, 1
        )
        with torch.no_grad():
            logits = self.model(mesh, force_grid)
            probs = torch.softmax(logits, dim=-1)[0]
        label_id = int(torch.argmax(probs).item())
        label_name = label_id_to_name(label_id)
        return CaptionerPrediction(
            label_id=label_id,
            label_name=label_name,
            caption=label_to_caption(label_name),
            probabilities=[float(value) for value in probs.detach().cpu().tolist()],
        )
