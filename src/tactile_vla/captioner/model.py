"""Tactile captioner network.

The v1 model consumes only mesh_motion and force grids. force_resultant is deliberately
excluded so training and runtime inference use the same modalities.
"""

from __future__ import annotations

import torch
from torch import nn

from tactile_vla.common.labels import LABEL_FIELDS
from tactile_vla.common.labels import LABEL_MAPS


class FrameGridEncoder(nn.Module):
    def __init__(self, input_channels: int = 18, feature_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


class TactileCaptioner(nn.Module):
    def __init__(
        self,
        *,
        mesh_channels: int = 12,
        force_channels: int = 6,
        frame_feature_dim: int = 128,
        temporal_hidden_dim: int = 192,
        temporal_dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        head_num_classes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.mesh_channels = mesh_channels
        self.force_channels = force_channels
        provided_head_sizes = dict(head_num_classes or {field: len(LABEL_MAPS[field]) for field in LABEL_FIELDS})
        if set(provided_head_sizes) != set(LABEL_FIELDS):
            raise ValueError(f"head_num_classes fields must be {LABEL_FIELDS}, got {tuple(provided_head_sizes)}")
        self.head_num_classes = {field: int(provided_head_sizes[field]) for field in LABEL_FIELDS}
        if any(num_classes <= 1 for num_classes in self.head_num_classes.values()):
            raise ValueError(f"Each tactile classification head needs at least two classes: {self.head_num_classes}")
        self.frame_encoder = FrameGridEncoder(
            input_channels=mesh_channels + force_channels,
            feature_dim=frame_feature_dim,
            dropout=dropout,
        )
        self.temporal_in = nn.Sequential(
            nn.Linear(frame_feature_dim, temporal_hidden_dim),
            nn.LayerNorm(temporal_hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            *[TemporalBlock(temporal_hidden_dim, dilation=dilation, dropout=dropout) for dilation in temporal_dilations]
        )
        self.head_norm = nn.LayerNorm(temporal_hidden_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleDict(
            {
                field: nn.Linear(temporal_hidden_dim, num_classes)
                for field, num_classes in self.head_num_classes.items()
            }
        )

    def forward(self, mesh_motion: torch.Tensor, force: torch.Tensor) -> dict[str, torch.Tensor]:
        if mesh_motion.ndim != 5:
            raise ValueError(f"mesh_motion must have shape [B, T, C, H, W], got {tuple(mesh_motion.shape)}")
        if force.ndim != 5:
            raise ValueError(f"force must have shape [B, T, C, H, W], got {tuple(force.shape)}")
        if mesh_motion.shape[:2] != force.shape[:2] or mesh_motion.shape[-2:] != force.shape[-2:]:
            raise ValueError(f"mesh_motion and force window/grid shapes differ: {mesh_motion.shape} vs {force.shape}")
        if mesh_motion.shape[2] != self.mesh_channels:
            raise ValueError(f"Expected {self.mesh_channels} mesh channels, got {mesh_motion.shape[2]}")
        if force.shape[2] != self.force_channels:
            raise ValueError(f"Expected {self.force_channels} force channels, got {force.shape[2]}")

        batch_size, window_size = mesh_motion.shape[:2]
        grid = torch.cat([mesh_motion, force], dim=2)
        grid = grid.reshape(batch_size * window_size, grid.shape[2], grid.shape[3], grid.shape[4])
        frame_features = self.frame_encoder(grid).reshape(batch_size, window_size, -1)
        temporal_values = self.temporal_in(frame_features).transpose(1, 2)
        temporal_values = self.temporal(temporal_values)
        final_feature = temporal_values[:, :, -1]
        final_feature = self.head_dropout(self.head_norm(final_feature))
        return {field: classifier(final_feature) for field, classifier in self.classifiers.items()}

    def config_dict(self) -> dict[str, object]:
        return {
            "mesh_channels": self.mesh_channels,
            "force_channels": self.force_channels,
            "frame_feature_dim": self.temporal_in[0].in_features,
            "temporal_hidden_dim": self.temporal_in[0].out_features,
            "temporal_dilations": tuple(block.net[0].dilation[0] for block in self.temporal),
            "dropout": float(self.head_dropout.p),
            "head_num_classes": dict(self.head_num_classes),
        }
