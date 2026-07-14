"""Small PyTorch helpers shared by VLA training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import safetensors.torch
import torch


def tree_to_torch(tree: Any, device: torch.device) -> Any:
    def convert(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            return value.float() if value.is_floating_point() else value
        if isinstance(value, np.ndarray):
            tensor = torch.as_tensor(value, device=device)
            return tensor.float() if tensor.is_floating_point() else tensor
        return value

    return jax.tree.map(convert, tree)


def load_safetensors_model(model: torch.nn.Module, checkpoint: str | Path) -> None:
    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    safetensors.torch.load_model(model, model_path)


def set_trainable_lora_only(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name.lower()
        if param.requires_grad:
            trainable += param.numel()
    return trainable


def set_trainable_prefix(model: torch.nn.Module, prefixes: tuple[str, ...]) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)
        if param.requires_grad:
            trainable += param.numel()
    return trainable


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def head_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }
