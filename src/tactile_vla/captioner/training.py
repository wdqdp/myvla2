"""Training and evaluation utilities for the tactile captioner."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tactile_vla.captioner.model import TactileCaptioner
from tactile_vla.common.labels import ROTATION_LABELS
from tactile_vla.common.metrics import classification_report
from tactile_vla.common.seed import set_seed
from tactile_vla.data.tactile_captioner_dataset import TactileCaptionerDataset
from tactile_vla.data.tactile_captioner_dataset import label_counts


@dataclasses.dataclass
class CaptionerTrainConfig:
    dataset_dir: Path = Path("data/tactile_captioner_data")
    output_dir: Path = Path("outputs/tactile_captioner")
    run_name: str = "tcn_v1_balanced"
    batch_size: int = 128
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    balanced_train: bool = True
    normalize: bool = True
    label_smoothing: float = 0.03
    grad_clip: float = 1.0
    patience: int = 10
    frame_feature_dim: int = 128
    temporal_hidden_dim: int = 192
    dropout: float = 0.1
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(val) for key, val in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false")
    return resolved


def make_dataloader(
    dataset: TactileCaptionerDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def compute_class_weights(counts: dict[int, int], *, num_classes: int) -> torch.Tensor:
    max_count = max(counts.values())
    weights = []
    for label_id in range(num_classes):
        count = max(1, int(counts.get(label_id, 0)))
        weights.append(math.sqrt(max_count / count))
    values = torch.tensor(weights, dtype=torch.float32)
    return values / values.mean().clamp_min(1e-6)


def make_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(1.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mesh_motion = batch["mesh_motion"].to(device, non_blocking=True)
    force = batch["force"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return mesh_motion, force, labels


def _run_epoch(
    model: TactileCaptioner,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    grad_clip: float = 1.0,
    max_batches: int | None = None,
    desc: str,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    labels_all = []
    preds_all = []
    total_loss = 0.0
    total_samples = 0
    last_grad_norm = 0.0

    iterator = tqdm(loader, desc=desc, leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        mesh_motion, force, labels = _move_batch(batch, device)

        with torch.set_grad_enabled(training):
            logits = model(mesh_motion, force)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                last_grad_norm = float(grad_norm.detach().cpu())
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        labels_all.append(labels.detach().cpu())
        preds_all.append(torch.argmax(logits.detach(), dim=-1).cpu())
        iterator.set_postfix(loss=total_loss / max(1, total_samples))

    if not labels_all:
        raise RuntimeError(f"No batches were processed for {desc}")
    labels_np = torch.cat(labels_all).numpy()
    preds_np = torch.cat(preds_all).numpy()
    report = classification_report(
        labels_np,
        preds_np,
        num_classes=len(ROTATION_LABELS),
        class_names=ROTATION_LABELS,
    )
    report["loss"] = total_loss / max(1, total_samples)
    if training:
        report["grad_norm"] = last_grad_norm
        report["lr"] = optimizer.param_groups[0]["lr"]
    return report


def save_checkpoint(
    path: Path,
    *,
    model: TactileCaptioner,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: CaptionerTrainConfig,
    metrics: dict[str, Any],
    train_dataset: TactileCaptionerDataset,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _jsonable(config),
        "model_config": model.config_dict(),
        "metrics": _jsonable(metrics),
        "label_names": ROTATION_LABELS,
        "normalization": train_dataset.normalization_stats(),
        "dataset_meta": {
            "window_size": train_dataset.window_size,
            "label_map": train_dataset.meta.get("label_map"),
            "mesh_motion_shape": train_dataset.meta.get("mesh_motion_shape"),
            "force_shape": train_dataset.meta.get("force_shape"),
        },
        "saved_at": time.time(),
    }
    torch.save(payload, path)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[TactileCaptioner, dict[str, Any]]:
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model_config = dict(checkpoint["model_config"])
    if "temporal_dilations" in model_config:
        model_config["temporal_dilations"] = tuple(model_config["temporal_dilations"])
    model = TactileCaptioner(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint


def train(config: CaptionerTrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    _write_json(run_dir / "config.json", _jsonable(config))

    train_dataset = TactileCaptionerDataset(
        config.dataset_dir,
        split="train",
        balanced=config.balanced_train,
        normalize=config.normalize,
    )
    val_dataset = TactileCaptionerDataset(config.dataset_dir, split="val", balanced=False, normalize=config.normalize)
    test_dataset = TactileCaptionerDataset(config.dataset_dir, split="test", balanced=False, normalize=config.normalize)

    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        device=device,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device,
    )

    model = TactileCaptioner(
        frame_feature_dim=config.frame_feature_dim,
        temporal_hidden_dim=config.temporal_hidden_dim,
        dropout=config.dropout,
        num_classes=len(ROTATION_LABELS),
    ).to(device)

    counts = label_counts(train_dataset)
    class_weights = compute_class_weights(counts, num_classes=len(ROTATION_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_steps_per_epoch = config.max_train_batches or len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps=max(1, train_steps_per_epoch * config.epochs))

    best_macro_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=config.grad_clip,
            max_batches=config.max_train_batches,
            desc=f"train {epoch}/{config.epochs}",
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            max_batches=config.max_val_batches,
            desc=f"val {epoch}/{config.epochs}",
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "class_weights": class_weights.detach().cpu().tolist(),
            "device": str(device),
        }
        _append_jsonl(metrics_path, record)
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            metrics=record,
            train_dataset=train_dataset,
        )

        val_macro_f1 = float(val_metrics["macro_f1"])
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=record,
                train_dataset=train_dataset,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    best_model, _ = load_model_from_checkpoint(best_path, device=device)
    test_metrics = _run_epoch(
        best_model,
        test_loader,
        device=device,
        criterion=criterion,
        max_batches=config.max_test_batches,
        desc="test",
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "test": test_metrics,
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
    }
    _write_json(run_dir / "test_metrics.json", summary)
    return summary


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    split: str = "test",
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "auto",
    normalize: bool = True,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    resolved_device = next(model.parameters()).device
    if dataset_dir is None:
        dataset_dir = checkpoint["config"]["dataset_dir"]
    dataset = TactileCaptionerDataset(dataset_dir, split=split, balanced=False, normalize=normalize)
    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=resolved_device,
    )
    criterion = nn.CrossEntropyLoss()
    return _run_epoch(
        model,
        loader,
        device=resolved_device,
        criterion=criterion,
        max_batches=max_batches,
        desc=f"eval {split}",
    )
