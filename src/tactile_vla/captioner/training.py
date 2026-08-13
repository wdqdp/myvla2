"""Training and evaluation utilities for the V3 multi-head tactile captioner."""

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
from tactile_vla.common.labels import LABEL_FIELDS
from tactile_vla.common.labels import LABEL_MAPS
from tactile_vla.common.labels import LABEL_SCHEMA_VERSION
from tactile_vla.common.labels import class_names
from tactile_vla.common.metrics import classification_report
from tactile_vla.common.seed import set_seed
from tactile_vla.data.tactile_captioner_dataset import TactileCaptionerDataset
from tactile_vla.data.tactile_captioner_dataset import label_counts


CLASS_WEIGHTING_MODES = ("sqrt_inverse", "none")


@dataclasses.dataclass
class CaptionerTrainConfig:
    dataset_dir: Path = Path("data/tactile_captioner_data")
    output_dir: Path = Path("outputs/tactile_captioner")
    run_name: str = "tcn_v3_multifield"
    batch_size: int = 128
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    balanced_train: bool = True
    normalize: bool = True
    class_weighting: str = "sqrt_inverse"
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


def compute_class_weights(
    counts: dict[int, int],
    *,
    num_classes: int,
    mode: str = "sqrt_inverse",
) -> torch.Tensor:
    if mode not in CLASS_WEIGHTING_MODES:
        raise ValueError(f"Unknown class weighting mode {mode!r}; expected one of {CLASS_WEIGHTING_MODES}")
    ordered_counts = [int(counts.get(label_id, 0)) for label_id in range(num_classes)]
    if any(count <= 0 for count in ordered_counts):
        raise ValueError(f"Every training class needs samples before weighting, got counts={ordered_counts}")
    if mode == "none":
        return torch.ones((num_classes,), dtype=torch.float32)
    max_count = max(ordered_counts)
    values = torch.tensor([math.sqrt(max_count / count) for count in ordered_counts], dtype=torch.float32)
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
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    mesh_motion = batch["mesh_motion"].to(device, non_blocking=True)
    force = batch["force"].to(device, non_blocking=True)
    labels = {
        field: batch["labels"][field].to(device, non_blocking=True)
        for field in LABEL_FIELDS
    }
    return mesh_motion, force, labels


def _make_criteria(
    class_weights: dict[str, torch.Tensor],
    *,
    label_smoothing: float,
) -> dict[str, nn.CrossEntropyLoss]:
    return {
        field: nn.CrossEntropyLoss(weight=class_weights[field], label_smoothing=label_smoothing)
        for field in LABEL_FIELDS
    }


def _run_epoch(
    model: TactileCaptioner,
    loader: DataLoader,
    *,
    device: torch.device,
    criteria: dict[str, nn.CrossEntropyLoss],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    grad_clip: float = 1.0,
    max_batches: int | None = None,
    desc: str,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    labels_all: dict[str, list[torch.Tensor]] = {field: [] for field in LABEL_FIELDS}
    preds_all: dict[str, list[torch.Tensor]] = {field: [] for field in LABEL_FIELDS}
    total_head_losses = {field: 0.0 for field in LABEL_FIELDS}
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
            head_losses = {
                field: criteria[field](logits[field], labels[field])
                for field in LABEL_FIELDS
            }
            loss = torch.stack(tuple(head_losses.values())).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                last_grad_norm = float(grad_norm.detach().cpu())
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = int(labels[LABEL_FIELDS[0]].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        for field in LABEL_FIELDS:
            total_head_losses[field] += float(head_losses[field].detach().cpu()) * batch_size
            labels_all[field].append(labels[field].detach().cpu())
            preds_all[field].append(torch.argmax(logits[field].detach(), dim=-1).cpu())
        iterator.set_postfix(loss=total_loss / max(1, total_samples))

    if not labels_all[LABEL_FIELDS[0]]:
        raise RuntimeError(f"No batches were processed for {desc}")

    heads: dict[str, dict[str, Any]] = {}
    for field in LABEL_FIELDS:
        report = classification_report(
            torch.cat(labels_all[field]).numpy(),
            torch.cat(preds_all[field]).numpy(),
            num_classes=len(LABEL_MAPS[field]),
            class_names=class_names(field),
        )
        report["loss"] = total_head_losses[field] / max(1, total_samples)
        heads[field] = report
    result: dict[str, Any] = {
        "loss": total_loss / max(1, total_samples),
        "mean_macro_f1": float(np.mean([heads[field]["macro_f1"] for field in LABEL_FIELDS])),
        "heads": heads,
    }
    if training:
        result["grad_norm"] = last_grad_norm
        result["lr"] = optimizer.param_groups[0]["lr"]
    return result


def save_checkpoint(
    path: Path,
    *,
    model: TactileCaptioner,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: CaptionerTrainConfig,
    metrics: dict[str, Any],
    train_dataset: TactileCaptionerDataset,
    class_counts: dict[str, dict[int, int]],
    class_weights: dict[str, torch.Tensor],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _jsonable(config),
        "model_config": model.config_dict(),
        "metrics": _jsonable(metrics),
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_maps": LABEL_MAPS,
        "class_counts": class_counts,
        "class_weights": {field: class_weights[field].detach().cpu().tolist() for field in LABEL_FIELDS},
        "normalization": train_dataset.normalization_stats(),
        "dataset_meta": {
            "dataset_format": train_dataset.meta["dataset_format"],
            "window_size": train_dataset.window_size,
            "label_schema_version": train_dataset.meta["label_schema_version"],
            "label_maps": train_dataset.meta["label_maps"],
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
    if checkpoint.get("label_schema_version") != LABEL_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint uses label schema {checkpoint.get('label_schema_version')!r}; "
            f"V3 requires {LABEL_SCHEMA_VERSION!r}"
        )
    model_config = dict(checkpoint["model_config"])
    if "temporal_dilations" in model_config:
        model_config["temporal_dilations"] = tuple(model_config["temporal_dilations"])
    model = TactileCaptioner(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint


def train(config: CaptionerTrainConfig) -> dict[str, Any]:
    if config.class_weighting not in CLASS_WEIGHTING_MODES:
        raise ValueError(f"class_weighting must be one of {CLASS_WEIGHTING_MODES}")
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
    if not (train_dataset.head_num_classes == val_dataset.head_num_classes == test_dataset.head_num_classes):
        raise ValueError("train/val/test tactile label maps do not match")

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
        head_num_classes=train_dataset.head_num_classes,
    ).to(device)

    counts = label_counts(train_dataset)
    class_weights = {
        field: compute_class_weights(
            counts[field],
            num_classes=train_dataset.head_num_classes[field],
            mode=config.class_weighting,
        ).to(device)
        for field in LABEL_FIELDS
    }
    train_criteria = _make_criteria(class_weights, label_smoothing=config.label_smoothing)
    eval_criteria = _make_criteria(
        {
            field: torch.ones((train_dataset.head_num_classes[field],), dtype=torch.float32, device=device)
            for field in LABEL_FIELDS
        },
        label_smoothing=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_steps_per_epoch = config.max_train_batches or len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps=max(1, train_steps_per_epoch * config.epochs))

    best_mean_macro_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            criteria=train_criteria,
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
            criteria=eval_criteria,
            max_batches=config.max_val_batches,
            desc=f"val {epoch}/{config.epochs}",
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "class_counts": counts,
            "class_weights": {
                field: class_weights[field].detach().cpu().tolist()
                for field in LABEL_FIELDS
            },
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
            class_counts=counts,
            class_weights=class_weights,
        )

        val_mean_macro_f1 = float(val_metrics["mean_macro_f1"])
        if val_mean_macro_f1 > best_mean_macro_f1:
            best_mean_macro_f1 = val_mean_macro_f1
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
                class_counts=counts,
                class_weights=class_weights,
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
        criteria=eval_criteria,
        max_batches=config.max_test_batches,
        desc="test",
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_mean_macro_f1": best_mean_macro_f1,
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
    checkpoint_window = int(checkpoint["dataset_meta"]["window_size"])
    if dataset.window_size != checkpoint_window:
        raise ValueError(
            f"Dataset window_size={dataset.window_size} does not match checkpoint window_size={checkpoint_window}"
        )
    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=resolved_device,
    )
    criteria = _make_criteria(
        {
            field: torch.ones((len(LABEL_MAPS[field]),), dtype=torch.float32, device=resolved_device)
            for field in LABEL_FIELDS
        },
        label_smoothing=0.0,
    )
    return _run_epoch(
        model,
        loader,
        device=resolved_device,
        criteria=criteria,
        max_batches=max_batches,
        desc=f"eval {split}",
    )
