"""Fine-tune only the rotation classifier on tactile data from robot demonstrations."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable, Iterator, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from tactile_vla.captioner.model import TactileCaptioner
from tactile_vla.captioner.training import load_model_from_checkpoint, resolve_device
from tactile_vla.common.labels import LABEL_FIELDS, LABEL_MAPS, class_names
from tactile_vla.common.metrics import classification_report
from tactile_vla.common.seed import set_seed
from tactile_vla.data.tactile_captioner_dataset import TactileCaptionerDataset


PHYSICAL_TO_ROTATION = {
    "right": "clockwise",
    "front": "clockwise",
    "left": "counterclockwise",
    "back": "counterclockwise",
}
PHYSICAL_DIRECTIONS = ("right", "left", "front", "back")
ROTATION_CLASS_NAMES = class_names("rotation")
ROTATION_PARAMETER_NAMES = {
    "classifiers.rotation.weight",
    "classifiers.rotation.bias",
}
REQUIRED_HDF5_DATASETS = (
    "timestamp",
    "tactile/mesh_motion",
    "tactile/force_concat",
    "meta/episode_id",
    "meta/attempt_id",
    "meta/result",
    "meta/rotation_direction",
    "meta/shift_timestamp",
)


@dataclasses.dataclass(frozen=True)
class DemoSequence:
    hdf5_path: Path
    episode_id: int
    physical_direction: str
    rotation_label: int
    shift_timestamp: float
    num_frames: int


@dataclasses.dataclass(frozen=True)
class DemoWindow:
    sequence_index: int
    end_frame_index: int
    rotation_label: int


@dataclasses.dataclass
class RotationFineTuneConfig:
    base_checkpoint: Path
    demo_data_dir: Path = Path("/data1/tac_data/raw_data")
    pure_dataset_dir: Path | None = None
    output_dir: Path = Path("/data1/outputs/tactile_captioner")
    run_name: str | None = None
    episode_start: int = 41
    episode_end: int = 120
    forced_test_episodes: tuple[int, ...] = (75,)
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.03
    grad_clip: float = 1.0
    patience: int = 10
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    overwrite: bool = False
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None

    @property
    def run_dir(self) -> Path:
        if self.run_name:
            name = self.run_name
        else:
            base_name = self.base_checkpoint.parent.name if self.base_checkpoint.name == "best.pt" else self.base_checkpoint.stem
            name = f"{base_name}_rotation_head_demo"
        return self.output_dir / name


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_hdf5_scalar(root: h5py.File, key: str) -> Any:
    return _decode_scalar(root[key][()])


def _validate_demo_hdf5(
    hdf5_path: Path,
    *,
    expected_episode_id: int,
    expected_direction: str,
    expected_shift: float,
) -> tuple[int, float]:
    try:
        with h5py.File(hdf5_path, "r") as root:
            missing = [key for key in REQUIRED_HDF5_DATASETS if key not in root]
            if missing:
                raise ValueError(f"{hdf5_path}: missing datasets {missing}")
            episode_id = int(_read_hdf5_scalar(root, "meta/episode_id"))
            attempt_id = int(_read_hdf5_scalar(root, "meta/attempt_id"))
            result = str(_read_hdf5_scalar(root, "meta/result")).strip().lower()
            direction = str(_read_hdf5_scalar(root, "meta/rotation_direction")).strip().lower()
            shift_timestamp = float(_read_hdf5_scalar(root, "meta/shift_timestamp"))
            timestamps = np.asarray(root["timestamp"], dtype=np.float64)
            mesh_shape = tuple(root["tactile/mesh_motion"].shape)
            force_shape = tuple(root["tactile/force_concat"].shape)
    except OSError as exc:
        raise ValueError(f"Cannot open {hdf5_path}: {exc}") from exc

    if episode_id != expected_episode_id or attempt_id != 1:
        raise ValueError(
            f"{hdf5_path}: expected episode{expected_episode_id}/attempt1, "
            f"got episode{episode_id}/attempt{attempt_id}"
        )
    if result != "failure":
        raise ValueError(f"{hdf5_path}: ordinary rotation attempt1 must have result=failure, got {result!r}")
    if direction != expected_direction:
        raise ValueError(
            f"{hdf5_path}: rotation_direction differs between meta.json ({expected_direction}) "
            f"and data.hdf5 ({direction})"
        )
    if not math.isfinite(shift_timestamp) or not math.isclose(shift_timestamp, expected_shift, abs_tol=1e-6):
        raise ValueError(
            f"{hdf5_path}: shift_timestamp differs between meta.json ({expected_shift}) "
            f"and data.hdf5 ({shift_timestamp})"
        )
    if timestamps.ndim != 1 or timestamps.size == 0 or not np.isfinite(timestamps).all():
        raise ValueError(f"{hdf5_path}: timestamp must be a non-empty finite 1-D array")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{hdf5_path}: timestamp must be strictly increasing")
    if not float(timestamps[0]) <= shift_timestamp <= float(timestamps[-1]):
        raise ValueError(
            f"{hdf5_path}: shift_timestamp={shift_timestamp} lies outside "
            f"[{timestamps[0]}, {timestamps[-1]}]"
        )
    expected_mesh_shape = (len(timestamps), 35, 20, 12)
    expected_force_shape = (len(timestamps), 35, 20, 6)
    if mesh_shape != expected_mesh_shape:
        raise ValueError(f"{hdf5_path}: mesh_motion shape {mesh_shape} != {expected_mesh_shape}")
    if force_shape != expected_force_shape:
        raise ValueError(f"{hdf5_path}: force_concat shape {force_shape} != {expected_force_shape}")
    return int(timestamps.size), shift_timestamp


def scan_demo_sequences(
    dataset_dir: Path,
    *,
    episode_start: int = 41,
    episode_end: int = 120,
) -> list[DemoSequence]:
    if episode_start > episode_end:
        raise ValueError(f"episode_start={episode_start} exceeds episode_end={episode_end}")
    records: list[DemoSequence] = []
    for episode_id in range(episode_start, episode_end + 1):
        attempt_dir = dataset_dir / f"episode{episode_id}" / "attempt1"
        meta_path = attempt_dir / "meta.json"
        hdf5_path = attempt_dir / "data.hdf5"
        if not meta_path.is_file() or not hdf5_path.is_file():
            raise FileNotFoundError(
                f"episode{episode_id}/attempt1 requires both meta.json and data.hdf5 under {dataset_dir}"
            )
        meta = _read_json(meta_path)
        if int(meta.get("episode_id", -1)) != episode_id or int(meta.get("attempt_id", -1)) != 1:
            raise ValueError(f"{meta_path}: expected episode_id={episode_id}, attempt_id=1")
        if str(meta.get("capture_type", "")).strip().lower() != "demo":
            raise ValueError(f"{meta_path}: capture_type must be 'demo'")
        if str(meta.get("result", "")).strip().lower() != "failure":
            raise ValueError(f"{meta_path}: ordinary rotation attempt1 must have result='failure'")
        if meta.get("valid") is not True:
            raise ValueError(f"{meta_path}: valid must be true")
        physical_direction = str(meta.get("rotation_direction", "")).strip().lower()
        if physical_direction not in PHYSICAL_TO_ROTATION:
            raise ValueError(
                f"{meta_path}: rotation_direction must be one of {tuple(PHYSICAL_TO_ROTATION)}, "
                f"got {physical_direction!r}"
            )
        raw_shift = meta.get("shift_timestamp")
        if not isinstance(raw_shift, (int, float)) or not math.isfinite(float(raw_shift)):
            raise ValueError(f"{meta_path}: shift_timestamp must be finite")
        num_frames, shift_timestamp = _validate_demo_hdf5(
            hdf5_path,
            expected_episode_id=episode_id,
            expected_direction=physical_direction,
            expected_shift=float(raw_shift),
        )
        rotation_name = PHYSICAL_TO_ROTATION[physical_direction]
        records.append(
            DemoSequence(
                hdf5_path=hdf5_path,
                episode_id=episode_id,
                physical_direction=physical_direction,
                rotation_label=LABEL_MAPS["rotation"][rotation_name],
                shift_timestamp=shift_timestamp,
                num_frames=num_frames,
            )
        )
    return records


def stratified_episode_split(
    records: Sequence[DemoSequence],
    *,
    seed: int = 42,
    forced_test_episodes: Sequence[int] = (75,),
) -> dict[str, list[DemoSequence]]:
    forced = set(int(value) for value in forced_test_episodes)
    found_episode_ids = {record.episode_id for record in records}
    unknown = forced - found_episode_ids
    if unknown:
        raise ValueError(f"Forced test episodes are not present in the selected demo data: {sorted(unknown)}")

    result: dict[str, list[DemoSequence]] = {"train": [], "val": [], "test": []}
    for direction_index, direction in enumerate(PHYSICAL_DIRECTIONS):
        group = sorted(
            (record for record in records if record.physical_direction == direction),
            key=lambda record: record.episode_id,
        )
        if len(group) != 20:
            raise ValueError(
                f"Expected exactly 20 ordinary-rotation attempt1 sequences for {direction}, got {len(group)}"
            )
        forced_group = [record for record in group if record.episode_id in forced]
        if len(forced_group) > 2:
            raise ValueError(f"At most two forced test episodes are allowed per direction, got {forced_group}")
        remaining = [record for record in group if record.episode_id not in forced]
        random.Random(seed + direction_index * 1009).shuffle(remaining)
        test_needed = 2 - len(forced_group)
        test_records = forced_group + remaining[:test_needed]
        val_records = remaining[test_needed : test_needed + 2]
        train_records = remaining[test_needed + 2 :]
        if (len(train_records), len(val_records), len(test_records)) != (16, 2, 2):
            raise AssertionError("Internal 16/2/2 split construction failed")
        result["train"].extend(train_records)
        result["val"].extend(val_records)
        result["test"].extend(test_records)

    for split in result:
        result[split].sort(key=lambda record: record.episode_id)
    split_sets = [{record.episode_id for record in result[split]} for split in ("train", "val", "test")]
    if any(left & right for index, left in enumerate(split_sets) for right in split_sets[index + 1 :]):
        raise AssertionError("Episode leakage detected across demo splits")
    return result


def build_demo_windows(records: Sequence[DemoSequence], window_size: int) -> list[DemoWindow]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    windows: list[DemoWindow] = []
    for sequence_index, record in enumerate(records):
        with h5py.File(record.hdf5_path, "r") as root:
            timestamps = np.asarray(root["timestamp"], dtype=np.float64)
        if len(timestamps) != record.num_frames:
            raise ValueError(
                f"{record.hdf5_path}: timestamp length changed from {record.num_frames} to {len(timestamps)}"
            )
        for end_frame_index in range(window_size - 1, record.num_frames):
            label = 0 if float(timestamps[end_frame_index]) < record.shift_timestamp else record.rotation_label
            windows.append(
                DemoWindow(
                    sequence_index=sequence_index,
                    end_frame_index=end_frame_index,
                    rotation_label=label,
                )
            )
    return windows


def rotation_counts(windows: Iterable[DemoWindow]) -> dict[str, int]:
    values = [window.rotation_label for window in windows]
    return {
        name: int(sum(label == label_id for label in values))
        for label_id, name in enumerate(ROTATION_CLASS_NAMES)
    }


class DemoRotationDataset(Dataset):
    def __init__(self, records: Sequence[DemoSequence], windows: Sequence[DemoWindow], window_size: int) -> None:
        self.records = list(records)
        self.windows = list(windows)
        self.window_size = int(window_size)
        self._files: dict[Path, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def _open(self, path: Path) -> h5py.File:
        handle = self._files.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._files[path] = handle
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        record = self.records[window.sequence_index]
        start = window.end_frame_index - self.window_size + 1
        stop = window.end_frame_index + 1
        group = self._open(record.hdf5_path)
        mesh_motion = torch.from_numpy(group["tactile/mesh_motion"][start:stop]).float()
        force = torch.from_numpy(group["tactile/force_concat"][start:stop]).float()
        mesh_motion = mesh_motion.permute(0, 3, 1, 2).contiguous()
        force = force.permute(0, 3, 1, 2).contiguous()
        if mesh_motion.shape != (self.window_size, 12, 35, 20):
            raise IndexError(f"Incomplete mesh_motion window for {record.hdf5_path} at frame {window.end_frame_index}")
        if force.shape != (self.window_size, 6, 35, 20):
            raise IndexError(f"Incomplete force window for {record.hdf5_path} at frame {window.end_frame_index}")
        return {
            "mesh_motion": mesh_motion,
            "force": force,
            "rotation_label": torch.tensor(window.rotation_label, dtype=torch.long),
            "episode_id": record.episode_id,
            "end_frame_index": window.end_frame_index,
        }


class BalancedRotationSampler(Sampler[int]):
    """Use the same number of none/clockwise/counterclockwise samples each epoch."""

    def __init__(self, labels: Sequence[int], *, seed: int = 42) -> None:
        self.seed = int(seed)
        self.epoch = 0
        label_array = np.asarray(labels, dtype=np.int64)
        self.buckets = {
            label_id: np.flatnonzero(label_array == label_id)
            for label_id in range(len(ROTATION_CLASS_NAMES))
        }
        missing = [ROTATION_CLASS_NAMES[label_id] for label_id, values in self.buckets.items() if len(values) == 0]
        if missing:
            raise ValueError(f"Cannot balance training data because classes have no samples: {missing}")
        self.samples_per_class = min(len(values) for values in self.buckets.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_class * len(self.buckets)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        selected = [rng.choice(values, size=self.samples_per_class, replace=False) for values in self.buckets.values()]
        indices = np.concatenate(selected)
        rng.shuffle(indices)
        return iter(indices.tolist())


def freeze_for_rotation_head(model: TactileCaptioner) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifiers["rotation"].parameters():
        parameter.requires_grad_(True)
    trainable = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if set(trainable) != ROTATION_PARAMETER_NAMES:
        raise RuntimeError(f"Expected only rotation classifier parameters to train, got {trainable}")
    model.eval()
    model.classifiers["rotation"].train()
    return trainable


def assert_only_rotation_head_changed(
    base_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
) -> None:
    if set(base_state) != set(current_state):
        raise RuntimeError("Model state keys changed during rotation-head fine-tuning")
    changed_outside_rotation = [
        name
        for name in base_state
        if name not in ROTATION_PARAMETER_NAMES and not torch.equal(base_state[name].cpu(), current_state[name].cpu())
    ]
    if changed_outside_rotation:
        raise RuntimeError(f"Frozen model state changed during rotation fine-tuning: {changed_outside_rotation}")


def _normalization_tensors(checkpoint: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    normalization = checkpoint.get("normalization", {})
    result: dict[str, torch.Tensor] = {}
    for modality, channels in (("mesh_motion", 12), ("force", 6)):
        for statistic in ("mean", "std"):
            key = f"{modality}_{statistic}"
            values = torch.as_tensor(normalization.get(key), dtype=torch.float32, device=device)
            if values.numel() != channels or not torch.isfinite(values).all():
                raise ValueError(f"Checkpoint normalization {key} must contain {channels} finite values")
            if statistic == "std" and torch.any(values <= 0):
                raise ValueError(f"Checkpoint normalization {key} must be positive")
            result[key] = values
    return result


def _normalize_batch(
    mesh_motion: torch.Tensor,
    force: torch.Tensor,
    normalization: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    mesh_motion = (mesh_motion - normalization["mesh_motion_mean"].view(1, 1, -1, 1, 1)) / normalization[
        "mesh_motion_std"
    ].view(1, 1, -1, 1, 1)
    force = (force - normalization["force_mean"].view(1, 1, -1, 1, 1)) / normalization["force_std"].view(
        1, 1, -1, 1, 1
    )
    return mesh_motion, force


def _batch_rotation_labels(batch: dict[str, Any]) -> torch.Tensor:
    if "rotation_label" in batch:
        return batch["rotation_label"]
    return batch["labels"]["rotation"]


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
    )


def run_rotation_epoch(
    model: TactileCaptioner,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: dict[str, torch.Tensor],
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    max_batches: int | None = None,
    desc: str,
) -> dict[str, Any]:
    training = optimizer is not None
    model.eval()
    model.classifiers["rotation"].train(training)
    labels_all: list[torch.Tensor] = []
    predictions_all: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0
    last_grad_norm = 0.0

    for batch_index, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        mesh_motion = batch["mesh_motion"].to(device, non_blocking=True)
        force = batch["force"].to(device, non_blocking=True)
        labels = _batch_rotation_labels(batch).to(device, non_blocking=True)
        mesh_motion, force = _normalize_batch(mesh_motion, force, normalization)
        with torch.set_grad_enabled(training):
            logits = model(mesh_motion, force)["rotation"]
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.classifiers["rotation"].parameters(), grad_clip)
                last_grad_norm = float(grad_norm.detach().cpu())
                optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        labels_all.append(labels.detach().cpu())
        predictions_all.append(torch.argmax(logits.detach(), dim=-1).cpu())

    if not labels_all:
        raise RuntimeError(f"No batches were processed for {desc}")
    report = classification_report(
        torch.cat(labels_all),
        torch.cat(predictions_all),
        num_classes=len(ROTATION_CLASS_NAMES),
        class_names=ROTATION_CLASS_NAMES,
    )
    report["loss"] = total_loss / total_samples
    if training:
        report["grad_norm"] = last_grad_norm
        report["lr"] = float(optimizer.param_groups[0]["lr"])
    return report


def evaluate_all_heads(
    model: TactileCaptioner,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: dict[str, torch.Tensor],
    max_batches: int | None = None,
    desc: str,
) -> dict[str, Any]:
    model.eval()
    labels_all: dict[str, list[torch.Tensor]] = {field: [] for field in LABEL_FIELDS}
    predictions_all: dict[str, list[torch.Tensor]] = {field: [] for field in LABEL_FIELDS}
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break
            mesh_motion = batch["mesh_motion"].to(device, non_blocking=True)
            force = batch["force"].to(device, non_blocking=True)
            mesh_motion, force = _normalize_batch(mesh_motion, force, normalization)
            logits = model(mesh_motion, force)
            for field in LABEL_FIELDS:
                labels_all[field].append(batch["labels"][field].cpu())
                predictions_all[field].append(torch.argmax(logits[field], dim=-1).cpu())
    if not labels_all[LABEL_FIELDS[0]]:
        raise RuntimeError(f"No batches were processed for {desc}")
    heads = {
        field: classification_report(
            torch.cat(labels_all[field]),
            torch.cat(predictions_all[field]),
            num_classes=len(LABEL_MAPS[field]),
            class_names=class_names(field),
        )
        for field in LABEL_FIELDS
    }
    return {
        "mean_macro_f1": float(np.mean([heads[field]["macro_f1"] for field in LABEL_FIELDS])),
        "heads": heads,
    }


def _state_dict_cpu(model: TactileCaptioner) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _save_checkpoint(
    path: Path,
    *,
    source_checkpoint: dict[str, Any],
    source_checkpoint_path: Path,
    model: TactileCaptioner,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: RotationFineTuneConfig,
    metrics: dict[str, Any],
    trainable_parameters: Sequence[str],
) -> None:
    payload = copy.copy(source_checkpoint)
    payload.update(
        {
            "epoch": epoch,
            "model_state_dict": _state_dict_cpu(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": _jsonable(metrics),
            "fine_tune": {
                "type": "rotation_head_only",
                "source_checkpoint": str(source_checkpoint_path),
                "config": _jsonable(config),
                "trainable_parameters": list(trainable_parameters),
                "physical_to_rotation": dict(PHYSICAL_TO_ROTATION),
                "label_rule": "window_end_timestamp < shift_timestamp => none; otherwise mapped rotation",
            },
            "saved_at": time.time(),
        }
    )
    torch.save(payload, path)


def _manifest_for_splits(
    splits: dict[str, list[DemoSequence]],
    windows: dict[str, list[DemoWindow]],
    *,
    window_size: int,
    sampler: BalancedRotationSampler,
    config: RotationFineTuneConfig,
) -> dict[str, Any]:
    return {
        "demo_data_dir": str(config.demo_data_dir),
        "episode_range": [config.episode_start, config.episode_end],
        "excluded_episode40": True,
        "attempt_id": 1,
        "window_size": window_size,
        "seed": config.seed,
        "forced_test_episodes": list(config.forced_test_episodes),
        "physical_to_rotation": dict(PHYSICAL_TO_ROTATION),
        "label_rule": "window_end_timestamp < shift_timestamp => none; otherwise mapped rotation",
        "splits": {
            split: {
                "episode_ids": [record.episode_id for record in records],
                "physical_direction_counts": {
                    direction: sum(record.physical_direction == direction for record in records)
                    for direction in PHYSICAL_DIRECTIONS
                },
                "window_counts": rotation_counts(windows[split]),
            }
            for split, records in splits.items()
        },
        "balanced_train_samples_per_class_per_epoch": sampler.samples_per_class,
        "balanced_train_windows_per_epoch": len(sampler),
    }


def _prepare_run_dir(config: RotationFineTuneConfig) -> Path:
    run_dir = config.run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output directory is not empty: {run_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _metrics_delta(base: dict[str, Any], finetuned: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(finetuned["accuracy"] - base["accuracy"]),
        "macro_f1": float(finetuned["macro_f1"] - base["macro_f1"]),
    }


def fine_tune_rotation_head(config: RotationFineTuneConfig) -> dict[str, Any]:
    if not config.base_checkpoint.is_file():
        raise FileNotFoundError(f"Base tactile Captioner checkpoint not found: {config.base_checkpoint}")
    if config.episode_start != 41:
        raise ValueError("Ordinary rotation recovery data starts at episode41; episode40 must remain excluded")
    if config.episode_end != 120:
        raise ValueError("This first rotation-head fine-tune version requires the complete episode41-120 dataset")
    if config.epochs <= 0 or config.batch_size <= 0 or config.patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")

    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = _prepare_run_dir(config)
    metrics_path = run_dir / "metrics.jsonl"
    _write_json(run_dir / "config.json", _jsonable(config))

    model, source_checkpoint = load_model_from_checkpoint(config.base_checkpoint, device=device)
    window_size = int(source_checkpoint["dataset_meta"]["window_size"])
    if window_size not in (15, 30):
        raise ValueError(f"Rotation-head fine-tuning supports W15 or W30 checkpoints, got W{window_size}")
    if config.pure_dataset_dir is None:
        config.pure_dataset_dir = Path(f"/data1/tac_data/tac_cap_data/captioner_dataset_{window_size}")
    if not config.pure_dataset_dir.is_dir():
        raise FileNotFoundError(f"Pure tactile dataset not found: {config.pure_dataset_dir}")

    trainable_parameters = freeze_for_rotation_head(model)
    base_state = _state_dict_cpu(model)
    normalization = _normalization_tensors(source_checkpoint, device)

    records = scan_demo_sequences(
        config.demo_data_dir,
        episode_start=config.episode_start,
        episode_end=config.episode_end,
    )
    splits = stratified_episode_split(
        records,
        seed=config.seed,
        forced_test_episodes=config.forced_test_episodes,
    )
    split_windows = {
        split: build_demo_windows(split_records, window_size)
        for split, split_records in splits.items()
    }
    demo_datasets = {
        split: DemoRotationDataset(splits[split], split_windows[split], window_size)
        for split in ("train", "val", "test")
    }
    sampler = BalancedRotationSampler(
        [window.rotation_label for window in split_windows["train"]],
        seed=config.seed,
    )
    _write_json(
        run_dir / "split_manifest.json",
        _manifest_for_splits(splits, split_windows, window_size=window_size, sampler=sampler, config=config),
    )

    train_loader = _make_loader(
        demo_datasets["train"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        sampler=sampler,
    )
    val_loader = _make_loader(
        demo_datasets["val"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    test_loader = _make_loader(
        demo_datasets["test"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    pure_test_dataset = TactileCaptionerDataset(
        config.pure_dataset_dir,
        split="test",
        balanced=False,
        normalize=False,
    )
    if pure_test_dataset.window_size != window_size:
        raise ValueError(
            f"Pure tactile test dataset uses W{pure_test_dataset.window_size}, but checkpoint uses W{window_size}"
        )
    pure_test_loader = _make_loader(
        pure_test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    trainable_tensors = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_tensors, lr=config.lr, weight_decay=config.weight_decay)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    for epoch in range(1, config.epochs + 1):
        sampler.set_epoch(epoch)
        train_metrics = run_rotation_epoch(
            model,
            train_loader,
            device=device,
            normalization=normalization,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=config.grad_clip,
            max_batches=config.max_train_batches,
            desc=f"rotation train {epoch}/{config.epochs}",
        )
        val_metrics = run_rotation_epoch(
            model,
            val_loader,
            device=device,
            normalization=normalization,
            criterion=eval_criterion,
            max_batches=config.max_val_batches,
            desc=f"demo val {epoch}/{config.epochs}",
        )
        assert_only_rotation_head_changed(base_state, model.state_dict())
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "demo_val": val_metrics,
            "balanced_train_samples_per_class": sampler.samples_per_class,
            "trainable_parameters": list(trainable_parameters),
            "device": str(device),
        }
        _append_jsonl(metrics_path, record)
        _save_checkpoint(
            last_path,
            source_checkpoint=source_checkpoint,
            source_checkpoint_path=config.base_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            metrics=record,
            trainable_parameters=trainable_parameters,
        )
        if float(val_metrics["macro_f1"]) > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(
                best_path,
                source_checkpoint=source_checkpoint,
                source_checkpoint_path=config.base_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=record,
                trainable_parameters=trainable_parameters,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_model, base_checkpoint = load_model_from_checkpoint(config.base_checkpoint, device=device)
    base_normalization = _normalization_tensors(base_checkpoint, device)
    base_demo_test = run_rotation_epoch(
        base_model,
        test_loader,
        device=device,
        normalization=base_normalization,
        criterion=eval_criterion,
        max_batches=config.max_test_batches,
        desc="base demo test",
    )
    base_pure_test = evaluate_all_heads(
        base_model,
        pure_test_loader,
        device=device,
        normalization=base_normalization,
        max_batches=config.max_test_batches,
        desc="base pure tactile test",
    )
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    best_model, best_checkpoint = load_model_from_checkpoint(best_path, device=device)
    best_normalization = _normalization_tensors(best_checkpoint, device)
    assert_only_rotation_head_changed(base_state, best_model.state_dict())
    finetuned_demo_test = run_rotation_epoch(
        best_model,
        test_loader,
        device=device,
        normalization=best_normalization,
        criterion=eval_criterion,
        max_batches=config.max_test_batches,
        desc="finetuned demo test",
    )
    finetuned_pure_test = evaluate_all_heads(
        best_model,
        pure_test_loader,
        device=device,
        normalization=best_normalization,
        max_batches=config.max_test_batches,
        desc="finetuned pure tactile test",
    )

    summary = {
        "best_epoch": best_epoch,
        "best_demo_val_rotation_macro_f1": best_val_macro_f1,
        "window_size": window_size,
        "base_checkpoint": str(config.base_checkpoint),
        "best_checkpoint": str(best_path),
        "demo_test": {
            "base": base_demo_test,
            "finetuned": finetuned_demo_test,
            "delta": _metrics_delta(base_demo_test, finetuned_demo_test),
        },
        "pure_tactile_test": {
            "dataset_dir": str(config.pure_dataset_dir),
            "base": base_pure_test,
            "finetuned": finetuned_pure_test,
            "rotation_delta": _metrics_delta(
                base_pure_test["heads"]["rotation"],
                finetuned_pure_test["heads"]["rotation"],
            ),
        },
        "frozen_state_verified": True,
        "trainable_parameters": list(trainable_parameters),
    }
    _write_json(run_dir / "summary.json", summary)
    return summary
