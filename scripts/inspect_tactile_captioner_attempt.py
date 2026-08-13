#!/usr/bin/env python3
"""Inspect frame-aligned tactile-captioner predictions for one synchronized attempt.
测试纯触觉训练的captioner在演示数据上的准确性"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.captioner.predictor import TactileCaptionerPredictor  # noqa: E402
from tactile_vla.common.labels import DEFAULT_TACTILE_CAPTION  # noqa: E402
from tactile_vla.common.labels import LABEL_FIELDS  # noqa: E402
from tactile_vla.common.labels import LABEL_MAPS  # noqa: E402
from tactile_vla.common.labels import NEUTRAL_LABELS  # noqa: E402


DEFAULT_DATASET_DIR = Path("/data1/tac_data/raw_data")
DEFAULT_CHECKPOINT = Path("/data1/outputs/tactile_captioner/tcn_v3_w30_multifield/best.pt")
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/tactile_caption_inspection")


@dataclass(frozen=True)
class AttemptData:
    hdf5_path: Path
    episode_id: int
    attempt_id: int
    result: str
    timestamps: np.ndarray
    mesh_motion: np.ndarray
    force: np.ndarray
    shift_timestamp: float | None


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_scalar(root: h5py.File, name: str, default: Any = None) -> Any:
    if name not in root:
        return default
    return _decode_scalar(root[name][()])


def _validate_timestamps(timestamps: np.ndarray, hdf5_path: Path) -> None:
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError(f"{hdf5_path}: timestamp must be a non-empty 1-D array, got {timestamps.shape}")
    if not np.isfinite(timestamps).all():
        raise ValueError(f"{hdf5_path}: timestamp contains NaN or Inf")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{hdf5_path}: timestamp must be strictly increasing")


def _validate_tactile_array(
    values: np.ndarray,
    *,
    name: str,
    frame_count: int,
    channels: int,
    hdf5_path: Path,
) -> None:
    expected = (frame_count, 35, 20, channels)
    if values.shape != expected:
        raise ValueError(f"{hdf5_path}: {name} must have shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{hdf5_path}: {name} contains NaN or Inf")


def load_attempt_data(dataset_dir: Path, episode_index: int, attempt_index: int) -> AttemptData:
    hdf5_path = dataset_dir / f"episode{episode_index}" / f"attempt{attempt_index}" / "data.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(
            f"Missing {hdf5_path}. Run data_sync_attempts.py and data_to_hdf5_attempts.py first."
        )

    try:
        with h5py.File(hdf5_path, "r") as root:
            required = ("timestamp", "tactile/mesh_motion", "tactile/force_concat")
            missing = [name for name in required if name not in root]
            if missing:
                raise ValueError(f"{hdf5_path}: missing required datasets: {missing}")
            timestamps = np.asarray(root["timestamp"], dtype=np.float64)
            mesh_motion = np.asarray(root["tactile/mesh_motion"], dtype=np.float32)
            force = np.asarray(root["tactile/force_concat"], dtype=np.float32)
            stored_episode_id = int(_read_scalar(root, "meta/episode_id", episode_index))
            stored_attempt_id = int(_read_scalar(root, "meta/attempt_id", attempt_index))
            result = str(_read_scalar(root, "meta/result", ""))
            raw_shift = _read_scalar(root, "meta/shift_timestamp", math.nan)
    except OSError as exc:
        raise ValueError(f"Cannot open completed HDF5 file {hdf5_path}: {exc}") from exc

    if stored_episode_id != episode_index or stored_attempt_id != attempt_index:
        raise ValueError(
            f"{hdf5_path}: requested episode{episode_index}/attempt{attempt_index}, "
            f"but HDF5 metadata says episode{stored_episode_id}/attempt{stored_attempt_id}"
        )
    _validate_timestamps(timestamps, hdf5_path)
    _validate_tactile_array(
        mesh_motion,
        name="tactile/mesh_motion",
        frame_count=len(timestamps),
        channels=12,
        hdf5_path=hdf5_path,
    )
    _validate_tactile_array(
        force,
        name="tactile/force_concat",
        frame_count=len(timestamps),
        channels=6,
        hdf5_path=hdf5_path,
    )

    shift_timestamp = float(raw_shift) if isinstance(raw_shift, (int, float)) else math.nan
    if not math.isfinite(shift_timestamp):
        shift_timestamp = None
    elif not timestamps[0] <= shift_timestamp <= timestamps[-1]:
        raise ValueError(
            f"{hdf5_path}: shift_timestamp={shift_timestamp} lies outside "
            f"[{timestamps[0]}, {timestamps[-1]}]"
        )

    return AttemptData(
        hdf5_path=hdf5_path,
        episode_id=stored_episode_id,
        attempt_id=stored_attempt_id,
        result=result,
        timestamps=timestamps,
        mesh_motion=mesh_motion,
        force=force,
        shift_timestamp=shift_timestamp,
    )


def _neutral_label_ids() -> dict[str, int]:
    return {field: int(LABEL_MAPS[field][NEUTRAL_LABELS[field]]) for field in LABEL_FIELDS}


def _shift_relation(timestamp: float, shift_timestamp: float | None) -> str:
    if shift_timestamp is None:
        return "not_applicable"
    return "before" if timestamp < shift_timestamp else "after"


def predict_timeline(data: AttemptData, predictor: Any) -> list[dict[str, Any]]:
    window_size = int(predictor.window_size)
    if window_size <= 0:
        raise ValueError(f"Captioner window_size must be positive, got {window_size}")

    timestamps = data.timestamps
    first_timestamp = float(timestamps[0])
    neutral_ids = _neutral_label_ids()
    timeline: list[dict[str, Any]] = []
    for frame_index, raw_timestamp in enumerate(timestamps):
        timestamp = float(raw_timestamp)
        warmup = frame_index < window_size - 1
        if warmup:
            caption = DEFAULT_TACTILE_CAPTION
            label_ids = dict(neutral_ids)
            label_names = dict(NEUTRAL_LABELS)
            probabilities = {field: None for field in LABEL_FIELDS}
            confidence = {field: None for field in LABEL_FIELDS}
        else:
            start = frame_index - window_size + 1
            prediction = predictor.predict(
                data.mesh_motion[start : frame_index + 1],
                data.force[start : frame_index + 1],
            )
            caption = str(prediction.caption)
            label_ids = {field: int(prediction.label_ids[field]) for field in LABEL_FIELDS}
            label_names = {field: str(prediction.label_names[field]) for field in LABEL_FIELDS}
            probabilities = {
                field: [float(value) for value in prediction.probabilities[field]] for field in LABEL_FIELDS
            }
            confidence = {
                field: float(probabilities[field][label_ids[field]]) for field in LABEL_FIELDS
            }

        timeline.append(
            {
                "frame_index": frame_index,
                "frame_number": frame_index + 1,
                "timestamp": timestamp,
                "relative_time_sec": timestamp - first_timestamp,
                "shift_relation": _shift_relation(timestamp, data.shift_timestamp),
                "warmup": warmup,
                "caption": caption,
                "label_ids": label_ids,
                "label_names": label_names,
                "probabilities": probabilities,
                "confidence": confidence,
            }
        )
    return timeline


def locate_shift(timestamps: np.ndarray, shift_timestamp: float | None) -> dict[str, Any] | None:
    if shift_timestamp is None:
        return None
    nearest_index = int(np.argmin(np.abs(timestamps - shift_timestamp)))
    at_or_after_index = int(np.searchsorted(timestamps, shift_timestamp, side="left"))
    at_or_after_index = min(at_or_after_index, len(timestamps) - 1)
    nearest_timestamp = float(timestamps[nearest_index])
    at_or_after_timestamp = float(timestamps[at_or_after_index])
    return {
        "shift_timestamp": float(shift_timestamp),
        "nearest_frame_index": nearest_index,
        "nearest_frame_number": nearest_index + 1,
        "nearest_frame_timestamp": nearest_timestamp,
        "nearest_offset_sec": nearest_timestamp - shift_timestamp,
        "first_at_or_after_frame_index": at_or_after_index,
        "first_at_or_after_frame_number": at_or_after_index + 1,
        "first_at_or_after_timestamp": at_or_after_timestamp,
        "first_at_or_after_offset_sec": at_or_after_timestamp - shift_timestamp,
    }


def build_segments(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not timeline:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(timeline)):
        previous = timeline[index - 1]
        current = timeline[index]
        if current["caption"] != previous["caption"] or current["warmup"] != previous["warmup"]:
            ranges.append((start, index - 1))
            start = index
    ranges.append((start, len(timeline) - 1))

    segments: list[dict[str, Any]] = []
    for segment_index, (start, end) in enumerate(ranges):
        first = timeline[start]
        last = timeline[end]
        segments.append(
            {
                "segment_index": segment_index,
                "start_frame_index": int(first["frame_index"]),
                "end_frame_index": int(last["frame_index"]),
                "start_frame_number": int(first["frame_number"]),
                "end_frame_number": int(last["frame_number"]),
                "start_timestamp": float(first["timestamp"]),
                "end_timestamp": float(last["timestamp"]),
                "start_relative_time_sec": float(first["relative_time_sec"]),
                "end_relative_time_sec": float(last["relative_time_sec"]),
                "duration_sec": float(last["timestamp"] - first["timestamp"]),
                "frame_count": end - start + 1,
                "warmup": bool(first["warmup"]),
                "caption": str(first["caption"]),
            }
        )
    return segments


def _count_distribution(values: list[str]) -> dict[str, dict[str, float | int]]:
    counts = Counter(values)
    total = len(values)
    return {
        name: {"count": int(count), "ratio": float(count / total) if total else 0.0}
        for name, count in sorted(counts.items())
    }


def prediction_statistics(
    entries: list[dict[str, Any]],
    *,
    low_confidence_threshold: float,
) -> dict[str, Any]:
    predicted = [entry for entry in entries if not entry["warmup"]]
    label_distribution = {
        field: _count_distribution([str(entry["label_names"][field]) for entry in predicted])
        for field in LABEL_FIELDS
    }
    confidence: dict[str, Any] = {}
    for field in LABEL_FIELDS:
        values = np.asarray([float(entry["confidence"][field]) for entry in predicted], dtype=np.float64)
        low_count = int(np.count_nonzero(values < low_confidence_threshold))
        confidence[field] = {
            "min": float(np.min(values)) if values.size else None,
            "mean": float(np.mean(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "low_confidence_threshold": float(low_confidence_threshold),
            "low_confidence_count": low_count,
            "low_confidence_ratio": float(low_count / values.size) if values.size else 0.0,
        }

    changes: dict[str, int] = {}
    for field in LABEL_FIELDS:
        names = [str(entry["label_names"][field]) for entry in predicted]
        changes[field] = sum(left != right for left, right in zip(names, names[1:]))

    return {
        "num_frames": len(entries),
        "num_model_predictions": len(predicted),
        "label_distribution": label_distribution,
        "caption_distribution": _count_distribution([str(entry["caption"]) for entry in predicted]),
        "field_change_counts": changes,
        "confidence": confidence,
    }


def build_report(
    data: AttemptData,
    predictor: Any,
    *,
    low_confidence_threshold: float,
) -> dict[str, Any]:
    if not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError(
            f"low_confidence_threshold must be in [0, 1], got {low_confidence_threshold}"
        )
    timeline = predict_timeline(data, predictor)
    duration = float(data.timestamps[-1] - data.timestamps[0])
    effective_fps = float((len(data.timestamps) - 1) / duration) if duration > 0 else None
    before_shift = [entry for entry in timeline if entry["shift_relation"] == "before"]
    after_shift = [entry for entry in timeline if entry["shift_relation"] == "after"]
    statistics: dict[str, Any] = {
        "total_frames": len(timeline),
        "warmup_frames": sum(bool(entry["warmup"]) for entry in timeline),
        "model_prediction_frames": sum(not bool(entry["warmup"]) for entry in timeline),
        "start_timestamp": float(data.timestamps[0]),
        "end_timestamp": float(data.timestamps[-1]),
        "duration_sec": duration,
        "effective_fps": effective_fps,
        "all_predictions": prediction_statistics(
            timeline,
            low_confidence_threshold=low_confidence_threshold,
        ),
        "before_shift": None,
        "after_shift": None,
    }
    if data.shift_timestamp is not None:
        statistics["before_shift"] = prediction_statistics(
            before_shift,
            low_confidence_threshold=low_confidence_threshold,
        )
        statistics["after_shift"] = prediction_statistics(
            after_shift,
            low_confidence_threshold=low_confidence_threshold,
        )

    return {
        "statistics": statistics,
        "segments": build_segments(timeline),
        "shift_position": locate_shift(data.timestamps, data.shift_timestamp),
    }


def output_path(output_dir: Path, checkpoint_path: Path, episode_index: int, attempt_index: int) -> Path:
    checkpoint_run = checkpoint_path.parent.name if checkpoint_path.name == "best.pt" else checkpoint_path.stem
    return output_dir / checkpoint_run / f"episode{episode_index}_attempt{attempt_index}.json"


def write_report(report: dict[str, Any], destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episode_index < 0 or args.attempt_index < 1:
        raise ValueError("episode-index must be >= 0 and attempt-index must be >= 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Captioner checkpoint not found: {args.checkpoint}")

    data = load_attempt_data(args.dataset_dir, args.episode_index, args.attempt_index)
    predictor = TactileCaptionerPredictor(args.checkpoint, device=args.device)
    report = build_report(
        data,
        predictor,
        low_confidence_threshold=args.low_confidence_threshold,
    )
    destination = output_path(args.output_dir, args.checkpoint, args.episode_index, args.attempt_index)
    write_report(report, destination, overwrite=args.overwrite)

    summary = report["statistics"]
    print(f"Wrote tactile-caption inspection report: {destination}")
    print(
        f"frames={summary['total_frames']} predictions={summary['model_prediction_frames']} "
        f"warmup={summary['warmup_frames']} segments={len(report['segments'])}"
    )
    if report["shift_position"] is not None:
        shift = report["shift_position"]
        print(
            f"shift_timestamp={shift['shift_timestamp']:.6f} "
            f"first_at_or_after_frame_index={shift['first_at_or_after_frame_index']} "
            f"offset={shift['first_at_or_after_offset_sec'] * 1000.0:.3f} ms"
        )


if __name__ == "__main__":
    main()
