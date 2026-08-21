#!/usr/bin/env python3
"""Annotate synchronized V3 demonstrations with a fine-tuned W30 captioner."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import h5py
import numpy as np
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.captioner.predictor import TactileCaptionerPredictor  # noqa: E402
from tactile_vla.common.labels import DEFAULT_TACTILE_CAPTION  # noqa: E402
from tactile_vla.common.labels import LABEL_FIELDS  # noqa: E402
from tactile_vla.common.labels import NEUTRAL_LABELS  # noqa: E402


DEFAULT_DATASET_DIR = Path("/data1/tac_data/raw_data")
DEFAULT_CHECKPOINT = Path("/data1/outputs/tactile_captioner/tcn_v3_w30_rotation_head/best.pt")
ANNOTATION_SCHEMA_VERSION = "tactile_caption_annotation_v1"
EXPECTED_WINDOW_SIZE = 30


@dataclass(frozen=True)
class AttemptRef:
    episode_id: int
    attempt_id: int
    attempt_dir: Path

    @property
    def hdf5_path(self) -> Path:
        return self.attempt_dir / "data.hdf5"


@dataclass(frozen=True)
class AttemptTactileData:
    timestamps: np.ndarray
    mesh_motion: np.ndarray
    force: np.ndarray


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _parse_index(path: Path, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", path.name)
    if match is None:
        raise ValueError(f"Expected {prefix}<integer> directory, got {path}")
    return int(match.group(1))


def _selected_attempt_directories(selection_file: Path, dataset_dir: Path) -> list[Path]:
    """Lazily load the sibling V4 schema only for selection-based annotation."""
    scripts_dir = PROJECT_ROOT.parent / "tac_ws" / "src" / "data_tools" / "scripts_new"
    schema_path = scripts_dir / "v4_data_schema.py"
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"--selection-file requires the sibling tac_ws V4 schema: {schema_path}"
        )
    sys.path.insert(0, str(scripts_dir))
    from v4_data_schema import selected_attempt_directories

    return selected_attempt_directories(selection_file, dataset_dir)


def discover_attempts(
    dataset_dir: Path,
    *,
    episode_start: int | None = None,
    episode_end: int | None = None,
    attempt_ids: set[int] | None = None,
    selection_file: Path | None = None,
) -> list[AttemptRef]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Demo dataset directory not found: {dataset_dir}")
    if episode_start is not None and episode_end is not None and episode_start > episode_end:
        raise ValueError("episode-start cannot be greater than episode-end")
    if selection_file is not None:
        if episode_start is not None or episode_end is not None or attempt_ids is not None:
            raise ValueError(
                "selection-file cannot be combined with episode range or attempt-index filters"
            )
        return [
            AttemptRef(
                episode_id=_parse_index(attempt_dir.parent, "episode"),
                attempt_id=_parse_index(attempt_dir, "attempt"),
                attempt_dir=attempt_dir,
            )
            for attempt_dir in _selected_attempt_directories(selection_file, dataset_dir)
        ]

    attempts: list[AttemptRef] = []
    episode_dirs = sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and re.fullmatch(r"episode\d+", path.name)],
        key=lambda path: _natural_key(path.name),
    )
    for episode_dir in episode_dirs:
        episode_id = _parse_index(episode_dir, "episode")
        if episode_start is not None and episode_id < episode_start:
            continue
        if episode_end is not None and episode_id > episode_end:
            continue
        attempt_dirs = sorted(
            [
                path
                for path in episode_dir.iterdir()
                if path.is_dir() and re.fullmatch(r"attempt\d+", path.name)
            ],
            key=lambda path: _natural_key(path.name),
        )
        for attempt_dir in attempt_dirs:
            attempt_id = _parse_index(attempt_dir, "attempt")
            if attempt_ids is not None and attempt_id not in attempt_ids:
                continue
            attempts.append(
                AttemptRef(
                    episode_id=episode_id,
                    attempt_id=attempt_id,
                    attempt_dir=attempt_dir,
                )
            )
    if not attempts:
        raise ValueError("No attempt directories matched the requested filters")
    return attempts


def _validate_tactile_array(
    values: np.ndarray,
    *,
    name: str,
    frame_count: int,
    channels: int,
    hdf5_path: Path,
) -> None:
    expected_shape = (frame_count, 35, 20, channels)
    if values.shape != expected_shape:
        raise ValueError(f"{hdf5_path}: {name} must have shape {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{hdf5_path}: {name} contains NaN or Inf")


def load_attempt_tactile_data(attempt: AttemptRef) -> AttemptTactileData:
    hdf5_path = attempt.hdf5_path
    if not hdf5_path.is_file():
        raise FileNotFoundError(
            f"Missing {hdf5_path}; run data_to_hdf5_attempts.py before offline annotation"
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
    except OSError as exc:
        raise ValueError(f"Cannot read completed HDF5 file {hdf5_path}: {exc}") from exc

    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError(f"{hdf5_path}: timestamp must be a non-empty 1-D array")
    if not np.isfinite(timestamps).all():
        raise ValueError(f"{hdf5_path}: timestamp contains NaN or Inf")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{hdf5_path}: timestamp must be strictly increasing")
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
    return AttemptTactileData(timestamps=timestamps, mesh_motion=mesh_motion, force=force)


def predict_frame_captions(
    data: AttemptTactileData,
    predictor: Any,
    *,
    batch_size: int,
) -> tuple[list[str], dict[str, Counter[str]]]:
    if batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {batch_size}")
    window_size = int(predictor.window_size)
    frame_count = len(data.timestamps)
    warmup_frames = min(frame_count, window_size - 1)
    captions = [DEFAULT_TACTILE_CAPTION] * warmup_frames
    field_counts = {field: Counter({NEUTRAL_LABELS[field]: warmup_frames}) for field in LABEL_FIELDS}

    offsets = np.arange(window_size - 1, -1, -1, dtype=np.int64)
    for batch_start in range(window_size - 1, frame_count, batch_size):
        batch_end = min(frame_count, batch_start + batch_size)
        end_indices = np.arange(batch_start, batch_end, dtype=np.int64)
        window_indices = end_indices[:, None] - offsets[None, :]
        predictions = predictor.predict_batch(
            data.mesh_motion[window_indices],
            data.force[window_indices],
        )
        if len(predictions) != len(end_indices):
            raise RuntimeError(
                f"Captioner returned {len(predictions)} predictions for {len(end_indices)} windows"
            )
        for prediction in predictions:
            captions.append(str(prediction.caption))
            for field in LABEL_FIELDS:
                field_counts[field][str(prediction.label_names[field])] += 1

    if len(captions) != frame_count:
        raise RuntimeError(f"Generated {len(captions)} captions for {frame_count} frames")
    return captions, field_counts


def _read_existing_labels(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read existing label file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Existing label file must contain a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_label_payload(
    *,
    existing: dict[str, Any] | None,
    captions: list[str],
    data: AttemptTactileData,
    checkpoint: Path,
    window_size: int,
    field_counts: dict[str, Counter[str]],
) -> dict[str, Any]:
    payload = dict(existing or {})
    payload["tactile_caption"] = captions
    payload["_tactile_caption_annotation"] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "window_size": int(window_size),
        "warmup_policy": "first window_size-1 frames use the neutral caption",
        "num_frames": len(captions),
        "first_timestamp": float(data.timestamps[0]),
        "last_timestamp": float(data.timestamps[-1]),
        "field_counts": {
            field: {name: int(count) for name, count in sorted(field_counts[field].items())}
            for field in LABEL_FIELDS
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def _merge_counts(
    destination: dict[str, Counter[str]],
    source: dict[str, Counter[str]],
) -> None:
    for field in LABEL_FIELDS:
        destination[field].update(source[field])


def _preflight_destinations(
    attempts: Iterable[AttemptRef],
    *,
    label_file_name: str,
    overwrite: bool,
    skip_existing: bool,
) -> None:
    if overwrite or skip_existing:
        return
    existing = [attempt.attempt_dir / label_file_name for attempt in attempts if (attempt.attempt_dir / label_file_name).exists()]
    if existing:
        preview = "\n".join(f"  - {path}" for path in existing[:10])
        suffix = f"\n  ... and {len(existing) - 10} more" if len(existing) > 10 else ""
        raise FileExistsError(
            "Label files already exist. Use --skip-existing to resume or --overwrite to replace only "
            f"their tactile_caption field:\n{preview}{suffix}"
        )


def _validate_label_file_name(value: str) -> str:
    path = Path(value)
    if path.name != value or path.suffix.lower() != ".json":
        raise ValueError("label-file-name must be a JSON filename without directory components")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--episode-start", type=int)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument(
        "--attempt-index",
        type=int,
        action="append",
        dest="attempt_indices",
        help="Only annotate this attempt index; repeat to select multiple indices.",
    )
    parser.add_argument("--label-file-name", default="labels.json")
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="Strict V4 selection.json; cannot be combined with range/attempt filters.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Summary JSON path (default: <dataset-dir>/tactile_caption_annotation_summary.json).",
    )
    existing_group = parser.add_mutually_exclusive_group()
    existing_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace tactile_caption in existing label files while preserving all other keys.",
    )
    existing_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip attempts whose label file already exists, for interrupted-run recovery.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    label_file_name = _validate_label_file_name(args.label_file_name)
    attempt_ids = set(args.attempt_indices) if args.attempt_indices else None
    attempts = discover_attempts(
        args.dataset_dir,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        attempt_ids=attempt_ids,
        selection_file=args.selection_file,
    )
    missing_hdf5 = [attempt.hdf5_path for attempt in attempts if not attempt.hdf5_path.is_file()]
    if missing_hdf5:
        preview = "\n".join(f"  - {path}" for path in missing_hdf5[:10])
        suffix = f"\n  ... and {len(missing_hdf5) - 10} more" if len(missing_hdf5) > 10 else ""
        raise FileNotFoundError(
            "Some selected attempts do not contain data.hdf5; finish HDF5 conversion first:\n"
            f"{preview}{suffix}"
        )
    _preflight_destinations(
        attempts,
        label_file_name=label_file_name,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
    )

    existing_count = sum((attempt.attempt_dir / label_file_name).exists() for attempt in attempts)
    print(
        f"Selected {len(attempts)} attempts under {args.dataset_dir}; "
        f"existing label files={existing_count}"
    )
    if args.dry_run:
        print("Dry run complete; no checkpoint was loaded and no files were written.")
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Captioner checkpoint not found: {args.checkpoint}")

    predictor = TactileCaptionerPredictor(args.checkpoint, device=args.device)
    if int(predictor.window_size) != EXPECTED_WINDOW_SIZE:
        raise ValueError(
            f"This script requires a W{EXPECTED_WINDOW_SIZE} checkpoint, but "
            f"{args.checkpoint} declares window_size={predictor.window_size}"
        )
    print(
        f"Loaded W{predictor.window_size} captioner on {predictor.device}; "
        f"offline batch_size={args.batch_size}"
    )

    aggregate_counts = {field: Counter() for field in LABEL_FIELDS}
    attempt_reports: list[dict[str, Any]] = []
    annotated_count = 0
    skipped_count = 0
    total_frames = 0
    for attempt in tqdm(attempts, desc="Annotating attempts"):
        destination = attempt.attempt_dir / label_file_name
        if destination.exists() and args.skip_existing:
            skipped_count += 1
            attempt_reports.append(
                {
                    "episode_id": attempt.episode_id,
                    "attempt_id": attempt.attempt_id,
                    "status": "skipped_existing",
                    "label_file": str(destination),
                }
            )
            continue

        existing = _read_existing_labels(destination) if destination.exists() else None
        data = load_attempt_tactile_data(attempt)
        captions, field_counts = predict_frame_captions(
            data,
            predictor,
            batch_size=args.batch_size,
        )
        payload = make_label_payload(
            existing=existing,
            captions=captions,
            data=data,
            checkpoint=args.checkpoint,
            window_size=predictor.window_size,
            field_counts=field_counts,
        )
        _atomic_write_json(destination, payload)
        _merge_counts(aggregate_counts, field_counts)
        annotated_count += 1
        total_frames += len(captions)
        attempt_reports.append(
            {
                "episode_id": attempt.episode_id,
                "attempt_id": attempt.attempt_id,
                "status": "annotated",
                "num_frames": len(captions),
                "warmup_frames": min(len(captions), predictor.window_size - 1),
                "label_file": str(destination),
            }
        )

    report_path = args.report or args.dataset_dir / "tactile_caption_annotation_summary.json"
    report = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "window_size": int(predictor.window_size),
        "warmup_policy": "first window_size-1 frames use the neutral caption",
        "selected_attempts": len(attempts),
        "annotated_attempts": annotated_count,
        "skipped_attempts": skipped_count,
        "annotated_frames": total_frames,
        "field_counts": {
            field: {name: int(count) for name, count in sorted(aggregate_counts[field].items())}
            for field in LABEL_FIELDS
        },
        "attempts": attempt_reports,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(report_path, report)
    print(
        f"Completed: annotated_attempts={annotated_count} skipped_attempts={skipped_count} "
        f"annotated_frames={total_frames}"
    )
    print(f"Summary: {report_path}")
    print(
        "Next: rerun data_to_hdf5_attempts.py with --labelFile "
        f"{label_file_name} --overwrite true"
    )


if __name__ == "__main__":
    main()
