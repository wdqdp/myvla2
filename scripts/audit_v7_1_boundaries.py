#!/usr/bin/env python3
"""Summarize V7.1 excluded intervals and their boundary timestamps."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_ARTIFACT_DIR = DEFAULT_ROOT / "outputs/rotation_v7_1_adjustment"
DEFAULT_OUTPUT_NAME = "boundary_exclusion_intervals.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--output-file",
        type=Path,
        help=f"Output JSON path; defaults to artifact-dir/{DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _timestamp_by_global_index(dataset_dir: Path) -> dict[int, float]:
    timestamps: dict[int, float] = {}
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files under {dataset_dir / 'data'}")
    for parquet_path in parquet_files:
        table = pq.read_table(
            parquet_path, columns=["index", "ros_timestamp"]
        ).to_pydict()
        for raw_index, raw_timestamp in zip(
            table["index"], table["ros_timestamp"], strict=True
        ):
            global_index = int(raw_index)
            if global_index in timestamps:
                raise ValueError(f"Duplicate LeRobot global index {global_index}")
            timestamps[global_index] = float(raw_timestamp)
    return timestamps


def exclusion_intervals(
    manifest: list[dict[str, Any]], timestamps: dict[int, float]
) -> list[dict[str, Any]]:
    excluded = sorted(
        (row for row in manifest if not bool(row.get("trainable", True))),
        key=lambda row: (
            int(row["episode_id"]),
            int(row["attempt_id"]),
            int(row["frame_index"]),
        ),
    )
    intervals: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in excluded:
        episode_id = int(row["episode_id"])
        attempt_id = int(row["attempt_id"])
        frame_index = int(row["frame_index"])
        global_index = int(row["global_index"])
        reason = str(row["exclusion_reason"])
        split = str(row["split"])
        timestamp = timestamps.get(global_index)
        if timestamp is None:
            raise ValueError(f"Missing ros_timestamp for global index {global_index}")
        continues = bool(
            current is not None
            and current["episode_id"] == episode_id
            and current["attempt_id"] == attempt_id
            and current["reason"] == reason
            and current["end_frame"] + 1 == frame_index
        )
        if not continues:
            if current is not None:
                intervals.append(current)
            current = {
                "split": split,
                "episode_id": episode_id,
                "attempt_id": attempt_id,
                "reason": reason,
                "start_frame": frame_index,
                "end_frame": frame_index,
                "start_timestamp": timestamp,
                "end_timestamp": timestamp,
                "skipped_action_start_count": 1,
            }
        else:
            current["end_frame"] = frame_index
            current["end_timestamp"] = timestamp
            current["skipped_action_start_count"] += 1
    if current is not None:
        intervals.append(current)
    for interval in intervals:
        interval["duration_seconds"] = (
            float(interval["end_timestamp"]) - float(interval["start_timestamp"])
        )
    return intervals


def build_summary(
    manifest: list[dict[str, Any]], timestamps: dict[int, float]
) -> dict[str, Any]:
    intervals = exclusion_intervals(manifest, timestamps)
    skipped_rows = [row for row in manifest if not bool(row.get("trainable", True))]
    reason_counts = Counter(str(row["exclusion_reason"]) for row in skipped_rows)
    split_counts = Counter(str(row["split"]) for row in skipped_rows)
    interval_reason_counts = Counter(str(row["reason"]) for row in intervals)
    return {
        "schema_version": "tactile_vla_v7_1_boundary_exclusion_intervals_v1",
        "data_profile": "rotation_phase_v7_1_adjustment",
        "timestamp_field": "ros_timestamp",
        "timestamp_unit": "seconds",
        "skipped_action_start_count": len(skipped_rows),
        "interval_count": len(intervals),
        "skipped_action_start_counts_by_reason": dict(sorted(reason_counts.items())),
        "skipped_action_start_counts_by_split": dict(sorted(split_counts.items())),
        "interval_counts_by_reason": dict(sorted(interval_reason_counts.items())),
        "intervals": intervals,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_file = (
        args.output_file.expanduser().resolve()
        if args.output_file is not None
        else artifact_dir / DEFAULT_OUTPUT_NAME
    )
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_file}")
    manifest = _load_manifest(artifact_dir / "action_phase_manifest.jsonl")
    dataset_dir = (
        args.data_root.expanduser().resolve()
        / "lerobot_data/tactile_vla_rotation_v4"
    )
    payload = build_summary(manifest, _timestamp_by_global_index(dataset_dir))
    _write_json(output_file, payload)
    print(
        json.dumps(
            {
                "output_file": str(output_file),
                "skipped_action_start_count": payload["skipped_action_start_count"],
                "interval_count": payload["interval_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
