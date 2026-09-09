#!/usr/bin/env python3
"""Detect V7.2 motion boundaries and write filter plus random audit JSON."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v4_data import SPLITS, V4Frame, validate_v4_index_dataset
from tactile_vla.vla.v7_1_adjustment_data import _load_lerobot_signals
from tactile_vla.vla.v7_2_boundary_filter import DEFAULT_DETECTOR_CONFIG
from tactile_vla.vla.v7_2_boundary_filter import V7_2_AUDIT_SCHEMA
from tactile_vla.vla.v7_2_boundary_filter import V7_2_FILTER_SCHEMA
from tactile_vla.vla.v7_2_boundary_filter import build_attempt_filter_row


DEFAULT_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_DATASET_DIR = DEFAULT_ROOT / "lerobot_data/tactile_vla_rotation_v4"
DEFAULT_V4_INDEX = DEFAULT_ROOT / "outputs/rotation_v4/v4_training_index.json"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "outputs/rotation_v7_2_adjustment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--v4-index-file", type=Path, default=DEFAULT_V4_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _timestamps(dataset_dir: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    for path in sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(path, columns=["index", "ros_timestamp"]).to_pydict()
        for raw_index, raw_timestamp in zip(
            table["index"], table["ros_timestamp"], strict=True
        ):
            index = int(raw_index)
            if index in result:
                raise ValueError(f"Duplicate global index {index}")
            result[index] = float(raw_timestamp)
    return result


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.audit_count < 1:
        raise ValueError("--audit-count must be positive")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    v4_index_file = args.v4_index_file.expanduser().resolve()
    v4_index = json.loads(v4_index_file.read_text())
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    actions, _ = _load_lerobot_signals(dataset_dir, len(global_lookup))
    timestamps = _timestamps(dataset_dir)
    grouped: dict[tuple[int, int], list[V4Frame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.attempt_key].append(frame)
    candidate_indices = {
        int(index)
        for split in SPLITS
        for index in v4_index["splits"][split]["execution_indices"]
    }
    split_by_episode = {
        global_lookup[int(index)].episode_id: split
        for split in SPLITS
        for index in v4_index["splits"][split]["execution_indices"]
    }

    attempt_rows: list[dict[str, Any]] = []
    for (episode_id, attempt_id), attempt_frames in sorted(grouped.items()):
        if attempt_id != 2:
            continue
        timing = v4_index["attempt_timing"][f"episode{episode_id}/attempt2"]
        row = build_attempt_filter_row(
            episode_id=episode_id,
            attempt_frames=attempt_frames,
            actions=actions,
            timestamps=timestamps,
            move_start_frame=int(timing["move_start_frame_index"]),
            rexecution_frame=int(timing["rexecution_frame_index"]),
            candidate_global_indices=candidate_indices,
        )
        row["split"] = split_by_episode[episode_id]
        row["annotated_timestamps"] = {
            "move_start_timestamp": float(timing["move_start_timestamp"]),
            "rexecution_timestamp": float(timing["rexecution_timestamp"]),
        }
        attempt_rows.append(row)

    reason_counts: Counter[str] = Counter()
    excluded_indices: set[int] = set()
    for row in attempt_rows:
        for interval in row["excluded_intervals"]:
            reason_counts[interval["reason"]] += interval[
                "excluded_candidate_action_start_count"
            ]
            excluded_indices.update(interval["excluded_global_indices"])
    payload = {
        "schema_version": V7_2_FILTER_SCHEMA,
        "data_profile": "rotation_phase_v7_2_adjustment",
        "detector_config": DEFAULT_DETECTOR_CONFIG,
        "dataset_dir": str(dataset_dir),
        "source_v4_index": {
            "path": str(v4_index_file),
            "sha256": sha256_file(v4_index_file),
        },
        "summary": {
            "attempt2_count": len(attempt_rows),
            "detected_attempt_count": len(attempt_rows),
            "excluded_candidate_action_start_count": len(excluded_indices),
            "excluded_counts_by_reason": dict(sorted(reason_counts.items())),
        },
        "attempts": attempt_rows,
    }

    rng = random.Random(args.seed)
    sampled = rng.sample(attempt_rows, min(args.audit_count, len(attempt_rows)))
    audit_attempts = []
    for row in sampled:
        compact_intervals = [
            {
                key: value
                for key, value in interval.items()
                if key != "excluded_global_indices"
            }
            for interval in row["excluded_intervals"]
        ]
        audit_attempts.append(
            {
                "episode_id": row["episode_id"],
                "attempt_id": row["attempt_id"],
                "split": row["split"],
                "annotated_timestamps": row["annotated_timestamps"],
                "gripper_motion_stop": row["events"]["gripper_motion_stop"],
                "arm_adjustment_start": row["events"]["arm_adjustment_start"],
                "arm_adjustment_stop": row["events"]["arm_adjustment_stop"],
                "gripper_close_start": row["events"]["gripper_close_start"],
                "event_offsets_from_anchor": row["event_offsets_from_anchor"],
                "excluded_intervals": compact_intervals,
                "manual_review_status": "pending",
            }
        )
    audit = {
        "schema_version": V7_2_AUDIT_SCHEMA,
        "data_profile": "rotation_phase_v7_2_adjustment",
        "seed": args.seed,
        "sample_count": len(audit_attempts),
        "timestamp_field": "ros_timestamp",
        "timestamp_unit": "seconds",
        "event_order_constraints": [
            "gripper_motion_stop < arm_adjustment_start",
            "arm_adjustment_stop < gripper_close_start",
        ],
        "attempts": audit_attempts,
    }
    output_dir = args.output_dir.expanduser().resolve()
    _write_json(output_dir / "v7_2_boundary_filter.json", payload, overwrite=args.overwrite)
    _write_json(output_dir / "boundary_audit_random.json", audit, overwrite=args.overwrite)
    print(json.dumps({"output_dir": str(output_dir), **payload["summary"], "audit_count": len(audit_attempts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
