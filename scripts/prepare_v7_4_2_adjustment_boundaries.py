#!/usr/bin/env python3
"""Offline-label V7.4.2 arm adjustment start/stop boundaries."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v4_data import validate_v4_index_dataset
from tactile_vla.vla.v7_4_2_adjustment_boundaries import AUDIT_SCHEMA
from tactile_vla.vla.v7_4_2_adjustment_boundaries import BOUNDARY_SCHEMA
from tactile_vla.vla.v7_4_2_adjustment_boundaries import DETECTOR_CONFIG
from tactile_vla.vla.v7_4_2_adjustment_boundaries import detect_adjustment_boundaries


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_DATASET_DIR = ROOT / "lerobot_data/tactile_vla_rotation_v4_1_new_env"
DEFAULT_V4_INDEX = ROOT / "outputs/rotation_v4_1_new_env/v4_training_index.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/rotation_v7_4_2_adjustment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--v4-index-file", type=Path, default=DEFAULT_V4_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_attempt2(dataset_dir: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(
            path,
            columns=["episode_id", "attempt_id", "frame_index", "index", "ros_timestamp", "action"],
        ).to_pydict()
        episode_id = int(table["episode_id"][0])
        attempt_id = int(table["attempt_id"][0])
        if attempt_id != 2:
            continue
        order = np.argsort(np.asarray(table["frame_index"], dtype=np.int64))
        if episode_id in result:
            raise ValueError(f"Duplicate attempt2 episode {episode_id}")
        result[episode_id] = {
            key: np.asarray(value)[order]
            for key, value in table.items()
        }
    return result


def main() -> int:
    args = parse_args()
    if args.audit_count < 1:
        raise ValueError("--audit-count must be positive")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    v4_index_file = args.v4_index_file.expanduser().resolve()
    v4_index = json.loads(v4_index_file.read_text())
    frames, _ = validate_v4_index_dataset(v4_index, dataset_dir)
    expected_attempt2 = sorted(
        {frame.episode_id for frame in frames if frame.attempt_id == 2}
    )
    attempts = _load_attempt2(dataset_dir)
    if sorted(attempts) != expected_attempt2:
        raise ValueError("LeRobot attempt2 set differs from the validated V4 frame set")

    rows: list[dict[str, Any]] = []
    for episode_id, table in sorted(attempts.items()):
        detected = detect_adjustment_boundaries(
            frame_indices=table["frame_index"],
            timestamps=table["ros_timestamp"],
            actions=np.stack(table["action"]),
        )
        for event in detected["events"].values():
            frame_index = int(event["frame_index"])
            event["global_index"] = int(table["index"][frame_index])
        rows.append(
            {
                "episode_id": episode_id,
                "attempt_id": 2,
                **detected,
            }
        )

    pre_idle = [row["diagnostics"]["pre_adjustment_idle_frames"] for row in rows]
    post_idle = [row["diagnostics"]["post_adjustment_idle_frames"] for row in rows]
    overlaps = [row["diagnostics"]["arm_close_overlap_frames"] for row in rows]
    summary = {
        "attempt2_count": len(rows),
        "pre_adjustment_idle_frames": {
            "min": min(pre_idle), "max": max(pre_idle), "mean": float(np.mean(pre_idle)),
            "counts": dict(sorted(Counter(pre_idle).items())),
        },
        "post_adjustment_idle_frames": {
            "min": min(post_idle), "max": max(post_idle), "mean": float(np.mean(post_idle)),
            "counts": dict(sorted(Counter(post_idle).items())),
        },
        "arm_close_overlap_frames": {
            "attempt_count": sum(value > 0 for value in overlaps),
            "max": max(overlaps),
            "mean": float(np.mean(overlaps)),
        },
    }
    payload = {
        "schema_version": BOUNDARY_SCHEMA,
        "data_profile": "rotation_phase_v7_4_2_adjustment",
        "dataset_dir": str(dataset_dir),
        "source_v4_index": {
            "path": str(v4_index_file),
            "sha256": sha256_file(v4_index_file),
            "training_data_hash": v4_index["training_data_hash"],
        },
        "detector_config": DETECTOR_CONFIG,
        "summary": summary,
        "attempts": rows,
    }
    payload["content_sha256"] = sha256_json(payload)

    sampled_ids = random.Random(args.seed).sample(
        expected_attempt2, min(args.audit_count, len(expected_attempt2))
    )
    by_episode = {row["episode_id"]: row for row in rows}
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "data_profile": "rotation_phase_v7_4_2_adjustment",
        "seed": args.seed,
        "sample_count": len(sampled_ids),
        "sample_episode_ids": sampled_ids,
        "review_status": "offline_signal_reviewed",
        "attempts": [by_episode[episode_id] for episode_id in sampled_ids],
    }
    print(json.dumps({"summary": summary, "audit_episode_ids": sampled_ids}, indent=2))
    if args.dry_run:
        print("dry_run=true; no boundary artifacts were written")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    _write_json(
        output_dir / "adjustment_boundaries.json", payload, overwrite=args.overwrite
    )
    _write_json(
        output_dir / "boundary_audit_10.json", audit, overwrite=args.overwrite
    )
    print(f"wrote V7.4.2 offline boundaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
