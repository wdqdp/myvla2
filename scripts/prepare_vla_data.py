#!/usr/bin/env python3
"""Build split and frame-index files for VLA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_DATASET_DIR = Path("/home/test/qxh/workspace/tac_ws/lerobot_data/tactile_vla")
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/vla/indices")
DEFAULT_SPLIT_FILE = Path("/data1/outputs/vla/indices/splits.json")
DEFAULT_INDEX_FILE = DEFAULT_OUTPUT_DIR / "vla_indices_h50.json"

from tactile_vla.vla.index import SplitConfig
from tactile_vla.vla.index import index_payload
from tactile_vla.vla.index import load_or_create_splits
from tactile_vla.vla.index import scan_lerobot_frames
from tactile_vla.vla.index import summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--status-negative-ratio", type=float, default=3.0)
    parser.add_argument("--reasoning-augment-after-frames", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_file = args.split_file or args.output_dir / "splits.json"
    index_file = args.index_file or args.output_dir / "vla_indices.json"

    records = scan_lerobot_frames(args.dataset_dir)
    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    splits = load_or_create_splits(records, split_file, split_config, overwrite=args.overwrite)
    payload = index_payload(
        records,
        splits,
        seed=args.seed,
        negative_ratio=args.status_negative_ratio,
        reasoning_augment_after_frames=args.reasoning_augment_after_frames,
        action_horizon=args.action_horizon,
    )
    payload["dataset_dir"] = str(args.dataset_dir)
    payload["global_summary"] = summarize_records(records)
    payload["global_summary"]["execution_action_horizon"] = args.action_horizon
    payload["global_summary"]["execution_indices"] = sum(
        len(split_payload["execution_indices"]) for split_payload in payload["splits"].values()
    )
    payload["split_file"] = str(split_file)

    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote_split_file={split_file}")
    print(f"wrote_index_file={index_file}")
    print(json.dumps({k: v["summary"] for k, v in payload["splits"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
