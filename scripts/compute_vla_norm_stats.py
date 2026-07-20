#!/usr/bin/env python3
"""Compute OpenPI state/action normalization stats for tactile VLA Stage A."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))

DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_expanded")
DEFAULT_SPLIT_FILE = Path("/data1/outputs/vla/indices/splits_h30_state_memory_expanded.json")
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/vla/assets/tactile_vla_h30_state_memory_expanded")

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

import numpy as np
import pyarrow.parquet as pq
import tqdm

from openpi.shared import normalize
from tactile_vla.vla.index import SplitConfig
from tactile_vla.vla.index import execution_indices
from tactile_vla.vla.index import load_or_create_splits
from tactile_vla.vla.index import records_for_split
from tactile_vla.vla.index import scan_lerobot_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--delta-action-dims", type=int, default=7)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite-splits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = scan_lerobot_frames(args.dataset_dir)
    splits = load_or_create_splits(
        records,
        args.split_file,
        SplitConfig(seed=args.seed),
        overwrite=args.overwrite_splits,
    )
    train_records = records_for_split(records, splits["train"])
    indices = execution_indices(train_records, action_horizon=args.action_horizon)
    if args.max_frames is not None:
        indices = indices[: args.max_frames]

    selected = set(indices)
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    parquet_files = sorted((args.dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    seen = 0
    for parquet_file in tqdm.tqdm(parquet_files, desc="Computing VLA norm stats"):
        table = pq.read_table(parquet_file, columns=["index", "observation.state", "action"])
        data = table.to_pydict()
        row_indices = [int(value) for value in data["index"]]
        states = np.asarray(data["observation.state"], dtype=np.float32)
        actions = np.asarray(data["action"], dtype=np.float32)
        for row_id, global_index in enumerate(row_indices):
            if global_index not in selected:
                continue
            chunk = actions[row_id : row_id + args.action_horizon]
            if chunk.shape[0] != args.action_horizon:
                continue
            dims = min(args.delta_action_dims, states.shape[-1], chunk.shape[-1])
            delta_chunk = chunk.copy()
            delta_chunk[..., :dims] -= states[row_id, :dims]
            stats["state"].update(states[row_id])
            stats["actions"].update(delta_chunk)
            seen += 1

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    normalize.save(args.output_dir, norm_stats)
    payload = {
        "dataset_dir": str(args.dataset_dir),
        "split_file": str(args.split_file),
        "num_frames": seen,
        "action_horizon": args.action_horizon,
        "action_space": "delta_joint_position",
        "delta_action_dims": args.delta_action_dims,
        "keys": ["state", "actions"],
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote_norm_stats={args.output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
