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

DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_v3")
DEFAULT_PROFILE_DIR = Path("/data1/outputs/vla/rotation_moderately_success_v1")
DEFAULT_INDEX_FILE = DEFAULT_PROFILE_DIR / "vla_indices_v3.json"
DEFAULT_SPLIT_FILE = DEFAULT_PROFILE_DIR / "splits.json"
DEFAULT_OUTPUT_DIR = Path(
    "/data1/outputs/vla/assets/tactile_vla_rotation_moderately_success_v1"
)

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

import numpy as np
import pyarrow.parquet as pq
import tqdm

from openpi.shared import normalize
from tactile_vla.vla.artifacts import artifact_identity
from tactile_vla.vla.data_profiles import ROTATION_MODERATELY_SUCCESS_V1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--data-profile", default=ROTATION_MODERATELY_SUCCESS_V1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--delta-action-dims", type=int, default=7)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite-splits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite_splits:
        raise ValueError("Versioned norm stats never overwrite or regenerate split files")
    if args.max_frames is not None:
        raise ValueError("Profile-bound norm stats must use every persisted train action index")
    if not args.index_file.is_file():
        raise FileNotFoundError(args.index_file)
    index = json.loads(args.index_file.read_text())
    if int(index.get("action_horizon", -1)) != args.action_horizon:
        raise ValueError(
            f"Index action_horizon={index.get('action_horizon')!r}, "
            f"requested={args.action_horizon}"
        )
    identity = artifact_identity(
        index,
        index_path=args.index_file,
        prompt_profile="not_applicable",
        requested_data_profile=args.data_profile,
    )
    indices = [int(value) for value in index["splits"]["train"]["execution_indices"]]

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

    if seen != len(indices):
        raise ValueError(
            f"Norm stats consumed {seen} frames, but the persisted train action index has {len(indices)}"
        )

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    normalize.save(args.output_dir, norm_stats)
    payload = {
        "dataset_dir": str(args.dataset_dir),
        "split_file": str(args.split_file),
        "index_file": str(args.index_file),
        "data_profile": args.data_profile,
        "artifact_identity": identity,
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
