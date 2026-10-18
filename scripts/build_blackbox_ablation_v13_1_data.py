#!/usr/bin/env python3
"""Build paired plan ablations; does not require or load a Stage A model."""
# ruff: noqa: E402
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tactile_vla.vla.blackbox_ablation_v13_1_data import DATASET, NORM, OUTPUT, build_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, default=DATASET)
    p.add_argument('--source-dir', type=Path)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    p.add_argument('--norm-stats-dir', type=Path, default=NORM)
    p.add_argument('--raw-image-dir', type=Path, help='Alternative root for raw_data image references')
    p.add_argument('--lerobot-dir', type=Path, help='LeRobot root; auto-detects dataset-dir/lerobot_data/tactile_vla_black_box_ablation')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    index, summary = build_data(**vars(args))
    import json
    print(json.dumps({'index': str(args.output_dir / 'index.json'),
                      'training_data_hash': index['training_data_hash'], 'summary': summary},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
