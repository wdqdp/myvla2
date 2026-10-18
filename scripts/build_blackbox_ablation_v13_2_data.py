#!/usr/bin/env python3
"""Build V13.2 global-reversal, random-grasp plan ablations from frozen V13.1 observations."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tactile_vla.vla.blackbox_ablation_v13_2_data import OUTPUT, build_data, prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-index', type=Path, default=prior.INDEX)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    p.add_argument('--seed', type=int, default=42)
    index, summary = build_data(**vars(p.parse_args()))
    print(json.dumps({'index': str(Path(index['events_file']).parent / 'index.json'), 'training_data_hash': index['training_data_hash'],
                      'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
