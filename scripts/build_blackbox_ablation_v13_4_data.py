#!/usr/bin/env python3
"""Derive V13.4 feedback-only prompts without changing frozen V13.2 observations/labels."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]

from tactile_vla.vla.blackbox_ablation_v13_4_data import OUTPUT, build_data, source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-index', type=Path, default=source.INDEX)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = p.parse_args()
    _, summary = build_data(source_index=args.source_index, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
