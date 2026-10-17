#!/usr/bin/env python3
"""Build checkpoint-independent joint blackbox-new-env / book V9.5 training data."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from tactile_vla.vla.v12_1_data import OUTPUT, SOURCES, build, read_json, validate_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--blackbox-source-index", type=Path, default=SOURCES["blackbox"])
    parser.add_argument("--book-source-index", type=Path, default=SOURCES["book"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true", help="Read-only validation of an already built index")
    args = parser.parse_args()
    if args.check_only:
        index = read_json(args.output_dir / "v12_1_training_index.json")
        validate_index(index)
    else:
        index = build(
            args.output_dir,
            source_paths={"blackbox": args.blackbox_source_index, "book": args.book_source_index},
            seed=args.seed,
        )
    print(json.dumps({"training_data_hash": index["training_data_hash"], "counts": index["counts"]}, indent=2))


if __name__ == "__main__":
    main()
