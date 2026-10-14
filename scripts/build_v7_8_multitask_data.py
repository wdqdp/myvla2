#!/usr/bin/env python3
"""Derive V7.8 visual/state manifests exactly from V7.7.2."""

# ruff: noqa: E402
from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts.build_v7_7_multitask_data import _write_json, _write_jsonl
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_8_multitask_data import (
    DEFAULT_OUTPUT, DEFAULT_SOURCE_INDEX, MANIFEST_NAMES, derive, load_source, summary, validate_index,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.source_index = args.source_index.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run and args.output_dir.exists():
        raise FileExistsError("V7.8 output exists; choose a new directory, no implicit overwrite")
    original, original_rows = load_source(args.source_index)
    index, rows = derive(original, original_rows, args.source_index)
    report = summary(index, rows)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    for task, name in MANIFEST_NAMES.items():
        path = args.output_dir / name
        _write_jsonl(path, rows[task])
        index[f"{task}_manifest_file"] = str(path)
        index[f"{task}_manifest_sha256"] = sha256_file(path)
    index["training_data_hash"] = sha256_json(index)
    validate_index(index)
    _write_json(args.output_dir / "v7_8_multitask_training_index.json", index)
    report["training_data_hash"] = index["training_data_hash"]
    _write_json(args.output_dir / "summary.json", report)
    _write_json(args.output_dir / "artifact_hashes.json", {
        p.name: sha256_file(p) for p in args.output_dir.iterdir() if p.is_file()
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
