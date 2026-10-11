#!/usr/bin/env python3
"""Derive Sponge V10.2 visual-only inputs from the verified V10.1 manifests."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts.build_v7_7_multitask_data import _write_jsonl
from scripts.prepare_book_stage_a_index import write_json
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.sponge_v10_2_multitask_data import (
    ROOT,
    MANIFEST_NAMES,
    derive,
    load_source,
    summary,
    validate_index,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sponge-root", type=Path, default=ROOT)
    p.add_argument("--source-index", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.source_index = (
        args.source_index
        or args.sponge_root / "outputs/sponge_v10_1_multitask/sponge_v10_1_multitask_training_index.json"
    ).resolve()
    args.output_dir = (args.output_dir or args.sponge_root / "outputs/sponge_v10_2_multitask").resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run and args.output_dir.exists():
        raise FileExistsError("V10.2 output exists; choose a new directory (no implicit overwrite)")
    original, original_rows = load_source(args.source_index)
    index, rows = derive(original, original_rows, args.source_index)
    report = summary(index, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    for task, filename in MANIFEST_NAMES.items():
        path = args.output_dir / filename
        _write_jsonl(path, rows[task])
        index[f"{task}_manifest_file"] = str(path)
        index[f"{task}_manifest_sha256"] = sha256_file(path)
    index["training_data_hash"] = sha256_json(index)
    write_json(args.output_dir / "sponge_v10_2_multitask_training_index.json", index)
    validate_index(index)
    report["training_data_hash"] = index["training_data_hash"]
    write_json(args.output_dir / "summary.json", report)
    write_json(args.output_dir / "boundary_audit.json", index["boundary_audit"])
    write_json(
        args.output_dir / "artifact_hashes.json",
        {p.name: sha256_file(p) for p in args.output_dir.iterdir() if p.is_file()},
    )
    return 0


if __name__ == "__main__":
    main()
