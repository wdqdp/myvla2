#!/usr/bin/env python3
"""Derive no-touch Bottle V11.3 from V11.2 without changing compressed targets/history."""

# ruff: noqa: E402
from pathlib import Path
import argparse
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src"), str(PROJECT / "openpi/src")]

from scripts.build_v7_7_multitask_data import _write_jsonl
from scripts.prepare_book_stage_a_index import write_json
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.bottle_v11_3_multitask_data import (
    ROOT,
    MANIFEST_NAMES,
    derive,
    load_source,
    summary,
    validate_index,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bottle-root", type=Path, default=ROOT)
    p.add_argument("--source-index", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.source_index = (
        args.source_index
        or args.bottle_root / "outputs/bottle_v11_2_multitask/bottle_v11_2_multitask_training_index.json"
    ).resolve()
    args.output_dir = (args.output_dir or args.bottle_root / "outputs/bottle_v11_3_multitask").resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run and args.output_dir.exists():
        raise FileExistsError("V11.3 output exists; choose a new directory, no implicit overwrite")
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
    write_json(args.output_dir / "bottle_v11_3_multitask_training_index.json", index)
    validate_index(index)
    write_json(args.output_dir / "summary.json", report | {"training_data_hash": index["training_data_hash"]})
    write_json(args.output_dir / "boundary_audit.json", index["boundary_audit"])
    write_json(args.output_dir / "idle_gap_audit.json", index["idle_gap_audit"])
    write_json(
        args.output_dir / "artifact_hashes.json",
        {p.name: sha256_file(p) for p in args.output_dir.iterdir() if p.is_file()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
