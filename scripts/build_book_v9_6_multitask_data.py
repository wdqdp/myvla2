#!/usr/bin/env python3
"""Convert existing V9.5 real data to V9.6 without constructing counterfactuals."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts.build_v7_7_multitask_data import _write_json, _write_jsonl
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_5_stage_a_data import reject_archive
from tactile_vla.vla.book_v9_6_multitask_data import (
    DEFAULT_MULTITASK_DIR,
    DEFAULT_SOURCE_INDEX,
    MANIFEST_NAMES,
    derive,
    load_source,
    summary,
    validate_index,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MULTITASK_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.source_index = reject_archive(args.source_index)
    args.output_dir = reject_archive(args.output_dir)
    if args.output_dir == args.source_index.parent or args.source_index.parent in args.output_dir.parents:
        parser.error("V9.6 output must be separate from the V9.5 source directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run and args.output_dir.exists():
        raise FileExistsError("V9.6 output already exists; use another --output-dir (no overwrite)")
    source_index, rows = load_source(args.source_index)
    for name in ("dataset_dir", "v4_dir"):
        protected = Path(source_index[name]).resolve()
        if args.output_dir == protected or protected in args.output_dir.parents:
            raise ValueError("V9.6 output must not be inside the source dataset/V4 artifacts")
    index, manifests, need_audit, reasoning_audit = derive(source_index, rows, args.source_index)
    report = summary(index, manifests, need_audit)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    paths = {task: args.output_dir / name for task, name in MANIFEST_NAMES.items()}
    paths.update(
        index=args.output_dir / "book_v9_6_multitask_training_index.json",
        summary=args.output_dir / "summary.json",
        hashes=args.output_dir / "artifact_hashes.json",
    )
    for task, rows in manifests.items():
        _write_jsonl(paths[task], rows)
        index[f"{task}_manifest_file"] = str(paths[task].resolve())
        index[f"{task}_manifest_sha256"] = sha256_file(paths[task])
    for name, audit in (("need_boundary_audit", need_audit), ("reasoning_boundary_audit", reasoning_audit)):
        paths[name] = args.output_dir / f"{name}.json"
        _write_json(paths[name], audit)
        index[f"{name}_file"] = str(paths[name].resolve())
        index[f"{name}_sha256"] = sha256_file(paths[name])
    index["training_data_hash"] = sha256_json(index)
    validate_index(index)
    report["training_data_hash"] = index["training_data_hash"]
    _write_json(paths["index"], index)
    _write_json(paths["summary"], report)
    _write_json(
        paths["hashes"],
        {
            "artifacts": {
                key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for key, path in paths.items()
                if key != "hashes"
            }
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
