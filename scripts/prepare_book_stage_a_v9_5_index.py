#!/usr/bin/env python3
"""Build an independent Book V9.5 Stage A index, excluding history9_4_5."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from scripts.prepare_book_stage_a_index import write_json
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_5_stage_a_data import ROOT, build_artifacts, reject_archive, validate_training_index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-root", type=Path, default=ROOT)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--v4-index-file", type=Path)
    parser.add_argument("--norm-stats-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    root = reject_archive(args.book_root)
    args.dataset_dir = args.dataset_dir or root / "lerobot_data/tactile_vla_rotation_v4"
    args.v4_index_file = args.v4_index_file or root / "outputs/rotation_v4/v4_training_index.json"
    args.norm_stats_dir = args.norm_stats_dir or root / "outputs/rotation_v4/norm_stats"
    args.output_dir = reject_archive(args.output_dir or root / "outputs/book_stage_a_v9_5")
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = {
        name: args.output_dir / name
        for name in (
            "action_phase_manifest.jsonl",
            "filter_summary.json",
            "book_stage_a_training_index.json",
            "artifact_hashes.json",
        )
    }
    if not args.dry_run and not args.overwrite and any(p.exists() for p in paths.values()):
        raise FileExistsError("V9.5 output exists; use a new directory or explicit --overwrite")
    rows, index, summary = build_artifacts(
        book_root=args.book_root,
        dataset_dir=args.dataset_dir,
        v4_index_file=args.v4_index_file,
        norm_stats_dir=args.norm_stats_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = paths["action_phase_manifest.jsonl"]
    temporary = manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(manifest)
    index["source_files"]["action_phase_manifest"] = {"path": str(manifest), "sha256": sha256_file(manifest)}
    index["action_phase_manifest_identity"]["file_sha256"] = sha256_file(manifest)
    index["training_data_hash"] = sha256_json(index)
    index_path = paths["book_stage_a_training_index.json"]
    write_json(index_path, index)
    validate_training_index(index, index_path=index_path, dataset_dir=args.dataset_dir, book_root=args.book_root)
    write_json(paths["filter_summary.json"], summary)
    hashes = {
        "schema_version": "tactile_vla_book_stage_a_v9_5_artifact_hashes_v1",
        "files": {
            p.name: {"path": str(p), "sha256": sha256_file(p)}
            for p in paths.values()
            if p.name != "artifact_hashes.json"
        },
    }
    hashes["sha256"] = sha256_json(hashes)
    write_json(paths["artifact_hashes.json"], hashes)
    print(f"wrote V9.5 Stage A training index to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
