#!/usr/bin/env python3
"""Build Bottle V11.2 compressed H30 targets without modifying source data."""

# ruff: noqa: E402
from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts.prepare_book_stage_a_index import write_json
from scripts.build_v7_7_multitask_data import _write_jsonl
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.bottle_v11_2_stage_a_data import (
    ROOT,
    DEFAULT_SETTLE_FRAMES,
    build_artifacts,
    validate_training_index,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bottle-root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path)
    p.add_argument("--v4-index-file", type=Path)
    p.add_argument("--norm-stats-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--settle-frames", type=int, default=DEFAULT_SETTLE_FRAMES)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.dataset_dir = (a.dataset_dir or a.bottle_root / "lerobot_data/tactile_vla_rotation_v4").resolve()
    a.v4_index_file = (a.v4_index_file or a.bottle_root / "outputs/rotation_v4/v4_training_index.json").resolve()
    a.norm_stats_dir = (a.norm_stats_dir or a.bottle_root / "outputs/rotation_v4/norm_stats").resolve()
    a.output_dir = (a.output_dir or a.bottle_root / "outputs/bottle_stage_a_v11_2").resolve()
    return a


def main(argv=None):
    a = parse_args(argv)
    if not a.dry_run and a.output_dir.exists():
        raise FileExistsError("V11.2 output exists; choose a new output directory")
    rows, index, summary = build_artifacts(
        dataset_dir=a.dataset_dir,
        v4_index_file=a.v4_index_file,
        norm_stats_dir=a.norm_stats_dir,
        settle_frames=a.settle_frames,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.dry_run:
        return 0
    manifest = a.output_dir / "action_phase_manifest.jsonl"
    index_path = a.output_dir / "stage_a_training_index.json"
    _write_jsonl(manifest, rows)
    index["source_files"]["action_phase_manifest"] = {"path": str(manifest), "sha256": sha256_file(manifest)}
    index["action_phase_manifest_identity"]["file_sha256"] = sha256_file(manifest)
    index["training_data_hash"] = sha256_json(index)
    write_json(index_path, index)
    validate_training_index(index, index_path=index_path, dataset_dir=a.dataset_dir)
    write_json(a.output_dir / "filter_summary.json", summary)
    write_json(a.output_dir / "boundary_audit.json", index["idle_gap_audit"])
    write_json(
        a.output_dir / "artifact_hashes.json", {p.name: sha256_file(p) for p in a.output_dir.iterdir() if p.is_file()}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
