#!/usr/bin/env python3
"""Build Sponge V10.1 adjustment/need/failure/plan data without Stage A weights."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts.prepare_book_stage_a_index import write_json
from scripts.build_v7_7_multitask_data import _write_jsonl
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.sponge_v10_1_multitask_data import ROOT, build_artifacts, validate_index


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sponge-root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path)
    p.add_argument("--v4-dir", type=Path)
    p.add_argument("--action-index", type=Path)
    p.add_argument("--incremental-state-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    root = a.sponge_root
    a.dataset_dir = (a.dataset_dir or root / "lerobot_data/tactile_vla_rotation_v4").resolve()
    a.v4_dir = (a.v4_dir or root / "outputs/rotation_v4").resolve()
    a.action_index = (a.action_index or root / "outputs/sponge_stage_a_v10_1/stage_a_training_index.json").resolve()
    a.incremental_state_dir = (a.incremental_state_dir or root / "outputs/incremental_state").resolve()
    a.output_dir = (a.output_dir or root / "outputs/sponge_v10_1_multitask").resolve()
    return a


def main(argv=None):
    a = parse_args(argv)
    if not a.dry_run and a.output_dir.exists():
        raise FileExistsError("V10.1 output exists; choose a new directory (no implicit overwrite)")
    index, summary, manifests = build_artifacts(
        dataset_dir=a.dataset_dir,
        v4_dir=a.v4_dir,
        action_index_file=a.action_index,
        incremental_state_dir=a.incremental_state_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.dry_run:
        return 0
    filenames = {
        "adjustment": "adjustment_end_manifest.jsonl",
        "need": "need_recovery_manifest.jsonl",
        "failure": "failure_reason_manifest.jsonl",
        "plan": "recovery_plan_manifest.jsonl",
    }
    for task, filename in filenames.items():
        path = a.output_dir / filename
        _write_jsonl(path, manifests[task])
        index[f"{task}_manifest_file"] = str(path)
        index[f"{task}_manifest_sha256"] = sha256_file(path)
    index["training_data_hash"] = sha256_json(index)
    index_file = a.output_dir / "sponge_v10_1_multitask_training_index.json"
    write_json(index_file, index)
    validate_index(index)
    summary["training_data_hash"] = index["training_data_hash"]
    write_json(a.output_dir / "summary.json", summary)
    write_json(a.output_dir / "boundary_audit.json", index["boundary_audit"])
    write_json(
        a.output_dir / "artifact_hashes.json", {p.name: sha256_file(p) for p in a.output_dir.iterdir() if p.is_file()}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
