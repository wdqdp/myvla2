#!/usr/bin/env python3
"""Build Bottle V11.2 five-task manifests using the Stage A idle boundary audit."""

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
from tactile_vla.vla.bottle_v11_2_multitask_data import ROOT, build_artifacts, validate_index


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bottle-root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path)
    p.add_argument("--v4-dir", type=Path)
    p.add_argument("--action-index", type=Path)
    p.add_argument("--incremental-state-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    root = a.bottle_root
    a.dataset_dir = (a.dataset_dir or root / "lerobot_data/tactile_vla_rotation_v4").resolve()
    a.v4_dir = (a.v4_dir or root / "outputs/rotation_v4").resolve()
    a.action_index = (a.action_index or root / "outputs/bottle_stage_a_v11_2/stage_a_training_index.json").resolve()
    a.incremental_state_dir = (a.incremental_state_dir or root / "outputs/incremental_state").resolve()
    a.output_dir = (a.output_dir or root / "outputs/bottle_v11_2_multitask").resolve()
    return a


def main(argv=None):
    a = parse_args(argv)
    if not a.dry_run and a.output_dir.exists():
        raise FileExistsError("V11.2 output exists; choose a new output directory")
    index, summary, manifests = build_artifacts(
        dataset_dir=a.dataset_dir,
        v4_dir=a.v4_dir,
        action_index_file=a.action_index,
        incremental_state_dir=a.incremental_state_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.dry_run:
        return 0
    for task, filename in {
        "adjustment": "adjustment_end_manifest.jsonl",
        "need": "need_recovery_manifest.jsonl",
        "failure": "failure_reason_manifest.jsonl",
        "plan": "recovery_plan_manifest.jsonl",
    }.items():
        path = a.output_dir / filename
        _write_jsonl(path, manifests[task])
        index[f"{task}_manifest_file"] = str(path)
        index[f"{task}_manifest_sha256"] = sha256_file(path)
    index["training_data_hash"] = sha256_json(index)
    write_json(a.output_dir / "bottle_v11_2_multitask_training_index.json", index)
    validate_index(index)
    write_json(a.output_dir / "summary.json", summary | {"training_data_hash": index["training_data_hash"]})
    write_json(a.output_dir / "boundary_audit.json", index["boundary_audit"])
    write_json(a.output_dir / "idle_gap_audit.json", index["idle_gap_audit"])
    write_json(
        a.output_dir / "artifact_hashes.json", {p.name: sha256_file(p) for p in a.output_dir.iterdir() if p.is_file()}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
