#!/usr/bin/env python3
"""Build the V7.4.2 phase-prompt H30 index from new-environment V4 data."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_4_2_adjustment_data import build_artifacts, validate_training_index

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_DATASET = ROOT / "lerobot_data/tactile_vla_rotation_v4_1_new_env"
DEFAULT_V4 = ROOT / "outputs/rotation_v4_1_new_env/v4_training_index.json"
DEFAULT_BOUNDARY = ROOT / "outputs/rotation_v7_4_2_adjustment/adjustment_boundaries.json"
DEFAULT_NORM = ROOT / "outputs/rotation_v4_1_new_env/norm_stats"
DEFAULT_OUTPUT = ROOT / "outputs/rotation_v7_4_2_adjustment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--v4-index-file", type=Path, default=DEFAULT_V4)
    parser.add_argument("--boundary-file", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    rows, index, summary = build_artifacts(
        dataset_dir=args.dataset_dir,
        v4_index_file=args.v4_index_file,
        boundary_file=args.boundary_file,
        norm_stats_dir=args.norm_stats_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0
    out = args.output_dir.expanduser().resolve()
    manifest = out / "action_phase_manifest.jsonl"
    summary_file = out / "filter_summary.json"
    index_file = out / "v7_4_2_prompt_training_index.json"
    hashes_file = out / "artifact_hashes.json"
    outputs = (manifest, summary_file, index_file, hashes_file)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"V7.4.2 output exists; use --overwrite: {existing[0]}")
    out.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(manifest)
    write_json(summary_file, summary)
    index["source_files"]["action_phase_manifest"] = {
        "path": str(manifest), "sha256": sha256_file(manifest),
    }
    index["action_phase_manifest_identity"]["file_sha256"] = sha256_file(manifest)
    index["training_data_hash"] = sha256_json(index)
    write_json(index_file, index)
    hashes = {
        "schema_version": "tactile_vla_v7_4_2_adjustment_artifact_hashes_v1",
        "files": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in (manifest, summary_file, index_file)
        },
    }
    hashes["sha256"] = sha256_json(hashes)
    write_json(hashes_file, hashes)
    validate_training_index(index, index_path=index_file, dataset_dir=args.dataset_dir)
    print(f"wrote V7.4.2 training index to {index_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
