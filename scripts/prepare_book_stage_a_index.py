#!/usr/bin/env python3
"""Build book Stage A H30 phase index from the native V4 rexecution event."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_stage_a_data import build_artifacts, validate_training_index

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book")
DEFAULT_DATASET = ROOT / "lerobot_data/tactile_vla_rotation_v4"
DEFAULT_V4 = ROOT / "outputs/rotation_v4/v4_training_index.json"
DEFAULT_NORM = ROOT / "outputs/rotation_v4/norm_stats"
DEFAULT_OUTPUT = ROOT / "outputs/book_stage_a_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--v4-index-file", type=Path, default=DEFAULT_V4)
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
        norm_stats_dir=args.norm_stats_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0
    out = args.output_dir.expanduser().resolve()
    manifest = out / "action_phase_manifest.jsonl"
    summary_file = out / "filter_summary.json"
    index_file = out / "book_stage_a_training_index.json"
    hashes_file = out / "artifact_hashes.json"
    existing = [path for path in (manifest, summary_file, index_file, hashes_file) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Book Stage A output exists; use --overwrite: {existing[0]}")
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
        "schema_version": "tactile_vla_book_stage_a_v1_artifact_hashes_v1",
        "files": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in (manifest, summary_file, index_file)
        },
    }
    hashes["sha256"] = sha256_json(hashes)
    write_json(hashes_file, hashes)
    validate_training_index(index, index_path=index_file, dataset_dir=args.dataset_dir)
    print(f"wrote book Stage A training index to {index_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
