#!/usr/bin/env python3
"""Build the factual book V9.2 adjustment-end manifest and index."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/src"))

from openpi.models.tokenizer import PaligemmaTokenizer
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_2_adjustment_end_data import build_artifacts, load_indexed_manifest_rows
from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "lerobot_data/tactile_vla_rotation_v4")
    parser.add_argument("--v4-index-file", type=Path, default=ROOT / "outputs/rotation_v4/v4_training_index.json")
    parser.add_argument("--action-index-file", type=Path, default=ROOT / "outputs/book_stage_a_v1/book_stage_a_training_index.json")
    parser.add_argument("--norm-stats-file", type=Path, default=ROOT / "outputs/rotation_v4/norm_stats/norm_stats.json")
    parser.add_argument("--stage-a-checkpoint", type=Path, default=ROOT / "outputs/stage_a_action/pi05_delta_tac_book_stage_a_v1_no_history/15000")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/book_adjustment_end_v9_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Book V9.2 fixes the reproducible sampling seed to 42")
    tokenizer = PaligemmaTokenizer(max_len=PHASE_CHANGE_MAX_TOKEN_LEN)
    manifest, index, summary = build_artifacts(
        dataset_dir=args.dataset_dir, v4_index_file=args.v4_index_file,
        action_index_file=args.action_index_file, norm_stats_file=args.norm_stats_file,
        stage_a_checkpoint=args.stage_a_checkpoint, tokenizer=tokenizer, seed=args.seed,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "splits"}, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("dry_run=true; no book V9.2 artifacts were written")
        return 0
    output = args.output_dir.expanduser().resolve()
    paths = {
        "manifest": output / "adjustment_end_manifest.jsonl",
        "index": output / "adjustment_end_training_index.json",
        "summary": output / "adjustment_end_summary.json",
        "hashes": output / "artifact_hashes.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Book V9.2 output exists; use --overwrite: {existing[0]}")
    output.mkdir(parents=True, exist_ok=True)
    temporary = paths["manifest"].with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in manifest:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(paths["manifest"])
    index["manifest_identity"]["file_sha256"] = sha256_file(paths["manifest"])
    index["manifest_file"] = str(paths["manifest"])
    index["training_data_hash"] = sha256_json(index)
    _write_json(paths["index"], index)
    load_indexed_manifest_rows(index=index, manifest_path=paths["manifest"])
    summary.update({
        "training_data_hash": index["training_data_hash"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "training_index_sha256": sha256_file(paths["index"]),
    })
    _write_json(paths["summary"], summary)
    _write_json(paths["hashes"], {
        "schema_version": "book_v9_2_adjustment_end_artifact_hashes_v1",
        "files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items() if name != "hashes"},
    })
    print(f"wrote book V9.2 artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
