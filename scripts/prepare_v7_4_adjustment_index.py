#!/usr/bin/env python3
"""Build the V7.4 idle-compressed H30 training index from V7.3."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_4_adjustment_data import artifact_hash_payload
from tactile_vla.vla.v7_4_adjustment_data import build_v7_4_adjustment_artifacts
from tactile_vla.vla.v7_4_adjustment_data import validate_v7_4_adjustment_training_index


DEFAULT_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_DATASET_DIR = DEFAULT_ROOT / "lerobot_data/tactile_vla_rotation_v4"
DEFAULT_V7_3_DIR = DEFAULT_ROOT / "outputs/rotation_v7_3_adjustment"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "outputs/rotation_v7_4_adjustment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--v7-3-index-file",
        type=Path,
        default=DEFAULT_V7_3_DIR / "v7_3_prompt_training_index.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    rows, index, summary = build_v7_4_adjustment_artifacts(
        v7_3_index_file=args.v7_3_index_file,
        dataset_dir=args.dataset_dir,
    )
    print(
        json.dumps(
            {
                "candidate_action_count": summary["candidate_action_count"],
                "trainable_action_count": summary["trainable_action_count"],
                "excluded_action_count": summary["excluded_action_count"],
                "h30_modifications": summary["h30_modifications"],
            },
            indent=2,
        )
    )
    if args.dry_run:
        print("dry_run=true; no V7.4 training artifacts were written")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "action_phase_manifest.jsonl"
    summary_path = output_dir / "filter_summary.json"
    index_path = output_dir / "v7_4_prompt_training_index.json"
    hashes_path = output_dir / "artifact_hashes.json"
    output_paths = (manifest_path, summary_path, index_path, hashes_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {existing[0]}")

    _write_jsonl(manifest_path, rows)
    _write_json(summary_path, summary)
    index["source_files"]["action_phase_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    index["source_files"]["filter_summary"] = {
        "path": str(summary_path),
        "sha256": sha256_file(summary_path),
    }
    index["action_phase_manifest_identity"]["file_sha256"] = sha256_file(manifest_path)
    index["training_data_hash"] = sha256_json(index)
    _write_json(index_path, index)
    _write_json(
        hashes_path,
        artifact_hash_payload(
            {
                "action_phase_manifest": manifest_path,
                "filter_summary": summary_path,
                "source_v7_3_training_index": args.v7_3_index_file,
                "training_index": index_path,
            }
        ),
    )
    validate_v7_4_adjustment_training_index(
        index,
        index_path=index_path,
        dataset_dir=args.dataset_dir,
    )
    print(f"wrote V7.4 adjustment artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
