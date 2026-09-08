#!/usr/bin/env python3
"""Build V7 phase labels from native V4 rexecution_frame_index values."""

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
from tactile_vla.vla.v7_adjustment_data import build_v7_adjustment_artifacts
from tactile_vla.vla.v7_adjustment_data import validate_v7_adjustment_training_index


DEFAULT_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_DATASET_DIR = DEFAULT_ROOT / "lerobot_data/tactile_vla_rotation_v4"
DEFAULT_V4_DIR = DEFAULT_ROOT / "outputs/rotation_v4"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "outputs/rotation_v7_adjustment"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--v4-index-file", type=Path, default=DEFAULT_V4_DIR / "v4_training_index.json")
    parser.add_argument("--v4-norm-stats-dir", type=Path, default=DEFAULT_V4_DIR / "norm_stats")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-h30-targets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    actions, index = build_v7_adjustment_artifacts(
        dataset_dir=args.dataset_dir,
        v4_index_file=args.v4_index_file,
        v4_norm_stats_dir=args.v4_norm_stats_dir,
    )
    print(json.dumps(index["summary"] | {"h30_modifications": index["h30_modifications"]}, indent=2))
    if args.dry_run:
        print("dry_run=true; no V7 files were written")
        return 0

    manifest_path = args.output_dir / "action_phase_manifest.jsonl"
    index_path = args.output_dir / "v7_prompt_training_index.json"
    existing = [path for path in (manifest_path, index_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {existing[0]}")
    _write_jsonl(manifest_path, actions)
    index["source_files"]["action_phase_manifest"] = {
        "path": str(manifest_path.expanduser().resolve()),
        "sha256": sha256_file(manifest_path),
    }
    index["action_phase_manifest_identity"]["file_sha256"] = sha256_file(manifest_path)
    index["training_data_hash"] = sha256_json(index)
    _write_json(index_path, index)
    validate_v7_adjustment_training_index(
        index,
        index_path=index_path,
        dataset_dir=args.dataset_dir,
        revalidate_h30_targets=args.verify_h30_targets,
    )
    print(f"wrote V7 adjustment artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
