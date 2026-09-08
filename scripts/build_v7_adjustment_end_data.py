#!/usr/bin/env python3
"""Build the immutable V7 native-R adjustment-end dataset."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/src"))

from openpi.models.tokenizer import PaligemmaTokenizer
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v7_adjustment_end_data import artifact_hash_payload
from tactile_vla.vla.v7_adjustment_end_data import build_adjustment_end_artifacts
from tactile_vla.vla.v7_adjustment_end_data import validate_adjustment_end_artifacts


DEFAULT_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_DATASET_DIR = DEFAULT_ROOT / "lerobot_data/tactile_vla_rotation_v4"
DEFAULT_V4_DIR = DEFAULT_ROOT / "outputs/rotation_v4"
DEFAULT_ACTION_INDEX = DEFAULT_ROOT / "outputs/rotation_v7_adjustment/v7_prompt_training_index.json"
DEFAULT_STAGE_A = DEFAULT_ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_no_history/15000"
DEFAULT_OUTPUT = DEFAULT_ROOT / "outputs/rotation_v7_adjustment_end_r10_r0"


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--v4-index-file", type=Path, default=DEFAULT_V4_DIR / "v4_training_index.json")
    parser.add_argument("--action-index-file", type=Path, default=DEFAULT_ACTION_INDEX)
    parser.add_argument("--norm-stats-file", type=Path, default=DEFAULT_V4_DIR / "norm_stats/norm_stats.json")
    parser.add_argument("--caption-summary-file", type=Path, default=DEFAULT_V4_DIR / "caption_annotation_summary.json")
    parser.add_argument("--stage-a-checkpoint", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--stage-a-config-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_a_config = args.stage_a_config_file or args.stage_a_checkpoint.parent / "config.json"
    required = (
        args.v4_index_file,
        args.action_index_file,
        args.norm_stats_file,
        args.caption_summary_file,
        stage_a_config,
        args.stage_a_checkpoint / "params" / "_METADATA",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    tokenizer = PaligemmaTokenizer(max_len=PHASE_CHANGE_MAX_TOKEN_LEN)
    manifest, index, summary = build_adjustment_end_artifacts(
        dataset_dir=args.dataset_dir,
        v4_index_file=args.v4_index_file,
        action_index_file=args.action_index_file,
        norm_stats_file=args.norm_stats_file,
        caption_summary_file=args.caption_summary_file,
        stage_a_checkpoint=args.stage_a_checkpoint,
        stage_a_config_file=stage_a_config,
        tokenizer=tokenizer,
    )
    validate_adjustment_end_artifacts(index=index, manifest=manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("dry_run=true; no V7 adjustment_end files were written")
        return 0

    paths = {
        "manifest": args.output_dir / "adjustment_end_manifest.jsonl",
        "index": args.output_dir / "adjustment_end_training_index.json",
        "summary": args.output_dir / "adjustment_end_summary.json",
        "artifact_hashes": args.output_dir / "artifact_hashes.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {existing[0]}")
    _write_jsonl(paths["manifest"], manifest)
    index["manifest_identity"]["file_sha256"] = sha256_file(paths["manifest"])
    index["manifest_file"] = str(paths["manifest"].resolve())
    index["training_data_hash"] = sha256_json(
        {key: value for key, value in index.items() if key != "training_data_hash"}
    )
    validate_adjustment_end_artifacts(index=index, manifest=manifest)
    _write_json(paths["index"], index)
    summary.update({
        "training_data_hash": index["training_data_hash"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "training_index_sha256": sha256_file(paths["index"]),
    })
    _write_json(paths["summary"], summary)
    _write_json(paths["artifact_hashes"], artifact_hash_payload({
        name: paths[name] for name in ("manifest", "index", "summary")
    }))
    print(f"wrote V7 adjustment_end artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
