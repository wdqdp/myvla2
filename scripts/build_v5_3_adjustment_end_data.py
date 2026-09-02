#!/usr/bin/env python3
"""Build the immutable V5.3 adjustment-end classification dataset."""

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
from tactile_vla.vla.v5_3_adjustment_end_data import artifact_hash_payload
from tactile_vla.vla.v5_3_adjustment_end_data import build_adjustment_end_artifacts
from tactile_vla.vla.v5_3_adjustment_end_data import validate_adjustment_end_artifacts
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN


DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_rotation_v4")
DEFAULT_V5_2_INDEX = Path(
    "/data1/outputs/vla/rotation_v5_adjustment_v2/v5_prompt_training_index.json"
)
DEFAULT_NORM_STATS = Path("/data1/outputs/vla/assets/tactile_vla_rotation_v4/norm_stats.json")
DEFAULT_CAPTIONER = Path("/data1/outputs/tactile_captioner/tcn_v3_w30_rotation_head/best.pt")
DEFAULT_CAPTION_SUMMARY = Path("/data1/outputs/vla/rotation_v4/caption_annotation_summary.json")
DEFAULT_BACKBONE = Path(
    "/data1/outputs/vla/stage_a_action/pi05_delta_tac_rotation_phase_v5_2_1/15000"
)
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/vla/rotation_v5_adjustment_end_v2")


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
    parser.add_argument("--v5-2-index-file", type=Path, default=DEFAULT_V5_2_INDEX)
    parser.add_argument("--norm-stats-file", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--captioner-checkpoint", type=Path, default=DEFAULT_CAPTIONER)
    parser.add_argument("--caption-summary-file", type=Path, default=DEFAULT_CAPTION_SUMMARY)
    parser.add_argument("--backbone-checkpoint", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--backbone-config-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backbone_config = args.backbone_config_file or args.backbone_checkpoint.parent / "config.json"
    required_files = (
        args.v5_2_index_file,
        args.norm_stats_file,
        args.captioner_checkpoint,
        args.caption_summary_file,
        backbone_config,
        args.backbone_checkpoint / "params" / "_METADATA",
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    tokenizer = PaligemmaTokenizer(max_len=PHASE_CHANGE_MAX_TOKEN_LEN)
    manifest, index, summary = build_adjustment_end_artifacts(
        dataset_dir=args.dataset_dir,
        v5_2_index_file=args.v5_2_index_file,
        norm_stats_file=args.norm_stats_file,
        captioner_checkpoint=args.captioner_checkpoint,
        caption_summary_file=args.caption_summary_file,
        backbone_checkpoint=args.backbone_checkpoint,
        backbone_config_file=backbone_config,
        tokenizer=tokenizer,
    )
    validate_adjustment_end_artifacts(index=index, manifest=manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("dry_run=true; no V5.3 files were written")
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

    summary["training_data_hash"] = index["training_data_hash"]
    summary["manifest_sha256"] = sha256_file(paths["manifest"])
    summary["training_index_sha256"] = sha256_file(paths["index"])
    _write_json(paths["summary"], summary)
    hashes = artifact_hash_payload({name: paths[name] for name in ("manifest", "index", "summary")})
    _write_json(paths["artifact_hashes"], hashes)
    print(f"wrote V5.3 adjustment-end artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
