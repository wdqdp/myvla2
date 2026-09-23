#!/usr/bin/env python3
"""Build V8.2 Stage-A data with detected small-grasp motion boundaries."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v5_adjustment_data import DEFAULT_PIPER_URDF
from tactile_vla.vla.v8_1_adjustment_data import build_attempt_timing_artifact
from tactile_vla.vla.v8_2_adjustment_data import (
    artifact_hash_payload,
    build_boundary_artifact,
    build_v8_2_adjustment_artifacts,
    validate_v8_2_adjustment_training_index,
)


DEFAULT_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_DATASET_DIR = DEFAULT_ROOT / "lerobot_data/tactile_vla_rotation_v4"
DEFAULT_V4_DIR = DEFAULT_ROOT / "outputs/rotation_v4"
DEFAULT_REFERENCE_V7_2 = (
    DEFAULT_ROOT / "outputs/rotation_v7_2_adjustment/v7_2_boundary_filter.json"
)
DEFAULT_REFERENCE_V7_4 = (
    DEFAULT_ROOT / "outputs/rotation_v7_4_adjustment/v7_4_prompt_training_index.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "outputs/rotation_v8_2_adjustment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--hdf5-dir", type=Path, default=DEFAULT_ROOT / "hdf5")
    parser.add_argument(
        "--v4-index-file", type=Path, default=DEFAULT_V4_DIR / "v4_training_index.json"
    )
    parser.add_argument(
        "--v4-norm-stats-dir", type=Path, default=DEFAULT_V4_DIR / "norm_stats"
    )
    parser.add_argument(
        "--reference-v7-2-boundary-file", type=Path, default=DEFAULT_REFERENCE_V7_2
    )
    parser.add_argument(
        "--reference-v7-4-index-file", type=Path, default=DEFAULT_REFERENCE_V7_4
    )
    parser.add_argument("--piper-urdf", type=Path, default=DEFAULT_PIPER_URDF)
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


def _build(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    timing_path = output_dir / "attempt_timing.json"
    boundary_path = output_dir / "boundary_filter.json"
    manifest_path = output_dir / "action_phase_manifest.jsonl"
    summary_path = output_dir / "filter_summary.json"
    index_path = output_dir / "v8_2_prompt_training_index.json"
    hashes_path = output_dir / "artifact_hashes.json"

    timing = build_attempt_timing_artifact(
        dataset_dir=args.dataset_dir,
        v4_index_file=args.v4_index_file,
        hdf5_dir=args.hdf5_dir,
    )
    _write_json(timing_path, timing)
    boundary = build_boundary_artifact(
        dataset_dir=args.dataset_dir,
        v4_index_file=args.v4_index_file,
        timing_payload=timing,
        reference_v7_2_boundary_file=args.reference_v7_2_boundary_file,
        piper_urdf=args.piper_urdf,
    )
    boundary["source_files"]["attempt_timing"]["path"] = str(timing_path.resolve())
    _write_json(boundary_path, boundary)
    rows, index, summary = build_v8_2_adjustment_artifacts(
        dataset_dir=args.dataset_dir,
        v4_index_file=args.v4_index_file,
        v4_norm_stats_dir=args.v4_norm_stats_dir,
        timing_file=timing_path,
        boundary_file=boundary_path,
        reference_v7_4_index_file=args.reference_v7_4_index_file,
    )
    _write_jsonl(manifest_path, rows)
    _write_json(summary_path, summary)
    index["source_files"].update(
        {
            "action_phase_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "filter_summary": {
                "path": str(summary_path.resolve()),
                "sha256": sha256_file(summary_path),
            },
        }
    )
    index["action_phase_manifest_identity"]["file_sha256"] = sha256_file(manifest_path)
    index["training_data_hash"] = sha256_json(index)
    _write_json(index_path, index)
    _write_json(
        hashes_path,
        artifact_hash_payload(
            {
                "attempt_timing": timing_path,
                "boundary_filter": boundary_path,
                "action_phase_manifest": manifest_path,
                "filter_summary": summary_path,
                "training_index": index_path,
                "reference_v7_2_boundary": args.reference_v7_2_boundary_file,
                "reference_v7_4_training_index": args.reference_v7_4_index_file,
                "piper_urdf": args.piper_urdf,
            }
        ),
    )
    validate_v8_2_adjustment_training_index(
        index, index_path=index_path, dataset_dir=args.dataset_dir
    )
    paths = {
        "attempt_timing": timing_path,
        "boundary_filter": boundary_path,
        "action_phase_manifest": manifest_path,
        "filter_summary": summary_path,
        "training_index": index_path,
        "artifact_hashes": hashes_path,
    }
    return paths, summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    expected = [
        output_dir / name
        for name in (
            "attempt_timing.json",
            "boundary_filter.json",
            "action_phase_manifest.jsonl",
            "filter_summary.json",
            "v8_2_prompt_training_index.json",
            "artifact_hashes.json",
        )
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output exists; use --overwrite: {existing[0]}")

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="v8_2_adjustment_") as temporary:
            _, summary = _build(args, Path(temporary))
        print("dry_run=true; no V8.2 artifacts were retained")
    else:
        paths, summary = _build(args, output_dir)
        print(f"wrote V8.2 adjustment artifacts to {output_dir}")
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    print(
        json.dumps(
            {
                "candidate_action_count": summary["candidate_action_count"],
                "trainable_action_count": summary["trainable_action_count"],
                "excluded_action_count": summary["excluded_action_count"],
                "exclusion_reason_counts": summary["exclusion_reason_counts"],
                "h30_modifications": summary["h30_modifications"],
                "splits": summary["splits"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
