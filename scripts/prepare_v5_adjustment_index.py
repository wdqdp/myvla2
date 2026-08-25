#!/usr/bin/env python3
"""Build V5.2 two-phase sidecars and terminal-held H30 identities on V4."""

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
from tactile_vla.vla.v5_adjustment_data import DEFAULT_PIPER_URDF
from tactile_vla.vla.v5_adjustment_data import build_v5_adjustment_artifacts
from tactile_vla.vla.v5_adjustment_data import validate_v5_adjustment_training_index
from tactile_vla.vla.v5_phase_data import EXPECTED_ATTEMPT_COUNT, EXPECTED_EPISODE_COUNT


DEFAULT_V4_DIR = Path("/data1/outputs/vla/rotation_v4")
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/vla/rotation_v5_adjustment_v2")
DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_rotation_v4")
DEFAULT_V4_NORM_DIR = Path("/data1/outputs/vla/assets/tactile_vla_rotation_v4")
DEFAULT_OVERRIDES = (
    PROJECT_ROOT / "configs/rotation_v5_adjustment_v2/phase_boundary_overrides.json"
)


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
    parser.add_argument("--overrides-file", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--v4-norm-stats-dir", type=Path, default=DEFAULT_V4_NORM_DIR)
    parser.add_argument("--piper-urdf", type=Path, default=DEFAULT_PIPER_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-episode-count", type=int, default=EXPECTED_EPISODE_COUNT)
    parser.add_argument("--expected-attempt-count", type=int, default=EXPECTED_ATTEMPT_COUNT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-raw-streams", action="store_true")
    parser.add_argument("--verify-h30-targets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    boundaries, actions, index, summary = build_v5_adjustment_artifacts(
        dataset_dir=args.dataset_dir,
        v4_index_file=args.v4_index_file,
        overrides_file=args.overrides_file,
        v4_norm_stats_dir=args.v4_norm_stats_dir,
        piper_urdf=args.piper_urdf,
        expected_episode_count=args.expected_episode_count,
        expected_attempt_count=args.expected_attempt_count,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("dry_run=true; no V5.2 files were written")
        return 0

    output_paths = {
        "phase_boundaries": args.output_dir / "phase_boundaries.jsonl",
        "action_phase_manifest": args.output_dir / "action_phase_manifest.jsonl",
        "phase_boundary_overrides": args.output_dir / "phase_boundary_overrides.json",
        "summary": args.output_dir / "phase_label_summary.json",
        "index": args.output_dir / "v5_prompt_training_index.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {existing[0]}")

    override_payload = json.loads(args.overrides_file.read_text())
    _write_json(output_paths["phase_boundary_overrides"], override_payload)
    _write_jsonl(output_paths["phase_boundaries"], boundaries)
    _write_jsonl(output_paths["action_phase_manifest"], actions)
    for name in ("phase_boundaries", "action_phase_manifest", "phase_boundary_overrides"):
        path = output_paths[name]
        index["source_files"][name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    index["phase_boundaries_identity"]["file_sha256"] = sha256_file(
        output_paths["phase_boundaries"]
    )
    index["action_phase_manifest_identity"]["file_sha256"] = sha256_file(
        output_paths["action_phase_manifest"]
    )
    index["training_data_hash"] = sha256_json(index)
    _write_json(output_paths["index"], index)

    validate_v5_adjustment_training_index(
        index,
        index_path=output_paths["index"],
        dataset_dir=args.dataset_dir,
        revalidate_raw_streams=args.verify_raw_streams,
        revalidate_h30_targets=args.verify_h30_targets,
    )
    summary.update(
        {
            "training_data_hash": index["training_data_hash"],
            "v5_prompt_training_index_sha256": sha256_file(output_paths["index"]),
            "phase_boundaries_sha256": sha256_file(output_paths["phase_boundaries"]),
            "action_phase_manifest_sha256": sha256_file(output_paths["action_phase_manifest"]),
            "phase_boundary_overrides_sha256": sha256_file(
                output_paths["phase_boundary_overrides"]
            ),
            "strict_revalidation": {
                "raw_streams": bool(args.verify_raw_streams),
                "h30_targets": bool(args.verify_h30_targets),
            },
        }
    )
    _write_json(output_paths["summary"], summary)
    print(f"wrote V5.2 adjustment artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
