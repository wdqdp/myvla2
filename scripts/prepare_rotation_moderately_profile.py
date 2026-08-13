#!/usr/bin/env python3
"""Create all immutable sidecars for rotation_moderately_success_v1."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import action_indices_identity
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.artifacts import sha256_json
from tactile_vla.vla.data_profiles import ROTATION_MODERATELY_SUCCESS_V1
from tactile_vla.vla.data_profiles import build_profile_splits
from tactile_vla.vla.data_profiles import action_attempt_coverage
from tactile_vla.vla.data_profiles import build_single_round_reasoning_samples
from tactile_vla.vla.data_profiles import data_config_hash
from tactile_vla.vla.data_profiles import direction_by_episode
from tactile_vla.vla.data_profiles import profile_config
from tactile_vla.vla.data_profiles import select_profile_records
from tactile_vla.vla.data_profiles import selection_summary
from tactile_vla.vla.data_profiles import validate_expected_action_counts
from tactile_vla.vla.data_profiles import validate_profile_metadata
from tactile_vla.vla.index import scan_lerobot_frames
from tactile_vla.vla.index import v3_index_payload


DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_v3")
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/vla/rotation_moderately_success_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--reasoning-window-frames", type=int, default=15)
    parser.add_argument("--status-negative-ratio", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w") as file:
        for sample_id, sample in enumerate(samples):
            file.write(
                json.dumps(
                    {"sample_id": sample_id, **sample},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _ensure_targets(output_dir: Path, *, overwrite: bool) -> None:
    targets = [
        output_dir / "profile.json",
        output_dir / "splits.json",
        output_dir / "vla_indices_v3.json",
        output_dir / "action_frame_manifest.json",
        output_dir / "selection_summary.json",
        output_dir / "artifact_manifest.json",
        *(output_dir / "reasoning" / f"{split}.jsonl" for split in ("train", "val", "test")),
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Profile artifacts already exist; use --overwrite: {[str(path) for path in existing]}"
        )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.reasoning_window_frames != 15:
        raise ValueError("rotation_moderately_success_v1 requires a 15-frame reasoning window")
    if args.action_horizon != 30:
        raise ValueError("rotation_moderately_success_v1 requires H30 actions")
    _ensure_targets(args.output_dir, overwrite=args.overwrite)

    all_records = scan_lerobot_frames(args.dataset_dir)
    records = select_profile_records(all_records)
    validate_profile_metadata(records)
    action_counts = validate_expected_action_counts(
        records,
        action_horizon=args.action_horizon,
    )
    splits, episode_groups = build_profile_splits(seed=args.seed)
    reasoning = build_single_round_reasoning_samples(records, splits)

    config = profile_config(seed=args.seed, action_horizon=args.action_horizon)
    config_hash = data_config_hash(seed=args.seed, action_horizon=args.action_horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reasoning_dir = args.output_dir / "reasoning"
    reasoning_dir.mkdir(parents=True, exist_ok=True)

    profile_payload = config | {
        "data_config_hash": config_hash,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
    }
    split_payload = {
        "schema_version": "tactile_vla_independent_group_split_v1",
        "split_unit": "original_episode_id",
        "data_profile": ROTATION_MODERATELY_SUCCESS_V1,
        "data_config_hash": config_hash,
        "seed": args.seed,
        "independent_group_random_streams": True,
        "original_episode_ids": splits,
        "episode_groups": episode_groups,
        "counts": {split: len(values) for split, values in splits.items()},
    }
    _write_json(args.output_dir / "profile.json", profile_payload)
    _write_json(args.output_dir / "splits.json", split_payload)

    for split, samples in reasoning.items():
        _write_jsonl(reasoning_dir / f"{split}.jsonl", samples)
    reasoning_summary = {
        "memory_length": 1,
        "donor_episode_ids": [],
        "counts": {split: len(samples) for split, samples in reasoning.items()},
        "directions": {
            split: dict(
                sorted(
                    Counter(
                        direction_by_episode()[
                            int(sample["current_observation"]["episode_id"])
                        ]
                        for sample in samples
                    ).items()
                )
            )
            for split, samples in reasoning.items()
        },
    }
    _write_json(reasoning_dir / "summary.json", reasoning_summary)

    index = v3_index_payload(
        records,
        splits,
        seed=args.seed,
        negative_ratio=args.status_negative_ratio,
        reasoning_window_frames=args.reasoning_window_frames,
        action_horizon=args.action_horizon,
    )
    index.update(
        {
            "data_profile": ROTATION_MODERATELY_SUCCESS_V1,
            "data_config_hash": config_hash,
            "dataset_dir": str(args.dataset_dir),
            "split_file": str(args.output_dir / "splits.json"),
            "reasoning_manifest_dir": str(reasoning_dir),
        }
    )
    index["action_indices_identity"] = action_indices_identity(index["splits"])
    if index["action_indices_identity"]["all"]["count"] != action_counts["all"]:
        raise AssertionError("Index action total differs from the validated profile total")
    _write_json(args.output_dir / "vla_indices_v3.json", index)

    action_manifest = {
        "schema_version": "tactile_vla_action_frame_manifest_v1",
        "data_profile": ROTATION_MODERATELY_SUCCESS_V1,
        "data_config_hash": config_hash,
        "index_file": str(args.output_dir / "vla_indices_v3.json"),
        "indices": index["action_indices_identity"],
        "category_counts": action_counts,
        "attempt_coverage": action_attempt_coverage(records),
    }
    # This is the canonical action-frame identity embedded in every training
    # checkpoint; the surrounding sidecar file also has a file hash in
    # artifact_manifest.json.
    action_manifest["manifest_hash"] = sha256_json(index["action_indices_identity"])
    _write_json(args.output_dir / "action_frame_manifest.json", action_manifest)

    summary = selection_summary(records) | {
        "data_profile": ROTATION_MODERATELY_SUCCESS_V1,
        "data_config_hash": config_hash,
        "split_counts": split_payload["counts"],
        "action_counts": action_counts,
        "reasoning": reasoning_summary,
    }
    _write_json(args.output_dir / "selection_summary.json", summary)

    artifact_paths = [
        "profile.json",
        "splits.json",
        "vla_indices_v3.json",
        "action_frame_manifest.json",
        "selection_summary.json",
        "reasoning/train.jsonl",
        "reasoning/val.jsonl",
        "reasoning/test.jsonl",
        "reasoning/summary.json",
    ]
    artifact_manifest = {
        "schema_version": "tactile_vla_profile_artifacts_v1",
        "data_profile": ROTATION_MODERATELY_SUCCESS_V1,
        "data_config_hash": config_hash,
        "files": {
            relative: sha256_file(args.output_dir / relative)
            for relative in artifact_paths
        },
    }
    _write_json(args.output_dir / "artifact_manifest.json", artifact_manifest)
    return summary


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
