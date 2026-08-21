#!/usr/bin/env python3
"""Join V4 profile/manifests to exact LeRobot global frame indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v4_data import ROTATION_V4
from tactile_vla.vla.v4_data import SPLITS
from tactile_vla.vla.v4_data import V4_TRAINING_INDEX_SCHEMA
from tactile_vla.vla.v4_data import build_need_rows
from tactile_vla.vla.v4_data import file_identity
from tactile_vla.vla.v4_data import load_jsonl
from tactile_vla.vla.v4_data import load_v4_sources
from tactile_vla.vla.v4_data import map_action_manifest
from tactile_vla.vla.v4_data import scan_v4_lerobot_frames
from tactile_vla.vla.v4_data import v4_action_identity
from tactile_vla.vla.v4_data import validate_direct_manifest_rows
from tactile_vla.vla.v4_data import validate_v4_lerobot_frames


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_index(
    *,
    dataset_dir: Path,
    selection_file: Path,
    profile_file: Path,
    split_file: Path,
    action_manifest_file: Path,
    reasoning_manifest_dir: Path,
    output_dir: Path,
    seed: int = 42,
    negative_ratio: float = 3.0,
    failure_window_length: int = 15,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selection, profile, split_payload = load_v4_sources(
        selection_file=selection_file,
        profile_file=profile_file,
        split_file=split_file,
    )
    if int(profile["config"]["failure_window_length"]) != failure_window_length:
        raise ValueError("Requested failure window differs from V4 profile")
    frames = scan_v4_lerobot_frames(dataset_dir)
    frame_lookup, by_attempt = validate_v4_lerobot_frames(frames, profile)
    action_rows = load_jsonl(action_manifest_file)
    action_splits = map_action_manifest(
        action_rows,
        frame_lookup=frame_lookup,
        profile=profile,
        splits=split_payload,
    )
    need_rows, need_summary = build_need_rows(
        profile=profile,
        splits=split_payload,
        by_attempt=by_attempt,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    profile_attempts = {
        (int(row["episode_id"]), int(row["attempt_id"])): row for row in profile["attempts"]
    }

    split_entries: dict[str, Any] = {}
    source_files: dict[str, Any] = {
        "selection": file_identity(selection_file),
        "profile": file_identity(profile_file),
        "splits": file_identity(split_file),
        "action_frame_manifest": file_identity(action_manifest_file),
        "lerobot_parquet": {
            path.relative_to(dataset_dir).as_posix(): sha256_file(path)
            for path in sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
        },
    }
    reasoning_summary_path = reasoning_manifest_dir / "summary.json"
    reasoning_summary = json.loads(reasoning_summary_path.read_text())
    if (
        reasoning_summary.get("schema_version") != "tactile_vla_v4_synthetic_reasoning_v1"
        or reasoning_summary.get("profile_config_hash") != profile["profile_config_hash"]
        or int(reasoning_summary.get("failure_window_length", -1)) != failure_window_length
    ):
        raise ValueError("V4 reasoning summary does not match profile/window")
    source_files["reasoning_summary"] = file_identity(reasoning_summary_path)
    for split in SPLITS:
        failure_path = reasoning_manifest_dir / "failure_reason" / f"{split}.jsonl"
        reasoning_path = reasoning_manifest_dir / "reasoning" / f"{split}.jsonl"
        failure_rows = validate_direct_manifest_rows(
            load_jsonl(failure_path),
            split=split,
            task="failure",
            frame_lookup=frame_lookup,
            profile_attempts=profile_attempts,
            failure_window_length=failure_window_length,
        )
        plan_rows = validate_direct_manifest_rows(
            load_jsonl(reasoning_path),
            split=split,
            task="plan",
            frame_lookup=frame_lookup,
            profile_attempts=profile_attempts,
            failure_window_length=failure_window_length,
        )
        failure_row_indices = (
            list(range(len(failure_rows)))
            if split == "train"
            else [index for index, row in enumerate(failure_rows) if int(row["frame_offset"]) == failure_window_length - 1]
        )
        plan_row_indices = (
            list(range(len(plan_rows)))
            if split == "train"
            else [index for index, row in enumerate(plan_rows) if int(row["frame_offset"]) == failure_window_length - 1]
        )
        failure_selected = [failure_rows[index] for index in failure_row_indices]
        plan_selected = [plan_rows[index] for index in plan_row_indices]
        if not failure_selected or not plan_selected:
            raise ValueError(f"V4 {split} failure/plan evaluation stream is empty")
        need_path = output_dir / "need" / f"{split}.jsonl"
        split_entries[split] = {
            "execution_indices": action_splits[split],
            "status_indices": [int(row["global_index"]) for row in need_rows[split]],
            "status_manifest_row_indices": list(range(len(need_rows[split]))),
            "status_manifest_file": str(need_path.resolve()),
            "status_manifest_sha256": sha256_json(need_rows[split]),
            "failure_reason_indices": [int(row["global_index"]) for row in failure_selected],
            "failure_reason_manifest_file": str(failure_path.resolve()),
            "failure_reason_manifest_sha256": sha256_file(failure_path),
            "failure_reason_manifest_row_indices": failure_row_indices,
            "reasoning_indices": [int(row["global_index"]) for row in plan_selected],
            "reasoning_manifest_file": str(reasoning_path.resolve()),
            "reasoning_manifest_sha256": sha256_file(reasoning_path),
            "reasoning_manifest_row_indices": plan_row_indices,
            "summary": {
                "action": len(action_splits[split]),
                "need": len(need_rows[split]),
                "failure": len(failure_selected),
                "plan": len(plan_selected),
                "failure_eval_last_frame_only": split != "train",
                "plan_eval_last_frame_only": split != "train",
            },
        }
        source_files[f"failure_reason_{split}"] = file_identity(failure_path)
        source_files[f"reasoning_{split}"] = file_identity(reasoning_path)

    action_identity = v4_action_identity(split_entries)
    payload: dict[str, Any] = {
        "schema_version": V4_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_V4,
        "profile_config_hash": profile["profile_config_hash"],
        "data_config_hash": profile["profile_config_hash"],
        "selection_hash": selection["selection_hash"],
        "dataset_dir": str(dataset_dir.resolve()),
        "seed": seed,
        "status_negative_ratio": negative_ratio,
        "action_horizon": int(profile["config"]["action_horizon"]),
        "failure_window_length": failure_window_length,
        "lerobot_identity": {
            "frame_count": len(frames),
            "attempt_count": len(by_attempt),
            "frame_key_sha256": sha256_json(
                [[frame.episode_id, frame.attempt_id, frame.frame_index, frame.global_index] for frame in frames]
            ),
        },
        "source_files": source_files,
        "splits": split_entries,
        "action_indices_identity": action_identity,
        "need_identity": {
            split: {"count": len(need_rows[split]), "sha256": sha256_json(need_rows[split])}
            for split in SPLITS
        },
        "failure_manifest_identity": {
            split: {
                "count": len(split_entries[split]["failure_reason_indices"]),
                "sha256": split_entries[split]["failure_reason_manifest_sha256"],
            }
            for split in SPLITS
        },
        "reasoning_manifest_identity": {
            split: {
                "count": len(split_entries[split]["reasoning_indices"]),
                "sha256": split_entries[split]["reasoning_manifest_sha256"],
            }
            for split in SPLITS
        },
    }
    payload["training_data_hash"] = sha256_json(
        {
            "schema_version": payload["schema_version"],
            "profile_config_hash": payload["profile_config_hash"],
            "selection_hash": payload["selection_hash"],
            "source_files": source_files,
            "action_indices_identity": action_identity,
            "need_identity": payload["need_identity"],
            "failure_manifest_identity": payload["failure_manifest_identity"],
            "reasoning_manifest_identity": payload["reasoning_manifest_identity"],
            "lerobot_identity": payload["lerobot_identity"],
        }
    )
    summary = {
        "schema_version": V4_TRAINING_INDEX_SCHEMA,
        "training_data_hash": payload["training_data_hash"],
        "profile_config_hash": payload["profile_config_hash"],
        "selection_hash": payload["selection_hash"],
        "action_indices_identity": action_identity,
        "need": need_summary,
        "splits": {split: split_entries[split]["summary"] for split in SPLITS},
    }
    return payload, need_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--profile-file", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--action-manifest-file", type=Path, required=True)
    parser.add_argument("--reasoning-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--status-negative-ratio", type=float, default=3.0)
    parser.add_argument("--failure-window-length", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index, need_rows, summary = build_index(
        dataset_dir=args.dataset_dir,
        selection_file=args.selection_file,
        profile_file=args.profile_file,
        split_file=args.split_file,
        action_manifest_file=args.action_manifest_file,
        reasoning_manifest_dir=args.reasoning_manifest_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        negative_ratio=args.status_negative_ratio,
        failure_window_length=args.failure_window_length,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("dry_run=true; no V4 training-index files were written")
        return 0
    outputs = [
        args.output_dir / "v4_training_index.json",
        args.output_dir / "training_index_summary.json",
        *(args.output_dir / "need" / f"{split}.jsonl" for split in SPLITS),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {existing[0]}")
    for split in SPLITS:
        _write_jsonl(args.output_dir / "need" / f"{split}.jsonl", need_rows[split])
    # Hash actual persisted need files and bind the final index to them.
    for split in SPLITS:
        path = args.output_dir / "need" / f"{split}.jsonl"
        index["splits"][split]["status_manifest_sha256"] = sha256_file(path)
        index["source_files"][f"need_{split}"] = file_identity(path)
    index["training_data_hash"] = sha256_json(
        {key: value for key, value in index.items() if key != "training_data_hash"}
    )
    summary["training_data_hash"] = index["training_data_hash"]
    _write_json(args.output_dir / "v4_training_index.json", index)
    _write_json(args.output_dir / "training_index_summary.json", summary)
    print(f"wrote V4 training index to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
