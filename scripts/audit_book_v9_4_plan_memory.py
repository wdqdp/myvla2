#!/usr/bin/env python3
"""Read-only V9.4 history/token audit, usable before the new Stage A exists."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from tactile_vla.vla.book_v9_4_memory import (
    MEMORY_POLICY,
    expand_plan_rows,
    plan_direction,
    validate_balanced_variants,
    validate_dataset_plan_tokens,
    validate_plan_row,
)
from tactile_vla.vla.book_v9_4_multitask_data import ROOT, load_captioner_provenance, validate_upload_metadata
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt
from tactile_vla.vla.v4_data import load_jsonl, scan_v4_lerobot_frames


def audit(dataset_dir: Path, v4_dir: Path, incremental_state_dir: Path) -> dict:
    profile = json.loads((v4_dir / "profile.json").read_text())
    validate_upload_metadata(dataset_dir, profile)
    identity, _ = load_captioner_provenance(incremental_state_dir, profile=profile)
    frames = {frame.key: frame for frame in scan_v4_lerobot_frames(dataset_dir)}
    plans, failures, sources = [], [], {}
    for split in ("train", "val", "test"):
        for source in load_jsonl(v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl"):
            obs = source["current_observation"]
            frame = frames[obs["episode_id"], obs["attempt_id"], source["frame_index"]]
            if frame.tactile_caption != obs["tactile_caption"]:
                raise ValueError("Source plan caption is not the current real caption")
            row = {
                "split": split,
                "episode_id": frame.episode_id,
                "attempt_id": frame.attempt_id,
                "global_index": frame.global_index,
                "frame_index": frame.frame_index,
                "frame_offset": source["frame_offset"],
                "memory_length": source["memory_length"],
                "failure_recovery_memory": source["failure_recovery_memory"],
                "target_recovery_plan": source["target_recovery_plan"],
                "prompt": build_recovery_prompt(
                    instruction=frame.instruction,
                    failed_tactile_caption=frame.tactile_caption,
                    failure_recovery_memory=source["failure_recovery_memory"],
                    prompt_profile=MINIMAL_PROMPT_PROFILE,
                ),
            }
            key = (split, frame.global_index)
            if key in sources:
                raise ValueError("Duplicate source plan observation")
            plans.append(row)
            failures.append(row | {"target_failure_reason": obs["failure_reason"]})
            sources[key] = source
    expanded = expand_plan_rows(plans, failures, sources=sources)
    validate_balanced_variants(expanded)
    for row in expanded:
        validate_plan_row(row, row["current_real_failure"])
    tokens = validate_dataset_plan_tokens(expanded, dataset_dir=dataset_dir, norm_stats_dir=v4_dir / "norm_stats")
    return {
        "read_only": True,
        "plan_memory_policy": MEMORY_POLICY,
        "captioner_identity": identity,
        "counts": {
            split: {
                "full_manifest": sum(row["split"] == split for row in expanded),
                "selected": sum(
                    row["split"] == split and (split == "train" or row["frame_offset"] == 14) for row in expanded
                ),
                "memory_length_direction": dict(
                    Counter(
                        f"{row['memory_length']}/{plan_direction(row['target_recovery_plan'])}"
                        for row in expanded
                        if row["split"] == split and (split == "train" or row["frame_offset"] == 14)
                    )
                ),
            }
            for split in ("train", "val", "test")
        },
        "plan_token_validation": tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "lerobot_data/tactile_vla_rotation_v4")
    parser.add_argument("--v4-dir", type=Path, default=ROOT / "outputs/rotation_v4")
    parser.add_argument("--incremental-state-dir", type=Path, default=ROOT / "outputs/incremental_state")
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_dir, args.v4_dir, args.incremental_state_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
