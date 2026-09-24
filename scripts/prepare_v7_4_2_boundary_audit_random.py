#!/usr/bin/env python3
"""Sample V7.4.2 offline arm boundaries into a V7.2-style review file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_BOUNDARIES = ROOT / "outputs/rotation_v7_4_2_adjustment/adjustment_boundaries.json"
DEFAULT_SPLITS = ROOT / "outputs/rotation_v4_1_new_env/splits.json"
DEFAULT_OUTPUT = ROOT / "outputs/rotation_v7_4_2_adjustment/boundary_audit_random.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_audit(boundaries: dict, splits: dict, *, seed: int, sample_count: int) -> dict:
    if boundaries.get("schema_version") != "tactile_vla_v7_4_2_adjustment_boundaries_v1":
        raise ValueError("Unexpected V7.4.2 boundary schema")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    attempts = boundaries["attempts"]
    by_episode = {int(row["episode_id"]): row for row in attempts}
    if len(by_episode) != len(attempts) or sample_count > len(attempts):
        raise ValueError("Duplicate episodes or sample_count exceeds attempt2 population")
    episode_split = {
        int(episode_id): split
        for split, episode_ids in splits["original_episode_ids"].items()
        for episode_id in episode_ids
    }
    episode_ids = random.Random(seed).sample(sorted(by_episode), sample_count)
    rows = []
    for episode_id in episode_ids:
        attempt = by_episode[episode_id]
        if int(attempt["attempt_id"]) != 2 or episode_id not in episode_split:
            raise ValueError(f"Invalid attempt2 identity for episode {episode_id}")
        events = attempt["events"]
        if set(events) != {"arm_adjustment_start", "arm_adjustment_stop"}:
            raise ValueError(f"Unexpected events for episode {episode_id}")
        start, stop = events["arm_adjustment_start"], events["arm_adjustment_stop"]
        if int(start["frame_index"]) >= int(stop["frame_index"]):
            raise ValueError(f"Invalid arm event order for episode {episode_id}")
        for name, event in (("start", start), ("stop", stop)):
            if set(event) != {"frame_index", "global_index", "timestamp"}:
                raise ValueError(f"Incomplete {name} event for episode {episode_id}")
        rows.append({
            "episode_id": episode_id,
            "attempt_id": 2,
            "split": episode_split[episode_id],
            "arm_adjustment_start": start,
            "arm_adjustment_stop": stop,
            "manual_review_status": "pending",
        })
    return {
        "schema_version": "tactile_vla_v7_4_2_boundary_audit_v1",
        "data_profile": "rotation_phase_v7_4_2_adjustment",
        "seed": seed,
        "sample_count": sample_count,
        "timestamp_field": "ros_timestamp",
        "timestamp_unit": "seconds",
        "event_order_constraints": ["arm_adjustment_start < arm_adjustment_stop"],
        "attempts": rows,
    }


def main() -> int:
    args = parse_args()
    boundaries = json.loads(args.boundaries.read_text())
    splits = json.loads(args.splits.read_text())
    audit = make_audit(boundaries, splits, seed=args.seed, sample_count=args.sample_count)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(f"wrote {len(audit['attempts'])} episodes to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
