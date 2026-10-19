#!/usr/bin/env python3
"""Offline FK distances from compatible V7.7.2/V7.9 adjustment recordings."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from tactile_vla.runtime.adjustment_measurements import SCHEMA
from tactile_vla.vla.v5_adjustment_data import load_piper_fk_chain, piper_fk_positions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument(
        "--boundary",
        choices=("publication", "phase"),
        default="publication",
        help="publication: first adjustment command to first next execution command; phase: enter to exit",
    )
    parser.add_argument("--max-feedback-age-ms", type=float, default=100.0)
    args = parser.parse_args()
    if not np.isfinite(args.max_feedback_age_ms) or args.max_feedback_age_ms <= 0:
        parser.error("--max-feedback-age-ms must be finite and positive")
    return args


def load_records(path):
    rows, issues = [], []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"Malformed JSON at {path}:{index + 1}") from None
            issues.append("Last JSON record is truncated; trial termination was not clean")
    return rows, issues


def validate_point(point, max_age_ms):
    feedback = point.get("feedback")
    if feedback is None:
        raise ValueError("Missing raw joint feedback")
    qpos = np.asarray(feedback.get("qpos", []), dtype=np.float64)
    if qpos.shape != (7,) or not np.isfinite(qpos).all():
        raise ValueError("Feedback qpos must contain seven finite values")
    clocks = np.asarray(
        [point["ros_time"], point["monotonic_time"], feedback["timestamp"], feedback["received_monotonic_time"]],
        dtype=np.float64,
    )
    if not np.isfinite(clocks).all():
        raise ValueError("Feedback or boundary timestamp is invalid")
    ros_age_ms = float((clocks[0] - clocks[2]) * 1000.0)
    receive_age_ms = float((clocks[1] - clocks[3]) * 1000.0)
    if not 0 <= ros_age_ms <= max_age_ms or not 0 <= receive_age_ms <= max_age_ms:
        raise ValueError(f"Stale/future feedback: ROS age={ros_age_ms:.3f}ms, received age={receive_age_ms:.3f}ms")
    return qpos[:6], ros_age_ms


def calculate(trial, boundary, max_age_ms):
    records, issues = load_records(trial / "adjustment_measurements.jsonl")
    metadata_rows = [r for r in records if r.get("event") == "run_start"]
    if len(metadata_rows) != 1 or metadata_rows[0].get("schema_version") != SCHEMA:
        raise ValueError(f"Expected one adjustment measurement run_start with schema {SCHEMA!r}")
    meta = metadata_rows[0]
    if meta.get("no_publish"):
        raise ValueError("This was a --no-publish run; physical distance statistics are not applicable")
    snapshot = Path(meta["urdf_snapshot"])
    if snapshot.name != str(snapshot):
        raise ValueError("URDF snapshot must be a filename inside the trial directory")
    urdf = trial / snapshot
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != meta["urdf_sha256"]:
        raise ValueError("URDF snapshot SHA256 differs from the recorded robot model")
    chain = load_piper_fk_chain(urdf, **meta["fk_reference"])
    by_segment = defaultdict(dict)
    for row in records:
        if row.get("segment_id") is not None:
            events = by_segment[int(row["segment_id"])]
            if row["event"] in events:
                raise ValueError(f"Duplicate {row['event']} for segment {row['segment_id']}")
            events[row["event"]] = row
    if not any(r.get("event") == "run_end" for r in records):
        issues.append("run_end is missing; only segments with both selected boundaries can be measured")
    start_key, end_key = (
        ("first_adjustment_command", "first_execution_command")
        if boundary == "publication"
        else ("adjustment_enter", "adjustment_exit")
    )
    results = []
    for segment, events in sorted(by_segment.items()):
        enter = events.get("adjustment_enter")
        if enter is None:
            issues.append(f"Segment {segment} has no adjustment_enter marker")
            continue
        row = {
            "segment_id": segment,
            "attempt_id": enter["attempt_id"],
            "recovery_plan": enter["recovery_plan"],
            "boundary": boundary,
            "status": "incomplete",
            "reason": "",
        }
        match = re.search(r"move horizontally (\w+) (\w+),", row["recovery_plan"])
        row.update(horizontal_direction=match[1] if match else None, magnitude=match[2] if match else None)
        start, end = events.get(start_key), events.get(end_key)
        if start is None or end is None:
            row["reason"] = f"Missing {start_key if start is None else end_key}"
            results.append(row)
            continue
        try:
            for item in (start, end):
                if item["attempt_id"] != enter["attempt_id"] or item["recovery_plan"] != enter["recovery_plan"]:
                    raise ValueError("Boundary plan/attempt identity mismatch")
            if "adjustment_exit" not in events:
                raise ValueError("No adjustment-to-execution transition")
            if boundary == "publication" and end.get("adjustment_command_recorded") is not True:
                raise ValueError("No actual adjustment command was recorded")
            start_point, end_point = start["point"], end["point"]
            q_start, age_start = validate_point(start_point, max_age_ms)
            q_end, age_end = validate_point(end_point, max_age_ms)
            if start_point["feedback"]["joint_names"] != end_point["feedback"]["joint_names"]:
                raise ValueError("Joint name/order changed between boundaries")
            duration = end_point["monotonic_time"] - start_point["monotonic_time"]
            if duration <= 0 or end_point["feedback"]["timestamp"] < start_point["feedback"]["timestamp"]:
                raise ValueError("Boundary/feedback timestamps are out of order")
            positions = piper_fk_positions(np.stack([q_start, q_end]), chain)
            delta = (positions[1] - positions[0]) * 1000.0
            row.update(
                status="complete",
                duration_seconds=duration,
                start_utc=start_point["utc_time"],
                end_utc=end_point["utc_time"],
                start_ros_time=start_point["ros_time"],
                end_ros_time=end_point["ros_time"],
                start_feedback_timestamp=start_point["feedback"]["timestamp"],
                end_feedback_timestamp=end_point["feedback"]["timestamp"],
                feedback_age_ms={"start": age_start, "end": age_end},
                start_joint_angles_rad=q_start.tolist(),
                end_joint_angles_rad=q_end.tolist(),
                start_position_m=positions[0].tolist(),
                end_position_m=positions[1].tolist(),
                delta_xyz_mm=delta.tolist(),
                dx_mm=float(delta[0]),
                dy_mm=float(delta[1]),
                dz_mm=float(delta[2]),
                net_distance_mm=float(np.linalg.norm(delta)),
                horizontal_xy_net_distance_mm=float(np.linalg.norm(delta[:2])),
            )
        except (ValueError, KeyError, TypeError) as exc:
            row.update(status="invalid", reason=str(exc))
        results.append(row)
    groups = defaultdict(list)
    for row in results:
        if row["status"] == "complete":
            groups[f"{row['horizontal_direction']}/{row['magnitude']}"].append(row)
    statistics = {}
    for key, rows in groups.items():
        stats = {"count": len(rows)}
        for metric in ("net_distance_mm", "horizontal_xy_net_distance_mm"):
            values = [row[metric] for row in rows]
            stats[metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        statistics[key] = stats
    return {
        "schema_version": "v7_7_2_adjustment_distance_summary_v1",
        "boundary": boundary,
        "raw_file": "adjustment_measurements.jsonl",
        "urdf_sha256": meta["urdf_sha256"],
        "reference_frame": chain.base_link,
        "measurement_point": chain.end_link,
        "tcp_offset_applied": False,
        "max_feedback_age_ms": max_age_ms,
        "segments": results,
        "statistics_by_plan": statistics,
        "issues": issues,
    }


def main():
    args = parse_args()
    trial = args.trial_dir.expanduser().resolve()
    summary = calculate(trial, args.boundary, args.max_feedback_age_ms)
    stem = trial / f"adjustment_distances_{args.boundary}"
    stem.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    fields = [
        "segment_id",
        "attempt_id",
        "recovery_plan",
        "boundary",
        "status",
        "reason",
        "horizontal_direction",
        "magnitude",
        "duration_seconds",
        "net_distance_mm",
        "horizontal_xy_net_distance_mm",
        "dx_mm",
        "dy_mm",
        "dz_mm",
        "start_utc",
        "end_utc",
    ]
    with stem.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["segments"])
    for row in summary["segments"]:
        if row["status"] == "complete":
            print(
                f"segment={row['segment_id']} plan={row['horizontal_direction']} {row['magnitude']} "
                f"net={row['net_distance_mm']:.2f}mm horizontal={row['horizontal_xy_net_distance_mm']:.2f}mm "
                f"delta_xyz={row['delta_xyz_mm']} duration={row['duration_seconds']:.2f}s"
            )
        else:
            print(f"segment={row['segment_id']} {row['status']}: {row['reason']}")
    print(f"Saved: {stem.with_suffix('.json')} and {stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
