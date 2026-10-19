#!/usr/bin/env python3
"""Offline failure lift heights and adjustment net distances from V7.7.2 recordings."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np

from scripts.calculate_adjustment_distances import calculate, load_records, validate_point
from tactile_vla.vla.v5_adjustment_data import load_piper_fk_chain, piper_fk_positions


def parse_args(*, description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--max-feedback-age-ms", type=float, default=100.0)
    parser.add_argument("--height-baseline", choices=("marker", "execution_start"), default="marker",
                        help="marker: latest b keypoint before detection; execution_start: explicit alternative")
    parser.add_argument("--lift-starts-file", type=Path,
                        help='Offline baseline selection: JSON mapping attempt IDs to ROS feedback timestamps')
    args = parser.parse_args()
    if not np.isfinite(args.max_feedback_age_ms) or args.max_feedback_age_ms <= 0:
        parser.error("--max-feedback-age-ms must be finite and positive")
    return args


def calculate_heights(trial, max_age_ms, *, height_baseline="marker", lift_starts=None):
    # Reuse boundary, schema, no-publish and URDF digest validation.
    adjustments = calculate(trial, "publication", max_age_ms)
    records, issues = load_records(trial / "adjustment_measurements.jsonl")
    meta = next(row for row in records if row.get("event") == "run_start")
    recorded_args = meta.get("args", {})
    if recorded_args.get("synchronous_phase_checks"):
        mode = "synchronous"
    elif recorded_args.get("asynchronous_lift_measurements"):
        mode = "asynchronous"
    else:
        raise ValueError("Expected height-enabled recording with b/execution/fault keypoints; old boundary-only logs are unsupported")
    chain = load_piper_fk_chain(trial / meta["urdf_snapshot"], **meta["fk_reference"])

    def position(point):
        qpos, _ = validate_point(point, max_age_ms)
        return piper_fk_positions(qpos[None, :], chain)[0] * 1000.0

    starts, plans, feedback, markers = {}, {}, [], []
    baseline = None
    for record in records:
        kind = record.get("event")
        if kind == "execution_start":
            attempt = int(record["attempt_id"])
            if attempt in starts:
                raise ValueError(f"Duplicate execution_start for attempt {attempt}")
            starts[attempt] = record
        elif kind == "trial_baseline":
            if baseline is not None:
                raise ValueError("Duplicate trial_baseline")
            baseline = record
        elif kind == "plan":
            plans[record.get("source_attempt_id")] = record
        elif kind == "action_feedback" and record.get("phase") == "execution":
            feedback.append(record)
        elif kind == "lift_baseline":
            markers.append(record)
    # Export FK points so a physical lift onset can also be annotated offline.
    trajectory = []
    for record in records:
        if record.get("event") not in {"action_feedback", "execution_start", "trial_baseline",
                                       "lift_baseline", "fault_detected"}:
            continue
        point = record["point"]
        sample = {"event": record["event"], "attempt_id": record.get("attempt_id"),
                  "phase": record.get("phase"), "published_steps": record.get("published_steps"),
                  "utc_time": point["utc_time"], "monotonic_time": point["monotonic_time"],
                  "status": "invalid", "reason": ""}
        try:
            xyz = position(point)
            sample.update(status="complete", feedback_timestamp=point["feedback"]["timestamp"],
                          x_mm=float(xyz[0]), y_mm=float(xyz[1]), z_mm=float(xyz[2]))
        except (ValueError, TypeError, KeyError) as exc:
            sample["reason"] = str(exc)
        trajectory.append(sample)
    point_records = [r for r in records if r.get("event") in {
        "action_feedback", "execution_start", "trial_baseline", "lift_baseline", "fault_detected",
    }]
    by_attempt = {row["attempt_id"]: row for row in adjustments["segments"]}
    failures, seen_attempts = [], set()
    for record in records:
        if record.get("event") != "fault_detected":
            continue
        attempt = int(record["attempt_id"])
        if attempt in seen_attempts:
            raise ValueError(f"Duplicate fault_detected for attempt {attempt}")
        seen_attempts.add(attempt)
        plan = plans.get(attempt, {})
        adjustment = by_attempt.get(attempt + 1)
        row = {
            "attempt_id": attempt, "request_id": record["request_id"], "status": "invalid", "reason": "",
            "failure_reason": plan.get("failure_reason"), "recovery_plan": plan.get("recovery_plan"),
            "adjustment_status": adjustment["status"] if adjustment else "not_started",
        }
        if adjustment:
            row["adjustment_reason"] = adjustment["reason"]
            row["adjustment_segment_id"] = adjustment["segment_id"]
            if adjustment["status"] == "complete":
                row.update(
                    adjustment_horizontal_xy_net_distance_mm=adjustment["horizontal_xy_net_distance_mm"],
                    adjustment_net_distance_mm=adjustment["net_distance_mm"],
                    adjustment_dx_mm=adjustment["dx_mm"], adjustment_dy_mm=adjustment["dy_mm"],
                    adjustment_dz_mm=adjustment["dz_mm"],
                    adjustment_duration_seconds=adjustment["duration_seconds"],
                )
        try:
            if baseline is None or attempt not in starts:
                raise ValueError("Missing trial_baseline or execution_start")
            start = starts[attempt]["point"]
            fault = record["point"]
            base = baseline["point"]
            names = start["feedback"]["joint_names"]
            if any(point["feedback"]["joint_names"] != names for point in (fault, base)):
                raise ValueError("Joint name/order changed between height boundaries")
            if fault["monotonic_time"] <= start["monotonic_time"] or (
                fault["feedback"]["timestamp"] < start["feedback"]["timestamp"]
            ):
                raise ValueError("Height boundary timestamps are out of order")
            start_xyz, fault_xyz, base_xyz = position(start), position(fault), position(base)
            samples = [start_xyz, fault_xyz]
            excluded_samples = 0
            for sample in feedback:
                point = sample["point"]
                if sample["attempt_id"] != attempt or not (
                    start["monotonic_time"] <= point["monotonic_time"] <= fault["monotonic_time"]
                ):
                    continue
                try:
                    if point["feedback"]["joint_names"] != names:
                        raise ValueError("Joint order changed")
                    samples.append(position(point))
                except (ValueError, TypeError, KeyError):
                    excluded_samples += 1
            peak_z = float(max(xyz[2] for xyz in samples))
            row.update(
                status="complete", fault_utc=fault["utc_time"],
                execution_start_z_mm=float(start_xyz[2]), trial_baseline_z_mm=float(base_xyz[2]),
                fault_z_mm=float(fault_xyz[2]), sampled_execution_peak_z_mm=peak_z,
                lift_at_fault_from_execution_start_mm=float(fault_xyz[2] - start_xyz[2]),
                lift_at_fault_from_trial_start_mm=float(fault_xyz[2] - base_xyz[2]),
                sampled_peak_lift_from_execution_start_mm=float(peak_z - start_xyz[2]),
                sampled_peak_lift_from_trial_start_mm=float(peak_z - base_xyz[2]),
                execution_duration_before_detection_seconds=fault["monotonic_time"] - start["monotonic_time"],
                valid_height_sample_count=len(samples), excluded_height_sample_count=excluded_samples,
            )
            if adjustment and adjustment["status"] == "complete":
                row.update(
                    fault_to_adjustment_start_dz_mm=float(adjustment["start_position_m"][2] * 1000 - fault_xyz[2]),
                    fault_to_adjustment_end_dz_mm=float(adjustment["end_position_m"][2] * 1000 - fault_xyz[2]),
                )
            selected_baseline = None
            source = height_baseline
            if lift_starts is not None and str(attempt) in lift_starts:
                requested_stamp = float(lift_starts[str(attempt)])
                if not np.isfinite(requested_stamp):
                    raise ValueError("Offline lift-start timestamp must be finite")
                candidates = [r for r in point_records if r["point"].get("feedback") is not None
                              and r["point"]["monotonic_time"] <= fault["monotonic_time"]]
                if not candidates:
                    raise ValueError("No recorded qpos before this fault for the requested lift start")
                selected_baseline = min(candidates, key=lambda r: abs(
                    r["point"]["feedback"]["timestamp"] - requested_stamp))
                if abs(selected_baseline["point"]["feedback"]["timestamp"] - requested_stamp) * 1000 > max_age_ms:
                    raise ValueError("Requested offline lift start has no nearby recorded qpos")
                source = "offline_ros_timestamp"
            elif height_baseline == "execution_start":
                selected_baseline = starts[attempt]
            else:
                candidates = [r for r in markers if r["point"]["monotonic_time"] <= fault["monotonic_time"]]
                if candidates:
                    selected_baseline = max(candidates, key=lambda r: r["point"]["monotonic_time"])
                    source = "keyboard_b"
            if selected_baseline is None:
                row.update(status="incomplete", reason="No pre-lift baseline; mark b or supply --lift-starts-file")
            else:
                lift_point = selected_baseline["point"]
                if lift_point["feedback"]["joint_names"] != names:
                    raise ValueError("Pre-lift joint name/order differs from fault feedback")
                if lift_point["feedback"]["timestamp"] > fault["feedback"]["timestamp"]:
                    raise ValueError("Pre-lift feedback timestamp is after fault feedback")
                lift_xyz = position(lift_point)
                row.update(
                    prelift_baseline_source=source, prelift_baseline_z_mm=float(lift_xyz[2]),
                    prelift_baseline_feedback_timestamp=lift_point["feedback"]["timestamp"],
                    prelift_baseline_attempt_id=selected_baseline.get("attempt_id"),
                    lift_at_fault_from_prelift_mm=float(fault_xyz[2] - lift_xyz[2]),
                )
        except (ValueError, TypeError, KeyError) as exc:
            row.update(status="invalid", reason=str(exc))
        failures.append(row)
    statistics = {}
    for metric in (
        "lift_at_fault_from_prelift_mm", "lift_at_fault_from_execution_start_mm", "sampled_peak_lift_from_execution_start_mm",
        "lift_at_fault_from_trial_start_mm", "sampled_peak_lift_from_trial_start_mm",
    ):
        values = [row[metric] for row in failures if row["status"] == "complete"]
        statistics[metric] = {
            "count": len(values), "mean": float(np.mean(values)) if values else None,
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    return {
        "schema_version": f"v7_7_2_{'sync' if mode == 'synchronous' else 'async'}_failure_measurements_summary_v1",
        "measurement_mode": mode,
        "reference_frame": chain.base_link, "measurement_point": chain.end_link,
        "tcp_offset_applied": False, "height_unit": "mm", "max_feedback_age_ms": max_age_ms,
        "failure_definition": (
            "receipt_of_need_recovery_true_after_complete_h30_not_physical_fault_onset"
            if mode == "synchronous" else "receipt_of_valid_need_recovery_true_during_async_execution_not_physical_fault_onset"
        ),
        "height_baseline": height_baseline,
        "offline_lift_starts": lift_starts,
        "marker_policy": "latest_explicit_marker_before_detection_persists_until_replaced",
        "peak_sampling": "per_action_feedback_plus_start_and_detection_keypoints_not_continuous_maximum",
        "height_status_counts": dict(Counter(row["status"] for row in failures)),
        "height_statistics": statistics, "failures": failures,
        "adjustment_summary": adjustments, "issues": issues, "fk_points": trajectory,
    }


def main(*, expected_mode=None, description=None):
    args = parse_args(description=description)
    trial = args.trial_dir.expanduser().resolve()
    lift_starts = json.loads(args.lift_starts_file.read_text()) if args.lift_starts_file else None
    if lift_starts is not None and not isinstance(lift_starts, dict):
        raise ValueError("--lift-starts-file must contain an object mapping attempt IDs to ROS timestamps")
    summary = calculate_heights(trial, args.max_feedback_age_ms,
                               height_baseline=args.height_baseline, lift_starts=lift_starts)
    if expected_mode is not None and summary["measurement_mode"] != expected_mode:
        raise ValueError(f"Expected {expected_mode} recordings, got {summary['measurement_mode']}")
    prefix = "sync" if summary["measurement_mode"] == "synchronous" else "async"
    stem = trial / f"{prefix}_failure_measurements"
    stem.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    fields = [
        "attempt_id", "request_id", "status", "reason", "failure_reason", "recovery_plan", "fault_utc",
        "prelift_baseline_source", "prelift_baseline_z_mm", "prelift_baseline_feedback_timestamp",
        "prelift_baseline_attempt_id", "lift_at_fault_from_prelift_mm",
        "execution_start_z_mm", "trial_baseline_z_mm", "fault_z_mm", "sampled_execution_peak_z_mm",
        "lift_at_fault_from_execution_start_mm", "lift_at_fault_from_trial_start_mm",
        "sampled_peak_lift_from_execution_start_mm", "sampled_peak_lift_from_trial_start_mm",
        "execution_duration_before_detection_seconds", "valid_height_sample_count", "excluded_height_sample_count",
        "adjustment_status", "adjustment_reason", "adjustment_segment_id",
        "adjustment_horizontal_xy_net_distance_mm", "adjustment_net_distance_mm",
        "adjustment_dx_mm", "adjustment_dy_mm", "adjustment_dz_mm", "adjustment_duration_seconds",
        "fault_to_adjustment_start_dz_mm", "fault_to_adjustment_end_dz_mm",
    ]
    with stem.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["failures"])
    points_path = trial / f"{prefix}_fk_points.csv"
    with points_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "event", "attempt_id", "phase", "published_steps", "utc_time", "monotonic_time",
            "status", "reason", "feedback_timestamp", "x_mm", "y_mm", "z_mm",
        ])
        writer.writeheader()
        writer.writerows(summary["fk_points"])
    for row in summary["failures"]:
        if row["status"] == "complete":
            print(f"attempt={row['attempt_id']} lift_at_detection={row['lift_at_fault_from_prelift_mm']:.2f}mm "
                  f"sampled_peak_lift={row['sampled_peak_lift_from_execution_start_mm']:.2f}mm "
                  f"adjustment_status={row['adjustment_status']} "
                  f"adjustment_xy_mm={row.get('adjustment_horizontal_xy_net_distance_mm')}")
        else:
            print(f"attempt={row['attempt_id']} {row['status']}: {row['reason']}")
    print(f"Saved: {stem.with_suffix('.json')} and {stem.with_suffix('.csv')}")
    print(f"FK points for offline lift-start annotation: {points_path}")


if __name__ == "__main__":
    main()
