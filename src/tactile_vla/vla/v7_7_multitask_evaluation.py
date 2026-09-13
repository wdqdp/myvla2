"""Offline V7.7 assessment metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


def source_false_positive_rates(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    result = {}
    for source in (
        "pre_failure_hard_negative", "one_success_easy_negative",
        "successful_recovery_easy_negative",
    ):
        values = [row for row in rows if row.get("source") == source and not bool(row["label"])]
        false_positive = sum(float(row["probability"]) >= threshold for row in values)
        result[source] = {
            "support": len(values),
            "false_positive_count": false_positive,
            "false_positive_rate": false_positive / len(values) if values else 0.0,
        }
    return result


def detection_delays(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    by_attempt = {}
    for row in rows:
        if not bool(row["label"]):
            continue
        key = (int(row["episode_id"]), int(row["attempt_id"]))
        by_attempt.setdefault(key, []).append(row)
    delays = []
    misses = 0
    for values in by_attempt.values():
        values.sort(key=lambda row: int(row["frame_index"]))
        detected = next((row for row in values if float(row["probability"]) >= threshold), None)
        if detected is None:
            misses += 1
        else:
            delays.append(int(detected["frame_index"]) - int(values[0]["frame_index"]))
    return {
        "attempt_count": len(by_attempt), "miss_count": misses,
        "mean_delay_frames": float(np.mean(delays)) if delays else None,
        "max_delay_frames": max(delays) if delays else None,
    }


def manifest_source_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("source", "unknown")) for row in rows).items()))


__all__ = ["detection_delays", "manifest_source_counts", "source_false_positive_rates"]
