"""Evaluation profile around the audited V7.5 arm-adjustment stop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tactile_vla.vla.v5_3_adjustment_end_evaluation import ranking_metrics as ranking_metrics
from tactile_vla.vla.v5_3_adjustment_end_evaluation import (
    select_max_recall_under_early_fpr as select_max_recall_under_early_fpr,
)
from tactile_vla.vla.v5_3_adjustment_end_evaluation import threshold_metrics as threshold_metrics


RELATIVE_FRAME_BINS = ((-30, -26), (-25, -21), (-20, -16), (-15, -11), (-10, -6), (-5, -1), (0, 4), (5, 10))


def relative_probability_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_id"]), []).append(row)
    if not by_episode:
        raise ValueError("Relative probability profile requires non-empty rows")
    bins = []
    for lower, upper in RELATIVE_FRAME_BINS:
        probabilities: list[float] = []
        episode_means: list[float] = []
        for episode_id, episode_rows in sorted(by_episode.items()):
            stops = {int(row["arm_adjustment_stop_frame"]) for row in episode_rows}
            if len(stops) != 1:
                raise ValueError(f"Episode {episode_id} has inconsistent arm_adjustment_stop")
            stop = stops.pop()
            values = [
                float(row["probability"])
                for row in episode_rows
                if lower <= int(row["frame_index"]) - stop <= upper
            ]
            if values:
                probabilities.extend(values)
                episode_means.append(float(np.mean(values)))
        if not probabilities:
            raise ValueError(f"No predictions cover arm-stop-relative bin [{lower}, {upper}]")
        bins.append(
            {
                "relative_frame_start_inclusive": lower,
                "relative_frame_end_inclusive": upper,
                "label": f"S{lower:+d}..S{upper:+d}",
                "sample_count": len(probabilities),
                "episode_count": len(episode_means),
                "sample_weighted_mean_probability": float(np.mean(probabilities)),
                "episode_balanced_mean_probability": float(np.mean(episode_means)),
            }
        )
    return {
        "probability": "P(adjustment_end=true)",
        "boundary": "arm_adjustment_stop",
        "range": {"start_inclusive": -30, "end_inclusive": 10},
        "positive_window": {"start_inclusive": -10, "end_inclusive": 10},
        "bins": bins,
    }
