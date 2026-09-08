"""V7 adjustment-end evaluation with profiles ending at native R."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tactile_vla.vla.v5_3_adjustment_end_evaluation import ranking_metrics as ranking_metrics
from tactile_vla.vla.v5_3_adjustment_end_evaluation import (
    select_max_recall_under_early_fpr as select_max_recall_under_early_fpr,
)
from tactile_vla.vla.v5_3_adjustment_end_evaluation import threshold_metrics as threshold_metrics


RELATIVE_FRAME_BINS = (
    (-30, -26),
    (-25, -21),
    (-20, -16),
    (-15, -11),
    (-10, -6),
    (-5, -1),
    (0, 0),
)


def relative_probability_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_id"]), []).append(row)
    if not by_episode:
        raise ValueError("Relative probability profile requires non-empty rows")
    rexecution_by_episode: dict[int, int] = {}
    for episode_id, episode_rows in by_episode.items():
        explicit = {int(row["rexecution_frame"]) for row in episode_rows if row.get("rexecution_frame") is not None}
        if len(explicit) != 1 or len(explicit) != len({row.get("rexecution_frame") for row in episode_rows}):
            raise ValueError(f"Episode {episode_id} has missing or inconsistent rexecution_frame")
        rexecution_by_episode[episode_id] = explicit.pop()
    bins = []
    for lower, upper in RELATIVE_FRAME_BINS:
        probabilities: list[float] = []
        episode_means: list[float] = []
        counts: dict[str, int] = {}
        for episode_id, episode_rows in sorted(by_episode.items()):
            rexecution = rexecution_by_episode[episode_id]
            values = [
                float(row["probability"])
                for row in episode_rows
                if lower <= int(row["frame_index"]) - rexecution <= upper
            ]
            if values:
                probabilities.extend(values)
                episode_means.append(float(np.mean(values)))
                counts[str(episode_id)] = len(values)
        if not probabilities:
            raise ValueError(f"No predictions cover relative frame bin [{lower}, {upper}]")
        bins.append({
            "relative_frame_start_inclusive": lower,
            "relative_frame_end_inclusive": upper,
            "label": f"R{lower:+d}..R{upper:+d}",
            "sample_count": len(probabilities),
            "episode_count": len(episode_means),
            "sample_weighted_mean_probability": float(np.mean(probabilities)),
            "episode_balanced_mean_probability": float(np.mean(episode_means)),
            "episode_sample_counts": counts,
        })
    return {
        "probability": "P(adjustment_end=true)",
        "range": {"start_inclusive": -30, "end_inclusive": 0},
        "positive_window": {"start_inclusive": -10, "end_inclusive": 0},
        "bins": bins,
    }
