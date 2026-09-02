"""Threshold evaluation helpers for the V5.3 adjustment-end classifier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


OFFICIAL_MIN_RECALL = 0.80
MAX_EARLY_FALSE_POSITIVE_RATE = 0.01

# Six bins contain five integer frame offsets each.  The final bin contains
# six frames so the report covers every frame from R-30 through R+5.
RELATIVE_FRAME_BINS = (
    (-30, -26),
    (-25, -21),
    (-20, -16),
    (-15, -11),
    (-10, -6),
    (-5, -1),
    (0, 5),
)


def ranking_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int32)
    probs = np.asarray([row["probability"] for row in rows], dtype=np.float64)
    if labels.size == 0 or not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("Ranking metrics require non-empty positive and negative rows")
    order = np.argsort(-probs, kind="stable")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    tpr = np.concatenate(([0.0], tp / positives, [1.0]))
    fpr = np.concatenate(([0.0], fp / negatives, [1.0]))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    auroc = float(np.trapz(tpr, fpr))
    auprc = float(np.sum((recall - np.concatenate(([0.0], recall[:-1]))) * precision))
    return {"auroc": auroc, "auprc": auprc}


def threshold_metrics(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    ranking: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.bool_)
    pred = np.asarray([row["probability"] >= threshold for row in rows], dtype=np.bool_)
    tp = int(np.count_nonzero(labels & pred))
    fp = int(np.count_nonzero(~labels & pred))
    fn = int(np.count_nonzero(labels & ~pred))
    tn = int(np.count_nonzero(~labels & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "early_false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    result.update(ranking if ranking is not None else ranking_metrics(rows))
    return result


def threshold_search(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the official and conservative probe operating points.

    ``official`` is present only when both fixed V5.3 constraints pass.  The
    experimental ``conservative_probe`` maximizes recall while retaining the
    same early-FPR ceiling.  Ties choose the higher threshold so the exported
    probe remains deterministic and conservative.
    """

    ranking = ranking_metrics(rows)
    candidates = sorted({0.0, 1.0, *(float(row["probability"]) for row in rows)})
    evaluated = [
        threshold_metrics(rows, threshold, ranking=ranking)
        for threshold in candidates
    ]
    official_candidates = [
        metrics
        for metrics in evaluated
        if metrics["recall"] >= OFFICIAL_MIN_RECALL
        and metrics["early_false_positive_rate"] <= MAX_EARLY_FALSE_POSITIVE_RATE
    ]
    fpr_safe = [
        metrics
        for metrics in evaluated
        if metrics["early_false_positive_rate"] <= MAX_EARLY_FALSE_POSITIVE_RATE
    ]
    recall_target = [
        metrics for metrics in evaluated if metrics["recall"] >= OFFICIAL_MIN_RECALL
    ]
    conservative = (
        max(fpr_safe, key=lambda item: (item["recall"], item["threshold"]))
        if fpr_safe
        else None
    )
    best_at_recall_target = (
        min(
            recall_target,
            key=lambda item: (item["early_false_positive_rate"], -item["threshold"]),
        )
        if recall_target
        else None
    )
    labels = np.asarray([row["label"] for row in rows], dtype=np.bool_)
    return {
        "requirements": {
            "minimum_positive_window_recall": OFFICIAL_MIN_RECALL,
            "maximum_early_false_positive_rate": MAX_EARLY_FALSE_POSITIVE_RATE,
        },
        "row_counts": {
            "total": int(labels.size),
            "positive_window": int(np.count_nonzero(labels)),
            "early_negative": int(np.count_nonzero(~labels)),
        },
        "ranking": ranking,
        "official_constraints_passed": bool(official_candidates),
        "official": max(official_candidates, key=lambda item: item["threshold"])
        if official_candidates
        else None,
        "conservative_probe": conservative,
        "best_operating_point_with_recall_at_least_80_percent": best_at_recall_target,
        "candidate_count": len(evaluated),
    }


def select_max_recall_under_early_fpr(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_early_false_positive_rate: float = MAX_EARLY_FALSE_POSITIVE_RATE,
) -> dict[str, Any]:
    """Select the highest-recall threshold under an early-FPR ceiling.

    There is intentionally no minimum-recall requirement.  Ties in recall are
    resolved toward the higher threshold, matching the conservative policy
    used by the earlier probe export while making that policy the only
    operating-point rule for joint V5.3 training.
    """

    if not 0.0 <= maximum_early_false_positive_rate <= 1.0:
        raise ValueError("maximum_early_false_positive_rate must be in [0, 1]")
    ranking = ranking_metrics(rows)
    candidates = sorted({0.0, 1.0, *(float(row["probability"]) for row in rows)})
    eligible = [
        threshold_metrics(rows, threshold, ranking=ranking)
        for threshold in candidates
    ]
    eligible = [
        metrics
        for metrics in eligible
        if metrics["early_false_positive_rate"]
        <= maximum_early_false_positive_rate
    ]
    if not eligible:
        raise ValueError(
            "No threshold satisfies early FPR <= "
            f"{maximum_early_false_positive_rate:.6f}"
        )
    selected = max(eligible, key=lambda item: (item["recall"], item["threshold"]))
    return {
        "threshold_policy": "max_recall_subject_to_early_fpr_lte_0_01",
        "maximum_early_false_positive_rate": maximum_early_false_positive_rate,
        "minimum_recall": None,
        "selected": selected,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
    }


def relative_probability_profile(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize P(adjustment_end=true) from R-30 through R+5.

    Every prediction must carry the original ``rexecution_frame``.  Inferring
    R from the last positive frame is invalid now that the label window ends
    at R+5.  The report emits sample-weighted and episode-balanced means.
    """

    by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_id"]), []).append(row)
    if not by_episode:
        raise ValueError("Relative probability profile requires non-empty rows")

    rexecution_by_episode: dict[int, int] = {}
    for episode_id, episode_rows in by_episode.items():
        if any(row.get("rexecution_frame") is None for row in episode_rows):
            raise ValueError(f"Episode {episode_id} prediction is missing rexecution_frame")
        explicit = {int(row["rexecution_frame"]) for row in episode_rows}
        if len(explicit) != 1:
            raise ValueError(f"Episode {episode_id} has inconsistent rexecution_frame values")
        rexecution_by_episode[episode_id] = explicit.pop()

    bins = []
    for lower, upper in RELATIVE_FRAME_BINS:
        probabilities: list[float] = []
        episode_means: list[float] = []
        episode_sample_counts: dict[str, int] = {}
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
                episode_sample_counts[str(episode_id)] = len(values)
        if not probabilities:
            raise ValueError(f"No predictions cover relative frame bin [{lower}, {upper}]")
        bins.append(
            {
                "relative_frame_start_inclusive": lower,
                "relative_frame_end_inclusive": upper,
                "label": f"R{lower:+d}..R{upper:+d}",
                "sample_count": len(probabilities),
                "episode_count": len(episode_means),
                "sample_weighted_mean_probability": float(np.mean(probabilities)),
                "episode_balanced_mean_probability": float(np.mean(episode_means)),
                "episode_sample_counts": episode_sample_counts,
            }
        )
    return {
        "probability": "P(adjustment_end=true)",
        "range": {"start_inclusive": -30, "end_inclusive": 5},
        "binning_note": (
            "The first six bins have five frames and the final bin has six; "
            "the inclusive positive window [R-10,R+5] spans the final three bins."
        ),
        "bins": bins,
    }
