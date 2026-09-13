"""Evaluation helpers for V7.6 factual/counterfactual prompt pairs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tactile_vla.vla.v5_3_adjustment_end_evaluation import ranking_metrics
from tactile_vla.vla.v7_5_adjustment_end_evaluation import (
    relative_probability_profile as v7_5_relative_probability_profile,
)
from tactile_vla.vla.v7_6_adjustment_end_data import SAMPLE_VARIANT_IDS


VARIANT_NAMES = {value: key for key, value in SAMPLE_VARIANT_IDS.items()}


def factual_relative_probability_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    factual = [row for row in rows if int(row["sample_variant_id"]) == SAMPLE_VARIANT_IDS["factual"]]
    return v7_5_relative_probability_profile(factual)


def _variant_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    result: dict[str, Any] = {
        "sample_count": int(values.size),
        "positive_count": int(labels.sum()),
        "negative_count": int(values.size - labels.sum()),
        "mean_probability": float(values.mean()),
    }
    if labels.any():
        result["positive_mean_probability"] = float(values[labels == 1].mean())
    if np.any(labels == 0):
        result["negative_mean_probability"] = float(values[labels == 0].mean())
    if labels.any() and np.any(labels == 0):
        result["ranking"] = ranking_metrics(rows)
    return result


def counterfactual_pair_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_pair: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    by_variant: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = int(row["sample_variant_id"])
        by_pair[int(row["pair_id"])][variant] = row
        by_variant[variant].append(row)
    variants = {
        VARIANT_NAMES[variant]: _variant_profile(values)
        for variant, values in sorted(by_variant.items())
    }
    pair_rows = []
    disagreement_rows = []
    for pair_id, variants_for_pair in by_pair.items():
        factual = variants_for_pair.get(SAMPLE_VARIANT_IDS["factual"])
        counterfactuals = [
            row for variant, row in variants_for_pair.items()
            if variant != SAMPLE_VARIANT_IDS["factual"]
        ]
        if factual is None or len(counterfactuals) > 1:
            if counterfactuals:
                raise ValueError(f"Invalid V7.6 pair composition for pair_id={pair_id}")
            continue
        if not counterfactuals:
            continue
        counterfactual = counterfactuals[0]
        delta = float(counterfactual["probability"]) - float(factual["probability"])
        pair = {
            "pair_id": pair_id,
            "counterfactual_variant": VARIANT_NAMES[int(counterfactual["sample_variant_id"])],
            "factual_label": int(factual["label"]),
            "counterfactual_label": int(counterfactual["label"]),
            "counterfactual_minus_factual_probability": delta,
        }
        pair_rows.append(pair)
        if pair["factual_label"] != pair["counterfactual_label"]:
            desired_sign = 1.0 if pair["counterfactual_label"] > pair["factual_label"] else -1.0
            pair["signed_margin_toward_correct_order"] = desired_sign * delta
            disagreement_rows.append(pair)

    margins = np.asarray(
        [row["signed_margin_toward_correct_order"] for row in disagreement_rows], dtype=np.float64
    )
    deltas = np.asarray(
        [row["counterfactual_minus_factual_probability"] for row in pair_rows], dtype=np.float64
    )
    return {
        "variant_profiles": variants,
        "paired_sample_count": len(pair_rows),
        "mean_absolute_prompt_probability_delta": float(np.abs(deltas).mean()) if deltas.size else None,
        "label_disagreement_pair_count": len(disagreement_rows),
        "label_disagreement_order_accuracy": float(np.mean(margins > 0)) if margins.size else None,
        "label_disagreement_tie_rate": float(np.mean(margins == 0)) if margins.size else None,
        "label_disagreement_mean_signed_margin": float(margins.mean()) if margins.size else None,
    }


__all__ = ["counterfactual_pair_metrics", "factual_relative_probability_profile"]
