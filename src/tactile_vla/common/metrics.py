"""Small classification metrics helper without a sklearn dependency."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def as_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def confusion_matrix(y_true: Any, y_pred: Any, num_classes: int) -> np.ndarray:
    true = as_numpy(y_true).astype(np.int64).reshape(-1)
    pred = as_numpy(y_pred).astype(np.int64).reshape(-1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, output in zip(true, pred, strict=False):
        if 0 <= target < num_classes and 0 <= output < num_classes:
            matrix[target, output] += 1
    return matrix


def classification_report(
    y_true: Any,
    y_pred: Any,
    *,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    recall_values = []
    precision_values = []
    for idx, name in enumerate(names):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        support = int(matrix[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "macro_recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "num_samples": total,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
