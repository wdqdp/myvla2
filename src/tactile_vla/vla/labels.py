"""Label maps for VLA status, diagnosis, and recovery planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

FAILURE_REASONS: tuple[str, ...] = (
    "tilted left",
    "tilted right",
    "tilted front",
    "tilted back",
)

RECOVERY_PLANS: tuple[str, ...] = (
    "Regrasp left.",
    "Regrasp right.",
    "Regrasp front.",
    "Regrasp back.",
)

FAILURE_REASON_TO_ID = {name: idx for idx, name in enumerate(FAILURE_REASONS)}
ID_TO_FAILURE_REASON = {idx: name for idx, name in enumerate(FAILURE_REASONS)}
RECOVERY_PLAN_TO_ID = {name: idx for idx, name in enumerate(RECOVERY_PLANS)}
ID_TO_RECOVERY_PLAN = {idx: name for idx, name in enumerate(RECOVERY_PLANS)}

_FAILURE_ALIASES = {
    "left": "tilted left",
    "right": "tilted right",
    "front": "tilted front",
    "back": "tilted back",
    "tilt left": "tilted left",
    "tilt right": "tilted right",
    "tilt front": "tilted front",
    "tilt back": "tilted back",
}

_RECOVERY_ALIASES = {
    "left": "Regrasp left.",
    "right": "Regrasp right.",
    "front": "Regrasp front.",
    "back": "Regrasp back.",
    "regrasp left": "Regrasp left.",
    "regrasp right": "Regrasp right.",
    "regrasp front": "Regrasp front.",
    "regrasp back": "Regrasp back.",
}


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().strip(".").lower().split())


def normalize_failure_reason(value: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""
    if normalized in FAILURE_REASON_TO_ID:
        return normalized
    if normalized in _FAILURE_ALIASES:
        return _FAILURE_ALIASES[normalized]
    raise ValueError(f"Unknown failure_reason: {value!r}")


def normalize_recovery_plan(value: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""
    if value.strip() in RECOVERY_PLAN_TO_ID:
        return value.strip()
    if normalized in _RECOVERY_ALIASES:
        return _RECOVERY_ALIASES[normalized]
    raise ValueError(f"Unknown recovery_plan: {value!r}")


def failure_reason_to_id(value: str) -> int:
    return FAILURE_REASON_TO_ID[normalize_failure_reason(value)]


def recovery_plan_to_id(value: str) -> int:
    return RECOVERY_PLAN_TO_ID[normalize_recovery_plan(value)]


def id_to_failure_reason(label_id: int) -> str:
    try:
        return ID_TO_FAILURE_REASON[int(label_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown failure_reason id: {label_id}") from exc


def id_to_recovery_plan(label_id: int) -> str:
    try:
        return ID_TO_RECOVERY_PLAN[int(label_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown recovery_plan id: {label_id}") from exc


def class_weights_from_counts(labels: Sequence[int], num_classes: int) -> list[float]:
    counts = [0] * num_classes
    for label in labels:
        if 0 <= int(label) < num_classes:
            counts[int(label)] += 1
    total = sum(counts)
    if total == 0:
        return [1.0] * num_classes
    nonzero = sum(1 for count in counts if count > 0)
    return [total / (nonzero * count) if count > 0 else 0.0 for count in counts]


def label_map_payload() -> Mapping[str, object]:
    return {
        "failure_reasons": list(FAILURE_REASONS),
        "recovery_plans": list(RECOVERY_PLANS),
        "failure_reason_to_id": FAILURE_REASON_TO_ID,
        "recovery_plan_to_id": RECOVERY_PLAN_TO_ID,
    }
