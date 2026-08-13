"""Shared V3 tactile-caption label definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LABEL_SCHEMA_VERSION = "tactile_multifield_v3"
LABEL_FIELDS = ("area", "fx_state", "fy_state", "fz_state", "rotation")
LABEL_MAPS = {
    "area": {"none": 0, "small": 1, "medium": 2, "full": 3},
    "fx_state": {"negative": 0, "near_zero": 1, "positive": 2},
    "fy_state": {"negative": 0, "near_zero": 1, "positive": 2},
    "fz_state": {"negative": 0, "near_zero": 1},
    "rotation": {"none": 0, "clockwise": 1, "counterclockwise": 2},
}
ID_TO_LABELS = {
    field: {label_id: name for name, label_id in label_map.items()}
    for field, label_map in LABEL_MAPS.items()
}
NEUTRAL_LABELS = {
    "area": "none",
    "fx_state": "near_zero",
    "fy_state": "near_zero",
    "fz_state": "near_zero",
    "rotation": "none",
}


def validate_label_maps(label_maps: Mapping[str, Any]) -> None:
    normalized = {
        str(field): {str(name): int(label_id) for name, label_id in values.items()}
        for field, values in label_maps.items()
    }
    if normalized != LABEL_MAPS:
        raise ValueError(f"Tactile label maps do not match {LABEL_SCHEMA_VERSION}: {normalized}")


def class_names(field: str) -> tuple[str, ...]:
    try:
        names = ID_TO_LABELS[field]
    except KeyError as exc:
        raise ValueError(f"Unknown tactile label field: {field!r}") from exc
    return tuple(names[label_id] for label_id in range(len(names)))


def label_id_to_name(field: str, label_id: int) -> str:
    try:
        return ID_TO_LABELS[field][int(label_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown {field} label id: {label_id}") from exc


def labels_to_caption(labels: Mapping[str, str]) -> str:
    missing = set(LABEL_FIELDS) - set(labels)
    if missing:
        raise ValueError(f"Missing tactile caption fields: {sorted(missing)}")
    normalized: dict[str, str] = {}
    for field in LABEL_FIELDS:
        value = str(labels[field]).strip().lower()
        if value not in LABEL_MAPS[field]:
            raise ValueError(f"Unknown {field} label: {value!r}")
        normalized[field] = value
    return (
        "Touch["
        f"area={normalized['area']}; "
        f"Fx={normalized['fx_state']}; "
        f"Fy={normalized['fy_state']}; "
        f"Fz={normalized['fz_state']}; "
        f"rotation={normalized['rotation']}"
        "]"
    )


DEFAULT_TACTILE_CAPTION = labels_to_caption(NEUTRAL_LABELS)
