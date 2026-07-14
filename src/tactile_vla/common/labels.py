"""Shared label definitions for tactile rotation states."""

from __future__ import annotations

ROTATION_LABELS = ("none", "clockwise", "counterclockwise")
LABEL_TO_ID = {name: idx for idx, name in enumerate(ROTATION_LABELS)}
ID_TO_LABEL = {idx: name for idx, name in enumerate(ROTATION_LABELS)}

TACTILE_CAPTIONS = {
    "none": "Tactile: no rotation.",
    "clockwise": "Tactile: clockwise rotation.",
    "counterclockwise": "Tactile: counterclockwise rotation.",
}


def label_id_to_name(label_id: int) -> str:
    try:
        return ID_TO_LABEL[int(label_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown rotation label id: {label_id}") from exc


def label_name_to_id(label_name: str) -> int:
    try:
        return LABEL_TO_ID[label_name]
    except KeyError as exc:
        raise ValueError(f"Unknown rotation label name: {label_name}") from exc


def label_to_caption(label: int | str) -> str:
    label_name = label_id_to_name(label) if isinstance(label, int) else label
    try:
        return TACTILE_CAPTIONS[label_name]
    except KeyError as exc:
        raise ValueError(f"Unknown rotation label: {label}") from exc
