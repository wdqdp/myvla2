"""Six-field labels for episode-constant, masked tactile supervision.

The V3 definitions remain in labels.py for existing five-head runtime users.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tactile_vla.common import labels as legacy

LABEL_SCHEMA_VERSION = "tactile_multifield_v4_masked"
DATASET_FORMAT = "tactile_captioner_shards_v3_masked"
INVALID_LABEL_ID = -1
LABEL_FIELDS = ("area", "fx_state", "fy_state", "fz_state", "fz_bias", "rotation")
LABEL_MAPS = {
    **{field: legacy.LABEL_MAPS[field] for field in LABEL_FIELDS if field != "fz_bias"},
    "fz_bias": {"left": 0, "right": 1, "balanced": 2},
}
LABEL_MAPS = {field: LABEL_MAPS[field] for field in LABEL_FIELDS}


def schema_for_version(version: str) -> tuple[tuple[str, ...], dict[str, dict[str, int]]]:
    if version == LABEL_SCHEMA_VERSION:
        return LABEL_FIELDS, LABEL_MAPS
    if version == legacy.LABEL_SCHEMA_VERSION:
        return legacy.LABEL_FIELDS, legacy.LABEL_MAPS
    raise ValueError(f"Unsupported tactile label schema: {version!r}")


def validate_label_maps(label_maps: Mapping[str, Any], *, schema_version: str = LABEL_SCHEMA_VERSION) -> None:
    _, expected = schema_for_version(schema_version)
    normalized = {
        str(field): {str(name): int(label_id) for name, label_id in values.items()}
        for field, values in label_maps.items()
    }
    if normalized != expected:
        raise ValueError(f"Tactile label maps do not match {schema_version}: {normalized}")


def class_names(field: str) -> tuple[str, ...]:
    mapping = LABEL_MAPS[field]
    return tuple(name for name, _ in sorted(mapping.items(), key=lambda item: item[1]))


def label_id_to_name(field: str, label_id: int) -> str:
    for name, value in LABEL_MAPS[field].items():
        if value == int(label_id):
            return name
    raise ValueError(f"Unknown {field} label id: {label_id}")


def labels_to_caption(labels: Mapping[str, str], *, schema_version: str = LABEL_SCHEMA_VERSION) -> str:
    if schema_version == legacy.LABEL_SCHEMA_VERSION:
        return legacy.labels_to_caption(labels)
    fields, maps = schema_for_version(schema_version)
    missing = set(fields) - set(labels)
    if missing:
        raise ValueError(f"Missing tactile caption fields: {sorted(missing)}")
    normalized = {field: str(labels[field]).strip().lower() for field in fields}
    for field, value in normalized.items():
        if value not in maps[field]:
            raise ValueError(f"Unknown {field} label: {value!r}")
    return (
        "Touch["
        f"area={normalized['area']}; "
        f"Fx={normalized['fx_state']}; "
        f"Fy={normalized['fy_state']}; "
        f"Fz={normalized['fz_state']}; "
        f"Fz_bias={normalized['fz_bias']}; "
        f"rotation={normalized['rotation']}"
        "]"
    )
