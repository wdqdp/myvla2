"""Paper five-field tactile labels; legacy V4 remains available separately."""
from typing import Any, Mapping

LABEL_SCHEMA_VERSION = 'tactile_multifield_v5_masked'
DATASET_FORMAT = 'tactile_captioner_shards_v4_masked'
LABEL_FIELDS = ('area', 'fx_state', 'fy_state', 'fz_state', 'rotation')
LABEL_MAPS = {
    'area': {'none': 0, 'small': 1, 'medium': 2, 'full': 3},
    'fx_state': {'negative': 0, 'near_zero': 1, 'positive': 2, 'pos_neg': 3, 'neg_pos': 4},
    'fy_state': {'negative': 0, 'near_zero': 1, 'positive': 2, 'pos_neg': 3, 'neg_pos': 4},
    'fz_state': {'negative': 0, 'near_zero': 1, 'left_stronger': 2, 'right_stronger': 3},
    'rotation': {'none': 0, 'clockwise': 1, 'counterclockwise': 2},
}
NEUTRAL_LABELS = dict(area='none', fx_state='near_zero', fy_state='near_zero', fz_state='near_zero', rotation='none')
INVALID_LABEL_ID = -1
MASK_VALUE = 'mask'


def validate_target_state(labels: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(labels, Mapping):
        raise ValueError('tactile_label must be an object')
    if set(labels) != set(LABEL_FIELDS):
        raise ValueError('V5 tactile_label must contain exactly the five V5 fields')
    result = {field: str(labels[field]).strip().lower() for field in LABEL_FIELDS}
    for field, value in result.items():
        if value not in {*LABEL_MAPS[field], MASK_VALUE}:
            raise ValueError(f'Invalid tactile_label.{field}: {value!r}')
    if sum(value != MASK_VALUE for value in result.values()) != 1:
        raise ValueError('V5 tactile_label requires exactly one non-mask field')
    return result


def infer_label_task(labels: Mapping[str, Any]) -> str:
    return next(field for field, value in validate_target_state(labels).items() if value != MASK_VALUE)


def labels_to_caption(labels):
    normalized = {field: str(labels[field]).strip().lower() for field in LABEL_FIELDS}
    for field, value in normalized.items():
        if value not in LABEL_MAPS[field]:
            raise ValueError(f"Invalid {field} caption label: {value!r}")
    names = ('area', 'Fx', 'Fy', 'Fz', 'rotation')
    return 'Touch[' + '; '.join(f'{name}={normalized[field]}' for name, field in zip(names, LABEL_FIELDS)) + ']'
