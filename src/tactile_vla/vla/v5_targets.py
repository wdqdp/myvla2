"""Lightweight target transforms shared by V5.2 data building and loading."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_h30_terminal_hold(
    actions: Any,
    *,
    terminal_hold_from_offset: int | None,
) -> np.ndarray:
    """Return an isolated copy of an H30 target with an optional terminal hold."""

    result = np.asarray(actions).copy()
    if result.ndim != 2:
        raise ValueError(f"Action chunk must be rank 2, got {result.shape}")
    if terminal_hold_from_offset is None:
        return result
    offset = int(terminal_hold_from_offset)
    if not 1 <= offset < len(result):
        raise ValueError(
            f"terminal_hold_from_offset must be in [1,{len(result) - 1}], got {offset}"
        )
    result[offset:] = result[offset - 1]
    return result
