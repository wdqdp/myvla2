"""Runtime H100 prompt construction for the V7.5 adjustment-end classifier."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v7_5_phase_change import QPOS_HISTORY_FRAMES
from tactile_vla.vla.v7_5_phase_change import QPOS_SAMPLED_FRAMES
from tactile_vla.vla.v7_5_phase_change import build_adjustment_end_prompt


RUNTIME_PAST_QPOS_FRAMES = QPOS_HISTORY_FRAMES - 1
RUNTIME_SAMPLE_OFFSETS = tuple(
    int(value)
    for value in np.rint(
        np.linspace(0, QPOS_HISTORY_FRAMES - 1, QPOS_SAMPLED_FRAMES)
    ).astype(np.int32)
)


def build_runtime_adjustment_end_prompt(
    *,
    instruction: str,
    tactile_caption: str,
    recovery_plan: str,
    past_qpos_h99: Sequence[Sequence[float]],
    current_qpos: Sequence[float],
    stats: StateQuantileStats,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Build V7.5 runtime prompt with p as both state and H100 endpoint.

    Runtime deliberately does not infer or compress the offline G/A idle gap.
    It uniformly samples eleven rows from the chronological raw H100.
    """

    past = np.asarray(past_qpos_h99, dtype=np.float32)
    current = np.asarray(current_qpos, dtype=np.float32)
    if past.shape != (RUNTIME_PAST_QPOS_FRAMES, 7):
        raise ValueError(f"past_qpos_h99 must have shape [99,7], got {past.shape}")
    if current.shape != (7,):
        raise ValueError(f"current_qpos must have shape [7], got {current.shape}")
    if not np.isfinite(past).all() or not np.isfinite(current).all():
        raise ValueError("Runtime H100 qpos must be finite")
    h100 = np.concatenate((past, current[None, :]), axis=0)
    sampled = h100[list(RUNTIME_SAMPLE_OFFSETS)]
    prompt, discrete = build_adjustment_end_prompt(
        instruction=instruction,
        tactile_caption=tactile_caption,
        recovery_plan=recovery_plan,
        sampled_qpos=sampled,
        stats=stats,
    )
    return prompt, discrete, h100
