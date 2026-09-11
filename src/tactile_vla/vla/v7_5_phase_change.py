"""V7.5 adjustment-end prompt helpers for an H100/11-point qpos history."""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v5_3_phase_change import QPOS_BIN_COUNT
from tactile_vla.vla.v5_3_phase_change import QPOS_HISTORY_DIM
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v5_3_phase_change import discretize_state_qpos
from tactile_vla.vla.v5_3_phase_change import normalize_state_qpos
from tactile_vla.vla.v5_3_phase_change import pi05_phase_change_token_length


PHASE_CHANGE_PROMPT_PROFILE = "phase_change_v7_5_h100_11"
PHASE_CHANGE_HELPER_SCHEMA = "tactile_vla_v7_5_phase_change_helper_v1"
QPOS_HISTORY_FRAMES = 100
QPOS_SAMPLED_FRAMES = 11


def compact_qpos_json(qpos_discrete: Sequence[Sequence[int]]) -> str:
    values = np.asarray(qpos_discrete)
    expected = (QPOS_SAMPLED_FRAMES, QPOS_HISTORY_DIM)
    if values.shape != expected:
        raise ValueError(f"qpos_h100_discrete must have shape {expected}, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            raise ValueError("qpos_h100_discrete must contain integers")
    return json.dumps([[int(value) for value in row] for row in values], separators=(",", ":"))


def build_adjustment_end_prompt(
    *,
    instruction: str,
    tactile_caption: str,
    recovery_plan: str,
    sampled_qpos: Sequence[Sequence[float]],
    stats: StateQuantileStats,
) -> tuple[str, np.ndarray]:
    values = np.asarray(sampled_qpos, dtype=np.float64)
    expected = (QPOS_SAMPLED_FRAMES, QPOS_HISTORY_DIM)
    if values.shape != expected:
        raise ValueError(f"sampled_qpos must have shape {expected}, got {values.shape}")
    discrete = discretize_state_qpos(values, stats)

    def period(value: str, field: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError(f"{field} must be non-empty")
        return cleaned if cleaned.endswith(".") else f"{cleaned}."

    caption = str(tactile_caption).strip()
    if not caption:
        raise ValueError("tactile_caption must be non-empty")
    prompt = "\n".join(
        (
            "Mode: adjustment.",
            f"Task: {period(instruction, 'instruction')}",
            f"Tac: {caption}",
            f"Recovery plan: {period(recovery_plan, 'recovery_plan')}",
            f"qpos_h100_11:{compact_qpos_json(discrete)}",
        )
    )
    return prompt, discrete


def helper_identity() -> dict[str, Any]:
    return {
        "schema_version": PHASE_CHANGE_HELPER_SCHEMA,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
        "qpos_history_frames": QPOS_HISTORY_FRAMES,
        "qpos_history_includes_current": True,
        "qpos_history_dim": QPOS_HISTORY_DIM,
        "qpos_sampled_frames": QPOS_SAMPLED_FRAMES,
        "qpos_bin_count": QPOS_BIN_COUNT,
        "qpos_discretization_extra_clip": False,
    }


__all__ = [
    "PHASE_CHANGE_MAX_TOKEN_LEN",
    "PHASE_CHANGE_PROMPT_PROFILE",
    "QPOS_HISTORY_FRAMES",
    "QPOS_SAMPLED_FRAMES",
    "build_adjustment_end_prompt",
    "helper_identity",
    "normalize_state_qpos",
    "pi05_phase_change_token_length",
]
