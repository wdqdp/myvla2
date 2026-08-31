"""Shared V5.3 phase-change prompt and qpos helpers.

This module is the only implementation used by data construction, training,
offline evaluation, replay, and robot inference.  Keep it dependency-light so
the AgileX client can import it without importing JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal, Sequence

import numpy as np


PHASE_CHANGE_PROMPT_PROFILE = "phase_change_v1"
PHASE_CHANGE_HELPER_SCHEMA = "tactile_vla_v5_3_phase_change_helper_v1"
PHASE_CHANGE_MAX_TOKEN_LEN = 512
ACTION_MAX_TOKEN_LEN = 200
QPOS_HISTORY_FRAMES = 30
QPOS_HISTORY_DIM = 7
QPOS_SAMPLED_FRAMES = 10
QPOS_BIN_COUNT = 256
QPOS_SAMPLE_OFFSETS = (0, 3, 6, 10, 13, 16, 19, 23, 26, 29)
PhaseChangeMode = Literal["execution", "adjustment"]


@dataclass(frozen=True)
class StateQuantileStats:
    q01: np.ndarray
    q99: np.ndarray

    def __post_init__(self) -> None:
        q01 = np.asarray(self.q01, dtype=np.float64)
        q99 = np.asarray(self.q99, dtype=np.float64)
        if q01.shape != (QPOS_HISTORY_DIM,) or q99.shape != (QPOS_HISTORY_DIM,):
            raise ValueError(f"state q01/q99 must both have shape [7], got {q01.shape}/{q99.shape}")
        if not np.isfinite(q01).all() or not np.isfinite(q99).all():
            raise ValueError("state q01/q99 contain non-finite values")
        if np.any(q99 <= q01):
            raise ValueError("each state q99 value must be greater than q01")
        object.__setattr__(self, "q01", q01)
        object.__setattr__(self, "q99", q99)


def _with_period(value: str, *, field: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field} must be non-empty")
    return cleaned if cleaned.endswith(".") else f"{cleaned}."


def sample_qpos_h30(qpos_h30: Sequence[Sequence[float]]) -> np.ndarray:
    """Uniformly sample the fixed ten rows from a strict 30x7 history."""

    history = np.asarray(qpos_h30, dtype=np.float64)
    if history.shape != (QPOS_HISTORY_FRAMES, QPOS_HISTORY_DIM):
        raise ValueError(f"qpos_h30 must have shape [30,7], got {history.shape}")
    if not np.isfinite(history).all():
        raise ValueError("qpos_h30 contains non-finite values")
    offsets = np.rint(np.linspace(0, QPOS_HISTORY_FRAMES - 1, QPOS_SAMPLED_FRAMES)).astype(np.int32)
    if tuple(int(value) for value in offsets) != QPOS_SAMPLE_OFFSETS:
        raise AssertionError(f"numpy sampling offsets changed unexpectedly: {offsets.tolist()}")
    return history[offsets]


def normalize_state_qpos(
    qpos: Sequence[Sequence[float]] | Sequence[float],
    stats: StateQuantileStats,
) -> np.ndarray:
    values = np.asarray(qpos, dtype=np.float64)
    if values.ndim not in {1, 2} or values.shape[-1] != QPOS_HISTORY_DIM:
        raise ValueError(f"qpos must end in dimension 7, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("qpos contains non-finite values")
    return (values - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


def discretize_normalized_state(normalized_qpos: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
    """Exactly mirror Pi0.5 state tokenization; deliberately do not clip."""

    values = np.asarray(normalized_qpos, dtype=np.float64)
    if values.ndim not in {1, 2} or values.shape[-1] != QPOS_HISTORY_DIM:
        raise ValueError(f"normalized qpos must end in dimension 7, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("normalized qpos contains non-finite values")
    bins = np.linspace(-1.0, 1.0, QPOS_BIN_COUNT + 1)[:-1]
    return (np.digitize(values, bins=bins) - 1).astype(np.int32)


def discretize_state_qpos(
    qpos: Sequence[Sequence[float]] | Sequence[float],
    stats: StateQuantileStats,
) -> np.ndarray:
    return discretize_normalized_state(normalize_state_qpos(qpos, stats))


def sampled_discrete_qpos_h10(
    qpos_h30: Sequence[Sequence[float]],
    stats: StateQuantileStats,
) -> np.ndarray:
    result = discretize_state_qpos(sample_qpos_h30(qpos_h30), stats)
    if result.shape != (QPOS_SAMPLED_FRAMES, QPOS_HISTORY_DIM):
        raise AssertionError(f"discrete qpos_h10 shape changed: {result.shape}")
    return result


def compact_qpos_json(qpos_h10_discrete: Sequence[Sequence[int]]) -> str:
    values = np.asarray(qpos_h10_discrete)
    if values.shape != (QPOS_SAMPLED_FRAMES, QPOS_HISTORY_DIM):
        raise ValueError(f"qpos_h10_discrete must have shape [10,7], got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            raise ValueError("qpos_h10_discrete must contain integers")
    rows = [[int(value) for value in row] for row in values]
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def build_phase_change_prompt(
    *,
    mode: PhaseChangeMode,
    instruction: str,
    tactile_caption: str,
    recovery_plan: str,
    qpos_h10_discrete: Sequence[Sequence[int]],
) -> str:
    if mode not in {"execution", "adjustment"}:
        raise ValueError(f"phase-change mode must be execution/adjustment, got {mode!r}")
    caption = str(tactile_caption).strip()
    if not caption:
        raise ValueError("tactile_caption must be non-empty")
    return "\n".join(
        (
            f"Mode: {mode}.",
            f"Task: {_with_period(instruction, field='instruction')}",
            f"Tac: {caption}",
            f"Recovery plan: {_with_period(recovery_plan, field='recovery_plan')}",
            f"qpos_h30:{compact_qpos_json(qpos_h10_discrete)}",
        )
    )


def build_adjustment_end_prompt(
    *,
    instruction: str,
    tactile_caption: str,
    recovery_plan: str,
    qpos_h30: Sequence[Sequence[float]],
    stats: StateQuantileStats,
) -> tuple[str, np.ndarray]:
    discrete = sampled_discrete_qpos_h10(qpos_h30, stats)
    prompt = build_phase_change_prompt(
        mode="adjustment",
        instruction=instruction,
        tactile_caption=tactile_caption,
        recovery_plan=recovery_plan,
        qpos_h10_discrete=discrete,
    )
    return prompt, discrete


def pi05_phase_change_prefix(
    *,
    prompt: str,
    normalized_current_qpos: Sequence[float],
) -> str:
    """Build the exact text passed to the Pi0.5 PaliGemma tokenizer.

    Historical qpos is already embedded in ``prompt``.  The current state is
    independently discretized here, exactly as ``PaligemmaTokenizer.tokenize``
    does for Pi0.5.
    """

    current = np.asarray(normalized_current_qpos, dtype=np.float64)
    if current.shape != (QPOS_HISTORY_DIM,):
        raise ValueError(f"normalized_current_qpos must have shape [7], got {current.shape}")
    discrete_current = discretize_normalized_state(current)
    cleaned_text = str(prompt).strip().replace("_", " ").replace("\n", " ")
    state_str = " ".join(map(str, discrete_current))
    return f"Task: {cleaned_text}, State: {state_str};\nAction: "


def pi05_phase_change_token_length(
    *,
    tokenizer: Any,
    prompt: str,
    normalized_current_qpos: Sequence[float],
) -> int:
    """Return the unpadded Pi0.5 prefix length without allowing truncation."""

    prefix = pi05_phase_change_prefix(
        prompt=prompt,
        normalized_current_qpos=normalized_current_qpos,
    )
    return len(tokenizer.encode_text(prefix, add_bos=True))


def runtime_reachable_endpoint(frame_index: int) -> bool:
    frame = int(frame_index)
    return frame >= QPOS_HISTORY_FRAMES and frame % QPOS_HISTORY_FRAMES == 0


def h30_endpoint_indices(*, start_offset: int, stop_frame: int) -> list[int]:
    start = int(start_offset)
    stop = int(stop_frame)
    if not 0 <= start < QPOS_HISTORY_FRAMES:
        raise ValueError(f"start_offset must be in [0,29], got {start}")
    if stop < 0:
        raise ValueError("stop_frame must be non-negative")
    return list(range(start + QPOS_HISTORY_FRAMES, stop + 1, QPOS_HISTORY_FRAMES))


def helper_identity() -> dict[str, Any]:
    return {
        "schema_version": PHASE_CHANGE_HELPER_SCHEMA,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
        "qpos_history_frames": QPOS_HISTORY_FRAMES,
        "qpos_history_dim": QPOS_HISTORY_DIM,
        "qpos_sampled_frames": QPOS_SAMPLED_FRAMES,
        "qpos_sample_offsets": list(QPOS_SAMPLE_OFFSETS),
        "qpos_bin_count": QPOS_BIN_COUNT,
        "qpos_discretization_extra_clip": False,
        "endpoint_rule": "t=start_offset+30*k",
    }
