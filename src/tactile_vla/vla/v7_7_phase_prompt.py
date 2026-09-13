"""Canonical V7.7 phase prompt and episode-continuous H100 helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

import numpy as np

from tactile_vla.vla.v5_3_phase_change import StateQuantileStats, discretize_state_qpos


PROMPT_PROFILE = "phase_v7_7_h100_11_answer"
HISTORY_FRAMES = 100
HISTORY_POINTS = 11
HISTORY_DIM = 7


@dataclass(frozen=True)
class TimelinePoint:
    episode_id: int
    attempt_id: int
    frame_index: int
    global_index: int
    timestamp: float
    qpos: np.ndarray


def compact_history(values: Sequence[Sequence[int]]) -> str:
    array = np.asarray(values)
    if array.shape != (HISTORY_POINTS, HISTORY_DIM):
        raise ValueError(f"history must be [{HISTORY_POINTS},{HISTORY_DIM}], got {array.shape}")
    if not np.equal(array, np.rint(array)).all():
        raise ValueError("discrete history contains non-integers")
    return json.dumps(array.astype(np.int32).tolist(), separators=(",", ":"))


def build_phase_prompt(
    *, instruction: str, tactile_caption: str, recovery_plan: str,
    qpos_h100_11_discrete: Sequence[Sequence[int]],
) -> str:
    fields = {
        "instruction": str(instruction).strip(),
        "tactile_caption": str(tactile_caption).strip(),
        "recovery_plan": str(recovery_plan).strip() or "none",
    }
    if not fields["instruction"] or not fields["tactile_caption"]:
        raise ValueError("instruction and tactile_caption must be non-empty")
    return "\n".join((
        "Mode: phase",
        f"Task: {fields['instruction']}",
        f"Touch: {fields['tactile_caption']}",
        f"Recovery plan: {fields['recovery_plan']}",
        "State history H100 sampled to 11 points: " + compact_history(qpos_h100_11_discrete),
    ))


def uniform_positions(length: int, points: int = HISTORY_POINTS) -> np.ndarray:
    if length <= 0 or points <= 0:
        raise ValueError("length and points must be positive")
    # Integer linspace is deterministic and always includes both endpoints.
    return np.linspace(0, length - 1, num=points, dtype=np.int64)


def sample_episode_history(
    timeline: Sequence[TimelinePoint], current_position: int, *,
    history_frames: int = HISTORY_FRAMES, points: int = HISTORY_POINTS,
) -> tuple[list[TimelinePoint], int]:
    if not 0 <= current_position < len(timeline):
        raise IndexError(current_position)
    current = timeline[current_position]
    if any(point.episode_id != current.episode_id for point in timeline):
        raise ValueError("episode timeline contains more than one episode")
    for left, right in zip(timeline, timeline[1:], strict=False):
        if (right.timestamp, right.global_index) < (left.timestamp, left.global_index):
            raise ValueError("episode timeline is not monotonic")
    start = max(0, current_position - history_frames + 1)
    window = list(timeline[start : current_position + 1])
    effective_length = len(window)
    if effective_length < history_frames:
        window = [timeline[0]] * (history_frames - effective_length) + window
    chosen = [window[int(index)] for index in uniform_positions(len(window), points)]
    if chosen[-1].global_index != current.global_index:
        raise AssertionError("H100 last sample is not the current frame")
    return chosen, effective_length


def discretize_history(points: Sequence[TimelinePoint], stats: StateQuantileStats) -> np.ndarray:
    qpos = np.asarray([point.qpos for point in points], dtype=np.float64)
    if qpos.shape != (HISTORY_POINTS, HISTORY_DIM):
        raise ValueError(f"sampled qpos has shape {qpos.shape}")
    return discretize_state_qpos(qpos, stats)


def history_sources(points: Sequence[TimelinePoint]) -> list[dict[str, Any]]:
    return [{
        "attempt_id": point.attempt_id,
        "frame_index": point.frame_index,
        "global_index": point.global_index,
        "timestamp": point.timestamp,
    } for point in points]


def helper_identity() -> dict[str, Any]:
    return {
        "prompt_profile": PROMPT_PROFILE,
        "history_frames": HISTORY_FRAMES,
        "sampled_points": HISTORY_POINTS,
        "history_dim": HISTORY_DIM,
        "includes_current": True,
        "attempt_policy": "continuous_within_episode",
        "episode_start_padding": "left_pad_episode_frame_0",
        "structured_suffix": "Answer:",
    }


__all__ = [
    "HISTORY_DIM", "HISTORY_FRAMES", "HISTORY_POINTS", "PROMPT_PROFILE",
    "TimelinePoint", "build_phase_prompt", "compact_history", "discretize_history",
    "helper_identity", "history_sources", "sample_episode_history", "uniform_positions",
]
