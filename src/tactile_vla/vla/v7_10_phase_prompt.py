"""V7.10: 11 qpos points from a 1/31/61-frame episode-continuous window."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tactile_vla.vla.v7_7_phase_prompt import TimelinePoint, compact_history, uniform_positions
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt as build_source_phase_prompt

WINDOW_FRAMES = {"0s": 1, "1s": 31, "2s": 61}
HISTORY_POINTS = 11
HISTORY_DIM = 7
NOMINAL_FPS = 30
SOURCE_HISTORY_PREFIX = "State history H100 sampled to 11 points: "


def history_frames(window: str) -> int:
    if window not in WINDOW_FRAMES:
        raise ValueError(f"V7.10 history window must be one of {tuple(WINDOW_FRAMES)}, got {window!r}")
    return WINDOW_FRAMES[window]


def prompt_profile(window: str) -> str:
    return f"phase_v7_10_h{history_frames(window)}_11_answer"


def history_prefix(window: str) -> str:
    return f"State history H{history_frames(window)} sampled to 11 points: "


def history_policy(window: str) -> dict:
    frames = history_frames(window)
    return {
        "prompt_profile": prompt_profile(window),
        "history_window": window,
        "history_frames": frames,
        "nominal_fps": NOMINAL_FPS,
        "nominal_span_seconds": (frames - 1) / NOMINAL_FPS,
        "window_policy": "last_n_raw_frames_including_current",
        "actual_time_span": "audit_source_timestamps_not_strict_wall_clock_cutoff",
        "sampled_points": HISTORY_POINTS,
        "history_dim": HISTORY_DIM,
        "sample_offsets": uniform_positions(frames).tolist(),
        "includes_current": True,
        "attempt_policy": "continuous_within_episode",
        "episode_start_padding": "left_pad_episode_frame_0",
        "idle_perturbation": "none_raw_contiguous",
        "structured_suffix": "Answer:",
        "affected_tasks": ["adjustment", "need", "failure"],
        "current_qpos": "unchanged",
    }


def sample_history(
    timeline: Sequence[TimelinePoint], current_position: int, window: str,
) -> tuple[list[TimelinePoint], int]:
    """Sample a prevalidated episode timeline, padding before subsampling.

    The integer offsets exactly follow V7.7. For 0s all 11 offsets are zero,
    so every point is the real current qpos, including at the episode start.
    """
    if not 0 <= current_position < len(timeline):
        raise IndexError(current_position)
    frames = history_frames(window)
    current = timeline[current_position]
    offsets = uniform_positions(frames)
    positions = np.maximum(0, current_position - frames + 1 + offsets)
    points = [timeline[int(position)] for position in positions]
    if any(point.episode_id != current.episode_id for point in points):
        raise ValueError("V7.10 history crosses episode boundaries")
    if any(point.timestamp > current.timestamp for point in points):
        raise ValueError("V7.10 history contains a future observation")
    if points[-1].global_index != current.global_index:
        raise AssertionError("V7.10 history must end at the current frame")
    return points, min(frames, current_position + 1)


def replace_history(prompt: str, values, window: str) -> str:
    """Change only the final history field, retaining Touch and recovery plan."""
    prefix, marker, previous = prompt.partition(SOURCE_HISTORY_PREFIX)
    if not marker or SOURCE_HISTORY_PREFIX in previous or "\n" in previous:
        raise ValueError("V7.10 expects one final V7.7.2 H100 history field")
    if not prefix.startswith("Mode: phase\n"):
        raise ValueError("V7.10 history substitution only accepts phase prompts")
    return prefix + history_prefix(window) + compact_history(values)


def build_phase_prompt(*, window: str, qpos_h100_11_discrete, **fields) -> str:
    """Adapt the shared runtime's keyword to the exact V7.10 training text."""
    original = build_source_phase_prompt(qpos_h100_11_discrete=qpos_h100_11_discrete, **fields)
    return replace_history(original, qpos_h100_11_discrete, window)
