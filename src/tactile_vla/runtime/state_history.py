"""Thread-safe dense robot-state history for online VLA inference."""

from __future__ import annotations

from collections import deque
import math
import threading

import numpy as np


DEFAULT_STATE_HISTORY_FPS = 30.0


def pad_state_history(
    states: np.ndarray,
    *,
    history_len: int,
    state_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the newest states and left-pad by repeating the earliest valid state."""

    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != state_dim:
        raise ValueError(f"Expected state sequence [N,{state_dim}], got {states.shape}")
    if states.shape[0] == 0:
        raise ValueError("At least one state is required to build a history window")
    states = states[-history_len:]
    valid_count = states.shape[0]
    history = np.repeat(states[:1], history_len, axis=0)
    history[-valid_count:] = states
    mask = np.zeros((history_len,), dtype=np.bool_)
    mask[-valid_count:] = True
    return history, mask


class StateHistoryBuffer:
    """Resample high-rate timestamped qpos callbacks onto the model history grid."""

    def __init__(
        self,
        *,
        history_len: int = 60,
        state_dim: int = 7,
        history_fps: float = DEFAULT_STATE_HISTORY_FPS,
        retention_margin_seconds: float = 0.5,
        max_sample_gap_seconds: float = 0.02,
    ) -> None:
        if history_len <= 0 or state_dim <= 0 or history_fps <= 0:
            raise ValueError("history_len, state_dim, and history_fps must be positive")
        if retention_margin_seconds < 0 or max_sample_gap_seconds <= 0:
            raise ValueError("retention_margin_seconds must be non-negative and max_sample_gap_seconds positive")
        self.history_len = int(history_len)
        self.state_dim = int(state_dim)
        self.history_fps = float(history_fps)
        self.history_span_seconds = (self.history_len - 1) / self.history_fps
        self.retention_seconds = self.history_span_seconds + float(retention_margin_seconds)
        self.max_sample_gap_seconds = float(max_sample_gap_seconds)
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def push(self, timestamp: float, state: np.ndarray) -> None:
        timestamp = float(timestamp)
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected state [{self.state_dim}], got {state.shape}")
        if not math.isfinite(timestamp) or not np.isfinite(state).all():
            raise ValueError("State-history timestamp and qpos must be finite")
        with self._lock:
            sample = (timestamp, state.copy())
            if not self._samples or timestamp > self._samples[-1][0]:
                self._samples.append(sample)
            elif timestamp == self._samples[-1][0]:
                self._samples[-1] = sample
            else:
                # A ROS publisher should normally be monotonic, but sorting a
                # short time window makes the runtime robust to reordered callbacks.
                samples = list(self._samples)
                insert_at = int(np.searchsorted([stamp for stamp, _ in samples], timestamp, side="left"))
                if insert_at < len(samples) and samples[insert_at][0] == timestamp:
                    samples[insert_at] = sample
                else:
                    samples.insert(insert_at, sample)
                self._samples = deque(samples)

            newest_timestamp = self._samples[-1][0]
            cutoff = newest_timestamp - self.retention_seconds
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

    def snapshot(self, *, current_timestamp: float, current_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest-neighbor history ending at the image-synchronized state.

        The output timestamps are ``current_timestamp + [-T+1, ..., 0] /
        history_fps``. Targets preceding the first callback after ``clear`` are
        left-padded and masked. A nearest callback farther than
        ``max_sample_gap_seconds`` is also masked.
        """

        current_timestamp = float(current_timestamp)
        current_state = np.asarray(current_state, dtype=np.float32)
        if current_state.shape != (self.state_dim,):
            raise ValueError(f"Expected current_state [{self.state_dim}], got {current_state.shape}")
        if not math.isfinite(current_timestamp) or not np.isfinite(current_state).all():
            raise ValueError("Current state-history timestamp and qpos must be finite")
        with self._lock:
            samples = [(stamp, state.copy()) for stamp, state in self._samples if stamp <= current_timestamp + 1e-6]

        if samples and abs(samples[-1][0] - current_timestamp) <= 1e-6:
            samples[-1] = (current_timestamp, current_state.copy())
        else:
            samples.append((current_timestamp, current_state.copy()))

        timestamps = np.asarray([stamp for stamp, _ in samples], dtype=np.float64)
        states = np.stack([state for _, state in samples], axis=0)
        target_timestamps = current_timestamp + (
            np.arange(self.history_len, dtype=np.float64) - (self.history_len - 1)
        ) / self.history_fps

        right = np.searchsorted(timestamps, target_timestamps, side="left")
        left = np.clip(right - 1, 0, timestamps.shape[0] - 1)
        right = np.clip(right, 0, timestamps.shape[0] - 1)
        left_distance = np.abs(target_timestamps - timestamps[left])
        right_distance = np.abs(timestamps[right] - target_timestamps)
        # Match the offline synchronization behavior: ties select the older
        # sample because its linear search only replaces on a strictly smaller distance.
        nearest = np.where(right_distance < left_distance, right, left)
        nearest_distance = np.minimum(left_distance, right_distance)

        history = states[nearest].astype(np.float32, copy=True)
        in_available_range = (target_timestamps >= timestamps[0] - 1e-6) & (
            target_timestamps <= timestamps[-1] + 1e-6
        )
        mask = in_available_range & (nearest_distance <= self.max_sample_gap_seconds)

        # The last element must exactly equal the state synchronized with the
        # current images, independent of timestamp roundoff or duplicate messages.
        history[-1] = current_state
        mask[-1] = True
        return history, mask.astype(np.bool_)
