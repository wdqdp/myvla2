from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vla.runtime.state_history import StateHistoryBuffer  # noqa: E402
from tactile_vla.runtime.state_history import pad_state_history  # noqa: E402
from tactile_vla.vla.openpi_bridge import NormalizeStateHistory  # noqa: E402


def test_pad_state_history_repeats_earliest_and_masks_padding() -> None:
    states = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    history, mask = pad_state_history(states, history_len=4, state_dim=2)

    np.testing.assert_array_equal(history, [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(mask, [False, False, True, True])


def test_state_history_buffer_resamples_200hz_callbacks_to_two_seconds_at_30hz() -> None:
    buffer = StateHistoryBuffer(history_len=60, state_dim=2, history_fps=30.0)
    for index in range(601):
        timestamp = index / 200.0
        buffer.push(timestamp, np.asarray([timestamp, 10.0 * timestamp]))

    history, mask = buffer.snapshot(current_timestamp=3.0, current_state=np.asarray([3.0, 30.0]))

    target_timestamps = 3.0 + np.arange(-59, 1) / 30.0
    np.testing.assert_allclose(history[:, 0], target_timestamps, atol=1.0 / 400.0 + 1e-6)
    np.testing.assert_allclose(history[:, 1], 10.0 * target_timestamps, atol=10.0 / 400.0 + 1e-6)
    np.testing.assert_array_equal(mask, np.ones(60, dtype=np.bool_))
    assert history[-1, 0] - history[0, 0] > 1.96


def test_state_history_buffer_uses_nearest_timestamp_and_prefers_older_on_tie() -> None:
    buffer = StateHistoryBuffer(
        history_len=3,
        state_dim=1,
        history_fps=2.0,
        max_sample_gap_seconds=0.1,
    )
    buffer.push(0.0, np.asarray([0.0]))
    buffer.push(0.48, np.asarray([48.0]))
    buffer.push(0.52, np.asarray([52.0]))
    buffer.push(1.2, np.asarray([120.0]))  # Newer than the synchronized image/state.

    history, mask = buffer.snapshot(current_timestamp=1.0, current_state=np.asarray([100.0]))

    np.testing.assert_array_equal(history[:, 0], [0.0, 48.0, 100.0])
    np.testing.assert_array_equal(mask, [True, True, True])


def test_state_history_buffer_left_pads_and_masks_before_attempt_start() -> None:
    buffer = StateHistoryBuffer(history_len=4, state_dim=1, history_fps=2.0)
    buffer.push(10.0, np.asarray([5.0]))

    history, mask = buffer.snapshot(current_timestamp=10.0, current_state=np.asarray([5.0]))

    np.testing.assert_array_equal(history[:, 0], [5.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_equal(mask, [False, False, False, True])


def test_state_history_buffer_masks_large_callback_gap() -> None:
    buffer = StateHistoryBuffer(
        history_len=3,
        state_dim=1,
        history_fps=2.0,
        max_sample_gap_seconds=0.02,
    )
    buffer.push(0.0, np.asarray([0.0]))
    buffer.push(0.4, np.asarray([4.0]))

    history, mask = buffer.snapshot(current_timestamp=1.0, current_state=np.asarray([10.0]))

    np.testing.assert_array_equal(history[:, 0], [0.0, 4.0, 10.0])
    np.testing.assert_array_equal(mask, [True, False, True])


def test_history_uses_current_state_quantile_statistics() -> None:
    stats = SimpleNamespace(
        mean=np.zeros((2,), dtype=np.float32),
        std=np.ones((2,), dtype=np.float32),
        q01=np.asarray([0.0, 10.0], dtype=np.float32),
        q99=np.asarray([10.0, 20.0], dtype=np.float32),
    )
    data = {"state_history": np.asarray([[0.0, 10.0], [5.0, 15.0], [10.0, 20.0]], dtype=np.float32)}

    result = NormalizeStateHistory({"state": stats}, use_quantiles=True)(data)

    np.testing.assert_allclose(result["state_history"], [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], atol=1e-6)
