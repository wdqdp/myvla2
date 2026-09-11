from __future__ import annotations

# ruff: noqa: E402

from collections import deque
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/inference/agilex/inference"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/packages/openpi-client/src"))

import agilex_inference_forced_phase_anlation_v7_5_sync as sync_runtime


class _Rate:
    def sleep(self) -> None:
        pass


class _Operator:
    def is_shutdown(self) -> bool:
        return False

    def rate(self, _hz: float) -> _Rate:
        return _Rate()


class _Policy:
    def get_server_metadata(self) -> dict[str, float]:
        return {"adjustment_end_threshold": 0.9}


class _Logger:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def record(self, row: dict[str, object]) -> None:
        self.rows.append(row)


def _observation() -> SimpleNamespace:
    return SimpleNamespace(qpos=np.zeros(7, dtype=np.float32))


def test_sync_classifier_runs_only_after_complete_adjustment_chunk_and_h99() -> None:
    common = {"completed_raw_actions": 30, "feedback_count": 99}
    assert sync_runtime.should_classify_adjustment_end(phase="adjustment", **common)
    assert not sync_runtime.should_classify_adjustment_end(phase="execution", **common)
    assert not sync_runtime.should_classify_adjustment_end(
        phase="adjustment", completed_raw_actions=29, feedback_count=99
    )
    assert not sync_runtime.should_classify_adjustment_end(
        phase="adjustment", completed_raw_actions=30, feedback_count=98
    )


def test_sync_cli_has_no_async_rate_or_pause_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["v7_5_sync", "--noise-seed", "42", "--gripper-min", "0.024"],
    )
    args, _ = sync_runtime.get_arguments()
    assert args.publish_rate == 30
    assert not hasattr(args, "adjustment_end_rate_hz")
    assert not hasattr(args, "continue_key")
    assert not hasattr(args, "history_freeze_delay_seconds")


def test_sync_classifier_converts_request_failure_to_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**_kwargs):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(sync_runtime.v75_async, "_run_async_adjustment_end_once", fail)
    history = deque(
        [(np.zeros(7, dtype=np.float32), float(index)) for index in range(sync_runtime.RUNTIME_PAST_QPOS_FRAMES)],
        maxlen=sync_runtime.RUNTIME_PAST_QPOS_FRAMES,
    )
    with pytest.raises(sync_runtime.v53.FailClosedError, match="synchronous V7.5"):
        sync_runtime._classify_adjustment_end_sync(
            args=SimpleNamespace(),
            policy=object(),
            operator=object(),
            captioner=object(),
            stats=object(),
            logger=object(),
            phase_index=3,
            published_steps=120,
            feedback_window=history,
        )


@pytest.mark.parametrize(
    ("classifier_result", "expected_phases", "expected_calls"),
    [
        (False, ["adjustment"] * 5, 2),
        (True, ["adjustment"] * 4 + ["execution"], 1),
    ],
)
def test_sync_false_continues_adjustment_and_true_switches_before_next_chunk(
    monkeypatch: pytest.MonkeyPatch,
    classifier_result: bool,
    expected_phases: list[str],
    expected_calls: int,
) -> None:
    requested_phases: list[str] = []
    classifier_calls: list[int] = []
    feedback_timestamp = 0

    def poll_key(*_args, **_kwargs):
        return "trigger" if not requested_phases else None

    def request_chunk(*, phase, **_kwargs):
        requested_phases.append(phase)
        return np.zeros((30, 32), dtype=np.float32), np.zeros((30, 7), dtype=np.float32)

    def publish_action(*, raw_action, **_kwargs):
        return np.asarray(raw_action).copy(), 1, None

    def wait_feedback(*_args, **_kwargs):
        nonlocal feedback_timestamp
        feedback_timestamp += 1
        return np.zeros(7, dtype=np.float32), float(feedback_timestamp)

    def classify(*, published_steps, **_kwargs):
        classifier_calls.append(published_steps)
        return classifier_result

    monkeypatch.setattr(sync_runtime.v75_async, "validate_server_metadata", lambda *_args: None)
    monkeypatch.setattr(sync_runtime, "load_state_quantiles", lambda *_args: object())
    monkeypatch.setattr(sync_runtime.async_base, "_poll_key", poll_key)
    monkeypatch.setattr(sync_runtime.async_base, "_enter_adjustment", lambda **_kwargs: _observation())
    monkeypatch.setattr(sync_runtime.v52, "_capture_observation", lambda *_args, **_kwargs: _observation())
    monkeypatch.setattr(sync_runtime.v52, "_request_action_chunk", request_chunk)
    monkeypatch.setattr(sync_runtime.v52, "_latest_feedback", lambda *_args: (None, None))
    monkeypatch.setattr(sync_runtime.v52, "_publish_raw_action", publish_action)
    monkeypatch.setattr(sync_runtime.v53, "_wait_feedback_after", wait_feedback)
    monkeypatch.setattr(sync_runtime, "_classify_adjustment_end_sync", classify)

    args = SimpleNamespace(
        max_publish_step=150,
        chunk_size=30,
        publish_rate=30,
        norm_stats_file=Path("unused.json"),
    )
    sync_runtime.run_v7_5_sync(
        args,
        _Operator(),
        _Policy(),
        object(),
        object(),
        _Logger(),
    )

    assert requested_phases == expected_phases
    assert len(classifier_calls) == expected_calls
    assert classifier_calls[0] == 120
