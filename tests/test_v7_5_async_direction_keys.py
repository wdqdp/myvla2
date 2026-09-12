from __future__ import annotations

# ruff: noqa: E402

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

import agilex_inference_forced_phase_anlation_v7_5_asyn_direction_keys as client


class _Keyboard:
    def __init__(self, key: str | None) -> None:
        self.key = key

    def get_key(self) -> str | None:
        return self.key


class _ImmediateFuture:
    def __init__(self, result) -> None:
        self._result = result

    def done(self) -> bool:
        return True

    def result(self):
        return self._result


class _ImmediateExecutor:
    def __init__(self, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass

    def submit(self, function, **kwargs):
        return _ImmediateFuture(function(**kwargs))


class _Operator:
    def is_shutdown(self) -> bool:
        return False

    def rate(self, _hz: float):
        return SimpleNamespace(sleep=lambda: None)


class _Policy:
    def get_server_metadata(self):
        return {"use_state_history": False, "adjustment_end_threshold": 0.8}


class _Logger:
    def __init__(self) -> None:
        self.rows = []

    def record(self, row) -> None:
        self.rows.append(row)


@pytest.mark.parametrize(
    ("key", "direction", "magnitude"),
    [
        ("a", "left", "moderately"),
        ("s", "left", "slightly"),
        ("d", "right", "slightly"),
        ("f", "right", "moderately"),
    ],
)
def test_direction_key_between_chunks_selects_plan(key: str, direction: str, magnitude: str) -> None:
    args = SimpleNamespace(quit_key="q")
    assert client._poll_key(args, _Keyboard(key), phase="execution") == "trigger"
    assert args.rotation_direction == direction
    assert args.rotation_magnitude == magnitude
    expected_failure, expected_plan = client.v52.rotation_targets(direction, magnitude)
    assert args.forced_failure_reason == expected_failure
    assert args.forced_recovery_plan == expected_plan
    assert args.adjustment_selection_key == key


@pytest.mark.parametrize("key", ["a", "s", "d", "f", " "])
def test_direction_and_space_keys_are_ignored_in_adjustment(key: str) -> None:
    args = SimpleNamespace(quit_key="q")
    assert client._poll_key(args, _Keyboard(key), phase="adjustment") is None
    assert client._poll_control_key(args, _Keyboard(key), phase="adjustment") is None


def test_direction_key_interrupts_execution_chunk_and_space_is_disabled() -> None:
    args = SimpleNamespace(quit_key="q")
    control = client._poll_control_key(args, _Keyboard("d"), phase="execution")
    assert control is not None
    assert control.kind == "trigger"
    assert control.key == "d"
    assert args.rotation_direction == "right"
    assert args.rotation_magnitude == "slightly"
    assert client._poll_control_key(args, _Keyboard(" "), phase="execution") is None


def test_q_quits_in_both_polling_locations() -> None:
    args = SimpleNamespace(quit_key="q")
    assert client._poll_key(args, _Keyboard("q"), phase="execution") == "quit"
    control = client._poll_control_key(args, _Keyboard("q"), phase="adjustment")
    assert control is not None and control.kind == "quit"


def test_direction_async_keeps_execution_feedback_for_first_adjustment_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_phases = []
    classifier_inputs = []
    feedback_timestamp = 0
    triggered = False

    def observation():
        return SimpleNamespace(qpos=np.zeros(7, dtype=np.float32))

    def poll_key(*_args, **_kwargs):
        nonlocal triggered
        if len(requested_phases) == 4 and not triggered:
            triggered = True
            return "trigger"
        return None

    def request_chunk(*, phase, **_kwargs):
        requested_phases.append(phase)
        return np.zeros((30, 32), dtype=np.float32), np.zeros((30, 7), dtype=np.float32)

    def wait_feedback(*_args, **_kwargs):
        nonlocal feedback_timestamp
        feedback_timestamp += 1
        return np.full(7, feedback_timestamp, dtype=np.float32), float(feedback_timestamp)

    def classify(**kwargs):
        classifier_inputs.append(kwargs)
        return SimpleNamespace(
            generation=kwargs["generation"],
            phase_index=kwargs["phase_index"],
            captured_step=kwargs["captured_step"],
            adjustment_end=False,
            submitted_monotonic=kwargs["submitted_monotonic"],
        )

    monkeypatch.setattr(client, "ThreadPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(client, "_poll_key", poll_key)
    monkeypatch.setattr(client.implementation, "validate_server_metadata", lambda *_args: None)
    monkeypatch.setattr(client.implementation.base, "load_state_quantiles", lambda *_args: object())
    monkeypatch.setattr(client.implementation.base, "_enter_adjustment", lambda **_kwargs: observation())
    monkeypatch.setattr(client.v52, "_capture_observation", lambda *_args, **_kwargs: observation())
    monkeypatch.setattr(client.v52, "_request_action_chunk", request_chunk)
    monkeypatch.setattr(client.v52, "_latest_feedback", lambda *_args: (None, None))
    monkeypatch.setattr(
        client.v52,
        "_publish_raw_action",
        lambda *, raw_action, **_kwargs: (np.asarray(raw_action).copy(), 1, None),
    )
    monkeypatch.setattr(client.implementation.v53, "_wait_feedback_after", wait_feedback)
    monkeypatch.setattr(client.implementation, "_run_async_adjustment_end_once", classify)
    monkeypatch.setattr(client.implementation, "_report_async_adjustment_end", lambda **_kwargs: None)
    monkeypatch.setattr(
        client.implementation.base,
        "should_submit_adjustment_end",
        lambda *, phase, feedback_count, request_active, **_kwargs: (
            phase == "adjustment" and feedback_count >= 99 and not request_active
        ),
    )

    args = SimpleNamespace(
        max_publish_step=122,
        chunk_size=30,
        publish_rate=30,
        adjustment_end_rate_hz=3.0,
        norm_stats_file=Path("unused.json"),
        quit_key="q",
    )
    logger = _Logger()
    client.run_v7_5_async_direction_keys(
        args,
        _Operator(),
        _Policy(),
        _Policy(),
        object(),
        object(),
        logger,
    )

    assert requested_phases[:5] == ["execution"] * 4 + ["adjustment"]
    assert classifier_inputs[0]["captured_step"] == 121
    assert len(classifier_inputs[0]["feedback_qpos_h30"]) == 99
    assert classifier_inputs[0]["feedback_timestamps"][0] == 23.0
    assert classifier_inputs[0]["feedback_timestamps"][-1] == 121.0
    run_start = next(row for row in logger.rows if row["event"] == "run_start")
    assert run_start["rolling_qpos_buffer_phases"] == ["execution", "adjustment"]
    assert run_start["rolling_qpos_buffer_reset_on_phase_transition"] is False
