from __future__ import annotations

# ruff: noqa: E402

import argparse
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
    assert sync_runtime.should_classify_adjustment_end(phase="adjustment", completed_raw_actions=30, feedback_count=100)


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


def test_classification_gripper_probe_remaps_only_open_seventh_dimension() -> None:
    qpos = np.asarray(
        [
            [1, 2, 3, 4, 5, 6, 0.078],
            [7, 8, 9, 10, 11, 12, 0.079],
            [13, 14, 15, 16, 17, 18, 0.080],
        ],
        dtype=np.float32,
    )
    mapped, count = sync_runtime.v75_async._remap_classification_gripper(
        qpos,
        open_threshold=0.079,
        open_value=0.0995,
    )
    np.testing.assert_allclose(mapped[:, :6], qpos[:, :6])
    np.testing.assert_allclose(mapped[:, 6], [0.078, 0.0995, 0.0995])
    np.testing.assert_array_equal(qpos[:, 6], np.asarray([0.078, 0.079, 0.080], dtype=np.float32))
    assert count == 2


def test_v7_5_async_cli_accepts_classification_gripper_probe() -> None:
    parser = argparse.ArgumentParser()
    sync_runtime.v75_async.add_classification_gripper_probe_arguments(parser)
    args = parser.parse_args(
        [
            "--classification-gripper-open-threshold",
            "0.079",
            "--classification-gripper-open-value",
            "0.0995",
        ]
    )
    sync_runtime.v75_async.validate_classification_gripper_probe_arguments(args, parser)
    assert args.classification_gripper_open_threshold == pytest.approx(0.079)
    assert args.classification_gripper_open_value == pytest.approx(0.0995)


def test_sync_classifier_converts_request_failure_to_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fail(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("connection lost")

    monkeypatch.setattr(sync_runtime.v75_async, "_run_async_adjustment_end_once", fail)
    history = deque(
        [
            (np.full(7, index, dtype=np.float32), float(index))
            for index in range(sync_runtime.ROLLING_QPOS_BUFFER_FRAMES)
        ],
        maxlen=sync_runtime.ROLLING_QPOS_BUFFER_FRAMES,
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
    forwarded = captured["feedback_qpos_h30"]
    assert len(forwarded) == sync_runtime.RUNTIME_PAST_QPOS_FRAMES
    np.testing.assert_array_equal(forwarded[0], np.full(7, 1, dtype=np.float32))
    np.testing.assert_array_equal(forwarded[-1], np.full(7, 99, dtype=np.float32))


@pytest.mark.parametrize(
    ("classifier_result", "expected_phases", "expected_calls"),
    [
        (False, ["execution"] * 4 + ["adjustment"] * 2, 2),
        (True, ["execution"] * 4 + ["adjustment", "execution"], 1),
    ],
)
def test_sync_preserves_execution_history_and_switches_only_on_true(
    monkeypatch: pytest.MonkeyPatch,
    classifier_result: bool,
    expected_phases: list[str],
    expected_calls: int,
) -> None:
    requested_phases: list[str] = []
    classifier_calls: list[int] = []
    feedback_timestamp = 0
    triggered = False

    def poll_key(*_args, **_kwargs):
        nonlocal triggered
        if len(requested_phases) == 4 and not triggered:
            triggered = True
            return "trigger"
        return None

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
        max_publish_step=180,
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
    assert classifier_calls[0] == 150
