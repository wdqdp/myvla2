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

from agilex_inference_forced_phase_anlation_5_3_asyn import _poll_key
from agilex_inference_forced_phase_anlation_5_3_asyn import _enter_adjustment
from agilex_inference_forced_phase_anlation_5_3_asyn import get_arguments
from agilex_inference_forced_phase_anlation_5_3_asyn import should_submit_adjustment_end
import agilex_inference_forced_phase_anlation_5_3_asyn as async_runtime


class _Keyboard:
    def __init__(self, key: str | None) -> None:
        self.key = key

    def get_key(self) -> str | None:
        return self.key


@pytest.mark.parametrize("feedback_count", [0, 1, 29])
def test_async_adjustment_end_waits_for_complete_h30(feedback_count: int) -> None:
    assert not should_submit_adjustment_end(
        phase="adjustment",
        feedback_count=feedback_count,
        request_active=False,
        now_monotonic=10.0,
        last_submit_monotonic=None,
        target_rate_hz=7.0,
    )


def test_async_adjustment_end_is_adjustment_only_and_one_request_at_a_time() -> None:
    common = {
        "feedback_count": 30,
        "now_monotonic": 10.0,
        "last_submit_monotonic": None,
        "target_rate_hz": 7.0,
    }
    assert should_submit_adjustment_end(phase="adjustment", request_active=False, **common)
    assert not should_submit_adjustment_end(phase="execution", request_active=False, **common)
    assert not should_submit_adjustment_end(phase="adjustment", request_active=True, **common)


def test_async_adjustment_end_rate_gate_is_seven_hz() -> None:
    interval = 1.0 / 7.0
    common = {
        "phase": "adjustment",
        "feedback_count": 30,
        "request_active": False,
        "last_submit_monotonic": 10.0,
        "target_rate_hz": 7.0,
    }
    assert not should_submit_adjustment_end(now_monotonic=10.0 + interval - 1e-6, **common)
    assert should_submit_adjustment_end(now_monotonic=10.0 + interval, **common)


def test_async_controls_ignore_s_and_space_is_execution_only() -> None:
    args = type("Args", (), {"quit_key": "q"})()
    assert _poll_key(args, _Keyboard("s"), phase="execution") is None
    assert _poll_key(args, _Keyboard("s"), phase="adjustment") is None
    assert _poll_key(args, _Keyboard(" "), phase="execution") == "trigger"
    assert _poll_key(args, _Keyboard(" "), phase="adjustment") is None
    assert _poll_key(args, _Keyboard("q"), phase="adjustment") == "quit"


def test_async_cli_removes_pause_and_history_freeze_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["v5_3_async", "--noise-seed", "42", "--gripper-min", "0.024"])
    args, _ = get_arguments()
    assert args.adjustment_end_rate_hz == 7.0
    assert not hasattr(args, "continue_key")
    assert not hasattr(args, "history_freeze_delay_seconds")


def test_space_transition_preserves_live_h60(monkeypatch: pytest.MonkeyPatch) -> None:
    observation = SimpleNamespace(
        img_front=np.zeros((2, 2, 3), dtype=np.uint8),
        img_left=np.zeros((2, 2, 3), dtype=np.uint8),
        qpos=np.zeros(7, dtype=np.float32),
        timestamp=123.0,
        tactile_caption="Touch[area=none]",
        state_history_mask=np.ones(60, dtype=np.bool_),
    )
    captured: dict[str, object] = {}

    def fake_capture(args, operator, captioner, *, reset_history, after_timestamp=None):
        captured["reset_history"] = reset_history
        captured["after_timestamp"] = after_timestamp
        return observation

    class Logger:
        def save_trigger_images(self, front, left):
            return {}

        def record(self, row):
            captured["record"] = row

    monkeypatch.setattr(async_runtime.v52, "_capture_observation", fake_capture)
    args = SimpleNamespace(
        rotation_direction="right",
        forced_failure_reason="failure_reason=rotate right,grasp appropriate.",
        forced_recovery_plan=(
            "recovery_plan=move horizontally right moderately, move vertically none moderately."
        ),
    )
    result = _enter_adjustment(
        args=args,
        operator=object(),
        captioner=object(),
        logger=Logger(),
        published_steps=12,
        discarded_raw_actions=18,
    )

    assert result is observation
    assert captured["reset_history"] is False
    assert captured["record"]["reset_state_history"] is False
