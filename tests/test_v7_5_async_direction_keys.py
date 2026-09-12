from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
from types import SimpleNamespace
import sys

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
