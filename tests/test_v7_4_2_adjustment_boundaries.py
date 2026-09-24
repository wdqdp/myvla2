from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.v7_4_2_adjustment_boundaries import detect_adjustment_boundaries


def test_detects_short_idle_and_caps_arm_close_overlap() -> None:
    actions = np.zeros((100, 7), dtype=np.float64)
    actions[:, 6] = 0.03
    actions[10:21, 6] = np.linspace(0.03, 0.10, 11)
    actions[21:70, 6] = 0.10
    actions[70:81, 6] = np.linspace(0.10, 0.03, 11)
    actions[81:, 6] = 0.03
    actions[22:73, 0] = np.linspace(0.0, 0.20, 51)
    actions[73:, 0] = 0.20

    result = detect_adjustment_boundaries(
        frame_indices=np.arange(100),
        timestamps=np.arange(100, dtype=np.float64) / 30.0,
        actions=actions,
    )

    events = result["events"]
    assert events["arm_adjustment_start"]["frame_index"] >= 20
    close_start = result["diagnostics"]["first_gripper_close"]["start"]
    assert events["arm_adjustment_stop"]["frame_index"] < close_start
    assert result["diagnostics"]["stop_capped_before_gripper_close"] is True


def test_rejects_missing_close() -> None:
    actions = np.zeros((80, 7), dtype=np.float64)
    actions[10:21, 6] = np.linspace(0.0, 0.10, 11)
    actions[21:, 6] = 0.10
    actions[25:60, 0] = np.linspace(0.0, 0.20, 35)

    try:
        detect_adjustment_boundaries(
            frame_indices=np.arange(80),
            timestamps=np.arange(80, dtype=np.float64),
            actions=actions,
        )
    except ValueError as error:
        assert "closing" in str(error)
    else:
        raise AssertionError("missing gripper close must be rejected")
