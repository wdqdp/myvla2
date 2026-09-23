from __future__ import annotations

import numpy as np
import pytest

from tactile_vla.vla.v8_2_boundary_filter import detect_small_grasp_boundaries


def _signals() -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((100, 7), dtype=np.float64)
    actions[:, 6] = 0.02
    actions[10:21, 6] = np.linspace(0.02, 0.1, 11)
    actions[21:76, 6] = 0.1
    actions[76:87, 6] = np.linspace(0.092, 0.02, 11)
    actions[87:, 6] = 0.02

    z = np.zeros(100, dtype=np.float64)
    z[35:46] = np.linspace(0.0, -0.006, 11)
    z[46:51] = -0.006
    z[51:62] = np.linspace(-0.006, -0.012, 11)
    z[62:] = -0.012
    return actions, z


def test_detects_first_gripper_maximum_and_complete_descent() -> None:
    actions, z = _signals()
    result = detect_small_grasp_boundaries(actions, z, rexecution_frame=70)

    assert result["events"] == {
        "gripper_motion_stop": 20,
        "arm_adjustment_start": 36,
        "arm_adjustment_stop": 61,
        "gripper_close_start": 76,
    }
    assert result["signals"]["gripper_max"] == pytest.approx(0.1)
    assert result["signals"]["downward_drop_m"] == pytest.approx(-0.012)


def test_rejects_small_grasp_without_downward_motion() -> None:
    actions, z = _signals()
    z[:] = 0.0

    with pytest.raises(ValueError, match="No sustained downward"):
        detect_small_grasp_boundaries(actions, z, rexecution_frame=70)
