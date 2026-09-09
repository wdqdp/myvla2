from __future__ import annotations

import numpy as np

from tactile_vla.vla.v7_2_boundary_filter import detect_attempt_boundaries


def test_detect_attempt_boundaries_and_strict_order() -> None:
    actions = np.zeros((180, 7), dtype=np.float32)
    actions[:, 6] = 0.02
    actions[20:41, 6] = np.linspace(0.024, 0.104, 21)
    actions[41:, 6] = 0.104
    actions[70:111, 0] = np.linspace(0.003, 0.123, 41)
    actions[111:, 0] = 0.123
    actions[140:161, 6] = np.linspace(0.100, 0.020, 21)
    actions[161:, 6] = 0.020

    result = detect_attempt_boundaries(
        actions,
        move_start_frame=65,
        rexecution_frame=120,
    )

    assert result["events"] == {
        "gripper_motion_stop_frame": 40,
        "arm_adjustment_start_frame": 70,
        "arm_adjustment_stop_frame": 110,
        "gripper_close_start_frame": 140,
    }


def test_short_noise_is_not_selected_as_motion() -> None:
    actions = np.zeros((180, 7), dtype=np.float32)
    actions[:, 6] = 0.02
    actions[20:41, 6] = np.linspace(0.024, 0.104, 21)
    actions[41:, 6] = 0.104
    actions[63, 0] = 0.001
    actions[64, 0] = 0.0
    actions[75:116, 0] = np.linspace(0.003, 0.123, 41)
    actions[116:, 0] = 0.123
    actions[130, 6] = 0.103
    actions[131:, 6] = 0.104
    actions[145:166, 6] = np.linspace(0.100, 0.020, 21)
    actions[166:, 6] = 0.020

    result = detect_attempt_boundaries(
        actions,
        move_start_frame=65,
        rexecution_frame=120,
    )

    assert result["events"]["arm_adjustment_start_frame"] == 75
    assert result["events"]["gripper_close_start_frame"] == 145
