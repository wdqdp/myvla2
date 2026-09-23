from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tactile_vla.vla.v8_3_adjustment_data import (
    DOWN_MODERATELY_PLAN,
    apply_v8_3_short_descent_targets,
)


def _event(frame: int) -> dict[str, int]:
    return {"frame_index": frame, "global_index": frame}


def test_short_descent_targets_replace_ten_g_adjacent_h30s() -> None:
    actions = [np.full(7, frame, dtype=np.float32) for frame in range(100)]
    rows = [
        {
            "episode_id": 1,
            "attempt_id": 2,
            "frame_index": frame,
            "global_index": frame,
            "task": "small_grasp",
            "phase": "adjustment",
            "split": "train",
            "trainable": False,
            "exclusion_reason": "pre_arm_adjustment_stop_h30",
            "action_target_offsets": None,
            "raw_chunk_phase_pure": True,
        }
        for frame in range(31, 41)
    ]
    boundary = {
        "attempts": [
            {
                "episode_id": 1,
                "attempt_id": 2,
                "task": "small_grasp",
                "rexecution_frame": 80,
                "events": {
                    "gripper_motion_stop": _event(40),
                    "arm_adjustment_start": _event(50),
                    "arm_adjustment_stop": _event(59),
                    "gripper_close_start": _event(90),
                },
            }
        ]
    }
    lookup = {
        frame: SimpleNamespace(input_recovery_plan=DOWN_MODERATELY_PLAN)
        for frame in range(100)
    }

    result, summary = apply_v8_3_short_descent_targets(
        rows,
        boundary_payload=boundary,
        actions=actions,
        global_lookup=lookup,
        action_horizon=30,
    )

    assert summary["modified_chunk_count"] == 10
    row = next(row for row in result if row["frame_index"] == 40)
    target = np.asarray(row["action_target_values"], dtype=np.float32)
    assert row["trainable"] is True
    assert row["action_target_offsets"] is None
    assert target.shape == (30, 7)
    assert np.array_equal(target[0], actions[40])
    assert np.array_equal(target[1], actions[50])
    assert np.array_equal(target[-1], actions[59])
    assert np.all(np.diff(target[1:, 0]) >= 0)
