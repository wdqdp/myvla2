from __future__ import annotations

from tactile_vla.vla.v7_4_adjustment_data import apply_v7_4_target_policy
from tactile_vla.vla.v8_2_adjustment_data import build_v8_2_exclusions


def _event(frame: int) -> dict[str, int]:
    return {"frame_index": frame, "global_index": frame}


def test_short_small_grasp_keeps_descent_in_pre_gap_compressed_h30() -> None:
    rows = [
        {
            "episode_id": 1,
            "attempt_id": 2,
            "task": "small_grasp",
            "split": "train",
            "global_index": frame,
            "frame_index": frame,
            "phase": "adjustment" if frame < 80 else "execution",
            "raw_chunk_phase_pure": frame + 29 < 80 or frame >= 80,
            "trainable": True,
        }
        for frame in range(101)
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
    idle = {
        **{frame: "post_gripper_motion_pre_arm_idle" for frame in range(41, 50)},
        **{frame: "post_arm_pre_close_idle" for frame in range(60, 90)},
    }

    exclusions, policies = build_v8_2_exclusions(
        rows,
        v7_2_exclusions=idle,
        boundary_payload=boundary,
        action_horizon=30,
    )
    assert policies[0]["mode"] == "short_small_grasp_supervised_by_pre_gap_compressed_h30"
    assert 29 not in exclusions
    assert 50 in exclusions

    filtered = [
        row
        | {
            "trainable": int(row["global_index"]) not in exclusions,
            "exclusion_reason": exclusions.get(int(row["global_index"])),
        }
        for row in rows
    ]
    transformed, _ = apply_v7_4_target_policy(
        filtered,
        boundary_payload=boundary,
        action_horizon=30,
    )
    row29 = transformed[29]
    assert row29["trainable"] is True
    assert row29["action_target_offsets"] == [*range(12), *range(21, 39)]
    assert 50 in [29 + offset for offset in row29["action_target_offsets"]]
