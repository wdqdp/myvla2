from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset
from tactile_vla.vla.v7_4_adjustment_data import apply_v7_4_target_policy
from tactile_vla.vla.v7_4_adjustment_data import compressed_h30_offsets


def _boundary() -> dict:
    return {
        "attempts": [
            {
                "episode_id": 7,
                "attempt_id": 2,
                "rexecution_frame": 180,
                "events": {
                    "gripper_motion_stop": {"frame_index": 100},
                    "arm_adjustment_start": {"frame_index": 130},
                },
            }
        ]
    }


def _row(frame: int, *, trainable: bool = True, attempt: int = 2) -> dict:
    return {
        "schema_version": "v7_3",
        "data_profile": "rotation_phase_v7_3_adjustment",
        "experiment_kind": "v7_3",
        "global_index": 1_000 + frame,
        "episode_id": 7,
        "attempt_id": attempt,
        "frame_index": frame,
        "split": "train",
        "phase": "adjustment",
        "trainable": trainable,
        "effective_h30_modified": False,
    }


def test_compressed_h30_offsets_skip_idle_and_resume_at_arm_start() -> None:
    offsets = compressed_h30_offsets(
        start_frame=90,
        gripper_motion_stop=100,
        arm_adjustment_start=130,
        action_horizon=30,
    )
    assert offsets == [*range(11), *range(40, 59)]
    assert len(offsets) == 30


def test_h30_ending_at_gripper_stop_does_not_need_rewrite() -> None:
    assert (
        compressed_h30_offsets(
            start_frame=71,
            gripper_motion_stop=100,
            arm_adjustment_start=130,
            action_horizon=30,
        )
        is None
    )


def test_policy_modifies_only_trainable_attempt2_window() -> None:
    rows = [_row(70), _row(71), _row(90), _row(100), _row(101), _row(95, trainable=False)]
    transformed, modifications = apply_v7_4_target_policy(
        rows,
        boundary_payload=_boundary(),
        action_horizon=30,
    )
    by_frame = {row["frame_index"]: row for row in transformed}
    assert by_frame[70]["action_target_offsets"] is None
    assert by_frame[71]["action_target_offsets"] is None
    assert by_frame[90]["action_target_offsets"] == [*range(11), *range(40, 59)]
    assert by_frame[100]["action_target_offsets"] == [0, *range(30, 59)]
    assert by_frame[101]["action_target_offsets"] is None
    assert by_frame[95]["action_target_offsets"] is None
    assert modifications["modified_chunk_count"] == 2
    assert modifications["modified_action_step_count"] == 48
    assert modifications["max_action_source_offset"] == 58


class _FakeLeRobotDataset:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action

    def __getitem__(self, index: int) -> dict:
        return {
            "index": index,
            "episode_id": 7,
            "attempt_id": 2,
            "frame_index": 90,
            "observation.images.front": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation.images.left": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation.state": np.zeros(7, dtype=np.float32),
            "instruction": "Pick object",
            "input_recovery_plan": (
                "recovery_plan=move horizontally left slightly, "
                "move vertically none moderately."
            ),
            "action": self.action,
        }


def test_frame_dataset_selects_non_contiguous_v7_4_targets() -> None:
    offsets = [*range(11), *range(40, 59)]
    actions = np.arange(59 * 7, dtype=np.float32).reshape(59, 7)
    phase_row = {
        **_row(90),
        "global_index": 1090,
        "action_target_offsets": offsets,
    }
    dataset = TactileVLAFrameDataset(
        dataset_dir="/unused",
        indices=[1090],
        stage="execution",
        action_horizon=30,
        prompt_profile="phase_v2",
        action_phase_by_global_index={1090: phase_row},
        lerobot_dataset=_FakeLeRobotDataset(actions),
    )
    sample = dataset[0]
    np.testing.assert_array_equal(sample["actions"], actions[offsets])
    assert sample["actions"].shape == (30, 7)
    assert dataset.max_action_offset == 58
