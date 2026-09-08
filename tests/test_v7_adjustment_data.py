from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.v4_data import V4Frame  # noqa: E402
from tactile_vla.vla.v7_adjustment_data import (  # noqa: E402
    native_reexecution_timing_identity,
)


def _frame(frame_index: int, *, attempt_id: int = 1) -> V4Frame:
    return V4Frame(
        global_index=frame_index,
        lerobot_episode_index=0,
        episode_id=7,
        attempt_id=attempt_id,
        frame_index=frame_index,
        ros_timestamp=100.0 + frame_index / 30.0,
        schema_version="test",
        result="success",
        rotation_direction="right",
        grasp_position="moderate",
        horizontal_direction="left",
        horizontal_magnitude="slightly",
        valid=True,
        stage_a_eligible=True,
        execution_eligible=True,
        action_chunk_valid=True,
        tactile_caption="area=small",
        instruction="Pick object.",
        input_recovery_plan="move left",
    )


def test_native_reexecution_timing_is_used_directly() -> None:
    attempt1 = [_frame(index) for index in range(4)]
    attempt2 = [
        replace(_frame(index, attempt_id=2), global_index=4 + index)
        for index in range(5)
    ]
    v4_index = {
        "attempt_timing": {
            "episode7/attempt1": {
                "rexecution_frame_index": None,
                "rexecution_timestamp": None,
            },
            "episode7/attempt2": {
                "rexecution_frame_index": 3,
                "rexecution_timestamp": 100.1,
            },
        }
    }

    identity, boundaries = native_reexecution_timing_identity(v4_index, attempt1 + attempt2)

    assert boundaries == {(7, 1): None, (7, 2): 3}
    assert identity["attempt_count"] == 2
    assert identity["attempt2_count"] == 1
    assert len(identity["content_sha256"]) == 64
    assert len(identity["sha256"]) == 64


def test_native_reexecution_timing_rejects_missing_attempt2_boundary() -> None:
    frames = [replace(_frame(index, attempt_id=2), global_index=index) for index in range(5)]
    v4_index = {
        "attempt_timing": {
            "episode7/attempt2": {
                "rexecution_frame_index": None,
                "rexecution_timestamp": None,
            }
        }
    }
    with pytest.raises(ValueError, match="rexecution_frame_index"):
        native_reexecution_timing_identity(v4_index, frames)
