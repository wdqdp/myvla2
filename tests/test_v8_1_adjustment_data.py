from __future__ import annotations

# ruff: noqa: E402

import copy
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v8_1_adjustment_data import (
    BOUNDARY_POLICY,
    ROTATION_PHASE_V8_1_ADJUSTMENT,
    V8_1_BOUNDARY_SCHEMA,
    validate_boundary_artifact,
)


def _event(frame: int) -> dict[str, int | float]:
    return {"frame_index": frame, "global_index": frame, "timestamp": float(frame)}


def _payload(v4_index: Path, timing_sha: str) -> dict:
    rows = [
        {
            "episode_id": 1,
            "attempt_id": 2,
            "task": "moderate_lift",
            "rexecution_frame": 40,
            "events": {
                "gripper_motion_stop": _event(10),
                "arm_adjustment_start": _event(20),
                "arm_adjustment_stop": _event(30),
                "gripper_close_start": _event(40),
            },
            "excluded_intervals": [],
        },
        {
            "episode_id": 2,
            "attempt_id": 2,
            "task": "small_grasp",
            "rexecution_frame": 50,
            "events": {
                "gripper_motion_stop": _event(0),
                "arm_adjustment_start": _event(0),
                "arm_adjustment_stop": _event(50),
                "gripper_close_start": _event(50),
            },
            "excluded_intervals": [
                {"reason": "post_gripper_motion_pre_arm_idle", "excluded_global_indices": []},
                {"reason": "post_arm_pre_close_idle", "excluded_global_indices": []},
            ],
        },
    ]
    return {
        "schema_version": V8_1_BOUNDARY_SCHEMA,
        "data_profile": ROTATION_PHASE_V8_1_ADJUSTMENT,
        "boundary_policy": BOUNDARY_POLICY,
        "content_sha256": sha256_json(rows),
        "source_files": {
            "v4_training_index": {"path": str(v4_index), "sha256": sha256_file(v4_index)},
            "attempt_timing": {"sha256": timing_sha},
        },
        "attempts": rows,
    }


def test_boundary_validation_accepts_v7_4_strict_and_small_grasp_equal(tmp_path: Path) -> None:
    v4_index = tmp_path / "v4.json"
    v4_index.write_text("{}")
    timing = {"sha256": "timing"}
    attempts, exclusions = validate_boundary_artifact(
        _payload(v4_index, timing["sha256"]),
        v4_index_file=v4_index,
        timing_payload=timing,
    )
    assert len(attempts) == 2
    assert exclusions == {}


def test_boundary_validation_rejects_non_equal_small_grasp(tmp_path: Path) -> None:
    v4_index = tmp_path / "v4.json"
    v4_index.write_text("{}")
    timing = {"sha256": "timing"}
    payload = _payload(v4_index, timing["sha256"])
    payload["attempts"][1]["events"]["arm_adjustment_start"] = _event(1)
    payload["content_sha256"] = sha256_json(payload["attempts"])
    with pytest.raises(ValueError, match="equal-boundary"):
        validate_boundary_artifact(
            payload,
            v4_index_file=v4_index,
            timing_payload=timing,
        )


def test_boundary_validation_keeps_old_tasks_strict(tmp_path: Path) -> None:
    v4_index = tmp_path / "v4.json"
    v4_index.write_text("{}")
    timing = {"sha256": "timing"}
    payload = _payload(v4_index, timing["sha256"])
    payload["attempts"][0]["events"]["arm_adjustment_start"] = copy.deepcopy(
        payload["attempts"][0]["events"]["gripper_motion_stop"]
    )
    payload["content_sha256"] = sha256_json(payload["attempts"])
    with pytest.raises(ValueError, match="inherited boundary"):
        validate_boundary_artifact(
            payload,
            v4_index_file=v4_index,
            timing_payload=timing,
        )
