from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "infer_vla_action_from_dataset.py"
SPEC = importlib.util.spec_from_file_location("infer_vla_action_from_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from tactile_vla.vla.v5_adjustment_data import FKJoint  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import PiperFKChain  # noqa: E402


def test_resolve_checkpoint_selects_latest_numbered_step(tmp_path: Path) -> None:
    for step in (5000, 10000):
        (tmp_path / str(step) / "params").mkdir(parents=True)
    assert MODULE.resolve_checkpoint(tmp_path) == tmp_path / "10000"
    assert MODULE.resolve_checkpoint(tmp_path / "5000") == tmp_path / "5000"
    assert MODULE.resolve_checkpoint(tmp_path / "5000" / "params") == tmp_path / "5000" / "params"


def test_resolve_checkpoint_selects_best_multitask_step(tmp_path: Path) -> None:
    for step in (4000, 5000):
        (tmp_path / str(step) / "full_params").mkdir(parents=True)
    (tmp_path / "best").mkdir()
    (tmp_path / "best" / "metrics.json").write_text('{"step": 4000}')
    assert MODULE.resolve_checkpoint(tmp_path) == tmp_path / "4000"
    assert MODULE.resolve_checkpoint(tmp_path / "5000" / "full_params") == tmp_path / "5000" / "full_params"


def test_checkpoint_kind_detects_v7_7_2() -> None:
    config = {"data_profile": MODULE.V7_7_2_DATA_PROFILE}
    assert MODULE.checkpoint_kind("auto", config) == "v7-7-2"
    with pytest.raises(ValueError, match="Checkpoint is v7-7-2"):
        MODULE.checkpoint_kind("stage-a", config)


def test_adjustment_prompt_uses_cli_direction_and_degree() -> None:
    prompt, plan = MODULE.build_cli_prompt(
        mode="adjustment",
        instruction="Pick up and transfer the object stably.",
        direction="front",
        degree="slightly",
        prompt_profile="phase_v2",
    )
    assert plan == (
        "recovery_plan=move horizontally front slightly, "
        "move vertically none moderately."
    )
    assert prompt == (
        "Mode: adjustment. Task: Pick up and transfer the object stably.\n"
        "Put the object back, and follow this recovery plan: "
        "recovery_plan=move horizontally front slightly, move vertically none moderately."
    )


def test_execution_prompt_has_no_adjustment_plan() -> None:
    prompt, plan = MODULE.build_cli_prompt(
        mode="execution",
        instruction="Pick object",
        direction=None,
        degree=None,
        prompt_profile="phase_v2",
    )
    assert plan == "none"
    assert prompt == "Mode: execution. Task: Pick object."


def _write_identity_parquet(root: Path) -> None:
    destination = root / "data" / "chunk-000" / "episode_000000.parquet"
    destination.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "index": [10, 11, 12],
                "episode_index": [0, 0, 0],
                "episode_id": [7, 7, 7],
                "attempt_id": [2, 2, 2],
                "frame_index": [0, 1, 2],
                "ros_timestamp": [100.0, 100.04, 100.08],
            }
        ),
        destination,
    )


def test_locate_nearest_frame_uses_attempt_and_timestamp(tmp_path: Path) -> None:
    _write_identity_parquet(tmp_path)
    selected = MODULE.locate_nearest_frame(
        tmp_path,
        episode_id=7,
        attempt_id=2,
        timestamp=100.039,
        max_timestamp_error=0.02,
    )
    assert selected.global_index == 11
    assert selected.frame_index == 1
    assert selected.ros_timestamp == pytest.approx(100.04)
    assert selected.timestamp_error_seconds == pytest.approx(0.001)


def test_locate_nearest_frame_rejects_large_timestamp_error(tmp_path: Path) -> None:
    _write_identity_parquet(tmp_path)
    with pytest.raises(ValueError, match="exceeds --max-timestamp-error"):
        MODULE.locate_nearest_frame(
            tmp_path,
            episode_id=7,
            attempt_id=2,
            timestamp=101.0,
            max_timestamp_error=0.02,
        )


def _one_joint_chain(tmp_path: Path) -> PiperFKChain:
    fixed_offset = np.eye(4)
    fixed_offset[0, 3] = 1.0
    return PiperFKChain(
        source_path=tmp_path / "robot.urdf",
        source_sha256="0" * 64,
        base_link="base",
        end_link="tool",
        joints=(
            FKJoint(
                name="joint1",
                joint_type="revolute",
                origin=np.eye(4),
                axis=np.array([0.0, 0.0, 1.0]),
            ),
            FKJoint(name="tool", joint_type="fixed", origin=fixed_offset, axis=None),
        ),
        revolute_joint_count=1,
    )


def test_cartesian_description_reports_zyx_change_in_degrees(tmp_path: Path) -> None:
    chain = _one_joint_chain(tmp_path)
    actions = np.zeros((30, 7), dtype=np.float64)
    actions[:, 0] = np.linspace(0.0, np.pi / 2, 30)
    actions[:, 6] = np.linspace(0.0, 0.03, 30)
    result = MODULE.describe_h30_cartesian_changes(
        actions,
        current_state=np.zeros(7),
        chain=chain,
    )
    endpoint = result["endpoint_change_from_current"]
    assert endpoint["translation_mm"] == pytest.approx({"x": -1000.0, "y": 1000.0, "z": 0.0})
    assert endpoint["rotation_zyx_deg"] == pytest.approx(
        {"z_yaw": 90.0, "y_pitch": 0.0, "x_roll": 0.0}
    )
    assert endpoint["gripper"] == pytest.approx(0.03)
    assert len(result["frames"]) == 30
    assert result["frames"][-1]["time_offset_seconds"] == pytest.approx(29 / 30)


@pytest.mark.parametrize(
    ("mode", "direction", "degree", "message"),
    [
        ("adjustment", None, "slightly", "requires both"),
        ("adjustment", "left", None, "requires both"),
        ("execution", "left", "slightly", "only valid"),
    ],
)
def test_prompt_argument_validation(
    mode: str,
    direction: str | None,
    degree: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE.validate_prompt_arguments(mode, direction, degree)
