from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "infer_vla_four_adjustments.py"
SPEC = importlib.util.spec_from_file_location("infer_vla_four_adjustments", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_jobs_cover_four_combinations_and_gpus() -> None:
    assert [(job.gpu, job.direction, job.degree) for job in MODULE.JOBS] == [
        (0, "left", "slightly"),
        (1, "left", "moderately"),
        (2, "right", "slightly"),
        (3, "right", "moderately"),
    ]


def test_build_command_calls_single_inference_script(tmp_path: Path) -> None:
    job = MODULE.JOBS[0]
    command = MODULE.build_command(
        job,
        checkpoint=tmp_path / "model",
        episode=59,
        attempt=2,
        timestamp=1788593614.443176,
        result_path=tmp_path / "child.json",
    )
    assert command[0] == sys.executable
    assert command[1] == str(MODULE.INFERENCE_SCRIPT)
    assert command[command.index("--checkpoint") + 1] == str(tmp_path / "model")
    assert command[command.index("--mode") + 1] == "adjustment"
    assert command[command.index("--direction") + 1] == "left"
    assert command[command.index("--degree") + 1] == "slightly"
    assert command[command.index("--noise-seed") + 1] == "0"


def test_resolve_model_checkpoint_uses_stage_a_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "stage_a_action"
    model = model_root / "pi05_test_model"
    model.mkdir(parents=True)
    monkeypatch.setattr(MODULE, "MODEL_ROOT", model_root)
    assert MODULE.resolve_model_checkpoint("pi05_test_model") == model


def test_resolve_model_checkpoint_accepts_v7_7_2_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "multitask_v7_7_2" / "run"
    run.mkdir(parents=True)
    monkeypatch.setattr(MODULE, "V7_7_2_RUN", run)
    assert MODULE.resolve_model_checkpoint("v7_7_2") == run


def test_resolve_model_checkpoint_accepts_explicit_step_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pi05_test_model" / "15000"
    checkpoint.mkdir(parents=True)
    assert MODULE.resolve_model_checkpoint(str(checkpoint)) == checkpoint


def test_resolve_model_checkpoint_rejects_current_directory_alias() -> None:
    with pytest.raises(ValueError, match="folder name or checkpoint path"):
        MODULE.resolve_model_checkpoint(".")


def test_endpoint_change_returns_only_final_delta(tmp_path: Path) -> None:
    endpoint = {
        "translation_mm": {"x": 1.0, "y": 2.0, "z": 3.0},
        "rotation_zyx_deg": {"z_yaw": 4.0, "y_pitch": 5.0, "x_roll": 6.0},
        "gripper": 0.01,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "h30_action": [[1, 2, 3]],
                "h30_cartesian_changes": {
                    "endpoint_change_from_current": endpoint,
                    "frames": [{"target_pose_in_base": {"position_m": {"x": 1}}}],
                },
            }
        ),
        encoding="utf-8",
    )
    assert MODULE.endpoint_change(result_path) == endpoint


def test_endpoint_change_rejects_incomplete_result(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "h30_cartesian_changes": {
                    "endpoint_change_from_current": {"translation_mm": {}}
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing"):
        MODULE.endpoint_change(result_path)
