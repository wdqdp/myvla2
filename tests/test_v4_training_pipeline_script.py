from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v4_training_pipeline.sh"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_v4_training_pipeline_shell_syntax_help_and_no_eval() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert os.access(SCRIPT, os.X_OK)
    help_result = run_script("--help")
    assert "training-index, norm, stage-a-dry-run, stage-b-dry-run" in help_result.stdout
    assert "never starts training" in help_result.stdout
    assert re.search(r"\beval\b", SCRIPT.read_text()) is None


def test_default_is_zero_write_and_prints_all_commands(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset with spaces"
    profile_dir = tmp_path / "profile with spaces"
    norm_dir = tmp_path / "norm with spaces"
    stage_a_dir = tmp_path / "stage a output"
    stage_b_dir = tmp_path / "stage b output"
    result = run_script(
        "--dataset-dir",
        str(dataset_dir),
        "--profile-dir",
        str(profile_dir),
        "--norm-stats-dir",
        str(norm_dir),
        "--stage-a-output-dir",
        str(stage_a_dir),
        "--stage-b-output-dir",
        str(stage_b_dir),
    )

    assert "mode=dry-run from=training-index to=stage-b-dry-run" in result.stdout
    assert "[training-index]" in result.stdout
    assert "[norm]" in result.stdout
    assert "[stage-a-dry-run]" in result.stdout
    assert "[stage-b-dry-run]" in result.stdout
    assert "dataset\\ with\\ spaces" in result.stdout
    assert not dataset_dir.exists()
    assert not profile_dir.exists()
    assert not norm_dir.exists()
    assert not stage_a_dir.exists()
    assert not stage_b_dir.exists()


def test_command_snapshot_is_v4_pinned() -> None:
    result = run_script()
    lines = result.stdout.splitlines()
    training_index = next(line for line in lines if line.startswith("[training-index]"))
    norm = next(line for line in lines if line.startswith("[norm]"))
    stage_a = next(line for line in lines if line.startswith("[stage-a-dry-run]"))
    stage_b = next(line for line in lines if line.startswith("[stage-b-dry-run]"))

    assert "/data1/outputs/vla/rotation_v4/v4_training_index.json" in norm
    assert "/data1/outputs/vla/rotation_v4/reasoning_manifests" in training_index
    assert "--data-profile rotation_v4" in norm
    for command in (stage_a, stage_b):
        assert "--data-profile rotation_v4" in command
        assert "--prompt-profile minimal_v1" in command
        assert "--batch-size 8" in command
        assert "--action-horizon 30" in command
        assert "--action-dim 32" in command
        assert "--state-history-len 60" in command
        assert "--state-history-dim 7" in command
        assert "--dry-run" in command
    assert (
        "--checkpoint /home/test/.cache/modelscope/hub/models/hairuoliu/pi05_base/params"
        in stage_a
    )
    assert (
        "--stage-a-checkpoint "
        "/data1/outputs/vla/stage_a_action/pi05_delta_tac_rotation_v4/10000"
        in stage_b
    )
    assert "--reasoning-window-frames 15" in stage_b
    assert "--status-negative-ratio 3.0" in stage_b


def test_step_range_and_invalid_reverse_range() -> None:
    result = run_script("--from-step", "norm", "--to-step", "stage-a-dry-run")
    assert "[training-index]" not in result.stdout
    assert "[norm]" in result.stdout
    assert "[stage-a-dry-run]" in result.stdout
    assert "[stage-b-dry-run]" not in result.stdout

    invalid = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--from-step",
            "stage-b-dry-run",
            "--to-step",
            "norm",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert "--from-step must not be after --to-step" in invalid.stderr
