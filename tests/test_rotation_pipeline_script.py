from __future__ import annotations

from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rotation_moderately_v1.sh"


def test_rotation_pipeline_shell_syntax_and_help() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "stage-a" in result.stdout
    assert "stage-b" in result.stdout
    assert "serve" in result.stdout
    assert "client" in result.stdout


def test_rotation_pipeline_print_config_is_version_pinned() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "print-config"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "data_profile=rotation_moderately_success_v1" in result.stdout
    assert "prompt_profile=minimal_v1" in result.stdout
    assert "pi05_delta_tac_rotation_moderately_v1/10000" in result.stdout
    assert "pi05_stage_b_rotation_moderately_v1/merged_best" in result.stdout
