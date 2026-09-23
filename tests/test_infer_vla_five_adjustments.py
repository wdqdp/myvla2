from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "infer_vla_five_adjustments.py"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("infer_vla_five_adjustments", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_jobs_cover_four_horizontal_and_one_vertical_adjustment() -> None:
    assert [job.name for job in MODULE.JOBS] == [
        "left_slightly",
        "left_moderately",
        "right_slightly",
        "right_moderately",
        "down_moderately",
    ]
    assert [job.gpu for job in MODULE.JOBS] == [0, 1, 2, 3, 0]


def test_down_command_uses_vertical_cli_arguments(tmp_path: Path) -> None:
    job = MODULE.JOBS[-1]
    command = MODULE.build_command(
        job,
        checkpoint=tmp_path / "15000",
        episode=1,
        attempt=2,
        timestamp=123.0,
        result_path=tmp_path / "down.json",
    )
    assert command[command.index("--vertical-direction") + 1] == "down"
    assert command[command.index("--vertical-degree") + 1] == "moderately"
    assert "--direction" not in command
