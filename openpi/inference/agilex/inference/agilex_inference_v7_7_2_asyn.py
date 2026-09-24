#!/usr/bin/env python3
"""Run the V7.7 async state machine with V7.7.2 identities and norm stats."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import agilex_inference_v7_7_asyn as runtime
from tactile_vla.vla.v7_7_2_multitask_data import DATA_PROFILE


def main() -> None:
    runtime.DATA_PROFILE = DATA_PROFILE
    runtime.DEFAULT_NORM_STATS = Path(
        "/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/rotation_v4_1_new_env/norm_stats/norm_stats.json"
    )
    runtime.DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/v7_7_2_async"
    runtime.main()


if __name__ == "__main__":
    main()
