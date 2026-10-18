#!/usr/bin/env python3
"""V7.9 identity wrapper; preserve the tactile V7.7 async state machine exactly."""
# ruff: noqa: E402
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import agilex_inference_v7_7_asyn as runtime
from tactile_vla.vla.v7_9_multitask_data import DATA_PROFILE
from tactile_vla.vla.v7_9_multitask_data import ROOT


def main():
    runtime.DATA_PROFILE = DATA_PROFILE
    runtime.DEFAULT_NORM_STATS = ROOT / "outputs/rotation_v4_1_new_env/norm_stats/norm_stats.json"
    runtime.DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/v7_9_async"
    runtime.main()


if __name__ == "__main__":
    main()
