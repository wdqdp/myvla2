#!/usr/bin/env python3
"""V7.9 tactile async inference with raw adjustment keypoints for offline FK."""

# ruff: noqa: E402
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import agilex_inference_v7_7_2_asyn_record_adjustment as recording
import agilex_inference_v7_9_asyn as deployment
from tactile_vla.vla.v7_9_multitask_data import DATA_PROFILE
from tactile_vla.vla.v7_9_multitask_data import ROOT

DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/v7_9_adjustment_measurements"


def main():
    recording.main(
        data_profile=DATA_PROFILE,
        default_norm_stats=ROOT / "outputs/rotation_v4_1_new_env/norm_stats/norm_stats.json",
        default_log_root=DEFAULT_LOG_ROOT,
        argument_parser=deployment.get_arguments,
        deployment_label="V7.9",
    )


if __name__ == "__main__":
    main()
