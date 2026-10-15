#!/usr/bin/env python3
"""V7.8/2000 asynchronous visual-state inference without tactile subscriptions."""

# ruff: noqa: E402
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import agilex_inference_v7_7_asyn as shared
from tactile_vla.vla.v7_8_phase_prompt import PROMPT_PROFILE
from tactile_vla.vla.v7_8_runtime import DATA_PROFILE
from tactile_vla.vla.v7_8_runtime import DEFAULT_NORM_DIR


def configure_version():
    shared.DATA_PROFILE = DATA_PROFILE
    shared.PROMPT_PROFILE = PROMPT_PROFILE
    shared.DEFAULT_NORM_STATS = DEFAULT_NORM_DIR / "norm_stats.json"
    shared.DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/v7_8_async"


def get_arguments(argv=None):
    configure_version()
    return shared.get_arguments(argv, visual_only=True)


def main():
    configure_version()
    shared.main(argument_parser=get_arguments, deployment_label="V7.8")


if __name__ == "__main__":
    main()
