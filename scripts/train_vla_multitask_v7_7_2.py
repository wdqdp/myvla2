#!/usr/bin/env python3
"""Train V7.7.2 with new-environment data from V7.4.2 step 10000."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_multitask_v7_7 as trainer
from tactile_vla.vla.v7_4_2_adjustment_data import validate_training_index as validate_action_index
from tactile_vla.vla.v7_7_2_multitask_data import DATA_PROFILE, validate_index


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")


def configure_version() -> None:
    trainer.DATA_PROFILE = DATA_PROFILE
    trainer.DEFAULT_INDEX = ROOT / "outputs/rotation_v7_7_2_multitask/v7_7_2_multitask_training_index.json"
    trainer.DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_2_no_history/10000"
    trainer.DEFAULT_OUTPUT = ROOT / "outputs/multitask_v7_7_2"
    trainer.DEFAULT_DATASET = ROOT / "lerobot_data/tactile_vla_rotation_v4_1_new_env"
    trainer.DEFAULT_NORM_STATS = ROOT / "outputs/rotation_v4_1_new_env/norm_stats"
    trainer.RUN_NAME = "pi05_rotation_v7_7_2_five_task_h100_no_history"
    trainer.VERSION_TAG = "v7_7_2"
    trainer.validate_index = validate_index
    trainer.validate_v7_4_adjustment_training_index = validate_action_index
    trainer.ROOT = ROOT


def main() -> None:
    configure_version()
    trainer.main()


if __name__ == "__main__":
    main()
