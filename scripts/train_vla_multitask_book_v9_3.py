#!/usr/bin/env python3
"""Train book V9.3 five tasks from book Stage A step 15000."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_multitask_v7_7 as trainer
from scripts import train_vla_stage_b_v3 as training_base
from tactile_vla.vla.book_stage_a_data import validate_training_index as validate_action_index
from tactile_vla.vla.book_v9_3_multitask_data import DATA_PROFILE, validate_index


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book")
_BASE_PARSE_ARGS = trainer.parse_args
_BASE_CONFIGURE = trainer.configure


def parse_args():
    defaults = {
        "--num-steps": "4000", "--eval-interval": "500",
        "--save-interval": "500", "--keep-period": "500",
        "--eval-max-need-samples": "4096",
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    args = _BASE_PARSE_ARGS()
    required = {
        "num_steps": 4000, "eval_interval": 500,
        "save_interval": 500, "keep_period": 500,
        "batch_size": 8, "seed": 42, "action_horizon": 30, "action_dim": 32,
        "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
        "max_token_len": 512, "reasoning_max_token_len": 320,
        "no_norm": False, "grammar_profile": "v3_full_v1",
        "paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora",
        "action_loss_weight": 1.0, "need_loss_weight": 1.0,
        "failure_loss_weight": 1.0, "plan_loss_weight": 1.0,
        "use_state_history": False, "state_history_len": 0,
    }
    mismatches = {
        name: (getattr(args, name), expected)
        for name, expected in required.items()
        if getattr(args, name) != expected
    }
    if mismatches:
        raise ValueError(f"Book V9.3 fixed training protocol mismatch: {mismatches}")
    return args


def configure_training():
    _BASE_CONFIGURE()
    training_base.EXTRA_CONFIG.update({
        "training_target_coverage": {
            "failure_reason": ["failure_reason=rotate right,grasp appropriate."],
            "recovery_plan": [
                "recovery_plan=move horizontally right moderately, move vertically none moderately."
            ],
        },
        "target_support_note": "book V4 has one real failure target, one real plan target, memory_length=1 only",
        "thresholds_status": "uncalibrated_placeholders_not_for_robot",
    })


def configure_version() -> None:
    # Reuse V7.7's five-task loop/model without editing its source file.
    trainer.DATA_PROFILE = DATA_PROFILE
    trainer.DEFAULT_INDEX = ROOT / "outputs/book_v9_3_multitask/book_v9_3_multitask_training_index.json"
    trainer.DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_book_stage_a_v1_no_history/15000"
    trainer.DEFAULT_OUTPUT = ROOT / "outputs/multitask_v9_3"
    trainer.DEFAULT_DATASET = ROOT / "lerobot_data/tactile_vla_rotation_v4"
    trainer.DEFAULT_NORM_STATS = ROOT / "outputs/rotation_v4/norm_stats"
    trainer.RUN_NAME = "pi05_book_v9_3_five_task_h100_no_history"
    trainer.VERSION_TAG = "book_v9_3"
    trainer.validate_index = validate_index
    trainer.validate_v7_4_adjustment_training_index = validate_action_index
    trainer.parse_args = parse_args
    trainer.configure = configure_training
    trainer.ROOT = ROOT


def main() -> None:
    configure_version()
    trainer.main()


if __name__ == "__main__":
    main()
