#!/usr/bin/env python3
"""Train Book V9.4 five tasks from the new Stage A, with configurable update steps."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_multitask_v7_7 as trainer
from scripts import train_vla_stage_b_v3 as training_base
from tactile_vla.vla.book_stage_a_data import validate_training_index as validate_action_index
from tactile_vla.vla.book_v9_4_memory import MEMORY_POLICY, merge_plan_group_metrics, plan_eval_groups
from tactile_vla.vla.book_v9_4_multitask_data import (
    ROOT,
    DEFAULT_INDEX,
    DEFAULT_STAGE_A,
    DATA_PROFILE,
    validate_index,
    validate_schedule,
    validate_stage_a_for_training,
)
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v4_data import scan_v4_lerobot_frames

_BASE_PARSE_ARGS = trainer.parse_args
_BASE_CONFIGURE = trainer.configure
_BASE_EVALUATE_TEXT = training_base.evaluate_text


def ensure_index(args, *, index_validator=None):
    index = json.loads(args.index_file.read_text())
    (validate_index if index_validator is None else index_validator)(index)
    if Path(index["dataset_dir"]).resolve() != args.dataset_dir.resolve():
        raise ValueError("V9.4 multitask index was built for a different LeRobot dataset")
    norm_file = (args.norm_stats_dir / "norm_stats.json").resolve()
    if index["source_hashes"].get(str(norm_file)) != sha256_file(norm_file):
        raise ValueError("V9.4 multitask index and norm stats differ")
    model_identity = validate_stage_a_for_training(args.stage_a_checkpoint, index)
    args.eval_max_need_samples = min(
        args.eval_max_need_samples,
        max(index["splits"]["val"][task]["sample_count"] for task in ("need", "adjustment")),
    )
    training_base.EXTRA_CONFIG.update(
        {
            "training_target_coverage": index["training_target_coverage"],
            "captioner_identity": index["captioner_identity"],
            "plan_token_validation": index["plan_token_validation"],
            "stage_a_initialization_identity": model_identity,
        }
    )
    return index, scan_v4_lerobot_frames(args.dataset_dir)


def evaluate_text(state, loader, task, grammar, data_sharding, *, max_samples):
    if task != "plan":
        return _BASE_EVALUATE_TEXT(state, loader, task, grammar, data_sharding, max_samples=max_samples)
    from torch.utils.data import Subset

    raw = loader.dataset.dataset
    groups = plan_eval_groups(raw.rows, raw.row_indices)
    metrics = {}
    for key, positions in sorted(groups.items()):
        group_loader = training_base._loader(
            Subset(loader.dataset, positions), batch_size=loader.batch_size, num_workers=0, shuffle=False
        )
        # Evaluate all F+14 variants, not a prefix that could exclude long histories.
        metrics[key] = _BASE_EVALUATE_TEXT(state, group_loader, task, grammar, data_sharding, max_samples=None)
    return merge_plan_group_metrics(metrics)


def parse_args():
    # The reused CLI inserts defaults using separate tokens; normalize --flag=value first.
    normalized = [sys.argv[0]]
    for argument in sys.argv[1:]:
        normalized.extend(argument.split("=", 1) if argument.startswith("--") and "=" in argument else [argument])
    sys.argv[:] = normalized
    for flag, value in {
        "--num-steps": "4000",
        "--eval-interval": "2000",
        "--save-interval": "2000",
        "--keep-period": "2000",
        "--eval-max-need-samples": "2147483647",
    }.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    args = _BASE_PARSE_ARGS()
    validate_schedule(
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
    )
    required = {
        "batch_size": 8,
        "seed": 42,
        "action_horizon": 30,
        "action_dim": 32,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "max_token_len": 512,
        "reasoning_max_token_len": 320,
        "no_norm": False,
        "grammar_profile": "v3_full_v1",
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
        "action_loss_weight": 1.0,
        "need_loss_weight": 1.0,
        "failure_loss_weight": 1.0,
        "plan_loss_weight": 1.0,
        "use_state_history": False,
        "state_history_len": 0,
    }
    mismatch = {key: (getattr(args, key), value) for key, value in required.items() if getattr(args, key) != value}
    if mismatch:
        raise ValueError(f"Book V9.4 training protocol mismatch: {mismatch}")
    return args


def configure_training():
    _BASE_CONFIGURE()
    training_base.EXTRA_CONFIG.update(
        {
            "target_support_note": "source-derived Book V9.4 targets; memory_length=1,2,3,4; synthetic text history only; no action/class counterfactuals",
            "plan_memory_policy": MEMORY_POLICY,
            "plan_eval_policy": "all_F_plus_14_variants_grouped_by_memory_length_and_left_right",
            "thresholds_status": "uncalibrated_placeholders_not_for_robot",
            "training_step_policy": "configurable_total_updates_divided_equally_across_five_tasks",
        }
    )


def configure_version():
    trainer.DATA_PROFILE = DATA_PROFILE
    trainer.DEFAULT_INDEX = DEFAULT_INDEX
    trainer.DEFAULT_STAGE_A = DEFAULT_STAGE_A
    trainer.DEFAULT_OUTPUT = ROOT / "outputs/multitask_v9_4"
    trainer.DEFAULT_DATASET = ROOT / "lerobot_data/tactile_vla_rotation_v4"
    trainer.DEFAULT_NORM_STATS = ROOT / "outputs/rotation_v4/norm_stats"
    trainer.RUN_NAME = "pi05_book_v9_4_five_task_h100_no_history"
    trainer.VERSION_TAG = "book_v9_4"
    trainer.validate_index = validate_index
    trainer.validate_v7_4_adjustment_training_index = validate_action_index
    trainer.parse_args = parse_args
    trainer.configure = configure_training
    trainer.ROOT = ROOT
    training_base.evaluate_text = evaluate_text
    trainer.ensure_index = ensure_index


def main():
    configure_version()
    trainer.main()


if __name__ == "__main__":
    main()
