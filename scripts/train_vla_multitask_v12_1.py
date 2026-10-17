#!/usr/bin/env python3
"""Joint five-task training from the joint V12.1 Stage A, not old multitask models."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_multitask_v7_7 as trainer
from scripts.train_vla_stage_a_v12_1 import RUN_NAME as STAGE_A_RUN, DEFAULT_OUTPUT as STAGE_A_OUTPUT, inject_defaults
from tactile_vla.vla.v12_1_data import (
    DATA_PROFILE,
    DEFAULT_INDEX,
    DOMAINS,
    OUTPUT,
    ROOT,
    artifact_identity,
    read_json,
    validate_index,
)
from tactile_vla.vla.v12_1_runtime import (
    build_loaders as joint_loaders,
    check_args,
    evaluate_action_report,
    print_dry_run,
)

base = trainer.base
ORIGINAL_PARSE = trainer._BASE_PARSE_ARGS
ORIGINAL_NEED, ORIGINAL_ADJUSTMENT, ORIGINAL_TEXT = base.evaluate_need, base.evaluate_adjustment, base.evaluate_text
ARGS = None
RUN_NAME = "pi05_blackbox_book_v12_1_five_task_h100_no_history"


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action-eval-samples-per-phase", type=int, default=256)
    extra, rest = parser.parse_known_args()
    sys.argv[:] = [sys.argv[0], *rest]
    if any(a in ("-h", "--help") for a in rest):
        print("V12.1 option: --action-eval-samples-per-phase 256 (per domain and phase)")
    inject_defaults(
        {
            "--dataset-dir": ROOT / "joint_blackbox_book",
            "--split-file": DEFAULT_INDEX,
            "--reasoning-manifest-dir": OUTPUT,
            "--index-file": DEFAULT_INDEX,
            "--norm-stats-dir": OUTPUT / "norm_stats",
            "--stage-a-checkpoint": STAGE_A_OUTPUT / STAGE_A_RUN / "15000",
            "--output-dir": ROOT / "joint_blackbox_book/outputs/multitask_v12_1",
            "--run-name": RUN_NAME,
            "--data-profile": DATA_PROFILE,
            "--prompt-profile": "phase_v7_7_h100_11_answer",
            "--num-steps": 4000,
            "--batch-size": 8,
            "--eval-interval": 1000,
            "--save-interval": 1000,
            "--keep-period": 1000,
            "--max-token-len": 512,
            "--reasoning-max-token-len": 320,
            "--state-history-len": 0,
            "--history-hidden-dim": 0,
            "--eval-max-need-samples": 10000,
        }
    )
    if not any(a in ("--use-state-history", "--no-use-state-history") for a in sys.argv):
        sys.argv.append("--no-use-state-history")
    args = ORIGINAL_PARSE()
    args.action_eval_samples_per_phase = extra.action_eval_samples_per_phase
    args.adjustment_loss_weight = 1.0
    check_args(args)
    if args.data_profile != DATA_PROFILE or args.prompt_profile != "phase_v7_7_h100_11_answer":
        raise ValueError("Wrong joint multitask protocol")
    if args.num_steps % 5 or args.save_interval % 5 or args.eval_interval % 5:
        raise ValueError("Intervals must finish complete five-task cycles")
    if args.action_eval_samples_per_phase <= 0:
        raise ValueError("Evaluation support must be positive")
    global ARGS
    ARGS = args
    return args


def ensure_index(args):
    index = read_json(args.index_file)
    validate_index(index)
    if args.norm_stats_dir.resolve() != Path(index["norm_stats_dir"]).resolve():
        raise ValueError("Index and shared norm stats differ")
    if not args.dry_run:
        checkpoint = args.stage_a_checkpoint
        config = read_json(checkpoint.parent / "config.json")
        if config.get("experiment_version") != "v12_1" or config["artifact_identity"] != artifact_identity(
            index, args.index_file
        ):
            raise ValueError("Must initialize from the matching joint V12.1 Stage A")
        if not (checkpoint / "params/_METADATA").is_file():
            raise FileNotFoundError("Joint Stage A checkpoint is incomplete")
        for key, required in {
            "action_horizon": 30,
            "action_dim": 32,
            "use_state_history": False,
            "paligemma_variant": args.paligemma_variant,
            "action_expert_variant": args.action_expert_variant,
        }.items():
            if config.get(key) != required:
                raise ValueError(f"Incompatible Stage A model: {key}")
    base.EXTRA_CONFIG.update(
        {
            "experiment_version": "v12_1",
            "training_data_hash": index["training_data_hash"],
            "joint_policy": index["policy"],
            "task_sampling": "1:1_per_batch",
            "training_target_coverage": {
                "failure_reason": [
                    "failure_reason=rotate left,grasp appropriate.",
                    "failure_reason=rotate right,grasp appropriate.",
                ],
                "recovery_plan": [
                    f"recovery_plan=move horizontally {d} {m}, move vertically none moderately."
                    for d in ("left", "right")
                    for m in ("slightly", "moderately")
                ],
            },
        }
    )
    return index, []


def identity(index, **kwargs):
    del kwargs
    return artifact_identity(index, ARGS.index_file)


def build_loaders(args, model_config, index, records, tokenizer, failure_codec, plan_codec):
    del records
    loaders = joint_loaders(
        args, model_config, index, tokenizer=tokenizer, failure_codec=failure_codec, plan_codec=plan_codec
    )
    base.EXTRA_CONFIG["action_eval_selection"] = loaders["action_eval_selection"]
    base.EXTRA_CONFIG["token_validation"] = loaders["token_validation"]
    return loaders


def grouped_classification(original, state, loader, data_sharding, *, max_samples):
    # Evaluate the complete val split in each task; no chronological-prefix cap hides a domain.
    del max_samples
    result = original(state, loader, data_sharding, max_samples=len(loader.dataset))
    groups = {
        d: original(state, domain_loader, data_sharding, max_samples=len(domain_loader.dataset))
        for d, domain_loader in loader.domain_loaders.items()
    }
    result["by_task"] = groups
    result["sample_weighted_macro_f1"] = result["macro_f1"]
    result["macro_f1"] = sum(v["macro_f1"] for v in groups.values()) / len(DOMAINS)
    for v in [result, *groups.values()]:
        cm = v["confusion_matrix"]
        negatives = sum(cm[0])
        v["false_positive_rate"] = cm[0][1] / negatives if negatives else None
    return result


def evaluate_need(state, loader, data_sharding, *, max_samples):
    return grouped_classification(ORIGINAL_NEED, state, loader, data_sharding, max_samples=max_samples)


def evaluate_adjustment(state, loader, data_sharding, *, max_samples):
    return grouped_classification(ORIGINAL_ADJUSTMENT, state, loader, data_sharding, max_samples=max_samples)


def evaluate_text(state, loader, task, grammar, data_sharding, *, max_samples):
    del max_samples
    result = ORIGINAL_TEXT(state, loader, task, grammar, data_sharding, max_samples=None)
    groups = {
        d: ORIGINAL_TEXT(state, domain_loader, task, grammar, data_sharding, max_samples=None)
        for d, domain_loader in loader.domain_loaders.items()
    }
    result["by_task"] = groups
    result["sample_weighted_exact_match"] = result["exact_match"]
    result["exact_match"] = sum(v["exact_match"] for v in groups.values()) / len(DOMAINS)
    return result


def validate_resume(saved, args, *, actual_precision):
    for key in ("lr", "weight_decay", "grad_clip", "seed", "fsdp_devices", "batch_size", "norm_stats_dir"):
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(f"V12.1 resume configuration differs: {key}")
    if saved.get("precision") != actual_precision:
        raise ValueError("V12.1 resume precision differs")


def configure():
    trainer.VERSION_TAG = "v12_1"
    trainer.configure()
    base.parse_args = parse_args
    base.ensure_v3_index = ensure_index
    base.artifact_identity = identity
    base.build_loaders = build_loaders
    base.ACTION_EVALUATION_HOOK = evaluate_action_report
    base.evaluate_need = evaluate_need
    base.evaluate_adjustment = evaluate_adjustment
    base.evaluate_text = evaluate_text
    base.print_dry_run_vlm_inputs = print_dry_run
    base.validate_v4_resume_config = validate_resume
    base.validate_reasoning_manifests = lambda *a, **kw: {
        "v12_1_training_data_hash": read_json(ARGS.index_file)["training_data_hash"]
    }


def main():
    configure()
    base.main()


if __name__ == "__main__":
    main()
