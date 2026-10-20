#!/usr/bin/env python3
"""V7.10: retrain five tasks with --history-window {0s,1s,2s} from V7.4.2."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_multitask_v7_7_2 as source_entry
from tactile_vla.vla import v7_10_multitask_data as data
from tactile_vla.vla.v7_9_multitask_eval import evaluate_adjustment
from tactile_vla.vla.v7_10_phase_prompt import WINDOW_FRAMES, prompt_profile

trainer, base = source_entry.trainer, source_entry.trainer.base
_BASE_PARSE = trainer.parse_args
_BASE_CONFIGURE = trainer.configure
_BASE_ENSURE_INDEX = trainer.ensure_index
_BASE_EXPORT = trainer.export_checkpoint
IDENTITY_KEYS = (
    "experiment_version", "history_window", "history_policy", "counterfactual_policy", "training_data_hash",
    "source_multitask_training_data_hash", "source_multitask_index_sha256", "adjustment_sampling_policy",
    "adjustment_label_policy", "need_successful_recovery_start", "stage_a_initialization_identity",
)


def parse_args(argv=None):
    saved = sys.argv
    supplied = list(saved[1:] if argv is None else argv)
    window_parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    window_parser.add_argument(
        "--history-window", choices=tuple(WINDOW_FRAMES), required=not any(a in {"--help", "-h"} for a in supplied),
        help="0s=current qpos repeated 11 times; 1s=31 raw frames; 2s=61 raw frames. Each run trains separately.",
    )
    custom, remaining = window_parser.parse_known_args(supplied)
    if any(a in {"--help", "-h"} for a in supplied):
        window_parser.print_help()
    window = custom.history_window or "2s"
    configure_version(window)
    normalized = []
    for argument in remaining:
        normalized.extend(argument.split("=", 1) if argument.startswith("--") and "=" in argument else [argument])
    defaults = {
        "--num-steps": "5000", "--batch-size": "8", "--fsdp-devices": "2", "--num-workers": "0",
        "--eval-interval": "1000", "--save-interval": "1000", "--keep-period": "1000",
        "--eval-max-need-samples": "4096", "--seed": "42", "--lr": "0.0001",
    }
    for flag, value in defaults.items():
        if flag not in normalized:
            normalized.extend([flag, value])
    try:
        sys.argv = [saved[0], *normalized]
        args = _BASE_PARSE()
    finally:
        sys.argv = saved
    args.history_window = window
    args.qpos_history_frames = WINDOW_FRAMES[window]
    required = {
        "data_profile": data.data_profile(window), "prompt_profile": prompt_profile(window),
        "use_state_history": False, "state_history_len": 0, "history_hidden_dim": 0,
        "action_horizon": 30, "action_dim": 32, "max_token_len": 512,
        "reasoning_max_token_len": 320, "no_norm": False,
    }
    mismatches = {key: (getattr(args, key), value) for key, value in required.items()
                  if getattr(args, key) != value}
    if mismatches:
        raise ValueError(f"V7.10 history/model protocol mismatch: {mismatches}")
    return args


def ensure_index(args):
    index, records = _BASE_ENSURE_INDEX(args)
    if index["history_window"] != args.history_window:
        raise ValueError("V7.10 requested history window and index differ")
    stage_a = data.validate_stage_a_for_training(args.stage_a_checkpoint, index)
    base.EXTRA_CONFIG.update({key: index[key] for key in IDENTITY_KEYS if key in index})
    base.EXTRA_CONFIG["stage_a_initialization_identity"] = stage_a
    base.EXTRA_CONFIG["training_target_coverage"] = {
        "failure_reason": sorted({row["target_failure_reason"]
                                  for row in data.load_jsonl(Path(index["failure_manifest_file"]))
                                  if row["split"] == "train"}),
        "recovery_plan": sorted({row["target_recovery_plan"]
                                 for row in data.load_jsonl(Path(index["plan_manifest_file"]))
                                 if row["split"] == "train"}),
    }
    return index, records


def validate_resume(saved, args, *, actual_precision):
    for key in IDENTITY_KEYS:
        if saved.get(key) != base.EXTRA_CONFIG.get(key):
            raise ValueError(f"V7.10 resume history/data identity mismatch: {key}")
    for key in (
        "data_profile", "prompt_profile", "batch_size", "seed", "lr", "weight_decay", "grad_clip",
        "action_loss_weight", "adjustment_loss_weight", "need_loss_weight", "failure_loss_weight",
        "plan_loss_weight", "eval_interval", "eval_max_need_samples", "eval_max_text_samples",
        "reasoning_max_token_len", "action_loss_degradation_limit", "grammar_profile", "fsdp_devices",
    ):
        if saved.get(key) != getattr(args, key):
            raise ValueError(f"V7.10 resume training config mismatch: {key}")
    if saved.get("precision") != actual_precision:
        raise ValueError("V7.10 resume precision mismatch")


def export_checkpoint(run_dir, state, step, filter_):
    _BASE_EXPORT(run_dir, state, step, filter_)
    path = run_dir / str(step) / "v7_10_export.json"
    metadata = json.loads(path.read_text())
    metadata.update(base.EXTRA_CONFIG | {"data_profile": trainer.DATA_PROFILE, "prompt_profile": trainer.PROMPT_PROFILE})
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def configure_training():
    _BASE_CONFIGURE()
    base.EXTRA_CONFIG.update(experiment_version=data.VERSION_TAG, counterfactual_policy=data.COUNTERFACTUAL_POLICY)
    base.validate_v4_resume_config = validate_resume
    base.CHECKPOINT_EXPORT_HOOK = export_checkpoint
    base.evaluate_adjustment = evaluate_adjustment


def configure_version(window="2s"):
    source_entry.configure_version()
    trainer.DATA_PROFILE, trainer.PROMPT_PROFILE = data.data_profile(window), prompt_profile(window)
    trainer.DEFAULT_INDEX, trainer.DEFAULT_STAGE_A = data.default_index(window), data.DEFAULT_STAGE_A
    trainer.DEFAULT_OUTPUT, trainer.RUN_NAME = data.ROOT / "outputs/multitask_v7_10", data.run_name(window)
    trainer.VERSION_TAG, trainer.validate_index = data.VERSION_TAG, data.validate_index
    trainer.parse_args, trainer.configure, trainer.ensure_index = parse_args, configure_training, ensure_index


def main():
    configure_version()
    trainer.main()


if __name__ == "__main__":
    main()
