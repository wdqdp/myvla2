#!/usr/bin/env python3
"""Train V7.11 Stage A with current tactile text as the recovery-action condition."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_stage_a_openpi as trainer
from tactile_vla.vla.v7_11_datasets import DirectActionDataset
from tactile_vla.vla.v7_11_stage_a import (
    DEFAULT_ACTION_INDEX, DEFAULT_DATASET, DEFAULT_NORM_STATS, DEFAULT_STAGE_A_OUTPUT,
    RUN_NAME, checkpoint_prompt_identity,
)

_BASE_PARSE = trainer.parse_args
_BASE_CONFIG = trainer.checkpoint_config_payload
_BASE_RESUME = trainer.validate_v4_resume_config
_BASE_LOADER = trainer.build_loader


def parse_args():
    defaults = {
        "--dataset-dir": DEFAULT_DATASET,
        "--index-file": DEFAULT_ACTION_INDEX,
        "--norm-stats-dir": DEFAULT_NORM_STATS,
        "--output-dir": DEFAULT_STAGE_A_OUTPUT,
        "--run-name": RUN_NAME,
        "--data-profile": trainer.ROTATION_PHASE_V7_4_2_ADJUSTMENT,
        "--prompt-profile": "phase_v2",
        "--experiment-kind": trainer.V7_4_2_EXPERIMENT_KIND,
        "--checkpoint": trainer.V7_4_2_SOURCE_CHECKPOINT,
        "--batch-size": 8,
        "--num-steps": 10000,
        "--lr": 1e-5,
        "--lr-final": 1e-6,
        "--lr-transition-steps": 10000,
        "--save-interval": 1000,
        "--keep-period": 5000,
        "--state-history-len": 0,
        "--history-hidden-dim": 0,
        "--max-token-len": 200,
        "--fsdp-devices": 2,
    }
    supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:]}
    for flag, value in defaults.items():
        if flag not in supplied:
            sys.argv.extend([flag, str(value)])
    if "--use-state-history" not in supplied and "--no-use-state-history" not in supplied:
        sys.argv.append("--no-use-state-history")
    args = _BASE_PARSE()
    if args.data_profile != trainer.ROTATION_PHASE_V7_4_2_ADJUSTMENT:
        raise ValueError("V7.11 Stage A must use the V7.4.2 action data profile")
    return args


def checkpoint_config_payload(args, identity):
    return _BASE_CONFIG(args, identity) | {"v7_11_stage_a": checkpoint_prompt_identity()}


def validate_resume(saved, args):
    if saved.get("v7_11_stage_a") != checkpoint_prompt_identity():
        raise ValueError("V7.11 Stage A resume requires the same direct-tactile action prompt code")
    _BASE_RESUME(saved, args)


def build_loader(args, model_config, payload=None):
    loader = _BASE_LOADER(args, model_config, payload)
    if args.dry_run:
        dataset = loader.dataset.dataset
        shown = set()
        for position, global_index in enumerate(dataset.indices):
            phase = dataset.action_phase_by_global_index[global_index]["phase"]
            if phase not in shown:
                sample = dataset[position]
                print(f"V7.11 Stage A {phase} action global_index={global_index}: {sample['prompt']}")
                print(f"V7.11 Stage A {phase} target shape={loader.dataset[position]['actions'].shape}")
                shown.add(phase)
            if {"execution", "adjustment"} <= shown:
                break
        if not {"execution", "adjustment"} <= shown:
            raise ValueError("V7.11 Stage A requires both execution and recovery action samples")
    return loader


def main():
    trainer.parse_args = parse_args
    trainer.TactileVLAFrameDataset = DirectActionDataset
    trainer.build_loader = build_loader
    trainer.checkpoint_config_payload = checkpoint_config_payload
    trainer.validate_v4_resume_config = validate_resume
    trainer.main()


if __name__ == "__main__":
    main()
