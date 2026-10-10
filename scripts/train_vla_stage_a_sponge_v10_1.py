#!/usr/bin/env python3
"""Train failure-safe Sponge V10.1 Stage A from pi05_base, without tactile/history."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_stage_a_openpi as trainer
from tactile_vla.vla.sponge_v10_1_stage_a_data import (
    DATA_PROFILE,
    EXPERIMENT_KIND,
    FAILURE_ACTION_POLICY,
    ROOT,
    RUN_NAME,
    VERSION_TAG,
    validate_training_index,
)

_BASE_PARSE_ARGS = trainer.parse_args
_BASE_CONFIG = trainer.checkpoint_config_payload
_BASE_RESUME = trainer.validate_v4_resume_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sponge-root", type=Path, default=ROOT)
    extra, rest = p.parse_known_args(sys.argv[1:] if argv is None else argv)
    root = extra.sponge_root.resolve()
    defaults = {
        "--dataset-dir": root / "lerobot_data/tactile_vla_rotation_v4",
        "--index-file": root / "outputs/sponge_stage_a_v10_1/stage_a_training_index.json",
        "--norm-stats-dir": root / "outputs/rotation_v4/norm_stats",
        "--output-dir": root / "outputs/stage_a_action",
        "--run-name": RUN_NAME,
        "--data-profile": DATA_PROFILE,
        "--prompt-profile": "phase_v2",
        "--experiment-kind": EXPERIMENT_KIND,
        "--num-steps": 15000,
        "--batch-size": 8,
        "--fsdp-devices": 2,
        "--state-history-len": 0,
        "--history-hidden-dim": 0,
    }
    for flag, value in defaults.items():
        if not any(x == flag or x.startswith(flag + "=") for x in rest):
            rest.extend([flag, str(value)])
    if not any(x in {"--use-state-history", "--no-use-state-history"} for x in rest):
        rest.append("--no-use-state-history")
    saved = sys.argv
    try:
        sys.argv = [saved[0], *rest]
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv = saved
    if args.data_profile != DATA_PROFILE or args.checkpoint != str(trainer.DEFAULT_BASE_CHECKPOINT):
        raise ValueError("V10.1 Stage A must use the native phase/H30 profile and pi05_base initialization")
    args.experiment_version = VERSION_TAG
    return args


def ensure_index(args):
    index = json.loads(args.index_file.read_text())
    _, args._v5_action_phase_lookup = validate_training_index(
        index, index_path=args.index_file, dataset_dir=args.dataset_dir
    )
    args._v10_1_source_scope = index["source_scope"]
    return index


def checkpoint_config_payload(args, identity):
    return _BASE_CONFIG(args, identity) | {
        "experiment_version": VERSION_TAG,
        "source_scope": args._v10_1_source_scope,
        "failure_action_policy": FAILURE_ACTION_POLICY,
    }


def validate_resume(saved, args):
    if (
        saved.get("experiment_version") != VERSION_TAG
        or saved.get("failure_action_policy") != FAILURE_ACTION_POLICY
        or saved.get("source_scope") != args._v10_1_source_scope
    ):
        raise ValueError("V10.1 cannot resume another version, failure policy or source scope")
    _BASE_RESUME(saved, args)


def configure_version():
    trainer.parse_args = parse_args
    trainer.ensure_index = ensure_index
    trainer.checkpoint_config_payload = checkpoint_config_payload
    trainer.validate_v4_resume_config = validate_resume


def main():
    configure_version()
    trainer.main()


if __name__ == "__main__":
    main()
