#!/usr/bin/env python3
"""Train Bottle V11.2 using the five-task engine and version-specific indices."""

# ruff: noqa: E402
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_multitask_bottle_v11_1 as previous
from tactile_vla.vla import bottle_v11_2_multitask_data as data
from tactile_vla.vla.bottle_v11_2_stage_a_data import validate_training_index


def parse_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--bottle-root", type=Path, default=data.ROOT)
    extra, _ = p.parse_known_args(arguments)
    root = extra.bottle_root
    for flag, value in {
        "--index-file": root / "outputs/bottle_v11_2_multitask/bottle_v11_2_multitask_training_index.json",
        "--stage-a-checkpoint": root / "outputs/stage_a_action/pi05_delta_tac_bottle_stage_a_v11_2_no_history/15000",
        "--output-dir": root / "outputs/multitask_v11_2",
        "--run-name": data.RUN_NAME,
    }.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in arguments):
            arguments.extend([flag, str(value)])
    return previous.parse_args(arguments)


def ensure_index(args):
    index, frames = previous.ensure_index(args)
    previous.base.EXTRA_CONFIG.update(
        {
            "action_target_policy": index["action_target_policy"],
            "idle_gap_audit_sha256": previous.sha256_json(index["idle_gap_audit"]),
        }
    )
    return index, frames


def identity(index, **kwargs):
    result = previous.identity(index, **kwargs)
    for key in ("bottle_v11_1_training_data_hash", "bottle_v11_1_multitask_index_sha256"):
        result[key.replace("v11_1", "v11_2")] = result.pop(key)
    return result


def validate_resume(saved, args, *, actual_precision):
    for key in ("action_target_policy", "idle_gap_audit_sha256"):
        if saved.get(key) != previous.base.EXTRA_CONFIG.get(key):
            raise ValueError(f"V11.2 resume mismatch: {key}")
    previous.validate_resume(saved, args, actual_precision=actual_precision)


def configure_training():
    previous.configure_training()
    previous.base.TEXT_GRAMMAR_FACTORY = data.build_training_grammars
    previous.base.validate_v4_resume_config = validate_resume


def configure_version():
    previous.DATA_PROFILE, previous.VERSION_TAG = data.DATA_PROFILE, data.VERSION_TAG
    previous.DEFAULT_INDEX, previous.DEFAULT_STAGE_A = data.DEFAULT_INDEX, data.DEFAULT_STAGE_A
    previous.RUN_NAME, previous.HISTORY_POLICY = data.RUN_NAME, data.HISTORY_POLICY
    previous.validate_index = data.validate_index
    previous.validate_stage_a_for_training = data.validate_stage_a_for_training
    previous.validate_action_index = validate_training_index
    previous.ACTION_POLICY = previous.ACTION_POLICY | {
        "action_candidates": "V11.2_compressed_phase_pure_failure_safe_H30"
    }
    previous.configure_version()
    previous.trainer.parse_args = parse_args
    previous.trainer.ensure_index = ensure_index
    previous.trainer.configure = configure_training
    previous.trainer.identity = identity
    previous.trainer.DEFAULT_OUTPUT = data.ROOT / "outputs/multitask_v11_2"


def main():
    configure_version()
    previous.trainer.main()


if __name__ == "__main__":
    main()
