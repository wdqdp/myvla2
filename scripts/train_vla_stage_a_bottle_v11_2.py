#!/usr/bin/env python3
"""Bottle V11.2 Stage A, using the existing trainer with compressed target offsets."""

# ruff: noqa: E402
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_stage_a_bottle_v11_1 as previous
from tactile_vla.vla import bottle_v11_2_stage_a_data as data

_BASE_CONFIG = previous.checkpoint_config_payload
_BASE_RESUME = previous.validate_resume
_BASE_PROTOCOL = previous.trainer.selected_stage_a_protocol
CONFIGURABLE_FIELDS = {"num_steps", "batch_size", "fsdp_devices", "lr_transition_steps"}


def parse_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--bottle-root", type=Path, default=data.ROOT)
    extra, _ = p.parse_known_args(arguments)
    for flag, value in {
        "--dataset-dir": extra.bottle_root / "lerobot_data/tactile_vla_rotation_v4",
        "--index-file": extra.bottle_root / "outputs/bottle_stage_a_v11_2/stage_a_training_index.json",
        "--norm-stats-dir": extra.bottle_root / "outputs/rotation_v4/norm_stats",
        "--output-dir": extra.bottle_root / "outputs/stage_a_action",
        "--run-name": data.RUN_NAME,
        "--data-profile": data.DATA_PROFILE,
        "--prompt-profile": "phase_v2",
        "--experiment-kind": data.EXPERIMENT_KIND,
        "--num-steps": 15000,
        "--batch-size": 8,
        "--fsdp-devices": 1,
        "--state-history-len": 0,
        "--history-hidden-dim": 0,
    }.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in arguments):
            arguments.extend([flag, str(value)])
    if not any(arg in {"--use-state-history", "--no-use-state-history"} for arg in arguments):
        arguments.append("--no-use-state-history")
    saved = sys.argv
    try:
        sys.argv = [saved[0], *arguments]
        # Bypass V11.1's entry-point-only requirement of exactly 15000 updates.
        args = previous._BASE_PARSE_ARGS()
    finally:
        sys.argv = saved
    if (
        args.data_profile != data.DATA_PROFILE
        or Path(args.checkpoint).expanduser().resolve()
        != previous.trainer.DEFAULT_BASE_CHECKPOINT.expanduser().resolve()
    ):
        raise ValueError("V11.2 Stage A requires the native phase/H30 profile and pi05_base initialization")
    for key in ("num_steps", "batch_size", "fsdp_devices"):
        if getattr(args, key) <= 0:
            raise ValueError(f"V11.2 {key} must be positive")
    if args.lr_transition_steps < 0:
        raise ValueError("V11.2 lr_transition_steps must be nonnegative")
    args.experiment_version = data.VERSION_TAG
    return args


def selected_stage_a_protocol(args):
    name, protocol = _BASE_PROTOCOL(args)
    if getattr(args, "experiment_version", None) == data.VERSION_TAG:
        protocol = {key: value for key, value in protocol.items() if key not in CONFIGURABLE_FIELDS}
    return name, protocol


def ensure_index(args):
    index = previous.ensure_index(args)
    args._v11_2_target_policy = index["target_policy"]
    args._v11_2_gap_audit = index["idle_gap_audit"]
    return index


def checkpoint_config_payload(args, identity):
    from tactile_vla.vla.artifacts import sha256_json

    return _BASE_CONFIG(args, identity) | {
        "action_target_policy": args._v11_2_target_policy,
        "idle_gap_audit_sha256": sha256_json(args._v11_2_gap_audit),
    }


def validate_resume(saved, args):
    from tactile_vla.vla.artifacts import sha256_json

    if saved.get("action_target_policy") != args._v11_2_target_policy or saved.get(
        "idle_gap_audit_sha256"
    ) != sha256_json(args._v11_2_gap_audit):
        raise ValueError("V11.2 resume compression policy/boundaries mismatch")
    # Free selection on a new run is different from changing an existing optimizer schedule.
    if saved.get("lr_transition_steps") != args.lr_transition_steps:
        raise ValueError("V11.2 resume learning-rate transition differs from the saved optimizer schedule")
    _BASE_RESUME(saved, args)


def configure_version():
    previous.VERSION_TAG, previous.RUN_NAME = data.VERSION_TAG, data.RUN_NAME
    previous.EXPERIMENT_KIND = data.EXPERIMENT_KIND
    previous.validate_training_index = data.validate_training_index
    previous.trainer.BOOK_STAGE_A_EXPERIMENT_KIND = data.EXPERIMENT_KIND
    previous.configure_version()
    previous.trainer.parse_args = parse_args
    previous.trainer.ensure_index = ensure_index
    previous.trainer.checkpoint_config_payload = checkpoint_config_payload
    previous.trainer.validate_v4_resume_config = validate_resume
    previous.trainer.selected_stage_a_protocol = selected_stage_a_protocol


def main():
    configure_version()
    previous.trainer.main()


if __name__ == "__main__":
    main()
