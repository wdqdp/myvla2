#!/usr/bin/env python3
"""Train Book V9.5 Stage A from pi05_base using only current, non-archived data."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_stage_a_openpi as trainer
from tactile_vla.vla.book_v9_5_stage_a_data import (
    DATA_PROFILE,
    EXPERIMENT_KIND,
    ROOT,
    RUN_NAME,
    VERSION_TAG,
    reject_archive,
    validate_training_index,
)

_BASE_PARSE_ARGS = trainer.parse_args
_BASE_CONFIG_PAYLOAD = trainer.checkpoint_config_payload
_BASE_RESUME_VALIDATE = trainer.validate_v4_resume_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--book-root", type=Path, default=ROOT)
    extra, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    root = reject_archive(extra.book_root)
    defaults = {
        "--dataset-dir": root / "lerobot_data/tactile_vla_rotation_v4",
        "--index-file": root / "outputs/book_stage_a_v9_5/book_stage_a_training_index.json",
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
        if not any(arg == flag or arg.startswith(flag + "=") for arg in rest):
            rest.extend([flag, str(value)])
    if not any(arg in {"--use-state-history", "--no-use-state-history"} for arg in rest):
        rest.append("--no-use-state-history")
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], *rest]
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv = saved_argv
    if args.data_profile != DATA_PROFILE:
        raise ValueError("V9.5 Stage A requires book_stage_a_v1")
    for path in (args.dataset_dir, args.index_file, args.norm_stats_dir, args.output_dir / args.run_name):
        reject_archive(path)
    args.book_root = root
    args.experiment_version = VERSION_TAG
    return args


def ensure_index(args):
    index = json.loads(args.index_file.read_text())
    _, lookup = validate_training_index(
        index,
        index_path=args.index_file,
        dataset_dir=args.dataset_dir,
        book_root=args.book_root,
    )
    args._v5_action_phase_lookup = lookup
    args._v9_5_source_scope = index["source_scope"]
    return index


def checkpoint_config_payload(args, identity):
    return _BASE_CONFIG_PAYLOAD(args, identity) | {
        "experiment_version": VERSION_TAG,
        "source_scope": args._v9_5_source_scope,
    }


def validate_resume_config(saved, args):
    if saved.get("experiment_version") != VERSION_TAG or saved.get("source_scope") != args._v9_5_source_scope:
        raise ValueError("V9.5 cannot resume a different version or source scope")
    _BASE_RESUME_VALIDATE(saved, args)


def configure_version():
    trainer.parse_args = parse_args
    trainer.ensure_index = ensure_index
    trainer.checkpoint_config_payload = checkpoint_config_payload
    trainer.validate_v4_resume_config = validate_resume_config


def main():
    configure_version()
    trainer.main()


if __name__ == "__main__":
    main()
