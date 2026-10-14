#!/usr/bin/env python3
"""Serve Bottle V11.3 five tasks without runtime captioner or tactile inputs."""

# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from scripts import serve_tactile_vla_bottle_v11_1 as shared
from tactile_vla.vla.bottle_v11_1_runtime import BOTTLE_ROOT, V113_VERSION

DEFAULT_RUN = BOTTLE_ROOT / "outputs/multitask_v11_3/pi05_bottle_v11_3_five_task_h100_no_history"
DEFAULT_CHECKPOINT = DEFAULT_RUN / "1000"


def parse_args(argv=None):
    args = shared.parse_args(
        argv,
        default_checkpoint=DEFAULT_CHECKPOINT,
        description=__doc__,
        captioner_enabled=False,
    )
    root = args.checkpoint.expanduser().resolve()
    if (root / "1000/full_params").is_dir() and not (root / "full_params").is_dir():
        args.checkpoint = root / "1000"
    return args


def main():
    shared.main(expected_version=V113_VERSION, argument_parser=parse_args)


if __name__ == "__main__":
    main()
