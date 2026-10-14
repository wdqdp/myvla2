#!/usr/bin/env python3
"""Serve Bottle V11.2 five tasks using its versioned compressed-history contract."""

# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from scripts import serve_tactile_vla_bottle_v11_1 as shared
from tactile_vla.vla.bottle_v11_1_runtime import BOTTLE_ROOT, V112_VERSION

DEFAULT_CHECKPOINT = BOTTLE_ROOT / "outputs/multitask_v11_2/pi05_bottle_v11_2_five_task_h100_no_history/1000"


def parse_args(argv=None):
    return shared.parse_args(argv, default_checkpoint=DEFAULT_CHECKPOINT, description=__doc__)


def main():
    shared.main(expected_version=V112_VERSION, argument_parser=parse_args)


if __name__ == "__main__":
    main()
