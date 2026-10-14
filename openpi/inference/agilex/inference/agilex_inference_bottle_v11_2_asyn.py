#!/usr/bin/env python3
"""Bottle V11.2 asynchronous deployment using raw H100 for all phase decisions."""

# ruff: noqa: E402
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import agilex_inference_bottle_v11_1_asyn as shared

DEFAULT_LOG_ROOT = shared.PROJECT_ROOT / "outputs/runtime/bottle_v11_2_async"


def get_arguments(argv=None):
    parser = shared.build_argument_parser()
    parser.description = __doc__
    parser.set_defaults(log_dir=DEFAULT_LOG_ROOT)
    return parser.parse_args(argv), parser


def run_bottle_v11_2_async(args, operator, action_policy, phase_client, captioner, keyboard, logger):
    shared.run_bottle_v11_1_async(
        args,
        operator,
        action_policy,
        phase_client,
        captioner,
        keyboard,
        logger,
        expected_version="bottle_v11_2",
        deployment_label="Bottle V11.2",
    )


def main():
    shared.main(
        argument_parser=get_arguments,
        run=run_bottle_v11_2_async,
        deployment_label="Bottle V11.2",
    )


if __name__ == "__main__":
    main()
