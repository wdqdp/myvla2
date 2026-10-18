#!/usr/bin/env python3
"""V13.3: train only plan with qpos masked during ALL train/validation inputs."""
# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]

from scripts import train_blackbox_ablation_v13_2 as previous
from tactile_vla.vla.blackbox_ablation_v13_3_runtime import (
    QPOS_PROTOCOL, SCHEMA, evaluate, make_loader, validate_tokens,
)

OUTPUT = previous.TRAIN_OUTPUT.parent / 'plan_v13_3'


def parse_args(argv=None):
    # Override only the default step budget; explicit user flags still take precedence.
    argv = list(sys.argv[1:] if argv is None else argv)
    args = previous.common.parse_args(['--num-steps', '400', *argv], index_file=previous.INDEX,
        output_dir=OUTPUT, run_prefix='pi05_blackbox_ablation_v13_3', description=__doc__)
    args.qpos_input_protocol = QPOS_PROTOCOL
    return args


def main():
    previous.common.main(parser=parse_args, component_builder=previous.components, evaluator=evaluate,
                         schema=SCHEMA, version_label='V13.3', loader_builder=make_loader,
                         token_validator=validate_tokens)


if __name__ == '__main__':
    main()
