#!/usr/bin/env python3
"""Train ONLY plan for V13.2: global-reversal latch and random historical grasps."""
# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]

from scripts import train_blackbox_ablation_v13_1 as common
from tactile_vla.vla.blackbox_ablation_v13_2_data import INDEX, SCHEMA, TRAIN_OUTPUT, validate_index
from tactile_vla.vla.blackbox_ablation_v13_2_runtime import evaluate


def parse_args(argv=None):
    return common.parse_args(argv, index_file=INDEX, output_dir=TRAIN_OUTPUT,
                             run_prefix='pi05_blackbox_ablation_v13_2', description=__doc__)


def components(args, *, verify_sources=True):
    return common.components(args, verify_sources=verify_sources, index_validator=validate_index)


def main():
    common.main(parser=parse_args, component_builder=components, evaluator=evaluate,
                schema=SCHEMA, version_label='V13.2')


if __name__ == '__main__':
    main()
