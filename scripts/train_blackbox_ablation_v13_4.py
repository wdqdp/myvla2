#!/usr/bin/env python3
"""V13.4: only plan; no executed-plan input; qpos masked only for this experiment."""
# ruff: noqa: E402
import copy
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]

from scripts import train_blackbox_ablation_v13_1 as common
from tactile_vla.vla import blackbox_ablation_v13_4_data as data
from tactile_vla.vla.blackbox_ablation_v13_4_runtime import evaluate, make_loader, validate_tokens


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = common.parse_args(['--num-steps', '400', *argv], index_file=data.INDEX,
        output_dir=data.TRAIN_OUTPUT, run_prefix='pi05_blackbox_ablation_v13_4', description=__doc__, groups=data.GROUPS)
    args.qpos_input_protocol, args.qpos_scope = data.QPOS_PROTOCOL, data.QPOS_SCOPE
    args.feedback_input_protocol = copy.deepcopy(data.FEEDBACK_PROTOCOL)
    return args


def components(args, *, verify_sources=True):
    return common.components(args, verify_sources=verify_sources, index_validator=data.validate_index)


def main():
    common.main(parser=parse_args, component_builder=components, evaluator=evaluate,
        schema=data.SCHEMA, version_label='V13.4', loader_builder=make_loader, token_validator=validate_tokens)


if __name__ == '__main__':
    main()
