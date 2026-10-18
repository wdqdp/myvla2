#!/usr/bin/env python3
"""Train A/B/C/D first, then test A/B/C/D, using isolated V13.2 defaults."""
# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]

from scripts import run_blackbox_ablation_v13_1 as common
from scripts.train_blackbox_ablation_v13_2 import parse_args as parse_training


def command_plan(args):
    python = str(PROJECT / 'openpi/.venv/bin/python')
    training, testing, dirs = [], [], []
    for group in args.groups:
        child_args = ['--group', group, *args.training_args]
        effective = parse_training(child_args)
        run = (PROJECT / effective.output_dir / effective.run_name).resolve()
        training.append([python, str(PROJECT / 'scripts/train_blackbox_ablation_v13_2.py'), *child_args])
        testing.append([python, str(PROJECT / 'scripts/eval_blackbox_ablation_v13_2.py'),
                        '--run-dir', str(run), '--checkpoint', 'best', '--split', 'test'])
        dirs.append(run)
    if len(set(dirs)) != len(dirs):
        raise ValueError('Groups resolve to the same run directory')
    return training, testing


def run_pipeline(args, launch=None):
    # No global mutation of V13.1's runner or parser.
    return common.run_pipeline(args, launch, planner=command_plan)


def main():
    args = common.parse_args(description=__doc__)
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        print('\nInterrupted; no further commands will be started.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
