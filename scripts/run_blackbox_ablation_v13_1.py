#!/usr/bin/env python3
"""Train ALL selected V13.1 groups first, then test ALL val-selected best models."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]


def parse_args(argv=None, *, description=__doc__):
    parser = argparse.ArgumentParser(description=description, epilog=(
        'Pass shared training options after --, e.g. -- --num-steps 1000 --batch-size 8. '
        'CUDA_VISIBLE_DEVICES and other environment settings are inherited by every child.'))
    parser.add_argument('--groups', choices=('A', 'B', 'C', 'D'), nargs='+', default=['A', 'B', 'C', 'D'])
    parser.add_argument('--dry-run', action='store_true', help='Only print the ordered commands; do not train or test')
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.training_args[:1] == ['--']:
        args.training_args = args.training_args[1:]
    if len(set(args.groups)) != len(args.groups):
        parser.error('Each group may appear only once')
    forbidden = {'--group', '--dry-run', '--audit-only', '--help', '-h'}
    if any(value.split('=', 1)[0] in forbidden for value in args.training_args):
        parser.error('Use --groups to select groups; child training must not be dry-run/audit-only/help')
    return args


def command_plan(args):
    # Use the actual trainer parser so defaults, seed, output-dir and run-name
    # cannot drift between training and evaluation. No weights/data are loaded.
    from scripts.train_blackbox_ablation_v13_1 import parse_args as parse_training
    python = str(PROJECT / 'openpi/.venv/bin/python')
    train_commands, test_commands, run_dirs = [], [], []
    for group in args.groups:
        training_args = ['--group', group, *args.training_args]
        effective = parse_training(training_args)
        run_dir = (PROJECT / effective.output_dir / effective.run_name).resolve()
        train_commands.append([python, str(PROJECT / 'scripts/train_blackbox_ablation_v13_1.py'), *training_args])
        test_commands.append([python, str(PROJECT / 'scripts/eval_blackbox_ablation_v13_1.py'),
                              '--run-dir', str(run_dir), '--checkpoint', 'best', '--split', 'test'])
        run_dirs.append(run_dir)
    if len(set(run_dirs)) != len(run_dirs):
        raise ValueError('Groups resolve to the same run directory; omit --run-name or select a single group')
    return train_commands, test_commands


def run_pipeline(args, launch=None, *, planner=command_plan):
    launch = subprocess.run if launch is None else launch
    training, testing = planner(args)
    if not args.dry_run and not Path(training[0][0]).is_file():
        raise FileNotFoundError(training[0][0])
    for stage, commands in (('TRAIN', training), ('TEST', testing)):
        print(f'\n=== {stage}: {", ".join(args.groups)} ===', flush=True)
        for group, command in zip(args.groups, commands, strict=True):
            print(f'[{stage} {group}] {shlex.join(command)}', flush=True)
            if args.dry_run:
                continue
            result = launch(command, cwd=PROJECT, check=False)
            if result.returncode:
                print(f'STOPPED: {stage} {group} exited with {result.returncode}; no further commands.',
                      file=sys.stderr, flush=True)
                return result.returncode if result.returncode > 0 else 128 - result.returncode
    print('\n' + ('Dry-run complete; nothing launched.' if args.dry_run else
                  'All training and best-checkpoint test evaluations completed.'), flush=True)
    return 0


def main():
    args = parse_args()
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        print('\nInterrupted; no further commands will be started.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
