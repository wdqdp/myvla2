#!/usr/bin/env python3
"""GPU3 trains A/C; GPU4 trains B; test all three best models after all training."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]

from scripts import run_blackbox_ablation_v13_3 as common  # noqa: E402

GPU_GROUPS = (('3', 'AC'), ('4', 'B'))


def busy_gpu_processes():
    ids = ','.join(gpu for gpu, _ in GPU_GROUPS)
    result = subprocess.run(['nvidia-smi', f'--id={ids}', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def wait_for_gpus(args, runner):
    label = '/'.join(gpu for gpu, _ in GPU_GROUPS)
    while not runner.stopped.is_set():
        busy = busy_gpu_processes()
        if not busy:
            return True
        if not args.wait_for_gpus:
            raise RuntimeError(f'GPU {label} are occupied; not starting. Use --wait-for-gpus.\n{busy}')
        print(f'[{datetime.now():%F %T}] Waiting for GPU {label} (no jobs will be stopped): {busy!r}', flush=True)
        runner.stopped.wait(args.poll_seconds)
    return False


def command_plan(args):
    from scripts.train_blackbox_ablation_v13_4 import parse_args as training_args
    python = str(PROJECT / 'openpi/.venv/bin/python')
    options = ['--num-steps', '400', '--batch-size', '8', '--fsdp-devices', '1', '--num-workers', '0',
               '--seed', '42', '--eval-interval', '200', '--save-interval', '200', '--keep-period', '200']
    if args.output_dir is not None:
        options.extend(['--output-dir', str(args.output_dir)])
    groups = {}
    for group in 'ABC':
        parsed = training_args(['--group', group, *options])
        run = (PROJECT / parsed.output_dir / parsed.run_name).resolve()
        groups[group] = {'run_dir': run,
            'train': [python, str(PROJECT / 'scripts/train_blackbox_ablation_v13_4.py'), '--group', group, *options],
            'test': [python, str(PROJECT / 'scripts/eval_blackbox_ablation_v13_4.py'),
                     '--run-dir', str(run), '--checkpoint', 'best', '--split', 'test']}
    phases = {stage: [{'group': f'GPU{gpu}', 'gpu': gpu, 'run_dir': groups[selected[0]]['run_dir'].parent,
        'stages': [(f'{stage}_{g}', groups[g][stage]) for g in selected]}
        for gpu, selected in GPU_GROUPS] for stage in ('train', 'test')}
    return groups, phases


def main(argv=None):
    # Same safe wait/dry-run options; task groups and commands are independently built.
    args = common.parse_args(argv, description=__doc__)
    groups, phases = command_plan(args)
    for lanes in phases.values():
        for lane in lanes:
            for name, command in lane['stages']:
                print(f'[{name} GPU {lane["gpu"]}] {shlex.join(command)}', flush=True)
    if args.dry_run:
        return 0
    from scripts.run_blackbox_ablation_v13_2_parallel import ParallelRunner
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('OPENPI_DATA_HOME', '/home/qxh/.cache/openpi')
    common.preflight(groups)
    root = groups['A']['run_dir'].parent
    log_dir = root / f'parallel_{datetime.now():%Y%m%d_%H%M%S_%f}'
    log_dir.mkdir(parents=True, exist_ok=False)
    runner = ParallelRunner(log_dir)
    previous = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if not wait_for_gpus(args, runner):
            return 130
        common.preflight(groups)
        # B may finish early and release GPU4; do not collide with another user
        # if that GPU is occupied before the global test phase starts.
        results = common.run_phases(phases, runner, description='GPU3 A/C; GPU4 B',
            before_phase=lambda stage: stage != 'test' or wait_for_gpus(args, runner))
        (log_dir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(f'Summary: {log_dir / "summary.json"}', flush=True)
        return 130 if runner.stopped.is_set() else int(any(r['returncode'] for r in results))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
