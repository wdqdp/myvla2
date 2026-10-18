#!/usr/bin/env python3
"""V13.3: GPU6 trains A/C, GPU7 trains B/D; test all best models after all training."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--wait-for-gpus', action='store_true', help='Wait until BOTH GPUs have no compute processes; never kill jobs')
    p.add_argument('--poll-seconds', type=float, default=30.)
    args = p.parse_args(argv)
    if args.poll_seconds <= 0:
        p.error('--poll-seconds must be positive')
    return args


def command_plan(args):
    from scripts.train_blackbox_ablation_v13_3 import parse_args as training_args
    python = str(PROJECT / 'openpi/.venv/bin/python')
    options = ['--num-steps', '400', '--batch-size', '8', '--fsdp-devices', '1',
               '--num-workers', '0', '--seed', '42', '--eval-interval', '200',
               '--save-interval', '200', '--keep-period', '200']
    if args.output_dir is not None:
        options.extend(['--output-dir', str(args.output_dir)])
    groups = {}
    for group in 'ABCD':
        parsed = training_args(['--group', group, *options])
        run = (PROJECT / parsed.output_dir / parsed.run_name).resolve()
        groups[group] = {'run_dir': run,
            'train': [python, str(PROJECT / 'scripts/train_blackbox_ablation_v13_3.py'), '--group', group, *options],
            'test': [python, str(PROJECT / 'scripts/eval_blackbox_ablation_v13_3.py'),
                     '--run-dir', str(run), '--checkpoint', 'best', '--split', 'test']}
    phases = {}
    for stage in ('train', 'test'):
        phases[stage] = [{'group': f'GPU{gpu}', 'gpu': gpu, 'run_dir': groups[selected[0]]['run_dir'].parent,
                          'stages': [(f'{stage}_{g}', groups[g][stage]) for g in selected]}
                         for gpu, selected in (('6', 'AC'), ('7', 'BD'))]
    return groups, phases


def preflight(groups):
    for group, job in groups.items():
        if job['run_dir'].exists():
            raise FileExistsError(f'{group}: {job["run_dir"]} exists; refusing to overwrite/retrain')


def busy_gpu_processes():
    result = subprocess.run(['nvidia-smi', '--id=6,7', '--query-compute-apps=gpu_uuid,pid',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def wait_for_gpus(args, runner):
    while not runner.stopped.is_set():
        busy = busy_gpu_processes()
        if not busy:
            return True
        if not args.wait_for_gpus:
            raise RuntimeError(f'GPU 6/7 are occupied; not starting. Use --wait-for-gpus.\n{busy}')
        print(f'[{datetime.now():%F %T}] Waiting for GPU 6/7 (no jobs will be stopped): {busy!r}', flush=True)
        runner.stopped.wait(args.poll_seconds)
    return False


def run_phases(phases, runner):
    results = []
    # Global barrier: no test starts unless ALL FOUR training jobs have succeeded.
    for stage in ('train', 'test'):
        print(f'=== {stage.upper()}: GPU6 A/C; GPU7 B/D ===', flush=True)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(runner.run_group, lane) for lane in phases[stage]]
            phase = []
            for lane, future in zip(phases[stage], futures, strict=True):
                try:
                    result = future.result()
                except Exception as error:
                    result = {'group': lane['group'], 'returncode': 1, 'error': str(error)}
                phase.append({'phase': stage, **result})
        results.extend(phase)
        if runner.stopped.is_set() or any(r['returncode'] for r in phase):
            break
    return results


def main(argv=None):
    args = parse_args(argv)
    groups, phases = command_plan(args)
    for stage, lanes in phases.items():
        for lane in lanes:
            for name, command in lane['stages']:
                print(f'[{name} GPU {lane["gpu"]}] {shlex.join(command)}', flush=True)
    if args.dry_run:
        return 0
    from scripts.run_blackbox_ablation_v13_2_parallel import ParallelRunner

    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('OPENPI_DATA_HOME', '/home/qxh/.cache/openpi')
    preflight(groups)
    root = groups['A']['run_dir'].parent
    log_dir = root / f'parallel_{datetime.now():%Y%m%d_%H%M%S_%f}'
    log_dir.mkdir(parents=True, exist_ok=False)
    runner = ParallelRunner(log_dir)
    previous = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if not wait_for_gpus(args, runner):
            return 130
        preflight(groups)  # Re-check after a potentially long wait.
        results = run_phases(phases, runner)
        (log_dir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(f'Summary: {log_dir / "summary.json"}', flush=True)
        return 130 if runner.stopped.is_set() else int(any(r['returncode'] for r in results))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
