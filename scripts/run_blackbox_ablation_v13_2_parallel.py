#!/usr/bin/env python3
"""GPU 4/5: test existing A/B best; GPU 6/7: train C/D for 400 steps, then test."""
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
import threading

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print commands without loading weights or writing files')
    parser.add_argument('--output-dir', type=Path)
    return parser.parse_args(argv)


def command_plan(args):
    from scripts.train_blackbox_ablation_v13_2 import parse_args as training_args

    python = str(PROJECT / 'openpi/.venv/bin/python')
    options = ['--num-steps', '400', '--batch-size', '8', '--fsdp-devices', '1',
               '--num-workers', '0', '--seed', '42', '--eval-interval', '200',
               '--save-interval', '200', '--keep-period', '200']
    if args.output_dir is not None:
        options.extend(['--output-dir', str(args.output_dir)])
    jobs = []
    for group, gpu in zip('ABCD', ('4', '5', '6', '7'), strict=True):
        effective = training_args(['--group', group, *options])
        run = (PROJECT / effective.output_dir / effective.run_name).resolve()
        stages = []
        if group in 'CD':
            stages.append(('train', [python, str(PROJECT / 'scripts/train_blackbox_ablation_v13_2.py'),
                                      '--group', group, *options]))
        stages.append(('test', [python, str(PROJECT / 'scripts/eval_blackbox_ablation_v13_2.py'),
                                '--run-dir', str(run), '--checkpoint', 'best', '--split', 'test']))
        jobs.append({'group': group, 'gpu': gpu, 'run_dir': run, 'stages': stages})
    return jobs


def preflight(jobs):
    """Never overwrite C/D or start a partial launch with missing A/B weights."""
    for job in jobs:
        run = job['run_dir']
        if job['group'] in 'AB':
            for path in (run / 'config.json', run / 'best/metrics.json', run / 'best/delta_params/_METADATA'):
                if not path.is_file():
                    raise FileNotFoundError(path)
        elif run.exists():
            raise FileExistsError(f'{run} already exists; refusing to overwrite or silently resume C/D')


class ParallelRunner:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.processes = {}

    def stop(self, signum=None, frame=None):
        del signum, frame
        self.stopped.set()
        with self.lock:
            for process in self.processes.values():
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def run_group(self, job):
        group, gpu = job['group'], job['gpu']
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, USE_TF='0', PYTHONUNBUFFERED='1')
        env.setdefault('OPENPI_DATA_HOME', '/home/qxh/.cache/openpi')
        result = {'group': group, 'gpu': gpu, 'run_dir': str(job['run_dir']), 'stages': []}
        for stage, command in job['stages']:
            if self.stopped.is_set():
                result['returncode'] = 130
                return result
            log_path = self.log_dir / f'{group}_{stage}.log'
            print(f'[{group} GPU {gpu}] START {stage}: {log_path}', flush=True)
            with log_path.open('x') as log:
                log.write(f'CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}\n')
                log.flush()
                with self.lock:
                    if self.stopped.is_set():
                        result['returncode'] = 130
                        return result
                    process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    self.processes[group] = process
                try:
                    while True:
                        try:
                            code = process.wait(timeout=5)
                            break
                        except subprocess.TimeoutExpired:
                            if self.stopped.is_set():
                                try:
                                    code = process.wait(timeout=15)
                                except subprocess.TimeoutExpired:
                                    os.killpg(process.pid, signal.SIGKILL)
                                    code = process.wait()
                                break
                finally:
                    with self.lock:
                        self.processes.pop(group, None)
            result['stages'].append({'stage': stage, 'returncode': code, 'log': str(log_path)})
            print(f'[{group} GPU {gpu}] END {stage}: exit={code}', flush=True)
            if code:
                result['returncode'] = code
                return result  # A failed train never launches that group's test.
        result['returncode'] = 0
        return result


def main(argv=None):
    args = parse_args(argv)
    jobs = command_plan(args)
    for job in jobs:
        for stage, command in job['stages']:
            print(f'[{job["group"]} {stage}] CUDA_VISIBLE_DEVICES={job["gpu"]} {shlex.join(command)}', flush=True)
    if args.dry_run:
        return 0
    preflight(jobs)
    root = jobs[0]['run_dir'].parent
    log_dir = root / f'parallel_{datetime.now():%Y%m%d_%H%M%S_%f}'
    log_dir.mkdir(parents=True, exist_ok=False)
    runner = ParallelRunner(log_dir)
    previous = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(runner.run_group, job) for job in jobs]
            results = []
            for job, future in zip(jobs, futures, strict=True):
                try:
                    results.append(future.result())
                except Exception as error:
                    print(f'[{job["group"]}] FAILED: {error}', flush=True)
                    results.append({'group': job['group'], 'returncode': 1, 'error': str(error)})
        (log_dir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(f'All workers finished. Summary: {log_dir / "summary.json"}', flush=True)
        return 130 if runner.stopped.is_set() else int(any(r['returncode'] for r in results))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
