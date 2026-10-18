"""CPU-only checks for the fixed four-GPU V13.2 schedule."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_blackbox_ablation_v13_2_parallel as parallel  # noqa: E402


def test_gpu_assignment_training_budget_and_best_tests(tmp_path):
    jobs = parallel.command_plan(parallel.parse_args(['--output-dir', str(tmp_path)]))
    assert [(j['group'], j['gpu']) for j in jobs] == list(zip('ABCD', ('4', '5', '6', '7'), strict=True))
    assert [[s for s, _ in j['stages']] for j in jobs] == [['test'], ['test'], ['train', 'test'], ['train', 'test']]
    for job in jobs:
        assert job['run_dir'] == tmp_path / f'pi05_blackbox_ablation_v13_2_{job["group"]}_s42'
        test = job['stages'][-1][1]
        assert test[-4:] == ['--checkpoint', 'best', '--split', 'test']
        if job['group'] in 'CD':
            command = job['stages'][0][1]
            for key, expected in {'num-steps': '400', 'batch-size': '8', 'fsdp-devices': '1',
                                  'eval-interval': '200', 'save-interval': '200', 'keep-period': '200'}.items():
                assert command[command.index('--' + key) + 1] == expected
            assert '--resume' not in command


def test_preflight_protects_existing_runs_and_missing_best(tmp_path):
    jobs = parallel.command_plan(parallel.parse_args(['--output-dir', str(tmp_path)]))
    with pytest.raises(FileNotFoundError):
        parallel.preflight(jobs)
    for job in jobs[:2]:
        for name in ('config.json', 'best/metrics.json', 'best/delta_params/_METADATA'):
            path = job['run_dir'] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
    parallel.preflight(jobs)
    jobs[2]['run_dir'].mkdir()
    with pytest.raises(FileExistsError):
        parallel.preflight(jobs)


def test_dry_run_never_creates_output(tmp_path):
    output = tmp_path / 'absent'
    assert parallel.main(['--dry-run', '--output-dir', str(output)]) == 0
    assert not output.exists()


def test_worker_sets_single_gpu_and_stops_after_failed_training(tmp_path, monkeypatch):
    jobs = parallel.command_plan(parallel.parse_args(['--output-dir', str(tmp_path)]))
    calls = []

    class Process:
        pid = 123456

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))

        def wait(self, timeout=None):
            return 23

    monkeypatch.setattr(parallel.subprocess, 'Popen', Process)
    runner = parallel.ParallelRunner(tmp_path)
    result = runner.run_group(jobs[2])
    assert result['returncode'] == 23
    assert len(calls) == 1
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == '6'
    assert calls[0][1]['start_new_session']
    runner.stop()
    assert runner.run_group(jobs[3])['returncode'] == 130
