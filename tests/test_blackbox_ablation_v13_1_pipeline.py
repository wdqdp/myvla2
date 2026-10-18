"""No GPU work: verify train-all-before-test ordering and fail-fast behavior."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_blackbox_ablation_v13_1 as pipeline  # noqa: E402


def test_all_training_precedes_all_testing_and_paths_follow_options(tmp_path):
    args = pipeline.parse_args(['--', '--output-dir', str(tmp_path), '--seed', '43', '--num-steps', '777'])
    called = []
    def launch(command, **kwargs):
        called.append(command)
        assert kwargs == {'cwd': pipeline.PROJECT, 'check': False}
        return SimpleNamespace(returncode=0)
    assert pipeline.run_pipeline(args, launch) == 0
    assert [Path(c[1]).name for c in called] == ['train_blackbox_ablation_v13_1.py'] * 4 + [
        'eval_blackbox_ablation_v13_1.py'] * 4
    assert [c[c.index('--group') + 1] for c in called[:4]] == list('ABCD')
    for group, command in zip('ABCD', called[4:], strict=True):
        assert command[command.index('--run-dir') + 1] == str(tmp_path / f'pi05_blackbox_ablation_v13_1_{group}_s43')
        assert command[-4:] == ['--checkpoint', 'best', '--split', 'test']


def test_training_failure_never_starts_tests():
    called = []
    def launch(command, **kwargs):
        called.append(command)
        return SimpleNamespace(returncode=23 if len(called) == 2 else 0)
    assert pipeline.run_pipeline(pipeline.parse_args([]), launch) == 23
    assert len(called) == 2 and all('train_blackbox' in c[1] for c in called)


def test_test_failure_stops_remaining_tests():
    called = []
    def launch(command, **kwargs):
        called.append(command)
        return SimpleNamespace(returncode=9 if len(called) == 5 else 0)
    assert pipeline.run_pipeline(pipeline.parse_args([]), launch) == 9
    assert len(called) == 5


def test_single_group_custom_run_name(tmp_path):
    args = pipeline.parse_args(['--groups', 'C', '--', '--output-dir', str(tmp_path), '--run-name', 'custom'])
    train, test = pipeline.command_plan(args)
    assert len(train) == len(test) == 1
    assert test[0][test[0].index('--run-dir') + 1] == str(tmp_path / 'custom')
    with pytest.raises(ValueError, match='same run directory'):
        pipeline.command_plan(pipeline.parse_args(['--', '--run-name', 'collision']))


def test_dry_run_never_launches():
    def launch(*args, **kwargs):
        pytest.fail('dry-run must not launch subprocesses')
    assert pipeline.run_pipeline(pipeline.parse_args(['--dry-run']), launch) == 0


@pytest.mark.parametrize('argv', [['--groups', 'A', 'A'], ['--', '--group', 'B'], ['--', '--audit-only']])
def test_reject_duplicate_or_nontraining_children(argv):
    with pytest.raises(SystemExit):
        pipeline.parse_args(argv)
