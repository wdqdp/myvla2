"""V13.3: identical train/test masking, isolated runs, fixed GPU lanes and test barrier."""
# ruff: noqa: E402
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT), str(ROOT / 'src'), str(ROOT / 'openpi/src')]

from scripts import run_blackbox_ablation_v13_3 as pipeline
from scripts import train_blackbox_ablation_v13_3 as trainer
from scripts.eval_blackbox_ablation_v13_3 import validate_protocol
from tactile_vla.vla.blackbox_ablation_v13_3_runtime import QPOS_PROTOCOL, SCHEMA, audit_inputs, validate_tokens
from tactile_vla.vla.openpi_bridge import build_structured_text_transform
from tactile_vla.vla.qpos_masking import mask_structured_transform
from test_blackbox_ablation_v13_2_no_qpos import make_transform, raw


def test_defaults_new_protocol_isolation_and_resume_guard():
    old = trainer.previous.parse_args(['--group', 'A'])
    new = trainer.parse_args(['--group', 'A'])
    assert old.num_steps == 1000 and new.num_steps == 400
    assert old.run_name == 'pi05_blackbox_ablation_v13_2_A_s42'
    assert new.run_name == 'pi05_blackbox_ablation_v13_3_A_s42'
    assert new.output_dir.name == 'plan_v13_3' and new.index_file == old.index_file
    assert new.qpos_input_protocol == QPOS_PROTOCOL
    assert trainer.parse_args(['--group', 'B', '--num-steps', '800']).num_steps == 800
    validate_protocol({'experiment_version': SCHEMA, 'qpos_input_protocol': QPOS_PROTOCOL})
    with pytest.raises(ValueError):
        validate_protocol({'experiment_version': 'blackbox_plan_ablation_v13_2'})
    with pytest.raises(ValueError, match='qpos_input_protocol'):
        trainer.previous.common.resume_check({'qpos_input_protocol': None}, {'qpos_input_protocol': QPOS_PROTOCOL})


def test_training_mask_is_before_target_tokenization_and_keeps_supervision():
    from openpi.shared.normalize import NormStats
    from tactile_vla.vla.structured_text import recovery_grammar
    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer(384)
    grammar = recovery_grammar(lambda text: tokenizer.encode_text(text, add_eos=True))
    config = SimpleNamespace(action_dim=32, max_token_len=384)
    norm = {'state': NormStats(mean=np.full(7, 2.), std=np.ones(7), q01=np.ones(7), q99=np.full(7, 3.))}
    transform = mask_structured_transform(build_structured_text_transform(
        config, tokenizer=tokenizer, grammar=grammar, norm_stats=norm), inference=False)
    row = {**raw(2.5), 'target_text': 'recovery_plan=move horizontally left moderately, move vertically none moderately.'}
    second = {**row, 'observation/state': np.full(7, 1.5, dtype=np.float32)}
    a, b = transform(dict(row)), transform(second)
    assert not a['state'].any() and a['token_loss_mask'].any()
    for k in ('tokenized_prompt', 'structured_target_compact_ids', 'structured_allowed_token_mask'):
        assert np.array_equal(a[k], b[k])
    loader = SimpleNamespace(dataset=SimpleNamespace(raw=lambda i: row, transform=transform))
    assert audit_inputs(loader, inference=False)['state_all_zero']


def test_inference_mask_uses_same_protocol_and_token_audit_uses_zero_state():
    original, tokenizer = make_transform()
    transform = mask_structured_transform(original, inference=True)
    loader = SimpleNamespace(dataset=SimpleNamespace(raw=lambda i: raw(2.5), transform=transform))
    assert audit_inputs(loader, inference=True)['qpos_token_invariant']
    class Grammar:
        def sequence_for_text(self, text):
            return [1, 2]
    rows = [{'prompts': {g: 'same prompt' for g in 'ABCD'}, 'target_recovery_plan': 'target'}]
    audit = validate_tokens(rows, tokenizer=tokenizer, grammar=Grammar(), norm=None, max_len=384)
    assert all(r['count'] == 1 and r['qpos_input_protocol'] == QPOS_PROTOCOL for r in audit.values())
    assert all(not state.any() for _, state, _ in tokenizer.calls)


def test_two_gpu_lanes_never_overlap_same_gpu_and_all_training_precedes_tests(tmp_path):
    args = pipeline.parse_args(['--output-dir', str(tmp_path)])
    groups, phases = pipeline.command_plan(args)
    assert all('v13_3' in job['run_dir'].name for job in groups.values())
    assert [[name for name, _ in lane['stages']] for lane in phases['train']] == [['train_A', 'train_C'], ['train_B', 'train_D']]
    assert [lane['gpu'] for lane in phases['train']] == ['6', '7']
    calls = []
    class Runner:
        stopped = SimpleNamespace(is_set=lambda: False)
        def run_group(self, lane):
            calls.append([name for name, _ in lane['stages']])
            return {'returncode': 0}
    result = pipeline.run_phases(phases, Runner())
    assert len(result) == 4
    assert all(name.startswith('train') for lane in calls[:2] for name in lane)
    assert all(name.startswith('test') for lane in calls[2:] for name in lane)
    for job in groups.values():
        command = job['train']
        assert command[command.index('--num-steps') + 1] == '400'
        for flag in ('--eval-interval', '--save-interval', '--keep-period'):
            assert command[command.index(flag) + 1] == '200'
    pipeline.preflight(groups)
    groups['C']['run_dir'].mkdir()
    with pytest.raises(FileExistsError):
        pipeline.preflight(groups)


def test_training_failure_prevents_all_testing_and_dry_run_never_writes(tmp_path):
    _, phases = pipeline.command_plan(pipeline.parse_args(['--output-dir', str(tmp_path)]))
    class Runner:
        stopped = SimpleNamespace(is_set=lambda: False)
        def run_group(self, lane):
            assert all(name.startswith('train') for name, _ in lane['stages'])
            return {'returncode': 23}
    assert len(pipeline.run_phases(phases, Runner())) == 2
    output = tmp_path / 'absent'
    assert pipeline.main(['--dry-run', '--output-dir', str(output)]) == 0
    assert not output.exists()


def test_busy_gpu_check_never_stops_existing_jobs():
    args = pipeline.parse_args([])
    runner = SimpleNamespace(stopped=SimpleNamespace(is_set=lambda: False))
    from unittest.mock import patch
    with patch.object(pipeline, 'busy_gpu_processes', return_value='GPU-uuid, 91173'):
        with pytest.raises(RuntimeError, match='occupied'):
            pipeline.wait_for_gpus(args, runner)
