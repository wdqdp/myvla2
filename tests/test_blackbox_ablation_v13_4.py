"""V13.4 feedback-only inputs, exact source preservation, opt-in qpos mask and ABC schedule."""
# ruff: noqa: E402
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT), str(ROOT / 'src'), str(ROOT / 'openpi/src')]

from test_blackbox_ablation_v13_2 import built as source_built  # noqa: F401
from test_blackbox_ablation_v13_1 import fixture as v13_1_fixture  # noqa: F401
from tactile_vla.vla import blackbox_ablation_v13_4_data as data
from tactile_vla.vla.artifacts import sha256_file, sha256_json


@pytest.fixture(scope='module')
def built(source_built, tmp_path_factory):  # noqa: F811
    out, _, _, originals, _, _ = source_built
    target = tmp_path_factory.mktemp('v13_4') / 'data'
    index, summary = data.build_data(source_index=out / 'index.json', output_dir=target)
    checked, rows = data.validate_index(target / 'index.json', require_images=True)
    assert checked == index
    return target, index, summary, rows, originals


def test_exact_observations_targets_splits_and_feedback_preserved_without_plan(built):
    _, index, summary, rows, originals = built
    assert index['groups'] == ['A', 'B', 'C'] and len(rows) == 3600
    assert {s: v['samples'] for s, v in summary['splits'].items()} == {'train': 2880, 'val': 360, 'test': 360}
    for row, original in zip(rows, originals, strict=True):
        assert {k: v for k, v in row.items() if k not in ('events', 'prompts')} == {
            k: v for k, v in original.items() if k not in ('events', 'prompts')}
        for event, old_event in zip(row['events'], original['events'], strict=True):
            assert event == {k: v for k, v in old_event.items() if k != 'recovery_plan'}
        assert all('recovery_plan=' not in prompt and 'initial plan' not in prompt for prompt in row['prompts'].values())
        a, b, c = (row['prompts'][g] for g in data.GROUPS)
        assert a.endswith('Feedback history: none') and 'Touch[' not in a and 'failure_reason=' not in a
        assert b.count('Touch[') == row['memory_length'] and 'failure_reason=' not in b
        assert c.count('failure_reason=') == row['memory_length'] and 'Touch[' not in c
        assert b.split('Feedback history: ', 1)[1].split(' | ')[-1] == row['current_touch']
        assert c.split('Feedback history: ', 1)[1].split(' | ')[-1] == data.source.failure(row['horizontal_direction'])


def test_policy_is_isolated_and_source_prompts_remain_original(built):
    _, index, _, rows, originals = built
    assert index['qpos_scope'] == 'experiment_only_main_method_preserves_qpos'
    assert index['feedback_input_protocol'] == data.FEEDBACK_PROTOCOL
    assert set(originals[0]['prompts']) == set('ABCD')
    assert 'initial plan' in originals[0]['prompts']['C']
    changed = copy.deepcopy(rows)
    changed[0]['events'][0]['recovery_plan'] = 'initial plan'
    with pytest.raises(ValueError, match='changed'):
        data.validate_rows(changed, originals)


@pytest.mark.parametrize('field', ['prompt', 'target', 'touch', 'qpos', 'grasp', 'order'])
def test_semantic_rehash_cannot_hide_input_or_label_changes(built, tmp_path, field):
    out, index, _, rows, _ = built
    changed = copy.deepcopy(rows)
    if field == 'prompt':
        changed[0]['prompts']['B'] += ' recovery_plan=move left moderately'
    elif field == 'target':
        changed[0]['target_recovery_plan'] = data.source.plan('right', 'slightly')
    elif field == 'touch':
        changed[0]['events'][0]['tactile_caption'] = 'Touch[rotation=none]'
    elif field == 'qpos':
        changed[0]['qpos'][0] += 1
    elif field == 'grasp':
        changed[0]['history_grip_orientations'][0] = 'invalid'
    else:
        changed[0], changed[1] = changed[1], changed[0]
    event_path = tmp_path / 'events.jsonl'
    event_path.write_text(''.join(json.dumps(r) + '\n' for r in changed))
    edited = copy.deepcopy(index)
    edited['events_file'] = str(event_path)
    edited['generated_hashes'].pop(str(out / 'plan_events.jsonl'))
    edited['generated_hashes'][str(event_path)] = sha256_file(event_path)
    edited['training_data_hash'] = sha256_json({k: v for k, v in edited.items() if k != 'training_data_hash'})
    index_path = tmp_path / 'index.json'
    index_path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match='changed'):
        data.validate_index(index_path, verify_sources=False)


def test_overwrite_rejected_and_observed_aliasing_audit_is_not_claim_of_generalization(built):
    out, index, summary, rows, _ = built
    with pytest.raises(FileExistsError):
        data.build_data(source_index=index['source_index_file'], output_dir=out)
    for split in data.source.prior.SPLITS:
        a = summary['splits'][split]['input_aliasing_audit']['A']
        assert a['observed_deterministic_joint_ceiling'] == pytest.approx(2 / 3)
        assert summary['splits'][split]['input_aliasing_audit']['C']['conflicting_target_buckets'] == 0
    assert data.ambiguity_audit(rows)['A']['conflicting_target_buckets'] == 1200


def test_cli_three_groups_and_resume_protocol_guard():
    from scripts import train_blackbox_ablation_v13_4 as trainer
    from scripts import train_blackbox_ablation_v13_3 as previous
    from scripts.eval_blackbox_ablation_v13_4 import validate_protocol
    args = trainer.parse_args(['--group', 'C'])
    assert args.num_steps == 400 and args.batch_size == 8 and args.fsdp_devices == 1
    assert args.eval_interval == args.save_interval == args.keep_period == 200
    assert args.run_name == 'pi05_blackbox_ablation_v13_4_C_s42'
    assert args.qpos_scope == data.QPOS_SCOPE and args.feedback_input_protocol == data.FEEDBACK_PROTOCOL
    assert previous.parse_args(['--group', 'D']).run_name == 'pi05_blackbox_ablation_v13_3_D_s42'
    with pytest.raises(SystemExit):
        trainer.parse_args(['--group', 'D'])
    current = vars(args) | {'experiment_version': data.SCHEMA}
    validate_protocol(current)
    with pytest.raises(ValueError):
        validate_protocol(current | {'qpos_scope': 'main_method'})
    with pytest.raises(ValueError, match='feedback_input_protocol'):
        trainer.common.resume_check(current, current | {'feedback_input_protocol': {}})


def test_masked_real_training_and_inference_transform_and_all_three_token_audits(built):
    from scripts.train_blackbox_ablation_v13_4 import components, parse_args
    from tactile_vla.vla.blackbox_ablation_v13_4_runtime import make_loader, validate_tokens
    out, _, _, rows, _ = built
    args = parse_args(['--group', 'A', '--index-file', str(out / 'index.json')])
    args.precision = 'float32'
    _, _, config, tokenizer, grammar, norm = components(args, verify_sources=False)
    audit = validate_tokens(rows, tokenizer=tokenizer, grammar=grammar, norm=norm, max_len=384)
    assert set(audit) == set('ABC')
    assert all(v['count'] == 3600 and v['max_tokens'] < 384 for v in audit.values())
    for group in data.GROUPS:
        for inference in (False, True):
            loader = make_loader(rows[:3], group=group, model_config=config, tokenizer=tokenizer,
                grammar=grammar, norm=norm, batch_size=2, num_workers=0, inference=inference)
            batch = next(iter(loader))
            assert not batch['state'].any()
            assert bool(batch['token_loss_mask'].any()) != inference
            assert loader.qpos_audit['qpos_token_invariant']


def test_runner_only_three_groups_and_global_training_barrier(tmp_path):
    from scripts import run_blackbox_ablation_v13_4 as pipeline
    args = pipeline.common.parse_args(['--output-dir', str(tmp_path)])
    groups, phases = pipeline.command_plan(args)
    assert list(groups) == list('ABC')
    assert [lane['gpu'] for lane in phases['train']] == ['3', '4']
    assert [lane['gpu'] for lane in phases['test']] == ['3', '4']
    assert [[name for name, _ in lane['stages']] for lane in phases['train']] == [['train_A', 'train_C'], ['train_B']]
    calls = []
    class Runner:
        stopped = SimpleNamespace(is_set=lambda: False)
        def run_group(self, lane):
            calls.append([name for name, _ in lane['stages']])
            return {'returncode': 0}
    result = pipeline.common.run_phases(phases, Runner(), description='GPU3 A/C; GPU4 B')
    assert len(result) == 4
    assert all(name.startswith('train') for lane in calls[:2] for name in lane)
    assert all(name.startswith('test') for lane in calls[2:] for name in lane)
    assert pipeline.main(['--dry-run', '--output-dir', str(tmp_path / 'absent')]) == 0
    assert not (tmp_path / 'absent').exists()


def test_runner_phase_guard_can_cancel_testing_without_starting_it(tmp_path):
    from scripts import run_blackbox_ablation_v13_4 as pipeline
    groups, phases = pipeline.command_plan(pipeline.common.parse_args(['--output-dir', str(tmp_path)]))
    called = []
    class Runner:
        stopped = SimpleNamespace(is_set=lambda: False)
        def run_group(self, lane):
            called.extend(name for name, _ in lane['stages'])
            return {'returncode': 0}
    result = pipeline.common.run_phases(phases, Runner(), before_phase=lambda stage: stage != 'test')
    assert set(called) == {'train_A', 'train_B', 'train_C'}
    assert result[-1]['returncode'] == 130 and result[-1]['phase'] == 'test'


def test_gpu_occupancy_check_and_wait_target_only_three_four(monkeypatch):
    from scripts import run_blackbox_ablation_v13_4 as pipeline
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='GPU-test, 123\n')
    monkeypatch.setattr(pipeline.subprocess, 'run', query)
    assert pipeline.busy_gpu_processes() == 'GPU-test, 123'
    assert '--id=3,4' in calls[0] and '--id=6,7' not in calls[0]
    args = pipeline.common.parse_args([])
    runner = SimpleNamespace(stopped=SimpleNamespace(is_set=lambda: False))
    with pytest.raises(RuntimeError, match='GPU 3/4 are occupied'):
        pipeline.wait_for_gpus(args, runner)
    monkeypatch.setattr(pipeline, 'busy_gpu_processes', lambda: '')
    assert pipeline.wait_for_gpus(args, runner)
