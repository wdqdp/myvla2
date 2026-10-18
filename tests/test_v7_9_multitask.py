"""V7.9 removes only train adjustment CF; keeps factual selection and all evaluation rows."""
# ruff: noqa: E402, F811
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT), str(ROOT / 'src'), str(ROOT / 'openpi/src')]

from test_v7_8_multitask import source_artifacts  # noqa: F401
from tactile_vla.vla import v7_9_multitask_data as data
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_9_multitask_eval import report_by_variant


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows))


def persisted(source_artifacts):
    original, old_rows, source_file = source_artifacts
    index, rows = data.derive(original, old_rows, source_file)
    for task in data.TASKS:
        path = source_file.parent / f'v79_{task}.jsonl'
        write_rows(path, rows[task])
        index[f'{task}_manifest_file'], index[f'{task}_manifest_sha256'] = str(path), sha256_file(path)
    index['training_data_hash'] = sha256_json(index)
    return index, rows


def test_only_training_counterfactuals_removed_and_input_payloads_exact(source_artifacts):
    original, old_rows, source_file = source_artifacts
    before = copy.deepcopy((original, old_rows))
    index, rows = data.derive(original, old_rows, source_file)
    for task in ('need', 'failure', 'plan'):
        assert rows[task] == old_rows[task]
        for split in data.SPLITS:
            assert index['splits'][split][task] == original['splits'][split][task]
    assert rows['adjustment'] == [r for r in old_rows['adjustment']
        if r['split'] != 'train' or r['sample_variant'] == 'factual']
    for split in data.SPLITS:
        old_stream, new_stream = original['splits'][split], index['splits'][split]
        assert old_stream['action'] == new_stream['action']
        expected = [old_rows['adjustment'][i] for i in old_stream['adjustment']['manifest_row_indices']
            if split != 'train' or old_rows['adjustment'][i]['sample_variant'] == 'factual']
        assert [rows['adjustment'][i] for i in new_stream['adjustment']['manifest_row_indices']] == expected
        assert new_stream['adjustment']['global_indices'] == [r['global_index'] for r in expected]
    for key in ('prompt_profile', 'history_policy', 'adjustment_label_policy', 'need_successful_recovery_start',
                'stage_a_checkpoint', 'action_index_file', 'action_training_data_hash', 'selection_hash', 'seed'):
        assert index[key] == original[key]
    assert index['counterfactual_policy'] == data.COUNTERFACTUAL_POLICY
    assert (original, old_rows) == before
    assert all('Touch' in row['prompt'] for rows_ in rows.values() for row in rows_)


@pytest.mark.parametrize('tamper', ['label', 'history', 'plan', 'train_cf', 'selection', 'action', 'policy'])
def test_rehashed_semantic_changes_rejected(source_artifacts, tamper):
    index, rows = persisted(source_artifacts)
    data.validate_index(index)
    task = 'adjustment'
    if tamper == 'label':
        rows[task][0]['adjustment_end'] = not rows[task][0]['adjustment_end']
    elif tamper == 'history':
        rows[task][0]['qpos_h100_11_discrete'][0][0] += 1
    elif tamper == 'plan':
        task = 'plan'
        rows[task][0]['target_recovery_plan'] = 'changed'
    elif tamper == 'train_cf':
        rows[task][0]['sample_variant'] = 'moderately_to_slightly'
    elif tamper == 'selection':
        index['splits']['val'][task]['manifest_row_indices'].pop()
    elif tamper == 'action':
        index['splits']['train']['action']['indices'].reverse()
    else:
        index['counterfactual_policy']['val_test'] = 'drop_cf'
    path = Path(index[f'{task}_manifest_file'])
    write_rows(path, rows[task])
    index[f'{task}_manifest_sha256'] = sha256_file(path)
    index['manifest_content_hashes'][task] = sha256_json(rows[task])
    index['training_data_hash'] = sha256_json({k: v for k, v in index.items() if k != 'training_data_hash'})
    with pytest.raises(ValueError):
        data.validate_index(index)


def test_source_change_and_index_hash_change_rejected(source_artifacts):
    index, _ = persisted(source_artifacts)
    with pytest.raises(ValueError, match='index hash'):
        data.validate_index(index | {'seed': 43})
    original, _, source_file = source_artifacts
    data.load_source(source_file)
    Path(original['need_manifest_file']).write_text('{}\n')
    with pytest.raises(ValueError, match='manifest hash'):
        data.load_source(source_file)


@pytest.mark.parametrize('tamper', ['profile', 'weights', 'norm', 'checkpoint'])
def test_initialization_requires_source_v742_stage_a(source_artifacts, tamper):
    original, old_rows, source_file = source_artifacts
    index, _ = data.derive(original, old_rows, source_file)
    checkpoint = Path(index['stage_a_checkpoint'])
    assert data.validate_stage_a_for_training(checkpoint, index)['step'] == 10000
    file = checkpoint.parent / 'config.json'
    config = json.loads(file.read_text())
    if tamper == 'profile':
        config['data_profile'] = data.DATA_PROFILE
    elif tamper == 'weights':
        config['artifact_identity']['training_data_hash'] = 'changed'
    elif tamper == 'norm':
        config['artifact_identity']['v4_norm_stats_sha256'] = 'changed'
    else:
        checkpoint = checkpoint.parent / '5000'
    file.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Stage A'):
        data.validate_stage_a_for_training(checkpoint, index)


def test_builder_dry_run_does_not_write_and_existing_output_protected(source_artifacts, tmp_path):
    from scripts import build_v7_9_multitask_data as builder
    original, old_rows, source = source_artifacts
    output = tmp_path / 'output'
    assert builder.main(['--source-index', str(source), '--output-dir', str(output), '--dry-run']) == 0
    assert not output.exists()
    assert builder.main(['--source-index', str(source), '--output-dir', str(output)]) == 0
    built = json.loads((output / 'v7_9_multitask_training_index.json').read_text())
    data.validate_index(built)
    counts = json.loads((output / 'summary.json').read_text())
    assert counts['adjustment']['train']['by_variant'] == {'factual': 1}
    assert counts['adjustment']['val']['by_variant']['slightly_to_moderately'] == 1
    for task in ('need', 'failure', 'plan'):
        assert Path(built[f'{task}_manifest_file']).read_bytes() == Path(original[f'{task}_manifest_file']).read_bytes()
    with pytest.raises(FileExistsError):
        builder.main(['--source-index', str(source), '--output-dir', str(output)])


@pytest.fixture
def training_entry(monkeypatch):
    from scripts import train_vla_multitask_v7_9 as entry
    for obj in (entry.trainer, entry.base):
        for name, value in list(vars(obj).items()):
            if not name.startswith('__'):
                monkeypatch.setattr(obj, name, value)
    entry.configure_version()
    return entry


def test_training_defaults_and_shared_five_task_model(training_entry):
    entry = training_entry
    args = entry.parse_args([])
    assert (args.num_steps, args.batch_size, args.fsdp_devices, args.eval_interval,
            args.save_interval, args.keep_period, args.eval_max_need_samples) == (2000, 8, 2, 1000, 1000, 1000, 4096)
    assert args.stage_a_checkpoint == data.DEFAULT_STAGE_A
    assert args.dataset_dir.name == 'tactile_vla_rotation_v4_1_new_env'
    assert (args.lr, args.weight_decay, args.grad_clip, args.seed) == (1e-4, 1e-4, 1., 42)
    assert args.run_name == data.RUN_NAME and args.data_profile == data.DATA_PROFILE
    assert args.prompt_profile == data.PROMPT_PROFILE and not args.use_state_history
    entry.configure_training()
    assert entry.base.StageBV3Model is entry.trainer.V77MultitaskModel
    assert entry.base.TASK_CYCLE == data.TASK_CYCLE
    assert entry.base.build_loaders is entry.trainer.build_loaders
    assert entry.base.evaluate_adjustment is entry.evaluate_adjustment
    assert entry.base.EXTRA_CONFIG['counterfactual_policy'] == data.COUNTERFACTUAL_POLICY


def test_cli_overrides_and_equals_form(training_entry):
    args = training_entry.parse_args(['--num-steps=3000', '--batch-size=16', '--fsdp-devices=1'])
    assert (args.num_steps, args.batch_size, args.fsdp_devices) == (3000, 16, 1)


@pytest.mark.parametrize('key', ['counterfactual_policy', 'source_multitask_training_data_hash', 'lr'])
def test_resume_rejects_changed_ablation_or_optimizer(training_entry, key):
    entry = training_entry
    args = entry.parse_args([])
    entry.configure_training()
    saved = vars(args) | copy.deepcopy(entry.base.EXTRA_CONFIG) | {'precision': 'float32'}
    entry.validate_resume(saved, args, actual_precision='float32')
    saved[key] = 'changed'
    with pytest.raises(ValueError, match='resume'):
        entry.validate_resume(saved, args, actual_precision='float32')


def test_export_carries_version_and_policy(training_entry, tmp_path, monkeypatch):
    entry = training_entry
    entry.configure_training()
    step_dir = tmp_path / '1000'
    step_dir.mkdir()
    def exported(*args):
        (step_dir / 'v7_9_export.json').write_text(json.dumps({'exports': {'delta_params': {}, 'full_params': {}}}))
    monkeypatch.setattr(entry, '_BASE_EXPORT', exported)
    entry.export_checkpoint(tmp_path, None, 1000, None)
    result = json.loads((step_dir / 'v7_9_export.json').read_text())
    assert result['data_profile'] == data.DATA_PROFILE
    assert result['counterfactual_policy'] == data.COUNTERFACTUAL_POLICY
    assert set(result['exports']) == {'delta_params', 'full_params'}


def test_subgroup_metrics_keep_original_aggregate():
    from tactile_vla.common.metrics import classification_report
    true, pred = [0, 1, 1, 0], [0, 0, 1, 0]
    report = report_by_variant(true, pred, ['factual', 'factual', 'moderately_to_slightly', 'slightly_to_moderately'])
    assert {k: v for k, v in report.items() if k != 'by_sample_variant'} == classification_report(
        true, pred, num_classes=2, class_names=['false', 'true'])
    assert report['by_sample_variant']['factual']['accuracy'] == .5
    assert report['by_sample_variant']['slightly_to_moderately']['accuracy'] == 1.
    with pytest.raises(ValueError):
        report_by_variant([1], [], [])


def test_both_training_cf_variants_removed_without_restoring_factual_rows(source_artifacts):
    original, rows, path = copy.deepcopy(source_artifacts)
    factual = [r for r in rows['adjustment'] if r['split'] == 'train' and r['sample_variant'] == 'factual']
    extra = copy.deepcopy(next(r for r in rows['adjustment'] if r['split'] == 'train'))
    extra.update(sample_variant='moderately_to_slightly', adjustment_end=True)
    position = len(rows['adjustment'])
    rows['adjustment'].append(extra)
    stream = original['splits']['train']['adjustment']
    stream['manifest_row_indices'].append(position)
    stream['global_indices'].append(extra['global_index'])
    stream['sample_count'] += 1
    index, result = data.derive(original, rows, path)
    assert [r for r in result['adjustment'] if r['split'] == 'train'] == factual
    assert index['splits']['train']['adjustment']['sample_count'] == len(factual)


def test_eval_variant_lookup_respects_manifest_indices_padding_and_limit(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from flax import nnx
    from openpi.shared import nnx_utils
    from scripts import train_vla_stage_b_v3 as base
    from tactile_vla.vla.v7_9_multitask_eval import evaluate_adjustment
    model = SimpleNamespace(eval=lambda: None, adjustment_end_logits=lambda batch: batch['logits'])
    monkeypatch.setattr(nnx, 'merge', lambda *args: model)
    monkeypatch.setattr(nnx_utils, 'module_jit', lambda fn: fn)
    monkeypatch.setattr(base, 'pad_eval_batch', lambda batch, multiple: (batch, batch['valid']))
    monkeypatch.setattr(base, 'batch_to_jax', lambda batch, task, shard: (batch, {'adjustment': batch['labels']}))
    rows = [{'sample_variant': variant} for variant in ('factual', 'moderately_to_slightly', 'slightly_to_moderately')]
    class Loader:
        dataset = SimpleNamespace(dataset=SimpleNamespace(rows=rows))
        def __iter__(self):
            yield {'manifest_row_index': np.array([2, 0]), 'labels': np.array([0, 1]),
                   'logits': np.array([[1., 0.], [0., 1.]]), 'valid': 2}
            yield {'manifest_row_index': np.array([1, 0]), 'labels': np.array([1, 0]),
                   'logits': np.array([[0., 1.], [0., 1.]]), 'valid': 1}
    report = evaluate_adjustment(SimpleNamespace(model_def=None, params=None), Loader(),
                                 SimpleNamespace(device_set={0}), max_samples=3)
    assert report['accuracy'] == 1.
    assert set(report['by_sample_variant']) == set(data.VARIANTS)
    assert all(values['accuracy'] == 1. for values in report['by_sample_variant'].values())
