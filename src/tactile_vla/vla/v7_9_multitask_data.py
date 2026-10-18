"""V7.9: drop ONLY selected training adjustment counterfactuals from frozen V7.7.2."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

from tactile_vla.vla import v7_7_2_multitask_data as source
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_7_multitask_data import TASK_CYCLE, load_jsonl
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE as PROMPT_PROFILE

ROOT = Path('/data1/qxh/tac_vla_new/tac_data/demon_data/black_box')
VERSION_TAG = 'v7_9'
DATA_PROFILE = 'rotation_phase_v7_9_five_task_h100_factual_train'
INDEX_SCHEMA = 'tactile_vla_v7_9_multitask_training_index_v1'
DEFAULT_SOURCE_INDEX = ROOT / 'outputs/rotation_v7_7_2_multitask/v7_7_2_multitask_training_index.json'
DEFAULT_OUTPUT = ROOT / 'outputs/rotation_v7_9_multitask'
DEFAULT_INDEX = DEFAULT_OUTPUT / 'v7_9_multitask_training_index.json'
DEFAULT_STAGE_A = ROOT / 'outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_2_no_history/10000'
RUN_NAME = 'pi05_rotation_v7_9_five_task_h100_no_history'
TASKS = TASK_CYCLE[1:]
SPLITS = ('train', 'val', 'test')
MANIFEST_NAMES = {'adjustment': 'adjustment_end_manifest.jsonl', 'need': 'need_recovery_manifest.jsonl',
                  'failure': 'failure_reason_manifest.jsonl', 'plan': 'recovery_plan_manifest.jsonl'}
COUNTERFACTUAL_POLICY = {
    'train_adjustment': 'drop_moderately_to_slightly_and_slightly_to_moderately',
    'factual_sampling': 'retain_source_selected_factual_no_resampling',
    'val_test': 'unchanged_source_including_counterfactual',
    'other_tasks': 'unchanged_source',
}
ADJUSTMENT_SAMPLING_POLICY = 'source_selected_factual_only_no_resampling'
VARIANTS = ('factual', 'moderately_to_slightly', 'slightly_to_moderately')


def load_source(path):
    index = json.loads(Path(path).read_text())
    source.validate_index(index)
    rows = {}
    for task in TASKS:
        manifest = Path(index[f'{task}_manifest_file'])
        rows[task] = load_jsonl(manifest)
        if (sha256_file(manifest) != index[f'{task}_manifest_sha256']
                or sha256_json(rows[task]) != index['manifest_content_hashes'][task]):
            raise ValueError(f'V7.9 source {task} manifest hash mismatch')
        for split in SPLITS:
            stream = index['splits'][split][task]
            for position, global_index in zip(stream['manifest_row_indices'], stream['global_indices'], strict=True):
                row = rows[task][position]
                if row['split'] != split or row['global_index'] != global_index:
                    raise ValueError(f'V7.9 source {split}/{task} selection mismatch')
    if any(r['sample_variant'] not in VARIANTS for r in rows['adjustment']):
        raise ValueError('Unknown source adjustment variant')
    return index, rows


def derive(original, original_rows, source_path):
    rows = copy.deepcopy(original_rows)
    kept, remap = [], {}
    for position, row in enumerate(rows['adjustment']):
        if row['split'] == 'train' and row['sample_variant'] != 'factual':
            continue
        remap[position] = len(kept)
        kept.append(row)
    rows['adjustment'] = kept
    index = copy.deepcopy(original)
    index.pop('training_data_hash', None)
    for task in TASKS:
        index.pop(f'{task}_manifest_file', None)
        index.pop(f'{task}_manifest_sha256', None)
    for split in SPLITS:
        old = original['splits'][split]['adjustment']
        positions = [i for i in old['manifest_row_indices'] if i in remap]
        index['splits'][split]['adjustment'] = {
            **old, 'manifest_row_indices': [remap[i] for i in positions],
            'global_indices': [original_rows['adjustment'][i]['global_index'] for i in positions],
            'sample_count': len(positions),
        }
    path = str(Path(source_path).resolve())
    index.update(schema_version=INDEX_SCHEMA, experiment_version=VERSION_TAG, data_profile=DATA_PROFILE,
        counterfactual_policy=copy.deepcopy(COUNTERFACTUAL_POLICY),
        adjustment_sampling_policy=ADJUSTMENT_SAMPLING_POLICY,
        source_multitask_index_file=path, source_multitask_index_sha256=sha256_file(path),
        source_multitask_training_data_hash=original['training_data_hash'],
        manifest_content_hashes={task: sha256_json(items) for task, items in rows.items()})
    index['source_hashes'][path] = sha256_file(path)
    for task in TASKS:
        index['source_hashes'][original[f'{task}_manifest_file']] = original[f'{task}_manifest_sha256']
    return index, rows


def validate_index(index):
    if (index.get('schema_version'), index.get('experiment_version'), index.get('data_profile')) != (
            INDEX_SCHEMA, VERSION_TAG, DATA_PROFILE):
        raise ValueError('V7.9 requires its own factual-train index')
    if index.get('training_data_hash') != sha256_json({k: v for k, v in index.items() if k != 'training_data_hash'}):
        raise ValueError('V7.9 index hash mismatch')
    path = Path(index['source_multitask_index_file'])
    if sha256_file(path) != index['source_multitask_index_sha256']:
        raise ValueError('V7.9 source V7.7.2 index changed')
    original, original_rows = load_source(path)
    expected, rows = derive(original, original_rows, path)
    excluded = {'training_data_hash'}
    for task in TASKS:
        excluded.update((f'{task}_manifest_file', f'{task}_manifest_sha256'))
        manifest = Path(index[f'{task}_manifest_file'])
        if sha256_file(manifest) != index[f'{task}_manifest_sha256'] or load_jsonl(manifest) != rows[task]:
            raise ValueError(f'V7.9 {task} is not the exact source-only-training-counterfactual removal')
    if {k: v for k, v in index.items() if k not in excluded} != expected:
        raise ValueError('V7.9 changes selection/labels/prompts/history/action or the ablation policy')


def validate_stage_a_for_training(checkpoint, index):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint != Path(index['stage_a_checkpoint']).resolve():
        raise ValueError('V7.9 must use the same Stage A checkpoint as V7.7.2')
    config_path, metadata = checkpoint.parent / 'config.json', checkpoint / 'params/_METADATA'
    config = json.loads(config_path.read_text())
    action_path = Path(index['action_index_file'])
    action = json.loads(action_path.read_text())
    identity = config.get('artifact_identity', {})
    norms = [Path(p) for p in index['source_hashes'] if Path(p).name == 'norm_stats.json']
    if len(norms) != 1:
        raise ValueError('V7.9 requires the original norm stats identity')
    if (config.get('data_profile') != 'rotation_phase_v7_4_2_adjustment'
            or config.get('use_state_history') is not False or config.get('prompt_profile') != 'phase_v2'
            or identity.get('training_data_hash') != index['action_training_data_hash']
            or identity.get('index_sha256') != sha256_file(action_path)
            or identity.get('v4_norm_stats_sha256') != sha256_file(norms[0])
            or action.get('training_data_hash') != index['action_training_data_hash'] or not metadata.is_file()):
        raise ValueError('V7.9 requires the source V7.4.2 Stage A action/data/norm identity')
    return {'path': str(checkpoint), 'step': int(checkpoint.name), 'config_sha256': sha256_file(config_path),
            'params_metadata_sha256': sha256_file(metadata), 'action_training_data_hash': index['action_training_data_hash']}


def summary(index, rows):
    return {'experiment_version': VERSION_TAG, 'counterfactual_policy': COUNTERFACTUAL_POLICY,
        'source_multitask_training_data_hash': index['source_multitask_training_data_hash'],
        'stage_a_checkpoint': index['stage_a_checkpoint'],
        'selected_counts': {split: {'action': len(index['splits'][split]['action']['indices'])}
            | {task: index['splits'][split][task]['sample_count'] for task in TASKS} for split in SPLITS},
        'adjustment': {split: {
            'positive': sum(bool(rows['adjustment'][i]['adjustment_end']) for i in index['splits'][split]['adjustment']['manifest_row_indices']),
            'negative': sum(not rows['adjustment'][i]['adjustment_end'] for i in index['splits'][split]['adjustment']['manifest_row_indices']),
            'by_variant': dict(Counter(rows['adjustment'][i]['sample_variant']
                for i in index['splits'][split]['adjustment']['manifest_row_indices'])),
        } for split in SPLITS}}
