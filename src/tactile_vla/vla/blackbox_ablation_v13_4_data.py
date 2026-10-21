"""V13.4: no plans in feedback memory; compare none/local Touch/global failure."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil

from tactile_vla.vla import blackbox_ablation_v13_2_data as source
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.blackbox_ablation_v13_3_runtime import QPOS_PROTOCOL

SCHEMA = 'blackbox_plan_ablation_v13_4'
GROUPS = ('A', 'B', 'C')
OUTPUT = source.OUTPUT.parent / 'ablation_v13_4'
INDEX = OUTPUT / 'index.json'
TRAIN_OUTPUT = source.TRAIN_OUTPUT.parent / 'plan_v13_4'
QPOS_SCOPE = 'experiment_only_main_method_preserves_qpos'
FEEDBACK_PROTOCOL = {
    'A': 'no_feedback_history',
    'B': 'local_touch_sequence_including_current',
    'C': 'global_failure_sequence_including_current_oracle',
    'past_plan': 'absent_in_all_groups',
    'standalone_current_touch': 'absent_in_all_groups',
    'order': 'oldest_to_current',
}


def build_prompt(group, instruction, events):
    if group not in GROUPS or not events:
        raise ValueError('Expected V13.4 group A/B/C and nonempty feedback events')
    if group == 'A':
        feedback = 'none'
    else:
        key = 'tactile_caption' if group == 'B' else 'failure_reason'
        feedback = ' | '.join(event[key] for event in events)
    prompt = f'Mode: reasoning.\nTask: {instruction.strip()}\nFeedback history: {feedback}'
    if 'recovery_plan=' in prompt or 'initial plan' in prompt:
        raise ValueError('Executed/initial plan leaked into V13.4 prompt')
    return prompt


def convert_row(original):
    row = copy.deepcopy(original)
    # Retain all observations/targets/provenance, but not the executed-plan shortcut.
    row['events'] = [{k: v for k, v in event.items() if k != 'recovery_plan'} for event in row['events']]
    row['prompts'] = {group: build_prompt(group, row['instruction'], row['events']) for group in GROUPS}
    return row


def validate_rows(rows, originals):
    if len(rows) != len(originals):
        raise ValueError('V13.4 must preserve every V13.2/V13.3 sample and its order')
    for row, original in zip(rows, originals, strict=True):
        if row != convert_row(original):
            raise ValueError('V13.4 changed source observation, label, sequence, prompt or provenance')
        if row['events'][-1]['tactile_caption'] != row['current_touch']:
            raise ValueError('B history must end with current Touch')
        if row['events'][-1]['failure_reason'] != source.failure(row['horizontal_direction']):
            raise ValueError('C history must end with current global failure')


def ambiguity_audit(rows):
    """Conservative observed aliasing: same current real observation AND same prompt."""
    report = {}
    for group in GROUPS:
        buckets = defaultdict(Counter)
        for row in rows:
            buckets[row['base_observation_id'], row['prompts'][group]][row['target_recovery_plan']] += 1
        report[group] = {
            'identical_input_buckets': len(buckets),
            'conflicting_target_buckets': sum(len(counts) > 1 for counts in buckets.values()),
            'samples_in_conflicting_buckets': sum(sum(c.values()) for c in buckets.values() if len(c) > 1),
            'observed_deterministic_joint_ceiling': sum(max(c.values()) for c in buckets.values()) / len(rows),
        }
    return report


def build_data(*, source_index=source.INDEX, output_dir=OUTPUT):
    source_index, output_dir = Path(source_index).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing overwrite: {output_dir}')
    original, originals = source.validate_index(source_index, require_images=True)
    rows = [convert_row(row) for row in originals]
    validate_rows(rows, originals)
    summary = {'schema_version': SCHEMA, 'source_training_data_hash': original['training_data_hash'],
        'feedback_input_protocol': FEEDBACK_PROTOCOL, 'qpos_input_protocol': QPOS_PROTOCOL, 'qpos_scope': QPOS_SCOPE,
        'history_is_synthetic': True, 'global_failure_is_oracle': True, 'images_ready': True,
        'magnitude_rule': 'first_opposite_GLOBAL_rotation_latches_slightly_forever', 'splits': {}}
    for split in source.prior.SPLITS:
        selected = [row for row in rows if row['split'] == split]
        summary['splits'][split] = {'episodes': original['splits'][split], 'samples': len(selected),
            'real_observations': len({row['base_observation_id'] for row in selected}),
            'kinds': dict(Counter(row['sample_kind'] for row in selected)),
            'reversal_stages': dict(Counter(row['reversal_stage'] for row in selected)),
            'classes': dict(Counter(f"{r['horizontal_direction']}/{r['horizontal_magnitude']}" for r in selected)),
            'input_aliasing_audit': ambiguity_audit(selected)}
    output_dir.mkdir(parents=True)
    (output_dir / 'norm_stats').mkdir()
    shutil.copyfile(Path(original['norm_stats_dir']) / 'norm_stats.json', output_dir / 'norm_stats/norm_stats.json')
    for name in ('splits.json', 'caption_audit.json', 'image_audit.json'):
        shutil.copyfile(Path(original['events_file']).parent / name, output_dir / name)
    (output_dir / 'plan_events.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    source.prior.write_json(output_dir / 'summary.json', summary)
    index = {'schema_version': SCHEMA, 'seed': original['seed'], 'groups': list(GROUPS),
        'feedback_input_protocol': copy.deepcopy(FEEDBACK_PROTOCOL), 'qpos_input_protocol': QPOS_PROTOCOL,
        'qpos_scope': QPOS_SCOPE, 'splits': original['splits'], 'source_index_file': str(source_index),
        'source_training_data_hash': original['training_data_hash'],
        'source_hashes': {**original['source_hashes'], **original['generated_hashes'], str(source_index): sha256_file(source_index)},
        'dataset_dir': original['dataset_dir'], 'lerobot_dataset_dir': original['lerobot_dataset_dir'],
        'mapping_by_stratum': original['mapping_by_stratum'], 'events_file': str(output_dir / 'plan_events.jsonl'),
        'norm_stats_dir': str(output_dir / 'norm_stats'),
        'generated_hashes': {str(output_dir / name): sha256_file(output_dir / name) for name in (
            'plan_events.jsonl', 'norm_stats/norm_stats.json', 'splits.json', 'caption_audit.json', 'image_audit.json', 'summary.json')}}
    index['training_data_hash'] = sha256_json(index)
    source.prior.write_json(output_dir / 'index.json', index)
    return index, summary


def validate_index(path, *, verify_sources=True, require_images=False):
    index = source.prior.read_json(path)
    if (index.get('schema_version') != SCHEMA or index.get('groups') != list(GROUPS)
            or index.get('feedback_input_protocol') != FEEDBACK_PROTOCOL
            or index.get('qpos_input_protocol') != QPOS_PROTOCOL or index.get('qpos_scope') != QPOS_SCOPE):
        raise ValueError('Expected isolated V13.4 three-group, experiment-only qpos-masked input policy')
    if index['training_data_hash'] != sha256_json({k: v for k, v in index.items() if k != 'training_data_hash'}):
        raise ValueError('V13.4 index identity changed')
    for hashes in (index['generated_hashes'], index['source_hashes'] if verify_sources else {}):
        for file, digest in hashes.items():
            if sha256_file(Path(file)) != digest:
                raise ValueError(f'V13.4 artifact changed: {file}')
    original, originals = source.validate_index(index['source_index_file'], verify_sources=False, require_images=require_images)
    if index['splits'] != original['splits'] or index['source_training_data_hash'] != original['training_data_hash']:
        raise ValueError('V13.4 must preserve the frozen V13.2/V13.3 split and source identity')
    rows = source.prior.load_jsonl(index['events_file'])
    validate_rows(rows, originals)
    return index, rows
