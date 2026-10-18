"""Global-reversal latch and independently oriented synthetic recovery events."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import shutil

from tactile_vla.vla import blackbox_ablation_v13_1_data as prior
from tactile_vla.vla.artifacts import sha256_file, sha256_json

SCHEMA = 'blackbox_plan_ablation_v13_2'
OUTPUT = prior.DATASET / 'outputs/ablation_v13_2'
INDEX = OUTPUT / 'index.json'
TRAIN_OUTPUT = prior.DATASET / 'outputs/plan_v13_2'
GROUPS = prior.GROUPS
ORIENTATIONS = ('normal', 'reversed')
TOUCH_ROTATION = {
    'normal': {'left': 'counterclockwise', 'right': 'clockwise'},
    'reversed': {'left': 'clockwise', 'right': 'counterclockwise'},
}
KINDS = ('initial', 'persistent', 'reversal')


def rng_for(seed, *key):
    return random.Random(int(sha256_json([seed, *key]), 16))


def other(direction):
    return {'left': 'right', 'right': 'left'}[direction]


def first_reversal(directions):
    if not directions or any(d not in ('left', 'right') for d in directions):
        raise ValueError('A nonempty left/right GLOBAL failure sequence is required')
    return next((i for i in range(1, len(directions)) if directions[i] != directions[i - 1]), None)


def magnitude(directions):
    return 'slightly' if first_reversal(directions) is not None else 'moderately'


def plan(direction, degree):
    return f'recovery_plan=move horizontally {direction} {degree}, move vertically none moderately.'


def failure(direction):
    return f'failure_reason=rotate {direction},grasp appropriate.'


def orientation(source_row, mappings):
    mapping = mappings[str(source_row['stratum'])]
    for name, expected in TOUCH_ROTATION.items():
        if mapping == expected:
            return name
    raise ValueError('Unknown collection orientation mapping')


def make_sequences(current, length, rng, *, stage=None):
    """Paired histories of the SAME length and terminal failure, different latch."""
    if current not in ('left', 'right') or not 2 <= length <= 5:
        raise ValueError('Multiround history length must be 2..5')
    persistent = [current] * length
    stage = stage or ('first_reversal_now' if length == 2 or rng.randrange(2) == 0 else 'after_reversal')
    if stage == 'first_reversal_now':
        reversed_ = [other(current)] * (length - 1) + [current]
    elif stage == 'after_reversal' and length >= 3:
        prefix = [rng.choice(('left', 'right')) for _ in range(length - 1)]
        if len(set(prefix)) == 1:
            prefix[rng.randrange(1, len(prefix))] = other(prefix[0])
        reversed_ = [*prefix, current]
    else:
        raise ValueError('A prior reversal requires at least three events')
    return persistent, reversed_


def build_events(source, directions, orientations, pools, seed, kind):
    if len(directions) != len(orientations):
        raise ValueError('One grasp orientation is required for every feedback event')
    events = []
    for i, (direction, grip) in enumerate(zip(directions, orientations, strict=True)):
        # The plan executed BEFORE observing F_i is based ONLY on F_0..F_{i-1}.
        executed = 'initial plan' if i == 0 else plan(directions[i - 1], magnitude(directions[:i]))
        if i == len(directions) - 1:
            donor = {'episode_id': source['episode_id'], 'frame_index': source['frame_index'],
                     'hdf5_path': source['hdf5_path'], 'split': source['split'],
                     'tactile_caption': source['current_touch'], 'grip_orientation': grip,
                     'global_direction': direction, 'source_type': 'real_current_observation'}
        else:
            candidates = pools[source['split'], grip, direction]
            donor = copy.deepcopy(prior.choose(candidates, seed, source['episode_id'], source['frame_index'], kind, i))
        events.append({'pair_index': i, 'recovery_plan': executed, 'failure_reason': failure(direction),
                       'global_direction': direction, 'grip_orientation': grip,
                       'tactile_caption': donor['tactile_caption'], 'touch_source': donor,
                       'has_global_reversal': first_reversal(directions[:i + 1]) is not None,
                       'source_type': 'synthetic_event_not_executed'})
    return events


def make_variants(source, *, current_orientation, pools, seed):
    key = (source['episode_id'], source['frame_index'])
    length = rng_for(seed, *key, 'length').randint(2, 5)
    persistent, reversed_ = make_sequences(source['horizontal_direction'], length, rng_for(seed, *key, 'global'))
    # Independent fair coin per historical instant. Use the SAME orientation
    # schedule for the paired histories to avoid orientation/length label cues.
    rng = rng_for(seed, *key, 'orientations')
    grips = [rng.choice(ORIENTATIONS) for _ in range(length - 1)] + [current_orientation]
    schedules = {'initial': ([source['horizontal_direction']], [current_orientation]),
                 'persistent': (persistent, grips), 'reversal': (reversed_, grips)}
    keep = ('episode_id', 'attempt_id', 'frame_index', 'split', 'stratum', 'hdf5_path', 'instruction',
            'current_touch', 'qpos', 'image_sources', 'lerobot_dataset_dir', 'lerobot_global_index',
            'lerobot_episode_index', 'lerobot_timestamp', 'horizontal_direction')
    result = []
    for kind, (directions, orientations) in schedules.items():
        row = {k: copy.deepcopy(source[k]) for k in keep if k in source}
        flip = first_reversal(directions)
        stage = ('initial' if kind == 'initial' else 'never_reversed' if flip is None else
                 'first_reversal_now' if flip == len(directions) - 1 else 'after_reversal')
        events = build_events(source, directions, orientations, pools, seed, kind)
        row.update({'base_observation_id': source['sample_id'], 'sample_kind': kind,
                    'pair_id': f'e{key[0]}-a1-f{key[1]}', 'memory_length': len(events),
                    'global_directions': directions, 'history_grip_orientations': orientations,
                    'current_grip_orientation': current_orientation,
                    'history_grip_changed': len(set(orientations)) > 1,
                    'first_global_reversal_index': flip, 'has_global_reversal': flip is not None,
                    'returned_to_initial_after_reversal': flip is not None and directions[0] in directions[flip + 1:],
                    'reversal_stage': stage, 'horizontal_magnitude': magnitude(directions),
                    'target_recovery_plan': plan(directions[-1], magnitude(directions)), 'events': events})
        row['prompts'] = {g: prior.build_prompt(g, row['instruction'], row['current_touch'], events) for g in GROUPS}
        result.append(row)
    return result


def validate_rows(rows, source_rows, source_index, mappings):
    source_lookup = {r['sample_id']: r for r in source_rows}
    if len(source_lookup) != len(source_rows) or len(rows) != 3 * len(source_rows):
        raise ValueError('Expected exactly three variants per source observation')
    if [r['sample_id'] for r in rows] != list(range(len(rows))):
        raise ValueError('Duplicate or non-contiguous sample IDs')
    by_source = defaultdict(list)
    template_lookup = {(r['episode_id'], r['frame_index']): r for r in source_rows}
    for row in rows:
        source = source_lookup[row['base_observation_id']]
        by_source[source['sample_id']].append(row)
        for k in ('episode_id', 'attempt_id', 'frame_index', 'split', 'stratum', 'hdf5_path', 'instruction',
                  'current_touch', 'qpos', 'image_sources', 'lerobot_dataset_dir', 'lerobot_global_index',
                  'lerobot_episode_index', 'lerobot_timestamp', 'horizontal_direction'):
            if row.get(k) != source.get(k):
                raise ValueError(f'Current observation changed: {k}')
        if row['episode_id'] not in source_index['splits'][row['split']]:
            raise ValueError('Episode crossed split')
        dirs, grips = row['global_directions'], row['history_grip_orientations']
        n = row['memory_length']
        if len(dirs) != n or len(grips) != n or len(row['events']) != n or not 1 <= n <= 5:
            raise ValueError('Invalid event/grasp history length')
        current_orientation = orientation(source, mappings)
        if grips[-1] != current_orientation or row['current_grip_orientation'] != current_orientation:
            raise ValueError('Real current grasp was randomized')
        if dirs[-1] != source['horizontal_direction']:
            raise ValueError('Current global failure changed')
        flip = first_reversal(dirs)
        expected_stage = ('initial' if n == 1 else 'never_reversed' if flip is None else
                          'first_reversal_now' if flip == n - 1 else 'after_reversal')
        if row['first_global_reversal_index'] != flip or row['has_global_reversal'] != (flip is not None):
            raise ValueError('Global reversal latch mismatch')
        if row['reversal_stage'] != expected_stage or row['horizontal_magnitude'] != magnitude(dirs):
            raise ValueError('Magnitude/stage differs from GLOBAL reversal rule')
        if row['target_recovery_plan'] != plan(dirs[-1], magnitude(dirs)):
            raise ValueError('Target plan differs from global history')
        if row['history_grip_changed'] != (len(set(grips)) > 1):
            raise ValueError('Grasp change audit mismatch')
        if row['returned_to_initial_after_reversal'] != (flip is not None and dirs[0] in dirs[flip + 1:]):
            raise ValueError('Return-after-reversal audit mismatch')
        for i, event in enumerate(row['events']):
            grip, d = grips[i], dirs[i]
            if event['pair_index'] != i or event['source_type'] != 'synthetic_event_not_executed':
                raise ValueError('Event order/source annotation mismatch')
            expected_plan = 'initial plan' if i == 0 else plan(dirs[i - 1], magnitude(dirs[:i]))
            if event['recovery_plan'] != expected_plan:
                raise ValueError('Historical plan used future feedback or wrong prefix')
            if event['failure_reason'] != failure(d) or event['global_direction'] != d or event['grip_orientation'] != grip:
                raise ValueError('Event semantic mismatch')
            if event['has_global_reversal'] != (first_reversal(dirs[:i + 1]) is not None):
                raise ValueError('Event latch mismatch')
            donor = event['touch_source']
            if donor['episode_id'] not in source_index['splits'][row['split']] or donor['split'] != row['split']:
                raise ValueError('Touch donor crossed split')
            template = template_lookup[donor['episode_id'], donor['frame_index']]
            if event['tactile_caption'] != template['current_touch'] or donor['tactile_caption'] != event['tactile_caption']:
                raise ValueError('Touch is not a complete unchanged real caption')
            if orientation(template, mappings) != grip or donor['grip_orientation'] != grip:
                raise ValueError('Touch from wrong grasp orientation')
            if donor['global_direction'] != d or template['horizontal_direction'] != d:
                raise ValueError('Touch from wrong global failure')
            if i == n - 1:
                if (donor['episode_id'], donor['frame_index']) != (source['episode_id'], source['frame_index']):
                    raise ValueError('Terminal feedback not current observation')
            elif prior.rotation(event['tactile_caption']) != TOUCH_ROTATION[grip][d]:
                raise ValueError('Local rotation does not match event grasp and global failure')
        if row['prompts'] != {g: prior.build_prompt(g, row['instruction'], row['current_touch'], row['events']) for g in GROUPS}:
            raise ValueError('Prompt policy mismatch')
    for variants in by_source.values():
        if Counter(r['sample_kind'] for r in variants) != dict.fromkeys(KINDS, 1):
            raise ValueError('Missing or duplicate paired variant')
        pair = {r['sample_kind']: r for r in variants}
        if pair['initial']['memory_length'] != 1 or pair['persistent']['memory_length'] < 2:
            raise ValueError('Initial/multiround length mismatch')
        if pair['persistent']['has_global_reversal'] or not pair['reversal']['has_global_reversal']:
            raise ValueError('Paired histories do not differ in reversal condition')
        for k in ('memory_length', 'history_grip_orientations', 'pair_id'):
            if pair['persistent'][k] != pair['reversal'][k]:
                raise ValueError('Paired histories changed length/orientation schedule')


def build_data(*, source_index=prior.INDEX, output_dir=OUTPUT, seed=42):
    source_index, output_dir = Path(source_index).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing overwrite: {output_dir}')
    source, observations = prior.validate_index(source_index, require_images=True)
    source_summary = prior.read_json(Path(source['events_file']).parent / 'summary.json')
    mappings = {str(k): v for k, v in source_summary['mapping_by_stratum'].items()}
    pools = defaultdict(list)
    for row in observations:
        grip = orientation(row, mappings)
        d = row['horizontal_direction']
        if prior.rotation(row['current_touch']) != TOUCH_ROTATION[grip][d]:
            continue
        pools[row['split'], grip, d].append({'episode_id': row['episode_id'], 'frame_index': row['frame_index'],
            'hdf5_path': row['hdf5_path'], 'split': row['split'], 'global_direction': d,
            'grip_orientation': grip, 'tactile_caption': row['current_touch'], 'source_type': 'real_caption_template'})
    for split in prior.SPLITS:
        for grip in ORIENTATIONS:
            for d in ('left', 'right'):
                if not pools[split, grip, d]:
                    raise ValueError(f'No same-split real Touch templates: {split}/{grip}/{d}')
    rows = []
    for obs in observations:
        for row in make_variants(obs, current_orientation=orientation(obs, mappings), pools=pools, seed=seed):
            row['sample_id'] = len(rows)
            rows.append(row)
    validate_rows(rows, observations, source, mappings)
    summary = {'schema_version': SCHEMA, 'seed': seed, 'source_training_data_hash': source['training_data_hash'],
               'mapping_by_stratum': mappings, 'history_is_synthetic': True,
               'magnitude_rule': 'first_opposite_GLOBAL_rotation_latches_slightly_forever',
               'grasp_rule': 'independent_historical_coins_paired_schedule_shared_real_current_fixed',
               'image_backend': source_summary['image_backend'], 'images_ready': True, 'splits': {}}
    for split in prior.SPLITS:
        selected = [r for r in rows if r['split'] == split]
        summary['splits'][split] = {'episodes': source['splits'][split], 'samples': len(selected),
            'real_observations': len(selected) // 3,
            'kinds': dict(Counter(r['sample_kind'] for r in selected)),
            'reversal_stages': dict(Counter(r['reversal_stage'] for r in selected)),
            'classes': dict(Counter(f"{r['horizontal_direction']}/{r['horizontal_magnitude']}" for r in selected)),
            'length_magnitude': dict(Counter(f"{r['memory_length']}/{r['horizontal_magnitude']}" for r in selected)),
            'multiround_grasp_changes': dict(Counter(str(r['history_grip_changed']) for r in selected if r['sample_kind'] != 'initial')),
            'historical_grasp_counts': dict(Counter(g for r in selected for g in r['history_grip_orientations'][:-1])),
            'returned_after_reversal': sum(r['returned_to_initial_after_reversal'] for r in selected)}
    output_dir.mkdir(parents=True)
    (output_dir / 'norm_stats').mkdir()
    shutil.copyfile(Path(source['norm_stats_dir']) / 'norm_stats.json', output_dir / 'norm_stats/norm_stats.json')
    for name in ('splits.json', 'caption_audit.json', 'image_audit.json'):
        shutil.copyfile(Path(source['events_file']).parent / name, output_dir / name)
    (output_dir / 'plan_events.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    prior.write_json(output_dir / 'summary.json', summary)
    source_hashes = {**source['source_hashes'], **source['generated_hashes'], str(source_index): sha256_file(source_index)}
    index = {'schema_version': SCHEMA, 'seed': seed, 'groups': list(GROUPS), 'splits': source['splits'],
             'source_index_file': str(source_index), 'source_training_data_hash': source['training_data_hash'],
             'source_hashes': source_hashes, 'dataset_dir': source['dataset_dir'],
             'lerobot_dataset_dir': source['lerobot_dataset_dir'], 'mapping_by_stratum': mappings,
             'events_file': str(output_dir / 'plan_events.jsonl'), 'norm_stats_dir': str(output_dir / 'norm_stats'),
             'generated_hashes': {str(output_dir / p): sha256_file(output_dir / p) for p in (
                 'plan_events.jsonl', 'norm_stats/norm_stats.json', 'splits.json', 'caption_audit.json',
                 'image_audit.json', 'summary.json')}}
    index['training_data_hash'] = sha256_json(index)
    prior.write_json(output_dir / 'index.json', index)
    return index, summary


def validate_index(path, *, verify_sources=True, require_images=False):
    index = prior.read_json(path)
    if index.get('schema_version') != SCHEMA or index.get('groups') != list(GROUPS):
        raise ValueError('Not a V13.2 plan ablation index')
    if index['training_data_hash'] != sha256_json({k: v for k, v in index.items() if k != 'training_data_hash'}):
        raise ValueError('V13.2 index identity changed')
    for hashes in (index['generated_hashes'], index['source_hashes'] if verify_sources else {}):
        for file, digest in hashes.items():
            if sha256_file(Path(file)) != digest:
                raise ValueError(f'Artifact changed: {file}')
    source, observations = prior.validate_index(index['source_index_file'], verify_sources=False, require_images=require_images)
    if index['splits'] != source['splits'] or index['source_training_data_hash'] != source['training_data_hash']:
        raise ValueError('V13.2 must reuse the exact V13.1 observation split')
    rows = prior.load_jsonl(index['events_file'])
    validate_rows(rows, observations, source, index['mapping_by_stratum'])
    return index, rows


def prediction_record(row, text):
    return prior.prediction_record(row, text) | {k: row[k] for k in (
        'base_observation_id', 'sample_kind', 'reversal_stage', 'has_global_reversal',
        'first_global_reversal_index', 'history_grip_changed', 'returned_to_initial_after_reversal',
        'current_grip_orientation', 'global_directions', 'history_grip_orientations')}


def metrics(predictions):
    result = prior.metrics(predictions)
    selectors = {
        'initial': lambda r: r['sample_kind'] == 'initial',
        'paired_multiround': lambda r: r['sample_kind'] != 'initial',
        'persistent': lambda r: r['sample_kind'] == 'persistent',
        'first_reversal_now': lambda r: r['reversal_stage'] == 'first_reversal_now',
        'after_reversal': lambda r: r['reversal_stage'] == 'after_reversal',
        'returned_after_reversal': lambda r: r['returned_to_initial_after_reversal'],
        'multiround_grasp_changed': lambda r: r['sample_kind'] != 'initial' and r['history_grip_changed'],
        'multiround_grasp_unchanged': lambda r: r['sample_kind'] != 'initial' and not r['history_grip_changed'],
    }
    result['subsets'] = {name: prior.metrics([r for r in predictions if select(r)]) for name, select in selectors.items()}
    pairs = defaultdict(list)
    for row in predictions:
        if row['sample_kind'] != 'initial':
            pairs[row['base_observation_id']].append(row)
    if any(len(v) != 2 or {r['sample_kind'] for r in v} != {'persistent', 'reversal'} for v in pairs.values()):
        raise ValueError('Evaluation omitted a member of a paired history')
    result['paired_both_correct'] = {'support_pairs': len(pairs),
        'accuracy': sum(all(r['joint_correct'] for r in pair) for pair in pairs.values()) / len(pairs) if pairs else None}
    return result
