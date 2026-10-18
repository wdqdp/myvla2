"""Paired, episode-disjoint plan-only ablations; never rewrite source labels."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

import h5py
import numpy as np

from tactile_vla.vla.artifacts import sha256_file, sha256_json

ROOT = Path('/data1/qxh/tac_vla_new/tac_data/demon_data')
DATASET = ROOT / 'black_box_ablation'
LEROBOT = DATASET / 'lerobot_data/tactile_vla_black_box_ablation'
OUTPUT = DATASET / 'outputs/ablation_v13_1'
INDEX = OUTPUT / 'index.json'
STAGE_A = ROOT / 'black_box/outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_2_no_history/10000'
NORM = ROOT / 'black_box/outputs/rotation_v4_1_new_env/norm_stats'
SCHEMA = 'blackbox_plan_ablation_v13_1'
GROUPS = ('A', 'B', 'C', 'D')
SPLITS = ('train', 'val', 'test')


def read_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rotation(caption):
    values = re.findall(r'(?<!\w)rotation=(none|clockwise|counterclockwise)(?=;|\])', caption)
    if len(values) != 1:
        raise ValueError('Expected one rotation in a full Touch caption')
    return values[0]


def failure_direction(text):
    match = re.fullmatch(r'failure_reason=rotate (left|right),grasp appropriate\.', text)
    if match is None:
        raise ValueError(f'Unsupported failure: {text}')
    return match[1]


def plan_fields(text):
    match = re.fullmatch(
        r'recovery_plan=move horizontally (left|right) (moderately|slightly), move vertically none moderately\.', text
    )
    if match is None:
        raise ValueError(f'Unsupported plan: {text}')
    return match.groups()


def stratum(episode):
    if not 1 <= episode <= 40:
        raise ValueError('V13.1 requires original episodes 1..40')
    return (episode - 1) // 10 + 1


def make_splits(seed=42, heldout_excluded=()):
    result = {s: [] for s in SPLITS}
    for group in range(1, 5):
        values = list(range(10 * group - 9, 10 * group + 1))
        random.Random(seed + group).shuffle(values)
        eligible = [e for e in values if e not in heldout_excluded]
        if len(eligible) < 2:
            raise ValueError('Not enough calibrated held-out episodes per stratum')
        val, test = eligible[-2:]
        for split, selected in zip(SPLITS, ([e for e in values if e not in (val, test)], [val], [test]), strict=True):
            result[split].extend(selected)
    return {k: sorted(v) for k, v in result.items()}


def opposite(value):
    return {'clockwise': 'counterclockwise', 'counterclockwise': 'clockwise'}[value]


def choose(candidates, *key):
    if not candidates:
        raise ValueError(f'No same-split, same-orientation real Touch donor for {key}')
    number = int(hashlib.sha256(json.dumps(key).encode()).hexdigest(), 16)
    return candidates[number % len(candidates)]


def build_prompt(group, instruction, current_touch, events):
    if group not in GROUPS or not events:
        raise ValueError('Invalid ablation group or empty event chain')
    parts = ['Mode: reasoning.', f'Task: {instruction.strip()}']
    if group != 'A':
        parts.append(current_touch)
    history = []
    if group in ('C', 'D'):
        for event in events:
            plan = event['recovery_plan']
            if not plan.startswith('recovery_plan='):
                plan = 'recovery_plan=' + plan
            feedback = event['tactile_caption'] if group == 'C' else event['failure_reason']
            history.append(f'{plan}; {feedback}')
    parts.append('Failure-recovery memory: ' + (' | '.join(history) if history else 'none'))
    return '\n'.join(parts)


def _text(h, key):
    value = h[key][()]
    return value.decode() if isinstance(value, bytes) else str(value)


def camera_path(value, hdf5_path, raw_image_dir=None):
    """Resolve path-backed images; None means JPEG/PNG bytes embedded in HDF5."""
    if isinstance(value, (bytes, str)):
        if isinstance(value, bytes) and value.startswith((b'\xff\xd8', b'\x89PNG')):
            return None
        text = value.decode() if isinstance(value, bytes) else value
        path = Path(text)
        if raw_image_dir is not None:
            parts = path.parts
            if 'raw_data' not in parts:
                raise ValueError('Cannot remap image reference without raw_data component')
            return (Path(raw_image_dir) / Path(*parts[parts.index('raw_data') + 1:])).resolve()
        return (Path(hdf5_path).parent / path).resolve()
    return None


def scan_lerobot_images(root):
    """Map ORIGINAL episode/attempt/frame identities, never assume episode_index=episode_id."""
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    info_path = root / 'meta/info.json'
    info = read_json(info_path)
    hashes = {str(info_path): sha256_file(info_path)}
    for name in ('episodes.jsonl', 'episodes_stats.jsonl', 'tasks.jsonl'):
        path = root / 'meta' / name
        if path.exists():
            hashes[str(path)] = sha256_file(path)
    if info['fps'] != 30 or any(info['features'][f'observation.images.{c}']['dtype'] != 'video' for c in ('front', 'left')):
        raise ValueError('V13.1 requires 30Hz LeRobot front/left video features')
    columns = ['index', 'episode_index', 'episode_id', 'attempt_id', 'frame_index', 'timestamp',
               'ros_timestamp', 'observation.state', 'tactile_caption', 'instruction']
    frames, indices, videos, missing = {}, set(), {}, []
    for path in sorted((root / 'data').rglob('*.parquet')):
        hashes[str(path)] = sha256_file(path)
        for item in pq.read_table(path, columns=columns).to_pylist():
            key = tuple(int(item[k]) for k in ('episode_id', 'attempt_id', 'frame_index'))
            if key in frames or item['index'] in indices:
                raise ValueError('Duplicate LeRobot original identity or global index')
            indices.add(item['index'])
            ep_index = int(item['episode_index'])
            paths = {}
            for camera in ('front', 'left'):
                video_key = f'observation.images.{camera}'
                video = root / info['video_path'].format(episode_chunk=ep_index // info['chunks_size'],
                                                       episode_index=ep_index, video_key=video_key)
                paths[camera] = str(video)
                videos[str(video)] = ep_index
            frames[key] = item | {'video_paths': paths}
    if len(frames) != info['total_frames'] or indices != set(range(len(frames))):
        raise ValueError('Incomplete LeRobot parquet/global index coverage')
    for video in sorted(videos):
        if Path(video).is_file():
            hashes[video] = sha256_file(Path(video))
        else:
            missing.append({'path': video})
    return frames, hashes, missing, len(videos)


def build_data(*, dataset_dir=DATASET, source_dir=None, output_dir=OUTPUT, norm_stats_dir=NORM, seed=42,
               raw_image_dir=None, lerobot_dir=None):
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    source_dir = Path(source_dir or dataset_dir / 'outputs/ablation_three_task').resolve()
    norm_file = Path(norm_stats_dir).resolve() / 'norm_stats.json'
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite: {output_dir}')
    profile = read_json(source_dir / 'profile.json')['attempts']
    profiles = {int(v['episode_id']): v for v in profile}
    if len(profile) != 40 or set(profiles) != set(range(1, 41)):
        raise ValueError('Expected exactly 40 source episodes')
    source_hashes = {str(source_dir / 'profile.json'): sha256_file(source_dir / 'profile.json'),
                     str(norm_file): sha256_file(norm_file)}
    # Explicit LeRobot source takes precedence over dangling raw JPEG references.
    automatic = dataset_dir / 'lerobot_data/tactile_vla_black_box_ablation'
    lerobot_dir = Path(lerobot_dir).resolve() if lerobot_dir is not None else automatic if automatic.exists() else None
    lerobot_frames, video_count = None, 0
    if lerobot_dir is not None:
        lerobot_frames, hashes, video_missing, video_count = scan_lerobot_images(lerobot_dir)
        source_hashes.update(hashes)
    old_rows = []
    for s in SPLITS:
        path = source_dir / f'plan_{s}.jsonl'
        source_hashes[str(path)] = sha256_file(path)
        old_rows.extend(load_jsonl(path))
    identities = [(int(r['episode_id']), int(r['frame_index'])) for r in old_rows]
    if len(old_rows) != 1200 or len(set(identities)) != 1200:
        raise ValueError('Expected 30 unique plan observations per episode')
    # Calibrate physical mapping using TRAIN episodes only, separately per collection stratum.
    episodes, audit, votes = {}, [], defaultdict(Counter)
    missing_images = list(video_missing) if lerobot_dir is not None else []
    for ep, p in sorted(profiles.items()):
        path = dataset_dir / 'hdf5' / p['hdf5_path']
        source_hashes[str(path)] = sha256_file(path)
        with h5py.File(path) as h:
            caps = list(h['label/tactile_caption'].asstr()[:])
            qpos = np.asarray(h['arm/jointStatePosition/puppetRight'][:], dtype=np.float32)
            direction = _text(h, 'meta/rotation_direction')
            start = int(p['failure_window_start'])
            if direction != p['rotation_direction'] or len(caps) != p['frame_count']:
                raise ValueError(f'Profile mismatch episode {ep}')
            if qpos.shape != (len(caps), 7) or not np.isfinite(qpos).all():
                raise ValueError('Invalid qpos')
            if not 0 <= start <= len(caps) - 30:
                raise ValueError('Invalid 30-frame failure window')
            instruction = _text(h, 'meta/instruction')
            if lerobot_frames is not None:
                for frame in range(len(caps)):
                    lr = lerobot_frames.get((ep, 1, frame))
                    if lr is None or lr['tactile_caption'] != caps[frame] or lr['instruction'] != instruction:
                        raise ValueError(f'LeRobot/HDF5 frame or label mismatch at {ep}/1/{frame}')
                    if not np.array_equal(np.asarray(lr['observation.state'], np.float32), qpos[frame]):
                        raise ValueError('LeRobot/HDF5 qpos mismatch')
                    if abs(lr['ros_timestamp'] - float(h['timestamp'][frame])) > 1e-6:
                        raise ValueError('LeRobot/HDF5 timestamp mismatch')
            image_sources = {}
            for camera in ('front', 'left'):
                if len(h[f'camera/color/{camera}']) != len(caps):
                    raise ValueError('Camera/frame mismatch')
                image_sources[camera] = {}
                for frame in range(start, start + 30):
                    if lerobot_frames is not None:
                        image_sources[camera][frame] = lerobot_frames[ep, 1, frame]['video_paths'][camera]
                        continue
                    image = camera_path(h[f'camera/color/{camera}'][frame], path, raw_image_dir)
                    image_sources[camera][frame] = str(image) if image is not None else None
                    if image is not None:
                        if image.is_file():
                            source_hashes[str(image)] = sha256_file(image)
                        else:
                            missing_images.append({'episode_id': ep, 'frame_index': frame,
                                                   'camera': camera, 'path': str(image)})
        rotations = [rotation(c) for c in caps[start:start + 30]]
        counts = Counter(r for r in rotations if r != 'none')
        if not counts:
            raise ValueError(f'No rotation in episode {ep}')
        dominant = counts.most_common(1)[0][0]
        episodes[ep] = dict(captions=caps, qpos=qpos, direction=direction, start=start,
                            instruction=instruction, path=str(path), dominant=dominant, image_sources=image_sources)
        label_path = path.parent / 'labels.json'
        if label_path.exists():
            source_hashes[str(label_path)] = sha256_file(label_path)
            if read_json(label_path)['tactile_caption'] != caps:
                raise ValueError('JSON and HDF5 Touch labels differ')
    # A held-out stratum has only ONE episode: it must contain a usable donor
    # for its own failure direction. Reserve majority-inconsistent episodes for
    # train (not dropped or relabelled), before any model evaluation exists.
    collection_left = {1: 'counterclockwise', 2: 'clockwise', 3: 'counterclockwise', 4: 'clockwise'}
    heldout_excluded = [ep for ep, info in episodes.items() if info['dominant'] != (
        collection_left[stratum(ep)] if info['direction'] == 'left' else opposite(collection_left[stratum(ep)]))]
    splits = make_splits(seed, heldout_excluded)
    split_of = {ep: s for s, eps in splits.items() for ep in eps}
    for ep in splits['train']:
        info = episodes[ep]
        votes[stratum(ep)][info['dominant'] if info['direction'] == 'left' else opposite(info['dominant'])] += 1
    mapping = {}
    for g in range(1, 5):
        ranking = votes[g].most_common()
        if not ranking or (len(ranking) > 1 and ranking[0][1] == ranking[1][1]):
            raise ValueError(f'Ambiguous physical mapping for stratum {g}')
        left = ranking[0][0]
        if left != collection_left[g]:
            raise ValueError('Train-calibrated mapping contradicts the collection orientation; inspect data')
        mapping[g] = {'left': left, 'right': opposite(left)}
    # Full real captions, not fabricated rotation-only replacements. Donors must also
    # have the matching global failure metadata. Inconsistent captions are audited.
    donors = defaultdict(list)
    for ep, info in episodes.items():
        m = mapping[stratum(ep)]
        expected = m[info['direction']]
        for frame in range(info['start'], info['start'] + 30):
            caption = info['captions'][frame]
            if rotation(caption) != expected:
                audit.append({'episode_id': ep, 'frame_index': frame, 'split': split_of[ep],
                              'observed': rotation(caption), 'expected': expected,
                              'policy': 'keep_current_unchanged_exclude_from_historical_donors'})
                continue
            donors[split_of[ep], m['left'], info['direction']].append({
                'episode_id': ep, 'frame_index': frame, 'tactile_caption': caption,
                'split': split_of[ep], 'source_type': 'real_caption_template',
                'hdf5_path': info['path'],
            })
    rows = []
    for old in sorted(old_rows, key=lambda r: (int(r['episode_id']), int(r['frame_index']))):
        ep, frame = int(old['episode_id']), int(old['frame_index'])
        info, split = episodes[ep], split_of[ep]
        if int(old['attempt_id']) != 1 or not info['start'] <= frame < info['start'] + 30:
            raise ValueError('Source observation identity mismatch')
        if old['current_observation']['tactile_caption'] != info['captions'][frame]:
            raise ValueError('Source current caption mismatch')
        target = str(old['target_recovery_plan'])
        direction, magnitude = plan_fields(target)
        memory = copy.deepcopy(old['failure_recovery_memory'])
        if len(memory) != old['memory_length'] or not 1 <= len(memory) <= 5:
            raise ValueError('Bad source memory length')
        if direction != info['direction'] or failure_direction(memory[-1]['failure_reason']) != direction:
            raise ValueError('Terminal failure/target mismatch')
        if magnitude != ('moderately' if len(memory) == 1 else 'slightly'):
            raise ValueError('Source magnitude rule changed')
        events = []
        for i, pair in enumerate(memory):
            failure = pair['failure_reason']
            d = failure_direction(failure)
            if i == 0:
                if pair['recovery_plan'] != 'initial plan':
                    raise ValueError('Missing initial placeholder')
            elif plan_fields(pair['recovery_plan']) != (
                failure_direction(memory[i - 1]['failure_reason']), 'moderately' if i == 1 else 'slightly'
            ):
                raise ValueError('Inconsistent source history chain')
            if i == len(memory) - 1:
                donor = {'episode_id': ep, 'frame_index': frame, 'split': split,
                         'tactile_caption': info['captions'][frame], 'hdf5_path': info['path'],
                         'source_type': 'real_current_observation'}
            else:
                donor = choose(donors[split, mapping[stratum(ep)]['left'], d], seed, ep, frame, i)
            events.append({'recovery_plan': pair['recovery_plan'], 'failure_reason': failure,
                           'tactile_caption': donor['tactile_caption'], 'touch_source': donor,
                           'source_type': 'synthetic_event_not_executed', 'pair_index': i})
        row = {'sample_id': len(rows), 'episode_id': ep, 'attempt_id': 1, 'frame_index': frame,
               'split': split, 'stratum': stratum(ep), 'hdf5_path': info['path'],
               'instruction': info['instruction'], 'current_touch': info['captions'][frame],
               'qpos': info['qpos'][frame].tolist(), 'target_recovery_plan': target,
               'horizontal_direction': direction, 'horizontal_magnitude': magnitude,
               'memory_length': len(events), 'events': events,
               'image_sources': {c: info['image_sources'][c][frame] for c in ('front', 'left')},
               'source_variant_id': old['variant_id'], 'original_split': old['split']}
        if lerobot_frames is not None:
            lr = lerobot_frames[ep, 1, frame]
            row.update({'lerobot_dataset_dir': str(lerobot_dir), 'lerobot_global_index': int(lr['index']),
                        'lerobot_episode_index': int(lr['episode_index']), 'lerobot_timestamp': lr['timestamp']})
        row['prompts'] = {g: build_prompt(g, info['instruction'], row['current_touch'], events) for g in GROUPS}
        rows.append(row)
    for ep in episodes:
        selected = [r for r in rows if r['episode_id'] == ep]
        if len(selected) != 30 or Counter(r['horizontal_magnitude'] for r in selected) != {
            'moderately': 15, 'slightly': 15
        }:
            raise ValueError('Per-episode magnitude balance changed')
    output_dir.mkdir(parents=True)
    (output_dir / 'norm_stats').mkdir()
    shutil.copyfile(norm_file, output_dir / 'norm_stats/norm_stats.json')
    (output_dir / 'plan_events.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_json(output_dir / 'splits.json', splits)
    write_json(output_dir / 'caption_audit.json', audit)
    write_json(output_dir / 'image_audit.json', {'ready': not missing_images,
        'required_observations': 1200, 'required_camera_frames': 2400,
        'backend': 'lerobot_video' if lerobot_dir is not None else 'hdf5_camera_references',
        'video_files': video_count, 'missing_count': len(missing_images), 'missing': missing_images})
    summary = {'schema_version': SCHEMA, 'mapping_calibration': 'train_episodes_only_majority_per_stratum',
               'heldout_excluded_episodes_kept_in_train': heldout_excluded,
               'heldout_eligibility': 'majority_caption_agrees_with_declared_collection_orientation_for_donor_coverage',
               'mapping_by_stratum': mapping, 'mapping_votes': {g: dict(v) for g, v in votes.items()},
               'caption_audit_count': len(audit), 'caption_audit_episode_counts': dict(Counter(a['episode_id'] for a in audit)),
               'history_is_synthetic': True, 'magnitude_rule': 'length1_moderately_length2to5_slightly',
               'normalization': 'reuse_v7_4_2_stage_a_fixed_stats_not_fit_on_ablation_val_test', 'splits': {}}
    summary['images_ready'] = not missing_images
    summary['image_backend'] = 'lerobot_video' if lerobot_dir is not None else 'hdf5_camera_references'
    summary['missing_image_files'] = len(missing_images)
    summary['lerobot_frames_aligned'] = len(lerobot_frames) if lerobot_frames is not None else 0
    summary['video_files'] = video_count
    for s in SPLITS:
        selected = [r for r in rows if r['split'] == s]
        summary['splits'][s] = {'episodes': splits[s], 'samples': len(selected),
            'classes': dict(Counter(f"{r['horizontal_direction']}/{r['horizontal_magnitude']}" for r in selected)),
            'memory_lengths': dict(Counter(r['memory_length'] for r in selected))}
    write_json(output_dir / 'summary.json', summary)
    index = {'schema_version': SCHEMA, 'seed': seed, 'groups': list(GROUPS), 'splits': splits,
             'source_hashes': source_hashes, 'dataset_dir': str(dataset_dir),
             'lerobot_dataset_dir': str(lerobot_dir) if lerobot_dir is not None else None,
             'events_file': str(output_dir / 'plan_events.jsonl'),
             'norm_stats_dir': str(output_dir / 'norm_stats'),
             'generated_hashes': {str(output_dir / p): sha256_file(output_dir / p) for p in (
                 'plan_events.jsonl', 'norm_stats/norm_stats.json', 'splits.json', 'caption_audit.json',
                 'image_audit.json', 'summary.json')}}
    index['training_data_hash'] = sha256_json(index)
    write_json(output_dir / 'index.json', index)
    return index, summary


def validate_index(path, *, verify_sources=True, require_images=False):
    index = read_json(path)
    if index.get('schema_version') != SCHEMA or index.get('groups') != list(GROUPS):
        raise ValueError('Not a V13.1 plan ablation index')
    if index['training_data_hash'] != sha256_json({k: v for k, v in index.items() if k != 'training_data_hash'}):
        raise ValueError('Index identity changed')
    sets = [set(index['splits'][s]) for s in SPLITS]
    if set.union(*sets) != set(range(1, 41)) or sum(map(len, sets)) != 40:
        raise ValueError('Episode split overlap or missing episodes')
    for split in SPLITS:
        expected = 8 if split == 'train' else 1
        if Counter(stratum(e) for e in index['splits'][split]) != dict.fromkeys(range(1, 5), expected):
            raise ValueError('Unbalanced physical strata')
    for hashes in (index['generated_hashes'], index['source_hashes'] if verify_sources else {}):
        for source, digest in hashes.items():
            if sha256_file(Path(source)) != digest:
                raise ValueError(f'Artifact changed: {source}')
    image_audit = read_json(Path(index['events_file']).parent / 'image_audit.json')
    if require_images and not image_audit['ready']:
        first = image_audit['missing'][0]['path']
        raise FileNotFoundError(f"{image_audit['missing_count']} camera frames missing when index was built; "
                                f'first: {first}. Supply LeRobot videos or raw images and rebuild in a new output directory.')
    rows = load_jsonl(index['events_file'])
    for r in rows:
        if r['episode_id'] not in index['splits'][r['split']]:
            raise ValueError('Event split mismatch')
        for e in r['events']:
            if e['touch_source']['episode_id'] not in index['splits'][r['split']]:
                raise ValueError('Touch donor crossed split')
        if r['prompts'] != {g: build_prompt(g, r['instruction'], r['current_touch'], r['events']) for g in GROUPS}:
            raise ValueError('Prompt policy changed or hidden input added')
    return index, rows


def metrics(predictions):
    def score(values):
        n = len(values)
        return {'support': n, **{k: sum(bool(r[k]) for r in values) / n if n else None
            for k in ('direction_correct', 'magnitude_correct', 'joint_correct', 'exact_match')}}
    result = score(predictions)
    for name in ('episode_id', 'stratum', 'memory_length', 'horizontal_direction', 'horizontal_magnitude'):
        groups = defaultdict(list)
        for r in predictions:
            groups[str(r[name])].append(r)
        result['by_' + name] = {k: score(v) for k, v in sorted(groups.items())}
    result['episode_macro_joint_correct'] = float(np.mean([
        v['joint_correct'] for v in result['by_episode_id'].values()
    ])) if predictions else None
    return result


def prediction_record(row, text):
    try:
        direction, magnitude = plan_fields(text)
    except ValueError:
        direction, magnitude = 'invalid', 'invalid'
    dc = direction == row['horizontal_direction']
    mc = magnitude == row['horizontal_magnitude']
    return {k: row[k] for k in ('sample_id', 'episode_id', 'frame_index', 'stratum', 'memory_length',
                               'horizontal_direction', 'horizontal_magnitude', 'target_recovery_plan')} | {
        'prediction': text, 'direction_correct': dc, 'magnitude_correct': mc,
        'joint_correct': dc and mc, 'exact_match': text == row['target_recovery_plan']}
