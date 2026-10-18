"""V13.2 paired history semantics, orientation grounding, causality and isolation."""
# ruff: noqa: E402
import copy
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT), str(ROOT / 'src'), str(ROOT / 'openpi/src')]

from test_blackbox_ablation_v13_1 import fixture as v13_1_fixture  # noqa: F401
from tactile_vla.vla import blackbox_ablation_v13_2_data as data


@pytest.fixture(scope='module')
def built(v13_1_fixture, tmp_path_factory):  # noqa: F811
    _, old_dir, _, _ = v13_1_fixture
    out = tmp_path_factory.mktemp('v13_2') / 'data'
    index, summary = data.build_data(source_index=old_dir / 'index.json', output_dir=out)
    checked, rows = data.validate_index(out / 'index.json', require_images=True)
    assert checked == index
    source, observations = data.prior.validate_index(old_dir / 'index.json')
    return out, index, summary, rows, source, observations


@pytest.mark.parametrize('sequence', [p for length in range(1, 6) for p in product(('left', 'right'), repeat=length)])
def test_latch_for_every_binary_global_history_up_to_length_five(sequence):
    expected = next((i for i in range(1, len(sequence)) if sequence[i] != sequence[0]), None)
    assert data.first_reversal(sequence) == expected
    assert data.magnitude(sequence) == ('moderately' if expected is None else 'slightly')
    for tail in ('left', 'right'):
        if expected is not None:
            assert data.magnitude([*sequence, tail]) == 'slightly'


@pytest.mark.parametrize('invalid', [[], ['none'], ['left', 'clockwise']])
def test_global_labels_not_local_rotation_drive_the_rule(invalid):
    with pytest.raises(ValueError):
        data.first_reversal(invalid)


def test_same_length_has_both_labels_and_exact_three_variants(built):
    _, index, summary, rows, source, observations = built
    assert index['splits'] == source['splits']
    assert len(rows) == 3600
    assert {s: v['samples'] for s, v in summary['splits'].items()} == {'train': 2880, 'val': 360, 'test': 360}
    for split in data.prior.SPLITS:
        counts = Counter((r['memory_length'], r['horizontal_magnitude']) for r in rows if r['split'] == split)
        for length in range(2, 6):
            assert counts[length, 'moderately'] == counts[length, 'slightly'] > 0
    assert summary['splits']['test']['kinds'] == {'initial': 120, 'persistent': 120, 'reversal': 120}
    assert len({r['base_observation_id'] for r in rows}) == len(observations)


def test_historical_grasps_are_random_not_constant_and_not_global_reversals(built):
    _, _, _, rows, _, _ = built
    assert any(r['sample_kind'] == 'persistent' and r['history_grip_changed'] and
               r['horizontal_magnitude'] == 'moderately' for r in rows)
    # Global failure can flip while the LOCAL rotation text remains identical.
    assert any(r['has_global_reversal'] and len({data.prior.rotation(e['tactile_caption'])
               for e in r['events']}) == 1 for r in rows)
    # Same-global histories can show opposite local rotations solely due to grasp changes.
    assert any(r['sample_kind'] == 'persistent' and len({data.prior.rotation(e['tactile_caption'])
               for e in r['events']}) == 2 for r in rows)
    grips = Counter(g for r in rows for g in r['history_grip_orientations'][:-1])
    assert set(grips) == {'normal', 'reversed'} and abs(grips['normal'] / sum(grips.values()) - .5) < .1


def test_full_real_touch_templates_follow_each_event_grasp_and_split(built):
    _, index, _, rows, _, observations = built
    templates = {(r['episode_id'], r['frame_index']): r for r in observations}
    for row in rows:
        for event in row['events']:
            donor = event['touch_source']
            template = templates[donor['episode_id'], donor['frame_index']]
            assert template['split'] == row['split']
            assert donor['episode_id'] in index['splits'][row['split']]
            assert event['tactile_caption'] == template['current_touch']
            if donor['source_type'] != 'real_current_observation':
                assert data.prior.rotation(event['tactile_caption']) == data.TOUCH_ROTATION[
                    event['grip_orientation']][event['global_direction']]
        assert row['events'][-1]['tactile_caption'] == row['current_touch']


def test_historical_plan_uses_only_feedback_available_before_it(built):
    _, index, _, rows, source, observations = built
    for row in rows:
        dirs = row['global_directions']
        for i, event in enumerate(row['events']):
            expected = 'initial plan' if i == 0 else data.plan(dirs[i - 1], data.magnitude(dirs[:i]))
            assert event['recovery_plan'] == expected
    changed = copy.deepcopy(rows)
    row = next(r for r in changed if r['reversal_stage'] == 'first_reversal_now')
    i = row['first_global_reversal_index']
    # The action preceding the first reverse observation must still be moderately.
    assert data.prior.plan_fields(row['events'][i]['recovery_plan'])[1] == 'moderately'
    row['events'][i]['recovery_plan'] = data.plan(row['global_directions'][i - 1], 'slightly')
    with pytest.raises(ValueError, match='future'):
        data.validate_rows(changed, observations, source, index['mapping_by_stratum'])


def test_reversing_back_never_resets_slightly(built):
    _, _, _, rows, _, _ = built
    returned = [r for r in rows if r['returned_to_initial_after_reversal']]
    assert returned
    for row in returned:
        assert row['horizontal_magnitude'] == 'slightly'
        assert row['reversal_stage'] == 'after_reversal'


def test_paired_current_inputs_and_grasp_schedules_are_identical(built):
    _, _, _, rows, _, _ = built
    for start in range(0, len(rows), 3):
        initial, persistent, reversal = rows[start:start + 3]
        for group in ('A', 'B'):
            assert initial['prompts'][group] == persistent['prompts'][group] == reversal['prompts'][group]
        assert persistent['memory_length'] == reversal['memory_length']
        assert persistent['history_grip_orientations'] == reversal['history_grip_orientations']
        assert 'failure_reason=' not in persistent['prompts']['C']
        assert 'normal' not in persistent['prompts']['C'] and 'reversed' not in persistent['prompts']['C']
        assert persistent['horizontal_magnitude'] == 'moderately' and reversal['horizontal_magnitude'] == 'slightly'


@pytest.mark.parametrize('mutation,match', [('current', 'Current observation'), ('grasp', 'grasp'), ('donor', 'split')])
def test_validation_rejects_hidden_current_or_donor_changes(built, mutation, match):
    _, index, _, rows, source, observations = built
    changed = copy.deepcopy(rows)
    row = next(r for r in changed if r['sample_kind'] == 'persistent')
    if mutation == 'current':
        row['qpos'][0] += 1
    elif mutation == 'grasp':
        row['history_grip_orientations'][-1] = 'reversed' if row['current_grip_orientation'] == 'normal' else 'normal'
    else:
        row['events'][0]['touch_source']['split'] = 'other'
    with pytest.raises(ValueError, match=match):
        data.validate_rows(changed, observations, source, index['mapping_by_stratum'])


def test_eval_separates_initial_paired_first_flip_and_returned_subsets(built):
    _, _, _, rows, _, _ = built
    selected = [r for r in rows if r['split'] == 'test']
    perfect = [data.prediction_record(r, r['target_recovery_plan']) for r in selected]
    report = data.metrics(perfect)
    assert report['joint_correct'] == report['paired_both_correct']['accuracy'] == 1
    assert report['subsets']['initial']['support'] == 120
    assert report['subsets']['paired_multiround']['support'] == 240
    assert report['subsets']['first_reversal_now']['support'] > 0
    assert report['subsets']['returned_after_reversal']['support'] > 0
    baseline = [data.prediction_record(r, data.plan(r['horizontal_direction'], 'moderately')) for r in selected]
    report = data.metrics(baseline)
    assert report['joint_correct'] == pytest.approx(2 / 3)
    assert report['subsets']['initial']['joint_correct'] == 1
    assert report['subsets']['paired_multiround']['joint_correct'] == .5
    assert report['paired_both_correct']['accuracy'] == 0


def test_v13_1_and_v13_2_cli_and_validators_remain_isolated(built):
    from scripts import train_blackbox_ablation_v13_1 as old
    from scripts import train_blackbox_ablation_v13_2 as new
    before = old.parse_args(['--group', 'A'])
    args = new.parse_args(['--group', 'B', '--num-steps', '777', '--batch-size', '16'])
    assert args.run_name == 'pi05_blackbox_ablation_v13_2_B_s42'
    assert args.index_file == data.INDEX and args.output_dir == data.TRAIN_OUTPUT and args.num_steps == 777
    after = old.parse_args(['--group', 'A'])
    assert vars(before) == vars(after)
    out, _, _, _, _, _ = built
    with pytest.raises(ValueError, match='V13.1'):
        data.prior.validate_index(out / 'index.json')


def test_build_is_deterministic_and_refuses_overwrite(built):
    out, _, _, rows, _, observations = built
    source_index = data.prior.read_json(out / 'index.json')['source_index_file']
    with pytest.raises(FileExistsError):
        data.build_data(source_index=source_index, output_dir=out)
    source, _ = data.prior.validate_index(source_index, verify_sources=False)
    summary = data.prior.read_json(Path(source['events_file']).parent / 'summary.json')
    pools = defaultdict(list)
    for obs in observations:
        grip = data.orientation(obs, summary['mapping_by_stratum'])
        d = obs['horizontal_direction']
        if data.prior.rotation(obs['current_touch']) == data.TOUCH_ROTATION[grip][d]:
            pools[obs['split'], grip, d].append({'episode_id': obs['episode_id'], 'frame_index': obs['frame_index'],
                'split': obs['split'], 'hdf5_path': obs['hdf5_path'], 'tactile_caption': obs['current_touch'],
                'grip_orientation': grip, 'global_direction': d, 'source_type': 'real_caption_template'})
    fresh = data.make_variants(observations[0], current_orientation=data.orientation(observations[0], summary['mapping_by_stratum']),
                               pools=pools, seed=42)
    assert fresh == [{k: v for k, v in r.items() if k != 'sample_id'} for r in rows[:3]]


def test_pipeline_trains_all_new_groups_before_testing_without_mutating_old_defaults(tmp_path):
    from scripts import run_blackbox_ablation_v13_1 as old
    from scripts import run_blackbox_ablation_v13_2 as new
    args = old.parse_args(['--', '--output-dir', str(tmp_path), '--seed', '43'])
    called = []
    def launch(command, **kwargs):
        called.append(command)
        return SimpleNamespace(returncode=0)
    assert new.run_pipeline(args, launch) == 0
    assert [Path(c[1]).name for c in called] == ['train_blackbox_ablation_v13_2.py'] * 4 + ['eval_blackbox_ablation_v13_2.py'] * 4
    for group, command in zip('ABCD', called[4:], strict=True):
        assert str(tmp_path / f'pi05_blackbox_ablation_v13_2_{group}_s43') in command
    old_commands, _ = old.command_plan(old.parse_args([]))
    assert Path(old_commands[0][1]).name == 'train_blackbox_ablation_v13_1.py'
