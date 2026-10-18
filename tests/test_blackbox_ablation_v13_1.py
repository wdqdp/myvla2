"""Plan-only input isolation, grouped splits, donor provenance and checkpoints."""
# ruff: noqa: E402
import copy
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('USE_TF', '0')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'openpi/src')]

import h5py
import numpy as np
from PIL import Image
import pytest

from tactile_vla.vla import blackbox_ablation_v13_1_data as data
from tactile_vla.vla.blackbox_ablation_v13_1_runtime import PlanDataset, StepBatchSampler


def touch(r):
    return f'Touch[area=medium; Fx=negative; Fy=positive; Fz=negative; Fz_bias=balanced; rotation={r}]'


def failure(d):
    return f'failure_reason=rotate {d},grasp appropriate.'


def plan(d, m):
    return f'recovery_plan=move horizontally {d} {m}, move vertically none moderately.'


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('v13_1')
    source = root / 'outputs/ablation_three_task'
    source.mkdir(parents=True)
    norm = root / 'norm'
    norm.mkdir()
    data.write_json(norm / 'norm_stats.json', {'norm_stats': {'state': {
        'mean': [0.] * 7, 'std': [1.] * 7, 'q01': [-2.] * 7, 'q99': [2.] * 7}}})
    profile, rows = [], []
    old_splits = data.make_splits(17)
    split_of = {ep: s for s, eps in old_splits.items() for ep in eps}
    jpeg = io.BytesIO()
    Image.fromarray(np.full((16, 24, 3), 130, np.uint8)).save(jpeg, format='JPEG')
    image_bytes = np.frombuffer(jpeg.getvalue(), np.uint8)
    for ep in range(1, 41):
        direction = 'left' if ep <= 20 else 'right'
        left = 'counterclockwise' if data.stratum(ep) in (1, 3) else 'clockwise'
        expected = left if direction == 'left' else data.opposite(left)
        caps = [touch('none')] * 4 + [touch(expected)] * 30
        if ep == 37:
            caps[4:24] = [touch(data.opposite(expected))] * 20
        qpos = np.repeat(np.arange(34, dtype=np.float32)[:, None] / 34, 7, axis=1)
        path = root / f'hdf5/episode{ep}/attempt1/data.hdf5'
        path.parent.mkdir(parents=True)
        with h5py.File(path, 'w') as h:
            h['meta/rotation_direction'] = direction
            h['meta/instruction'] = 'Pick up and transfer the object stably.'
            h.create_dataset('label/tactile_caption', data=caps, dtype=h5py.string_dtype())
            h['arm/jointStatePosition/puppetRight'] = qpos
            for camera in ('front', 'left'):
                ds = h.create_dataset(f'camera/color/{camera}', (34,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
                for i in range(34):
                    ds[i] = image_bytes
        data.write_json(path.parent / 'labels.json', {'tactile_caption': caps})
        profile.append({'episode_id': ep, 'hdf5_path': f'episode{ep}/attempt1/data.hdf5',
                        'rotation_direction': direction, 'frame_count': 34, 'failure_window_start': 4})
        for i in range(30):
            memory = [{'recovery_plan': 'initial plan', 'failure_reason': failure(direction)}]
            if i % 2:
                previous = 'right' if direction == 'left' else 'left'
                memory = [{'recovery_plan': 'initial plan', 'failure_reason': failure(previous)},
                          {'recovery_plan': plan(previous, 'moderately'), 'failure_reason': failure(direction)}]
            rows.append({'episode_id': ep, 'attempt_id': 1, 'frame_index': i + 4, 'split': split_of[ep],
                         'current_observation': {'tactile_caption': caps[i + 4]}, 'memory_length': len(memory),
                         'failure_recovery_memory': memory, 'variant_id': f'original-{ep}-{i}',
                         'target_recovery_plan': plan(direction, 'slightly' if i % 2 else 'moderately')})
    data.write_json(source / 'profile.json', {'attempts': profile})
    for split in data.SPLITS:
        (source / f'plan_{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows if r['split'] == split))
    out = root / 'built'
    index, summary = data.build_data(dataset_dir=root, output_dir=out, norm_stats_dir=norm)
    return root, out, index, summary


def test_split_is_grouped_balanced_deterministic():
    splits = data.make_splits()
    assert splits == data.make_splits()
    assert splits != data.make_splits(43)
    assert len(set.union(*(set(v) for v in splits.values()))) == 40
    for name, eps in splits.items():
        assert len(eps) == (32 if name == 'train' else 4)
        assert all(sum(data.stratum(ep) == g for ep in eps) == (8 if name == 'train' else 1) for g in range(1, 5))
    assert 37 in data.make_splits(42, [37])['train']


@pytest.mark.parametrize('group', data.GROUPS)
def test_prompt_isolates_available_information(group):
    events = [{'recovery_plan': 'initial plan', 'failure_reason': failure('right'), 'tactile_caption': touch('clockwise')},
              {'recovery_plan': plan('right', 'moderately'), 'failure_reason': failure('left'),
               'tactile_caption': touch('counterclockwise')}]
    prompt = data.build_prompt(group, 'transfer', touch('counterclockwise'), events)
    assert ('Touch[' in prompt) == (group != 'A')
    assert ('failure_reason=' in prompt) == (group == 'D')
    assert ('recovery_plan=initial plan' in prompt) == (group in ('C', 'D'))
    assert (plan('right', 'moderately') in prompt) == (group in ('C', 'D'))
    assert plan('left', 'slightly') not in prompt  # Target is never an input.
    assert all(word not in prompt for word in ('sample_id', 'frame_index', 'episode_id', 'variant_id'))


def test_build_preserves_current_labels_targets_and_donor_splits(fixture):
    root, out, index, summary = fixture
    checked, rows = data.validate_index(out / 'index.json')
    assert checked == index and len(rows) == 1200
    assert summary['heldout_excluded_episodes_kept_in_train'] == [37]
    assert 37 in index['splits']['train']
    assert summary['mapping_by_stratum'][1]['left'] == 'counterclockwise'
    assert summary['mapping_by_stratum'][2]['left'] == 'clockwise'
    assert summary['caption_audit_episode_counts'] == {37: 20}
    for row in rows:
        assert row['episode_id'] in index['splits'][row['split']]
        assert len(row['events']) == row['memory_length']
        for event in row['events']:
            donor = event['touch_source']
            assert donor['episode_id'] in index['splits'][row['split']]
            assert donor['split'] == row['split']
            with h5py.File(donor['hdf5_path']) as h:
                assert h['label/tactile_caption'].asstr()[donor['frame_index']] == event['tactile_caption']
        assert row['events'][-1]['tactile_caption'] == row['current_touch']
        assert row['events'][-1]['touch_source']['source_type'] == 'real_current_observation'
        assert row['horizontal_magnitude'] == ('moderately' if row['memory_length'] == 1 else 'slightly')
    noisy = next(r for r in rows if r['episode_id'] == 37 and r['frame_index'] == 4)
    assert data.rotation(noisy['current_touch']) == 'clockwise'  # No silent correction.
    with pytest.raises(FileExistsError):
        data.build_data(dataset_dir=root, output_dir=out, norm_stats_dir=root / 'norm')


def test_hash_check_detects_changed_generated_artifact(fixture):
    _, out, index, _ = fixture
    corrupted = copy.deepcopy(index)
    corrupted['seed'] = 123
    path = out / 'bad_index.json'
    data.write_json(path, corrupted)
    with pytest.raises(ValueError, match='identity'):
        data.validate_index(path)


def test_sampler_replays_exactly_after_resume_and_across_groups():
    all_batches = list(StepBatchSampler(30, 8, 42, 15))
    assert all_batches == list(StepBatchSampler(30, 8, 42, 15))
    assert all_batches[7:] == list(StepBatchSampler(30, 8, 42, 15, start_step=7))
    assert all(len(set(batch)) == 8 for batch in all_batches)
    with pytest.raises(ValueError):
        StepBatchSampler(7, 8, 42, 10)


def test_hdf5_dataset_inference_has_no_answer_and_keeps_both_images(fixture):
    _, out, _, _ = fixture
    _, rows = data.validate_index(out / 'index.json', verify_sources=False)
    train = PlanDataset(rows, group='A', transform=lambda x: x)
    infer = PlanDataset(rows, group='D', transform=lambda x: x, inference=True)
    a, d = train[0], infer[0]
    assert a['observation/image'].shape == d['observation/wrist_image'].shape == (16, 24, 3)
    assert a['observation/state'].shape == (7,)
    assert 'Touch[' not in a['prompt'] and 'failure_reason=' in d['prompt']
    assert 'target_text' in a and 'target_text' not in d


def test_camera_paths_and_missing_image_guard(fixture, tmp_path):
    h = tmp_path / 'hdf5/episode1/attempt1/data.hdf5'
    ref = b'../../../raw_data/episode1/attempt1/camera/color/front/1.jpg'
    assert data.camera_path(ref, h) == tmp_path / 'raw_data/episode1/attempt1/camera/color/front/1.jpg'
    assert data.camera_path(ref, h, tmp_path / 'elsewhere') == tmp_path / 'elsewhere/episode1/attempt1/camera/color/front/1.jpg'
    assert data.camera_path(b'\xff\xd8anything', h) is None
    _, out, index, _ = fixture
    audit = out / 'image_audit.json'
    original = audit.read_bytes()
    try:
        data.write_json(audit, {'ready': False, 'missing_count': 1, 'missing': [{'path': '/absent/image.jpg'}]})
        changed = copy.deepcopy(index)
        from tactile_vla.vla.artifacts import sha256_file, sha256_json
        changed['generated_hashes'][str(audit)] = sha256_file(audit)
        changed['training_data_hash'] = sha256_json({k: v for k, v in changed.items() if k != 'training_data_hash'})
        p = tmp_path / 'index.json'
        data.write_json(p, changed)
        with pytest.raises(FileNotFoundError, match='camera frames missing'):
            data.validate_index(p, require_images=True)
    finally:
        audit.write_bytes(original)


def test_three_metrics_are_independent():
    row = {'sample_id': 0, 'episode_id': 1, 'frame_index': 8, 'stratum': 1, 'memory_length': 2,
           'horizontal_direction': 'left', 'horizontal_magnitude': 'slightly',
           'target_recovery_plan': plan('left', 'slightly')}
    values = [data.prediction_record(row, plan('left', 'moderately')),
              data.prediction_record(row | {'episode_id': 2}, plan('right', 'slightly'))]
    report = data.metrics(values)
    assert report['direction_correct'] == report['magnitude_correct'] == 0.5
    assert report['joint_correct'] == report['exact_match'] == report['episode_macro_joint_correct'] == 0


def test_cli_does_not_pin_training_steps_or_batch():
    from scripts.train_blackbox_ablation_v13_1 import parse_args
    args = parse_args(['--group', 'B', '--num-steps', '777', '--batch-size', '16', '--fsdp-devices', '2'])
    assert args.num_steps == 777 and args.batch_size == 16 and args.fsdp_devices == 2


def test_resume_rejects_changed_group_base_or_sampling():
    from scripts.train_blackbox_ablation_v13_1 import resume_check
    original = {'group': 'A', 'stage_a_checkpoint': '/base', 'seed': 42, 'batch_size': 8}
    resume_check(original, original)
    for key, value in (('group', 'D'), ('stage_a_checkpoint', '/other'), ('batch_size', 16)):
        with pytest.raises(ValueError, match=key):
            resume_check(original, original | {key: value})


def test_model_has_no_classifier_and_filter_only_updates_shared_lora(tmp_path):
    from flax import nnx
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from tactile_vla.vla.blackbox_ablation_v13_1_runtime import model_class, trainable_filter, restore_overlay
    class Backbone(nnx.Module):
        def __init__(self):
            self.llm = nnx.Dict(lora=nnx.Param(jnp.ones(2)), lora_1=nnx.Param(jnp.ones(3)),
                                base=nnx.Param(jnp.ones(4)))
    model = model_class()(Backbone())
    assert not hasattr(model, 'need_head') and not hasattr(model, 'adjustment_head')
    selected = nnx.state(model).filter(trainable_filter())
    assert list(selected['backbone']['llm']) == ['lora']
    # Exercise the exact typed Orbax overlay used by best-checkpoint evaluation.
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(tmp_path / 'delta', {'params': selected})
    restored = restore_overlay(nnx.state(model), tmp_path / 'delta')
    assert np.array_equal(restored.to_pure_dict()['backbone']['llm']['lora'], np.ones(2))


def test_source_stage_a_validation_is_architecture_only(tmp_path):
    from scripts.train_blackbox_ablation_v13_1 import validate_stage_a
    p = tmp_path / '1234/params'
    p.mkdir(parents=True)
    (p / '_METADATA').touch()
    data.write_json(tmp_path / 'config.json', {'paligemma_variant': 'gemma_2b_lora',
        'action_expert_variant': 'gemma_300m_lora', 'action_dim': 32, 'action_horizon': 30,
        'use_state_history': False})
    validate_stage_a(p.parent)  # No required numeric checkpoint step.


def test_actual_openpi_train_and_answer_free_inference_transforms(fixture):
    from scripts.train_blackbox_ablation_v13_1 import parse_args, components
    from tactile_vla.vla.blackbox_ablation_v13_1_runtime import make_loader, validate_tokens
    _, out, _, _ = fixture
    args = parse_args(['--group', 'A', '--index-file', str(out / 'index.json')])
    args.precision = 'float32'
    _, rows, config, tokenizer, grammar, norm = components(args)
    audit = validate_tokens(rows, tokenizer=tokenizer, grammar=grammar, norm=norm, max_len=384)
    assert all(v['count'] == 1200 and v['max_tokens'] <= 384 for v in audit.values())
    for group in data.GROUPS:
        common = dict(group=group, model_config=config, tokenizer=tokenizer, grammar=grammar, norm=norm, batch_size=2)
        batch = next(iter(make_loader(rows[:2], **common)))
        infer = next(iter(make_loader(rows[:2], **common, inference=True)))
        assert batch['tokenized_prompt'].shape == infer['tokenized_prompt'].shape == (2, 384)
        assert batch['token_loss_mask'].any() and not infer['token_loss_mask'].any()
        assert 'structured_target_compact_ids' not in infer
        assert batch['image']['base_0_rgb'].shape == (2, 224, 224, 3)


def test_lerobot_reader_checks_original_identity_and_never_passes_target_to_inference(fixture, monkeypatch):
    from tactile_vla.vla import blackbox_ablation_v13_1_runtime as runtime
    _, out, _, _ = fixture
    row = copy.deepcopy(data.load_jsonl(out / 'plan_events.jsonl')[0])
    row.update(lerobot_dataset_dir='/fake/lerobot', lerobot_global_index=145,
               lerobot_episode_index=7, lerobot_timestamp=4.0)
    # Intentionally unrelated episode_index and original episode_id.
    item = {'index': 145, 'episode_index': 7, 'episode_id': row['episode_id'], 'attempt_id': 1,
            'frame_index': row['frame_index'], 'tactile_caption': row['current_touch'],
            'observation.state': np.asarray(row['qpos'], np.float32),
            'observation.images.front': np.zeros((3, 24, 32), np.float32),
            'observation.images.left': np.zeros((3, 16, 24), np.float32)}
    monkeypatch.setattr(runtime, 'lerobot_dataset', lambda *args: {145: item})
    ds = runtime.PlanDataset([row], group='C', transform=lambda x: x, inference=True)
    output = ds[0]
    assert output['observation/image'].shape == (3, 24, 32)
    assert 'target_text' not in output
    item['episode_index'] = 1
    with pytest.raises(ValueError, match='identity'):
        ds[0]


def test_tiny_plan_only_initialization_and_optimizer_update(monkeypatch):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.training import sharding
    from scripts import train_vla_stage_b_v3 as base
    from tactile_vla.vla import stage_b_v3_jax as language
    from tactile_vla.vla.blackbox_ablation_v13_1_runtime import init_state
    class Backbone(nnx.Module):
        def __init__(self):
            self.llm = nnx.Dict(lora=nnx.Param(jnp.zeros(2)), lora_1=nnx.Param(jnp.ones(3)),
                                base=nnx.Param(jnp.ones(4)))
    args = SimpleNamespace(lr=1e-4, weight_decay=1e-4, grad_clip=1.0, seed=42,
                           stage_a_checkpoint=Path('/authorized/10000'))
    def load(loader, reference):
        assert 'authorized/10000/params' in str(loader)
        return jax.tree.map(lambda p: jnp.zeros(p.shape, p.dtype), reference)
    monkeypatch.setattr(base, 'load_weights_and_validate', load)
    mesh = sharding.make_mesh(1)
    state, _, filter_ = init_state(args, SimpleNamespace(create=lambda rng: Backbone()), mesh)
    assert int(state.step) == 0 and set(state.params) == {'backbone'}
    before = state.params.to_pure_dict()
    def logits(backbone, observation, ids, **kwargs):
        del observation, ids, kwargs
        return jnp.broadcast_to(backbone.llm.lora.value, (1, 2, 2))
    monkeypatch.setattr(language, 'structured_token_logits', logits)
    targets = {'target_ids': jnp.array([[0, 1]]), 'allowed': jnp.ones((1, 2, 2), bool)}
    # Unequal targets ensure a nonzero gradient at zero logits.
    targets['target_ids'] = jnp.array([[1, 1]])
    updated, info = base.train_text_step(filter_, 1., jnp.array([10, 11]), jax.random.key(4), state, (None, targets))
    after = updated.params.to_pure_dict()
    assert int(updated.step) == 1 and float(info['grad_norm']) > 0
    assert not np.array_equal(before['backbone']['llm']['lora'], after['backbone']['llm']['lora'])
    for key in ('lora_1', 'base'):
        assert np.array_equal(before['backbone']['llm'][key], after['backbone']['llm'][key])
