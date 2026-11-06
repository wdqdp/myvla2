"""Contract checks for the classification-only V7.11 deployment."""

# ruff: noqa: E402
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'openpi/src'),
                str(ROOT / 'openpi/inference/agilex/inference')]

import agilex_inference_v7_11_asyn as client
from scripts import serve_tactile_vla_v7_11 as server
from tactile_vla.captioner.predictor import CaptionerPrediction
from tactile_vla.common.labels import LABEL_FIELDS as FIVE_FIELDS
from tactile_vla.common.labels_v4 import LABEL_FIELDS as SIX_FIELDS
from tactile_vla.vla.v7_11_data import DATA_PROFILE, PROMPT_PROFILE
from tactile_vla.vla.v7_11_stage_a import checkpoint_prompt_identity


def make_v4_prediction():
    names = {'area': 'medium', 'fx_state': 'positive', 'fy_state': 'negative',
             'fz_state': 'negative', 'fz_bias': 'left', 'rotation': 'clockwise'}
    return CaptionerPrediction(label_ids={field: 0 for field in names}, label_names=names,
                               caption='unused original V4 caption',
                               probabilities={field: [0.1, 0.9] for field in names})


def test_v4_caption_is_converted_to_original_five_fields():
    original = make_v4_prediction()
    predictor = SimpleNamespace(schema_version='tactile_multifield_v4_masked', label_fields=SIX_FIELDS,
                                window_size=30, predict=lambda mesh, force: original,
                                predict_batch=lambda mesh, force: [original])
    adapter = client.FiveFieldCaptioner(predictor)
    result = adapter.predict(None, None)
    assert result.caption == 'Touch[area=medium; Fx=positive; Fy=negative; Fz=negative; rotation=clockwise]'
    assert tuple(result.label_names) == FIVE_FIELDS
    assert tuple(result.label_ids) == FIVE_FIELDS and tuple(result.probabilities) == FIVE_FIELDS
    assert adapter.predict_batch(None, None)[0] == result
    assert 'fz_bias' in original.label_names
    assert adapter.schema_version == 'tactile_multifield_v3' and adapter.window_size == 30


def test_caption_adapter_rejects_incompatible_schema():
    predictor = SimpleNamespace(schema_version='tactile_multifield_v5_masked', label_fields=FIVE_FIELDS, window_size=30)
    with pytest.raises(ValueError, match='Unsupported V7.11 captioner'):
        client.FiveFieldCaptioner(predictor)


def test_legacy_v3_captioner_remains_compatible():
    original = make_v4_prediction()
    names = {field: original.label_names[field] for field in FIVE_FIELDS}
    prediction = CaptionerPrediction(label_ids={field: 0 for field in FIVE_FIELDS}, label_names=names,
                                    caption='old caption', probabilities={field: [1.0] for field in FIVE_FIELDS})
    predictor = SimpleNamespace(schema_version='tactile_multifield_v3', label_fields=FIVE_FIELDS,
                                window_size=30, predict=lambda mesh, force: prediction)
    result = client.FiveFieldCaptioner(predictor).predict(None, None)
    assert result.caption == 'Touch[area=medium; Fx=positive; Fy=negative; Fz=negative; rotation=clockwise]'
    assert result.label_names == names


def test_converted_caption_reaches_phase_and_adjustment(monkeypatch):
    prediction = make_v4_prediction()
    predictor = SimpleNamespace(schema_version='tactile_multifield_v4_masked', label_fields=SIX_FIELDS,
                                window_size=30, predict=lambda mesh, force: prediction)
    adapter = client.FiveFieldCaptioner(predictor)
    def capture(args, operator, captioner, *, after_timestamp):
        obs = SimpleNamespace(timestamp=11.0, qpos=np.zeros(7), tactile_caption=captioner.predict(None, None).caption,
                              img_front=None, img_left=None)
        return obs, {'qpos': 11.0}
    monkeypatch.setattr(client.v53, '_capture_classification_observation', capture)
    monkeypatch.setattr(client, 'discretize_state_qpos', lambda values, stats: np.zeros((11, 7), dtype=int))
    monkeypatch.setattr(client.runtime, 'build_payload', lambda **kwargs: dict(kwargs))
    monkeypatch.setattr(client.v52, 'deterministic_action_noise', lambda *a, **k: np.zeros((30, 32)))
    args = SimpleNamespace(classification_gripper_open_threshold=None, classification_gripper_open_value=None,
                           instruction='Pick up object', noise_seed=42, request_timeout_seconds=1)
    operator = SimpleNamespace(state_history=SimpleNamespace(
        snapshot=lambda **kwargs: (np.zeros((100, 7)), np.ones(100, dtype=bool))))
    captured = []
    class Phase:
        def events(self, payload):
            captured.append(payload['prompt'])
            yield {'event': 'phase_decision', 'request_id': 'r1', 'phase': 'execution', 'need_recovery': False}
    class Action:
        def infer_with_timeout(self, payload, timeout):
            captured.append(payload['prompt'])
            return {'actions': np.zeros((30, 7)), 'raw_model_actions': np.zeros((30, 32))}
    gate = client.V77AsyncControlGate(hold=lambda: None)
    gate.begin_phase_request('r1')
    result = client._run_phase_assessment(args=args, client=Phase(), operator=operator, captioner=adapter,
                                          stats=None, gate=gate, phase='execution', generation=0, attempt_id=1,
                                          request_id='r1', captured_step=1, after_timestamp=10,
                                          recovery_plan='', submitted_monotonic=0)
    client._request_action_chunk(args=args, policy=Action(), observation=result.observation,
                                 phase='adjustment', phase_index=0, recovery_plan='',
                                 logger=SimpleNamespace(record=lambda event: None))
    for prompt in captured:
        assert 'Touch[area=medium; Fx=positive; Fy=negative; Fz=negative; rotation=clockwise]' in prompt
        assert 'Fz_bias' not in prompt


def test_streamed_need_true_does_not_wait_for_failure():
    decision = {'event': 'phase_decision', 'request_id': 'r1', 'phase': 'execution', 'need_recovery': True}

    class Socket:
        def __init__(self):
            self.receives = 0
        def send(self, data):
            pass
        def recv(self, timeout):
            self.receives += 1
            if self.receives != 1:
                raise AssertionError('V7.11 attempted to read an untrained failure event')
            return client.msgpack_numpy.Packer().pack(decision)

    phase_client = client.StreamingPhaseClient.__new__(client.StreamingPhaseClient)
    phase_client._ws, phase_client._timeout = Socket(), 1
    phase_client._packer, phase_client._lock = client.msgpack_numpy.Packer(), threading.Lock()
    assert list(phase_client.events({'request_id': 'r1'})) == [decision]
    assert phase_client._ws.receives == 1


def test_phase_prompt_and_immediate_gate(monkeypatch):
    observation = SimpleNamespace(timestamp=11.0, qpos=np.arange(7, dtype=np.float32),
                                  tactile_caption='Touch[rotation=clockwise]', img_front=np.zeros((2, 2, 3)),
                                  img_left=np.zeros((2, 2, 3)))
    monkeypatch.setattr(client, 'capture_tactile_observation', lambda *a, **k: (observation, {'qpos': 11.0}))
    monkeypatch.setattr(client, 'discretize_state_qpos', lambda values, stats: np.zeros((11, 7), dtype=int))
    monkeypatch.setattr(client.runtime, 'build_payload', lambda **kwargs: dict(kwargs))
    history = SimpleNamespace(snapshot=lambda **kwargs: (np.zeros((100, 7)), np.ones(100, dtype=bool)))
    operator = SimpleNamespace(state_history=history)
    holds = []
    gate = client.V77AsyncControlGate(hold=lambda: holds.append(True))
    gate.begin_phase_request('r1')

    class PhaseClient:
        def events(self, payload):
            assert 'Recovery plan:' not in payload['prompt']
            assert 'Touch: Touch[rotation=clockwise]' in payload['prompt']
            assert 'State history H100 sampled to 11 points:' in payload['prompt']
            yield {'event': 'phase_decision', 'request_id': 'r1', 'phase': 'execution', 'need_recovery': True}

    args = SimpleNamespace(classification_gripper_open_threshold=None, classification_gripper_open_value=None,
                           instruction='Pick up object')
    result = client._run_phase_assessment(args=args, client=PhaseClient(), operator=operator, captioner=object(),
                                          stats=None, gate=gate, phase='execution', generation=0, attempt_id=1,
                                          request_id='r1', captured_step=1, after_timestamp=10,
                                          recovery_plan='', submitted_monotonic=0)
    assert result.triggered and holds and gate.snapshot()[-1]
    assert not gate.may_publish_action(0)
    gate.switch_phase('adjustment', increment_attempt=True)
    assert gate.snapshot()[:2] == ('adjustment', 2)
    assert gate.snapshot()[-1]


def test_adjustment_actions_use_each_fresh_caption(monkeypatch):
    monkeypatch.setattr(client.runtime, 'build_payload', lambda **kwargs: dict(kwargs))
    monkeypatch.setattr(client.v52, 'deterministic_action_noise', lambda *a, **k: np.zeros((30, 32)))
    args = SimpleNamespace(instruction='Pick up object', noise_seed=42, request_timeout_seconds=1)
    prompts = []

    class Policy:
        def infer_with_timeout(self, payload, timeout):
            prompts.append(payload['prompt'])
            return {'actions': np.zeros((30, 7)), 'raw_model_actions': np.zeros((30, 32))}

    logger = SimpleNamespace(record=lambda event: None)
    for caption in ('Touch[area=small]', 'Touch[area=large]'):
        observation = SimpleNamespace(tactile_caption=caption, img_front=None, img_left=None,
                                      qpos=np.zeros(7), timestamp=1.0)
        client._request_action_chunk(args=args, policy=Policy(), observation=observation,
                                     phase='adjustment', phase_index=0, recovery_plan='', logger=logger)
    assert prompts == ['Mode: adjustment. Task: Pick up object.\nTouch: Touch[area=small]',
                       'Mode: adjustment. Task: Pick up object.\nTouch: Touch[area=large]']


def test_rng_key_restores_from_abstract_shape():
    class Model(nnx.Module):
        def __init__(self):
            self.dropout = nnx.Dropout(0.1, rngs=nnx.Rngs(jax.random.key(0)))

    graph, state = nnx.split(nnx.eval_shape(Model))
    reference = state.to_pure_dict()
    raw = jax.tree.map(lambda leaf: jnp.zeros((2,), dtype=jnp.uint32)
                       if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
                       else jnp.zeros(leaf.shape, dtype=leaf.dtype), reference)
    state.replace_by_pure_dict(server.restore_rng_keys(reference, raw))
    assert isinstance(nnx.merge(graph, state), Model)


def test_server_classifies_without_generating_failure(monkeypatch):
    policy = server.V711Policy.__new__(server.V711Policy)
    policy._assessment_transform = lambda request: request
    policy._need_logits = lambda observation: jnp.asarray([[0.0, 10.0]])
    policy._adjustment_logits = lambda observation: jnp.asarray([[0.0, 10.0]])
    policy._need_threshold, policy._adjustment_threshold = 0.5, 0.9
    monkeypatch.setattr(server.v3.TactileVLAPolicyV3, '_observation', staticmethod(lambda request: request))
    request = {'mode': 'phase', 'phase': 'execution', 'request_id': 'r1',
               'prompt': 'Mode: phase\nTouch: Touch[area=small]\nState history H100 sampled to 11 points: []'}
    events = list(policy.infer_events(request))
    assert len(events) == 1 and events[0]['need_recovery'] is True
    request['phase'] = 'adjustment'
    assert list(policy.infer_events(request))[0]['adjustment_end'] is True
    request['prompt'] += '\nRecovery plan: none'
    with pytest.raises(ValueError, match='without a plan'):
        list(policy.infer_events(request))


def test_client_main_does_not_write_recordings(monkeypatch, tmp_path):
    norm = tmp_path / 'norm_stats.json'
    norm.write_text('{}')
    checkpoint = tmp_path / 'captioner.pt'
    checkpoint.write_bytes(b'captioner')
    monkeypatch.setattr(client.sys, 'argv', ['client', '--noise-seed', '42', '--gripper-min', '0.035',
                                           '--norm-stats-file', str(norm), '--captioner_checkpoint', str(checkpoint),
                                           '--start-immediately'])
    monkeypatch.setattr(client.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(client.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(client.runtime, 'load_captioner', lambda args: SimpleNamespace(
        schema_version='tactile_multifield_v4_masked', label_fields=SIX_FIELDS, window_size=30))
    monkeypatch.setattr(client.runtime, 'RosOperator', lambda args: object())
    monkeypatch.setattr(client.v53, '_install_tactile_arrival_timestamps', lambda op: None)
    monkeypatch.setattr(client.v53, 'TimeoutWebsocketPolicy', lambda host, port: object())
    monkeypatch.setattr(client, 'StreamingPhaseClient', lambda *a, **k: SimpleNamespace(close=lambda: None))

    class Keyboard:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None

    monkeypatch.setattr(client.runtime, 'KeyboardPoller', Keyboard)
    def run(args, operator, policy, phase, captioner, keyboard, logger):
        assert args.publish_rate == 20 and args.max_attempts == 10
        logger.record({'event': 'test', 'payload': 'not persisted'})
    monkeypatch.setattr(client, 'run_v7_7_async', run)
    monkeypatch.chdir(tmp_path)
    client.main()
    assert set(tmp_path.iterdir()) == {norm, checkpoint}


def test_artifact_validation_rejects_incomplete_export(tmp_path):
    run = tmp_path / 'run'
    step = run / '2001'
    full = step / 'full_params'
    full.mkdir(parents=True)
    (full / '_METADATA').write_text('{}')
    norm = tmp_path / 'norm'
    norm.mkdir()
    (norm / 'norm_stats.json').write_text('{}')
    norm_sha = hashlib.sha256((norm / 'norm_stats.json').read_bytes()).hexdigest()
    config = {'experiment_version': 'v7_11', 'data_profile': DATA_PROFILE, 'prompt_profile': PROMPT_PROFILE,
              'task_cycle': ['action', 'adjustment', 'need'], 'phase_prefill_protocol': 'classification_only',
              'recovery_action_condition': 'current_tactile_no_plan_no_recovery_history',
              'phase_condition': 'current_tactile_h100_no_plan', 'use_state_history': False,
              'state_history_len': 0, 'history_hidden_dim': 0, 'action_horizon': 30, 'action_dim': 32,
              'state_history_dim': 7, 'state_history_fps': 30.0,
              'failure_grammar': [], 'recovery_grammar': [], 'artifact_identity': {'v4_norm_stats_sha256': norm_sha},
              'stage_a_initialization_identity': {'prompt_identity': checkpoint_prompt_identity()}}
    (run / 'config.json').write_text(json.dumps(config))
    (step / 'v7_11_export.json').write_text(json.dumps({'step': 2001, 'default_deployment': 'full_params',
                                                        'exports': {'full_params': {'parameter_tree_sha256': 'a' * 64}}}))
    args = server.parse_args(['--checkpoint', str(step), '--norm-stats-dir', str(norm), '--validate-only'])
    _, identity = server.inspect_artifacts(args)
    assert identity['checkpoint_step'] == 2001
    assert args.need_recovery_threshold == 0.5 and args.adjustment_end_threshold == 0.9
    (full / 'chunk.incomplete').write_text('')
    with pytest.raises(FileNotFoundError, match='unfinished'):
        server.inspect_artifacts(args)
