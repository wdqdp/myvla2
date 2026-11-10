"""Exercise the Book no-Plan wire format and asynchronous stop behavior."""

# ruff: noqa: E402, SLF001
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'openpi/src'),
                str(ROOT / 'openpi/inference/agilex/inference')]

import agilex_inference_book_v9_7_asyn as client
from scripts import serve_tactile_vla_book_v9_7 as server
from tactile_vla.common.labels_v4 import LABEL_FIELDS, LABEL_SCHEMA_VERSION
from tactile_vla.vla import book_v9_7_runtime as contract
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt
from tactile_vla.vla.v7_11_data import DATA_PROFILE, PROMPT_PROFILE, direct_recovery_prompt, phase_prompt_without_plan

CAPTION = 'Touch[area=medium; Fx=near_zero; Fy=positive; Fz=negative; Fz_bias=left; rotation=clockwise]'


def phase_prompt(history=None):
    return phase_prompt_without_plan(build_phase_prompt(
        instruction=contract.BOOK_INSTRUCTION, tactile_caption=CAPTION, recovery_plan='',
        qpos_h100_11_discrete=np.zeros((11, 7), dtype=int) if history is None else history,
    ))


def training_config(norm_sha):
    return {
        'experiment_version': 'book_v9_7', 'data_profile': DATA_PROFILE, 'prompt_profile': PROMPT_PROFILE,
        'task_cycle': ['action', 'adjustment', 'need'], 'phase_prefill_protocol': 'classification_only',
        'recovery_action_condition': 'current_tactile_no_plan_no_recovery_history',
        'phase_condition': 'current_tactile_h100_no_plan', 'use_state_history': False,
        'state_history_len': 0, 'history_hidden_dim': 0, 'action_horizon': 30, 'action_dim': 32,
        'state_history_dim': 7, 'state_history_fps': 30.0, 'max_token_len': 512,
        'reasoning_max_token_len': 320, 'failure_grammar': [], 'recovery_grammar': [], 'no_norm': False,
        'checkpoint_format': 'full_params_inference_only_v1', 'checkpoint_contents': ['full_params'],
        'stage_a_checkpoint_step': 10000,
        'artifact_identity': {'data_profile': 'book_stage_a_v1', 'v4_norm_stats_sha256': norm_sha,
                              'book_v9_7_training_data_hash': 'a' * 64, 'book_v9_7_index_sha256': 'b' * 64,
                              'training_data_hash': 'c' * 64},
        'stage_a_initialization_identity': {'step': 10000, 'action_training_data_hash': 'c' * 64,
                                          'config_sha256': 'd' * 64, 'params_metadata_sha256': 'e' * 64},
    }


def artifacts(tmp_path):
    run = tmp_path / 'run'
    step = run / '2000'
    full = step / 'full_params'
    full.mkdir(parents=True)
    (full / '_METADATA').write_text('{}')
    norm = tmp_path / 'norm'
    norm.mkdir()
    (norm / 'norm_stats.json').write_text('{}')
    config = training_config(hashlib.sha256(b'{}').hexdigest())
    (run / 'config.json').write_text(json.dumps(config))
    # Real no-Plan exports contain no exports/parameter_tree_sha256 field.
    export = {'step': 2000, 'default_deployment': 'full_params', 'resume_supported': False,
              'full_params': '/original/training/machine/full_params'}
    (step / 'book_v9_7_export.json').write_text(json.dumps(export))
    return run, step, full, norm, config


def test_actual_export_format_and_incomplete_download(tmp_path):
    _, step, full, norm, _ = artifacts(tmp_path)
    for checkpoint in (step, full):
        args = server.parse_args(['--checkpoint', str(checkpoint), '--norm-stats-dir', str(norm)])
        _, identity = server.inspect_artifacts(args)
        assert args.checkpoint == full and identity['checkpoint_step'] == 2000
        assert 'full_params_sha256' not in identity  # Computed only after a real restore.
        assert (args.need_recovery_threshold, args.adjustment_end_threshold) == (0.5, 0.9)
    (full / 'data.incomplete').write_bytes(b'partial')
    with pytest.raises(FileNotFoundError, match='unfinished'):
        server.inspect_artifacts(args)


@pytest.mark.parametrize('change', [
    {'experiment_version': 'v7_11'}, {'failure_grammar': ['failure']},
    {'phase_prefill_protocol': 'need_failure_shared_kv_v1'},
    {'stage_a_initialization_identity': {'step': 15000}},
])
def test_reject_wrong_model_family(tmp_path, change):
    run, step, _, norm, config = artifacts(tmp_path)
    (run / 'config.json').write_text(json.dumps(config | change))
    args = server.parse_args(['--checkpoint', str(step), '--norm-stats-dir', str(norm)])
    with pytest.raises(ValueError, match='config mismatch'):
        server.inspect_artifacts(args)


def test_norm_identity_and_threshold_overrides(tmp_path):
    _, step, _, norm, _ = artifacts(tmp_path)
    args = server.parse_args(['--checkpoint', str(step), '--norm-stats-dir', str(norm),
                              '--need-recovery-threshold', '0.6', '--adjustment-end-threshold', '0.85'])
    server.inspect_artifacts(args)
    assert (args.need_recovery_threshold, args.adjustment_end_threshold) == (0.6, 0.85)
    (norm / 'norm_stats.json').write_text('{"wrong": true}')
    with pytest.raises(ValueError, match='norm_stats_sha256'):
        server.inspect_artifacts(args)


def test_prompts_keep_six_fields_and_reject_plan_or_other_task():
    for mode, prompt in [('phase', phase_prompt()),
                         ('execution', direct_recovery_prompt(contract.BOOK_INSTRUCTION, CAPTION)),
                         ('execution', f'Mode: execution. Task: {contract.BOOK_INSTRUCTION}')]:
        contract.validate_request({'mode': mode, 'prompt': prompt})
        with pytest.raises(ValueError):
            contract.validate_request({'mode': mode, 'prompt': prompt.replace(contract.BOOK_INSTRUCTION, 'Other task.')})
    with pytest.raises(ValueError, match='without Plan'):
        contract.validate_request({'mode': 'phase', 'prompt': phase_prompt() + '\nRecovery plan: none'})
    with pytest.raises(ValueError, match='six-field'):
        contract.validate_request({'mode': 'phase', 'prompt': phase_prompt().replace('; Fz_bias=left', '')})


def test_captioner_schema_and_raw_qpos_arguments(tmp_path):
    captioner = SimpleNamespace(schema_version=LABEL_SCHEMA_VERSION, label_fields=LABEL_FIELDS, window_size=30)
    contract.validate_captioner(captioner)
    with pytest.raises(ValueError, match='six-field'):
        contract.validate_captioner(SimpleNamespace(schema_version='tactile_multifield_v3'))
    norm = tmp_path / 'norm.json'
    norm.write_text('{}')
    base = ['--noise-seed', '42', '--gripper-min', '0.025', '--norm-stats-file', str(norm)]
    args, parser = client.get_arguments(base)
    client.validate_args(args, parser)
    assert args.instruction == contract.BOOK_INSTRUCTION and args.publish_rate == 20 and args.max_attempts == 10
    args, parser = client.get_arguments(base + ['--classification-gripper-open-threshold', '0.0985',
                                               '--classification-gripper-open-value', '0.0995'])
    with pytest.raises(SystemExit):
        client.validate_args(args, parser)


def test_phase_payload_keeps_raw_qpos_h100_and_true_stops_chunk(monkeypatch):
    qpos = np.arange(7, dtype=np.float32)
    dense = np.arange(700, dtype=np.float32).reshape(100, 7)
    obs = SimpleNamespace(qpos=qpos, timestamp=11.0, tactile_caption=CAPTION, img_front=None, img_left=None)
    monkeypatch.setattr(client, 'capture_tactile_observation', lambda *a, **kw: (obs, {}))
    monkeypatch.setattr(client.runtime, 'build_payload', lambda **kw: kw)
    seen = []
    def discretize(values, stats):
        np.testing.assert_array_equal(values, dense[np.linspace(0, 99, 11, dtype=int)])
        return np.zeros((11, 7), dtype=int)
    monkeypatch.setattr(client, 'discretize_state_qpos', discretize)
    held = []
    gate = client.BookV97AsyncControlGate(hold=lambda: held.append(True))
    generation, _ = gate.begin_phase_request('r1')
    class Phase:
        def events(self, payload):
            seen.append(payload)
            yield {'event': 'phase_decision', 'request_id': 'r1', 'phase': 'execution', 'need_recovery': True}
    result = client._run_phase_assessment(
        args=SimpleNamespace(instruction=contract.BOOK_INSTRUCTION), client=Phase(),
        operator=SimpleNamespace(state_history=SimpleNamespace(snapshot=lambda **kw: (dense, None))),
        captioner=None, stats=None, gate=gate, phase='execution', generation=generation, attempt_id=1,
        request_id='r1', captured_step=3, after_timestamp=10.0, recovery_plan='', submitted_monotonic=0,
    )
    np.testing.assert_array_equal(seen[0]['qpos'], qpos)
    assert seen[0]['prompt'] == phase_prompt() and 'Fz_bias=left' in result.prompt
    assert held and result.triggered and not gate.may_publish_action(0)
    gate.switch_phase('adjustment', increment_attempt=True)
    assert not gate.accept_action_result(0, np.zeros((30, 7)))
    assert gate.snapshot()[:2] == ('adjustment', 2)


def test_each_adjustment_action_uses_fresh_six_field_caption(monkeypatch):
    monkeypatch.setattr(client.runtime, 'build_payload', lambda **kw: kw)
    prompts = []
    class Action:
        def infer_with_timeout(self, payload, timeout):
            prompts.append(payload['prompt'])
            return {'actions': np.zeros((30, 7)), 'raw_model_actions': np.zeros((30, 32))}
    args = SimpleNamespace(instruction=contract.BOOK_INSTRUCTION, noise_seed=42, request_timeout_seconds=1)
    for caption in (CAPTION, CAPTION.replace('Fz_bias=left', 'Fz_bias=right')):
        client._request_action_chunk(
            args=args, policy=Action(), observation=SimpleNamespace(qpos=np.zeros(7), timestamp=1,
                tactile_caption=caption, img_front=None, img_left=None),
            phase='adjustment', phase_index=0, recovery_plan='', logger=SimpleNamespace(record=lambda e: None),
        )
    assert prompts == [direct_recovery_prompt(args.instruction, c) for c in
                       (CAPTION, CAPTION.replace('Fz_bias=left', 'Fz_bias=right'))]


def test_two_heads_emit_single_event_without_failure_or_plan(monkeypatch):
    policy = server.BookV97Policy.__new__(server.BookV97Policy)
    policy._assessment_transform = lambda request: request
    policy._need_logits = lambda observation: jnp.asarray([[0.0, 2.0]])
    policy._adjustment_logits = policy._need_logits
    policy._need_threshold, policy._adjustment_threshold = 0.5, 0.9
    monkeypatch.setattr(server.v3.TactileVLAPolicyV3, '_observation', staticmethod(lambda r: r))
    request = {'mode': 'phase', 'request_id': 'r1', 'prompt': phase_prompt()}
    need = list(policy.infer_events(request | {'phase': 'execution'}))
    end = list(policy.infer_events(request | {'phase': 'adjustment'}))
    assert len(need) == len(end) == 1
    assert need[0]['need_recovery'] is True and end[0]['adjustment_end'] is False


@pytest.mark.parametrize('change', [None, {'phase': 'adjustment'}, {'need_recovery_probs': [float('nan'), 1]},
                                    {'need_recovery': False}])
def test_stream_reads_one_decision_and_validates_it(change):
    phase = client.StreamingPhaseClient.__new__(client.StreamingPhaseClient)
    decision = {'event': 'phase_decision', 'request_id': 'r1', 'phase': 'execution',
                'need_recovery': True, 'need_recovery_probs': [0.1, 0.9]}
    reads = []
    def receive(timeout):
        reads.append(True)
        assert len(reads) == 1, 'classification-only client must not wait for a second event'
        return client.msgpack_numpy.packb(decision | (change or {}))
    phase._ws = SimpleNamespace(send=lambda payload: None, recv=receive)
    phase._timeout = 1
    phase._packer = client.msgpack_numpy.Packer()
    phase._lock = threading.Lock()
    phase.metadata = {'need_recovery_threshold': 0.5}
    request = {'phase': 'execution', 'request_id': 'r1'}
    if change is None:
        assert list(phase.events(request)) == [decision]
    else:
        with pytest.raises(ValueError):
            list(phase.events(request))


def test_only_three_shared_paths_are_warmed_up():
    requests = []
    class Policy:
        def infer_events(self, request):
            contract.validate_request(request)
            requests.append(request)
            yield {}
    server.warm_up(Policy())
    assert [r.get('phase', 'action') for r in requests] == ['action', 'execution', 'adjustment']


def test_client_main_writes_no_recordings(monkeypatch, tmp_path):
    norm, checkpoint = tmp_path / 'norm.json', tmp_path / 'captioner.pt'
    norm.write_text('{}')
    checkpoint.write_bytes(b'captioner')
    monkeypatch.setattr(client.sys, 'argv', ['client', '--noise-seed', '42', '--gripper-min', '0.025',
        '--norm-stats-file', str(norm), '--captioner_checkpoint', str(checkpoint), '--start-immediately'])
    monkeypatch.setattr(client.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(client.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(client.runtime, 'load_captioner', lambda args: SimpleNamespace(
        schema_version=LABEL_SCHEMA_VERSION, label_fields=LABEL_FIELDS, window_size=30))
    monkeypatch.setattr(client.runtime, 'RosOperator', lambda args: object())
    monkeypatch.setattr(client.v53, '_install_tactile_arrival_timestamps', lambda op: None)
    monkeypatch.setattr(client.v53, 'TimeoutWebsocketPolicy', lambda *a: object())
    monkeypatch.setattr(client, 'StreamingPhaseClient', lambda *a, **kw: SimpleNamespace(close=lambda: None))
    class Keyboard:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
    monkeypatch.setattr(client.runtime, 'KeyboardPoller', Keyboard)
    monkeypatch.setattr(client, 'run_book_v9_7_async', lambda *a: a[-1].record({'test': True}))
    monkeypatch.chdir(tmp_path)
    client.main()
    assert set(tmp_path.iterdir()) == {norm, checkpoint}
