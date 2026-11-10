"""Sponge V10.3 direct-tactile deployment contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from tactile_vla.common.labels_v4 import LABEL_FIELDS, LABEL_SCHEMA_VERSION
from tactile_vla.vla.sponge_v10_1_runtime import validate_tactile_caption
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt
from tactile_vla.vla.v7_11_data import DATA_PROFILE, PROMPT_PROFILE, direct_recovery_prompt, phase_prompt_without_plan

SPONGE_ROOT = Path('/data1/qxh/tac_vla_new/tac_data/demon_data/sponge')
SPONGE_INSTRUCTION = 'Put the black sponge into the black frame.'
DEFAULT_RUN = SPONGE_ROOT / 'outputs/multitask_v10_3/pi05_sponge_v10_3_direct_tactile_h100'
DEFAULT_NORM_DIR = SPONGE_ROOT / 'outputs/rotation_v4/norm_stats'
DEFAULT_CAPTIONER = Path(
    '/data1/qxh/tac_vla_new/tac_data/tac_cap_data/tactile_captioner/'
    'tcn_v4_w30_masked_six_head_mixed/best.pt'
)
SERVER_NAME = 'tactile_vla_sponge_v10_3'


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_training_config(config, norm_sha):
    expected = {
        'experiment_version': 'sponge_v10_3', 'data_profile': DATA_PROFILE, 'prompt_profile': PROMPT_PROFILE,
        'task_cycle': ['action', 'adjustment', 'need'], 'phase_prefill_protocol': 'classification_only',
        'recovery_action_condition': 'current_tactile_no_plan_no_recovery_history',
        'phase_condition': 'current_tactile_h100_no_plan', 'use_state_history': False,
        'state_history_len': 0, 'history_hidden_dim': 0, 'action_horizon': 30, 'action_dim': 32,
        'state_history_dim': 7, 'state_history_fps': 30.0, 'max_token_len': 512,
        'reasoning_max_token_len': 320, 'failure_grammar': [], 'recovery_grammar': [],
        'checkpoint_format': 'full_params_inference_only_v1', 'checkpoint_contents': ['full_params'],
        'no_norm': False,
    }
    mismatch = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    artifact = config.get('artifact_identity', {})
    if artifact.get('data_profile') != 'book_stage_a_v1':
        mismatch['artifact_identity.data_profile'] = (artifact.get('data_profile'), 'book_stage_a_v1')
    if artifact.get('v4_norm_stats_sha256') != norm_sha:
        mismatch['norm_stats_sha256'] = (norm_sha, artifact.get('v4_norm_stats_sha256'))
    for key in ('sponge_v10_3_training_data_hash', 'sponge_v10_3_index_sha256', 'training_data_hash'):
        if not _digest(artifact.get(key)):
            mismatch[key] = (artifact.get(key), 'SHA256')
    stage = config.get('stage_a_initialization_identity', {})
    if stage.get('step') != 10000 or config.get('stage_a_checkpoint_step') != 10000:
        mismatch['stage_a_step'] = (stage.get('step'), 10000)
    if stage.get('action_training_data_hash') != artifact.get('training_data_hash'):
        mismatch['stage_a_action_data'] = (stage.get('action_training_data_hash'), artifact.get('training_data_hash'))
    for key in ('config_sha256', 'params_metadata_sha256'):
        if not _digest(stage.get(key)):
            mismatch['stage_a.' + key] = (stage.get(key), 'SHA256')
    if mismatch:
        raise ValueError(f'Sponge V10.3 config mismatch: {mismatch}')


def validate_captioner(captioner):
    if (getattr(captioner, 'schema_version', None) != LABEL_SCHEMA_VERSION
            or tuple(getattr(captioner, 'label_fields', ())) != tuple(LABEL_FIELDS)
            or getattr(captioner, 'window_size', None) != 30):
        raise ValueError('Sponge V10.3 requires the original six-field V4 W30 caption schema, including Fz_bias')


def validate_server_metadata(metadata):
    expected = {
        'name': SERVER_NAME, 'experiment_version': 'sponge_v10_3',
        'data_profile': DATA_PROFILE, 'phase_prompt_profile': PROMPT_PROFILE,
        'instruction': SPONGE_INSTRUCTION, 'phase_prefill_protocol': 'classification_only',
        'classification_qpos_policy': 'raw_no_gripper_remap',
        'tactile_label_fields': list(LABEL_FIELDS), 'tactile_label_schema': LABEL_SCHEMA_VERSION,
        'requires_captioner': True, 'requires_tactile_topics': True,
        'supports_streamed_phase_events': True, 'supports_action_noise': True, 'requires_action_noise': True,
        'supports_failure_generation': False, 'supports_recovery_generation': False,
        'action_horizon': 30, 'action_dim': 32, 'output_action_dim': 7,
        'use_state_history': False, 'state_history_len': 0,
    }
    mismatch = {k: (metadata.get(k), v) for k, v in expected.items() if metadata.get(k) != v}
    for key in ('need_recovery_threshold', 'adjustment_end_threshold'):
        value = metadata.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            mismatch[key] = (value, 'finite probability')
    for key in ('norm_stats_sha256', 'full_params_sha256', 'training_data_hash'):
        if not _digest(metadata.get(key)):
            mismatch[key] = (metadata.get(key), 'SHA256')
    if mismatch:
        raise ValueError(f'Sponge V10.3 server metadata mismatch: {mismatch}')


def validate_request(request):
    """Require the exact task and Touch format seen by the training wrappers."""
    if any(key in request for key in ('observation/state_history', 'observation/state_history_mask')):
        raise ValueError('Sponge V10.3 action expert does not accept dense state history')
    prompt = request.get('prompt', '')
    if not isinstance(prompt, str):
        raise ValueError('Sponge V10.3 prompt must be text')
    mode = request.get('mode')
    if mode in ('action', 'execution'):
        if prompt == f'Mode: execution. Task: {SPONGE_INSTRUCTION}':
            return
        prefix = direct_recovery_prompt(SPONGE_INSTRUCTION, '_').removesuffix('_')
        if not prompt.startswith(prefix):
            raise ValueError('Sponge V10.3 action requires its original instruction and current Touch')
        validate_tactile_caption(prompt[len(prefix):])
        return
    if mode != 'phase':
        raise ValueError('Sponge V10.3 supports only action and phase classification')
    lines = prompt.splitlines()
    if len(lines) != 4 or not lines[2].startswith('Touch: '):
        raise ValueError('Sponge V10.3 phase requires current Touch and H100 without Plan')
    caption = lines[2][len('Touch: '):]
    validate_tactile_caption(caption)
    history_prefix = 'State history H100 sampled to 11 points: '
    if not lines[3].startswith(history_prefix):
        raise ValueError('Sponge V10.3 phase requires H100 sampled to 11 points')
    history = json.loads(lines[3][len(history_prefix):])
    expected = phase_prompt_without_plan(build_phase_prompt(
        instruction=SPONGE_INSTRUCTION, tactile_caption=caption, recovery_plan='',
        qpos_h100_11_discrete=history,
    ))
    if prompt != expected:
        raise ValueError('Sponge V10.3 phase prompt differs from its training format')
