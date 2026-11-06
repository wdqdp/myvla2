"""V7.11 Stage A provenance and Stage B initialization checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tactile_vla.vla.artifacts import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path('/data1/qxh/tac_vla_new/tac_data/demon_data/black_box')
DEFAULT_DATASET = SOURCE_ROOT / 'lerobot_data/tactile_vla_rotation_v4_1_new_env'
DEFAULT_ACTION_INDEX = SOURCE_ROOT / 'outputs/rotation_v7_4_2_adjustment/v7_4_2_prompt_training_index.json'
DEFAULT_NORM_STATS = SOURCE_ROOT / 'outputs/rotation_v4_1_new_env/norm_stats'
DEFAULT_STAGE_A_OUTPUT = PROJECT_ROOT / 'outputs/stage_a_action'
RUN_NAME = 'pi05_rotation_v7_11_stage_a_direct_tactile'
DEFAULT_STAGE_A_CHECKPOINT = DEFAULT_STAGE_A_OUTPUT / RUN_NAME / '10000'


def checkpoint_prompt_identity() -> dict[str, str]:
    root = PROJECT_ROOT / 'src/tactile_vla/vla'
    return {
        'version': 'v7_11',
        'action_condition': 'execution_unchanged_adjustment_current_tactile_no_plan',
        'prompt_code_sha256': sha256_file(root / 'v7_11_data.py'),
        'dataset_code_sha256': sha256_file(root / 'v7_11_datasets.py'),
    }


def validate_stage_a_checkpoint(checkpoint: Path, multitask_index: Mapping[str, Any]) -> dict[str, Any]:
    """Reject legacy or mismatched checkpoints before Stage B loads any weights."""
    step = Path(checkpoint).expanduser().resolve()
    if step.name == 'params':
        raise ValueError('Stage B expects a numeric Stage A step directory, not its params child')
    if not step.name.isdigit() or int(step.name) <= 0 or str(int(step.name)) != step.name:
        raise ValueError(f'V7.11 Stage A checkpoint must be a positive numeric step directory: {step}')
    config_file, metadata = step.parent / 'config.json', step / 'params/_METADATA'
    if not config_file.is_file() or not metadata.is_file():
        raise FileNotFoundError(
            f'V7.11 Stage A checkpoint is incomplete: {step}. Train Stage A before starting Stage B.'
        )
    config = json.loads(config_file.read_text())
    identity = config.get('artifact_identity', {})
    action_file = Path(multitask_index['action_index_file']).resolve()
    norm_candidates = [Path(k) for k in multitask_index['source_hashes'] if Path(k).name == 'norm_stats.json']
    if len(norm_candidates) != 1:
        raise ValueError('V7.11 requires exactly one norm_stats.json in the source index')
    norm_file = norm_candidates[0]
    checks = {
        'v7_11_stage_a': (config.get('v7_11_stage_a'), checkpoint_prompt_identity()),
        'data_profile': (config.get('data_profile'), 'rotation_phase_v7_4_2_adjustment'),
        'prompt_profile': (config.get('prompt_profile'), 'phase_v2'),
        'stage_a_protocol': (config.get('stage_a_protocol'), 'v7_4_2_finetune_no_state_history'),
        'use_state_history': (config.get('use_state_history'), False),
        'state_history_len': (config.get('state_history_len'), 0),
        'history_hidden_dim': (config.get('history_hidden_dim'), 0),
        'action_horizon': (config.get('action_horizon'), 30),
        'action_dim': (config.get('action_dim'), 32),
        'train_lora_only': (config.get('train_lora_only'), True),
        'paligemma_variant': (config.get('paligemma_variant'), 'gemma_2b_lora'),
        'action_expert_variant': (config.get('action_expert_variant'), 'gemma_300m_lora'),
        'training_data_hash': (identity.get('training_data_hash'), multitask_index['action_training_data_hash']),
        'index_sha256': (identity.get('index_sha256'), sha256_file(action_file)),
        'v4_norm_stats_sha256': (identity.get('v4_norm_stats_sha256'), sha256_file(norm_file)),
    }
    mismatches = {key: {'saved': actual, 'expected': expected}
                  for key, (actual, expected) in checks.items() if actual != expected}
    if mismatches:
        raise ValueError(f'V7.11 Stage A checkpoint does not match the direct-tactile protocol: {mismatches}')
    return {'path': str(step), 'step': int(step.name), 'config_sha256': sha256_file(config_file),
            'params_metadata_sha256': sha256_file(metadata), 'prompt_identity': config['v7_11_stage_a']}
