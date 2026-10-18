#!/usr/bin/env python3
"""V7.9: V7.4.2 Stage A -> five tasks, without TRAIN adjustment counterfactuals."""
# ruff: noqa: E402
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / 'src'), str(PROJECT_ROOT / 'openpi/src')]

from scripts import train_vla_multitask_v7_7_2 as source_entry
from tactile_vla.vla import v7_9_multitask_data as data
from tactile_vla.vla.v7_9_multitask_eval import evaluate_adjustment

trainer, base = source_entry.trainer, source_entry.trainer.base
_BASE_PARSE = trainer.parse_args
_BASE_CONFIGURE = trainer.configure
_BASE_ENSURE_INDEX = trainer.ensure_index
_BASE_EXPORT = trainer.export_checkpoint
IDENTITY_KEYS = ('experiment_version', 'counterfactual_policy', 'training_data_hash',
    'source_multitask_training_data_hash', 'source_multitask_index_sha256', 'history_policy',
    'adjustment_sampling_policy', 'adjustment_label_policy', 'need_successful_recovery_start',
    'stage_a_initialization_identity')


def parse_args(argv=None):
    saved = sys.argv
    supplied = list(saved[1:] if argv is None else argv)
    normalized = []
    for argument in supplied:
        normalized.extend(argument.split('=', 1) if argument.startswith('--') and '=' in argument else [argument])
    defaults = {'--num-steps': '2000', '--batch-size': '8', '--fsdp-devices': '2',
                '--eval-interval': '1000', '--save-interval': '1000', '--keep-period': '1000',
                '--eval-max-need-samples': '4096', '--num-workers': '0', '--seed': '42', '--lr': '0.0001'}
    for flag, value in defaults.items():
        if flag not in normalized:
            normalized.extend([flag, value])
    try:
        sys.argv = [saved[0], *normalized]
        args = _BASE_PARSE()
    finally:
        sys.argv = saved
    required = {'data_profile': data.DATA_PROFILE, 'prompt_profile': data.PROMPT_PROFILE,
        'use_state_history': False, 'state_history_len': 0, 'history_hidden_dim': 0,
        'action_horizon': 30, 'action_dim': 32, 'max_token_len': 512,
        'reasoning_max_token_len': 320, 'no_norm': False}
    mismatches = {k: (getattr(args, k), v) for k, v in required.items() if getattr(args, k) != v}
    if mismatches:
        raise ValueError(f'V7.9 input/model protocol differs from V7.7.2: {mismatches}')
    return args


def ensure_index(args):
    index, records = _BASE_ENSURE_INDEX(args)
    stage_a = data.validate_stage_a_for_training(args.stage_a_checkpoint, index)
    base.EXTRA_CONFIG.update({key: index[key] for key in IDENTITY_KEYS if key in index})
    base.EXTRA_CONFIG['stage_a_initialization_identity'] = stage_a
    return index, records


def validate_resume(saved, args, *, actual_precision):
    for key in IDENTITY_KEYS:
        if saved.get(key) != base.EXTRA_CONFIG.get(key):
            raise ValueError(f'V7.9 resume data/ablation identity mismatch: {key}')
    for key in ('data_profile', 'prompt_profile', 'batch_size', 'seed', 'lr', 'weight_decay', 'grad_clip',
        'action_loss_weight', 'adjustment_loss_weight', 'need_loss_weight', 'failure_loss_weight',
        'plan_loss_weight', 'eval_interval', 'eval_max_need_samples', 'eval_max_text_samples',
        'reasoning_max_token_len', 'action_loss_degradation_limit', 'grammar_profile', 'fsdp_devices'):
        if saved.get(key) != getattr(args, key):
            raise ValueError(f'V7.9 resume training config mismatch: {key}')
    if saved.get('precision') != actual_precision:
        raise ValueError('V7.9 resume precision mismatch')


def export_checkpoint(run_dir, state, step, filter_):
    _BASE_EXPORT(run_dir, state, step, filter_)
    path = run_dir / str(step) / 'v7_9_export.json'
    metadata = json.loads(path.read_text())
    metadata.update(base.EXTRA_CONFIG | {'data_profile': data.DATA_PROFILE, 'prompt_profile': data.PROMPT_PROFILE})
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


def configure_training():
    _BASE_CONFIGURE()
    base.EXTRA_CONFIG.update(experiment_version=data.VERSION_TAG, counterfactual_policy=data.COUNTERFACTUAL_POLICY)
    base.validate_v4_resume_config = validate_resume
    base.CHECKPOINT_EXPORT_HOOK = export_checkpoint
    base.evaluate_adjustment = evaluate_adjustment


def configure_version():
    source_entry.configure_version()
    trainer.DATA_PROFILE, trainer.PROMPT_PROFILE = data.DATA_PROFILE, data.PROMPT_PROFILE
    trainer.DEFAULT_INDEX, trainer.DEFAULT_STAGE_A = data.DEFAULT_INDEX, data.DEFAULT_STAGE_A
    trainer.DEFAULT_OUTPUT, trainer.RUN_NAME = data.ROOT / 'outputs/multitask_v7_9', data.RUN_NAME
    trainer.VERSION_TAG, trainer.validate_index = data.VERSION_TAG, data.validate_index
    trainer.parse_args, trainer.configure, trainer.ensure_index = parse_args, configure_training, ensure_index


def main():
    configure_version()
    trainer.main()


if __name__ == '__main__':
    main()
