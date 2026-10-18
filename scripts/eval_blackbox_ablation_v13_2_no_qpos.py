#!/usr/bin/env python3
"""Evaluate an existing V13.2 model with normalized qpos AND its text tokens held constant."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('OPENPI_DATA_HOME', '/home/qxh/.cache/openpi')
os.environ.setdefault('HF_HOME', str(PROJECT / '.cache/huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', str(PROJECT / '.cache/huggingface/datasets'))

import numpy as np

from tactile_vla.vla.qpos_masking import MaskNormalizedQpos, mask_structured_transform  # noqa: F401


def mask_inference_transform(transform):
    """Insert after normalization but BEFORE qpos is embedded in the prompt tokens."""
    return mask_structured_transform(transform, inference=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='best')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--dry-run', action='store_true', help='Audit real input masking on CPU; no model weights/output writes')
    parser.add_argument('--overwrite', action='store_true', help='Replace only existing *_qpos_masked result files')
    args = parser.parse_args(argv)
    value = args.checkpoint
    if value != 'best' and (not value.isdigit() or value != str(int(value))):
        parser.error('--checkpoint must be best or a numeric step')
    return args


def output_paths(run, split, checkpoint):
    stem = f'{split}_{checkpoint}_qpos_masked'
    return run / f'{stem}.metrics.json', run / f'{stem}.predictions.json'


def audit_batch(loader):
    """Verify the REAL transform has no qpos-dependent state or text, while preserving images."""
    row = loader.dataset.raw(0)
    first = loader.dataset.transform(dict(row))
    changed = {**row, 'observation/state': np.asarray(row['observation/state'], dtype=np.float32) + .123}
    second = loader.dataset.transform(changed)
    for key in ('state', 'tokenized_prompt', 'tokenized_prompt_mask', 'token_ar_mask', 'token_loss_mask'):
        if not np.array_equal(first[key], second[key]):
            raise AssertionError(f'Qpos information survived masking: {key}')
    if np.any(first['state']) or np.any(first['token_loss_mask']) or 'target_text' in first:
        raise AssertionError('Expected zero state and a target-free inference transform')
    return {'state_shape': list(first['state'].shape), 'state_all_zero': True,
            'qpos_token_invariant': True, 'teacher_forcing': False,
            'prompt': row['prompt'], 'sample_id': int(row['global_index'])}


def main(argv=None):
    extra = parse_args(argv)
    if extra.dry_run:
        # Set before importing JAX/OpenPI so a diagnostic never reserves a GPU.
        os.environ['JAX_PLATFORMS'] = 'cpu'
    from scripts.train_blackbox_ablation_v13_2 import components, common
    from tactile_vla.vla.artifacts import sha256_file
    from tactile_vla.vla.blackbox_ablation_v13_1_data import read_json, write_json
    from tactile_vla.vla.blackbox_ablation_v13_1_runtime import init_state, make_loader, restore_overlay
    from tactile_vla.vla.blackbox_ablation_v13_2_data import SCHEMA
    from tactile_vla.vla.blackbox_ablation_v13_2_runtime import evaluate

    run = extra.run_dir.resolve()
    metrics_path, predictions_path = output_paths(run, extra.split, extra.checkpoint)
    if not extra.dry_run and not extra.overwrite:
        for path in (metrics_path, predictions_path):
            if path.exists():
                raise FileExistsError(f'{path}; use --overwrite to replace only masked-qpos results')
    saved = read_json(run / 'config.json')
    if saved.get('experiment_version') != SCHEMA:
        raise ValueError('Expected an independent V13.2 run')
    args = argparse.Namespace(**saved)
    for key in ('index_file', 'stage_a_checkpoint', 'output_dir'):
        setattr(args, key, Path(getattr(args, key)))
    common.validate_stage_a(args.stage_a_checkpoint)
    if sha256_file(args.stage_a_checkpoint / 'params/_METADATA') != saved['stage_a_metadata_sha256']:
        raise ValueError('Stage A metadata changed')
    index, rows, config, tokenizer, grammar, norm = components(args)
    if index['training_data_hash'] != saved['training_data_hash']:
        raise ValueError('Evaluation data differs from training')
    selected = [row for row in rows if row['split'] == extra.split]
    loader = make_loader(selected, group=args.group, model_config=config, tokenizer=tokenizer,
                         grammar=grammar, norm=norm, batch_size=args.batch_size,
                         num_workers=args.num_workers, inference=True)
    loader.dataset.transform = mask_inference_transform(loader.dataset.transform)
    audit = audit_batch(loader)
    print(json.dumps({'group': args.group, 'samples': len(selected), 'input_audit': audit,
                      'weights_loaded': False}, ensure_ascii=False, indent=2), flush=True)
    if extra.dry_run:
        return 0

    import jax
    from openpi.training import sharding

    if args.batch_size % jax.device_count() or jax.device_count() % args.fsdp_devices:
        raise ValueError('Batch must divide over visible devices; FSDP must divide device count')
    mesh = sharding.make_mesh(args.fsdp_devices)
    state, _, _ = init_state(args, config, mesh)
    checkpoint = extra.checkpoint
    path = run / checkpoint / ('delta_params' if checkpoint == 'best' else 'params')
    state = dataclasses.replace(state, params=restore_overlay(state.params, path, include_rng=checkpoint != 'best'))
    checkpoint_step = read_json(run / 'best/metrics.json')['step'] if checkpoint == 'best' else int(checkpoint)
    report, predictions = evaluate(state, loader, grammar)
    write_json(metrics_path, {'group': args.group, 'checkpoint': checkpoint, 'checkpoint_step': checkpoint_step,
               'split': extra.split, 'training_data_hash': index['training_data_hash'],
               'intervention': {'name': 'normalized_qpos_zero_before_tokenization', 'masked_dimensions': list(range(7)),
                                'images_unchanged': True, 'prompt_unchanged': True,
                                'state_token_slots_retained': True, 'retrained': False},
               'input_audit': audit, 'metrics': report})
    write_json(predictions_path, predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Results: {metrics_path}\nPredictions: {predictions_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
