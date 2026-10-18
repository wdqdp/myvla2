#!/usr/bin/env python3
"""Train ONLY plan for one of the four paired V13.1 ablation groups."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('OPENPI_DATA_HOME', '/home/qxh/.cache/openpi')
os.environ.setdefault('HF_HOME', str(PROJECT / '.cache/huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', str(PROJECT / '.cache/huggingface/datasets'))

from tactile_vla.vla.blackbox_ablation_v13_1_data import (
    GROUPS, INDEX, OUTPUT, SCHEMA, STAGE_A, read_json, validate_index, write_json,
)
from tactile_vla.vla.blackbox_ablation_v13_1_runtime import (
    StepBatchSampler, evaluate, init_state, make_loader, validate_tokens,
)


def parse_args(argv=None, *, index_file=INDEX, output_dir=OUTPUT.parent / 'plan_v13_1',
               run_prefix='pi05_blackbox_ablation_v13_1', description=__doc__):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--group', choices=GROUPS, required=True)
    p.add_argument('--index-file', type=Path, default=index_file)
    p.add_argument('--stage-a-checkpoint', type=Path, default=STAGE_A)
    p.add_argument('--output-dir', type=Path, default=output_dir)
    p.add_argument('--run-name')
    p.add_argument('--num-steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--eval-interval', type=int, default=200)
    p.add_argument('--save-interval', type=int, default=200)
    p.add_argument('--keep-period', type=int, default=200)
    p.add_argument('--log-interval', type=int, default=20)
    p.add_argument('--max-token-len', type=int, default=384)
    p.add_argument('--fsdp-devices', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--precision', choices=('auto', 'bfloat16', 'float32'), default='auto')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='CPU data/token/transform audit, no model weights loaded')
    p.add_argument('--audit-only', action='store_true', help='Validate labels/tokens without reading images or weights')
    args = p.parse_args(argv)
    if any(getattr(args, k) <= 0 for k in ('num_steps', 'batch_size', 'eval_interval', 'save_interval',
                                         'keep_period', 'log_interval', 'max_token_len', 'fsdp_devices')):
        p.error('Steps, batch size, intervals and sharding must be positive')
    if args.lr <= 0 or args.grad_clip <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        p.error('Invalid optimizer/workers setting')
    if args.run_name is None:
        args.run_name = f'{run_prefix}_{args.group}_s{args.seed}'
    if Path(args.run_name).name != args.run_name:
        p.error('--run-name must be a single directory name')
    return args


def components(args, *, verify_sources=True, index_validator=validate_index):
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from tactile_vla.vla.structured_text import recovery_grammar
    index, rows = index_validator(args.index_file, verify_sources=verify_sources,
                                 require_images=not args.audit_only)
    config = Pi0Config(dtype=args.precision, pi05=True, paligemma_variant='gemma_2b_lora',
                      action_expert_variant='gemma_300m_lora', action_dim=32, action_horizon=30,
                      max_token_len=args.max_token_len, use_state_history=False,
                      state_history_len=0, history_hidden_dim=0, pytorch_compile_mode=None)
    tokenizer = PaligemmaTokenizer(args.max_token_len)
    grammar = recovery_grammar(lambda text: tokenizer.encode_text(text, add_eos=True))
    norm = normalize.load(Path(index['norm_stats_dir']))
    return index, rows, config, tokenizer, grammar, norm


def validate_stage_a(path):
    path = Path(path)
    if not (path / 'params/_METADATA').is_file():
        raise FileNotFoundError(f'Missing Stage A params: {path}')
    config = read_json(path.parent / 'config.json')
    required = {'paligemma_variant': 'gemma_2b_lora', 'action_expert_variant': 'gemma_300m_lora',
                'action_dim': 32, 'action_horizon': 30, 'use_state_history': False}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f'Incompatible Stage A architecture: {key}')
    return config


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def resume_check(saved, current):
    # All input semantics and optimizer settings remain fixed; total steps may extend.
    for key in ('group', 'seed', 'batch_size', 'fsdp_devices', 'lr', 'weight_decay', 'grad_clip',
                'max_token_len', 'precision', 'training_data_hash', 'stage_a_checkpoint', 'stage_a_metadata_sha256',
                'experiment_version', 'qpos_input_protocol'):
        if saved.get(key) != current.get(key):
            raise ValueError(f'Resume mismatch: {key}')


def main(*, parser=parse_args, component_builder=components, evaluator=evaluate,
         schema=SCHEMA, version_label='V13.1', loader_builder=make_loader, token_validator=validate_tokens):
    args = parser()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import tqdm
    from openpi.training import checkpoints, sharding
    from tactile_vla.vla.artifacts import sha256_file
    from scripts import train_vla_stage_b_v3 as base
    base.init_logging()
    args.precision = ('bfloat16' if jax.default_backend() in ('gpu', 'tpu') else 'float32') if args.precision == 'auto' else args.precision
    index, rows, config, tokenizer, grammar, norm = component_builder(args)
    audit = token_validator(rows, tokenizer=tokenizer, grammar=grammar, norm=norm, max_len=args.max_token_len)
    if args.audit_only:
        print(json.dumps({'group': args.group, 'token_validation': audit,
            'images': read_json(Path(index['events_file']).parent / 'image_audit.json')['ready'],
            'weights_loaded': False}, ensure_ascii=False, indent=2))
        return
    selected = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val')}
    common = dict(group=args.group, model_config=config, tokenizer=tokenizer, grammar=grammar,
                  norm=norm, batch_size=args.batch_size, num_workers=args.num_workers)
    if args.dry_run:
        loader = loader_builder(selected['train'], **common)
        batch = next(iter(loader))
        inference = next(iter(loader_builder(selected['val'], **common, inference=True)))
        assert not np.any(inference['token_loss_mask'])
        print(json.dumps({'group': args.group, 'train_samples': len(selected['train']), 'val_samples': len(selected['val']),
            'token_validation': audit, 'prompt': selected['train'][0]['prompts'][args.group],
            'batch_shapes': {k: list(np.shape(v)) for k, v in batch.items() if not isinstance(v, dict)},
            'weights_loaded': False, 'task_cycle': ['plan']}, ensure_ascii=False, indent=2))
        return
    validate_stage_a(args.stage_a_checkpoint)
    if args.batch_size % jax.device_count() or jax.device_count() % args.fsdp_devices:
        raise ValueError('Batch must divide over visible devices; FSDP must divide device count')
    run_dir = (args.output_dir / args.run_name).resolve()
    current = vars(args) | {'experiment_version': schema, 'training_data_hash': index['training_data_hash'],
        'stage_a_checkpoint': str(args.stage_a_checkpoint.resolve()),
        'stage_a_metadata_sha256': sha256_file(args.stage_a_checkpoint / 'params/_METADATA'),
        'index_file': str(args.index_file.resolve()), 'task_cycle': ['plan'], 'updates_per_task': args.num_steps,
        'trainable_components': ['paligemma_lora'], 'frozen_components': ['action_expert', 'paligemma_non_lora'],
        'token_validation': audit, 'norm_stats_dir': index['norm_stats_dir'],
        'selection_metric': 'val.episode_macro_joint_correct', 'recovery_grammar': list(grammar.texts)}
    current = json.loads(json.dumps(current, default=str))
    if args.resume:
        resume_check(read_json(run_dir / 'config.json'), current)
    elif run_dir.exists():
        raise FileExistsError('Run already exists; use a new name or --resume, never overwrite')
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        run_dir, keep_period=args.keep_period, overwrite=False, resume=args.resume, enable_async_checkpointing=False)
    try:
        jax.config.update('jax_compilation_cache_dir', str(PROJECT / '.cache/jax'))
        mesh = sharding.make_mesh(args.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        state, state_sharding, filter_ = init_state(args, config, mesh)
        rng = jax.random.key(args.seed + 1)
        if resuming:
            state, rng = base.restore_state(manager, state, filter_, rng)
        if not resuming:
            write_json(run_dir / 'config.json', current)
        start = int(jax.device_get(state.step))
        if start >= args.num_steps:
            raise ValueError('Requested total steps already reached')
        sampler = StepBatchSampler(len(selected['train']), args.batch_size, args.seed, args.num_steps, start)
        loader = loader_builder(selected['train'], **common, sampler=sampler)
        val_loader = loader_builder(selected['val'], **common, inference=True)
        compact_ids = jnp.asarray(grammar.compact_token_ids, dtype=jnp.int32)
        update = jax.jit(lambda key, st, batch: base.train_text_step(filter_, 1.0, compact_ids, key, st, batch),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        print(f'{version_label} group={args.group} devices={jax.devices()} trainable_params={base.count_state_params(state.params.filter(filter_))}')
        best_path = run_dir / 'best/metrics.json'
        best = read_json(best_path)['val']['episode_macro_joint_correct'] if best_path.exists() else -1.0
        losses, begun = [], time.monotonic()
        for batch in tqdm.tqdm(loader, total=args.num_steps - start, desc=f'{version_label} {args.group}'):
            rng, key = jax.random.split(rng)
            with sharding.set_mesh(mesh):
                state, info = update(key, state, base.batch_to_jax(batch, 'plan', data_sharding))
            step = int(jax.device_get(state.step))
            info = jax.device_get(info)
            losses.append(float(info['loss']))
            if step % args.log_interval == 0 or step == args.num_steps:
                value = {'step': step, 'plan_loss': float(np.mean(losses)), 'grad_norm': float(info['grad_norm']),
                         'elapsed_sec': time.monotonic() - begun}
                append(run_dir / 'metrics.jsonl', value)
                tqdm.tqdm.write(str(value))
                losses.clear()
            if step % args.save_interval == 0 or step == args.num_steps:
                base.save_state(manager, state, step, filter_, rng)
                manager.wait_until_finished()
            if step % args.eval_interval == 0 or step == args.num_steps:
                report, predictions = evaluator(state, val_loader, grammar)
                value = {'step': step, 'group': args.group, 'split': 'val', 'val': report}
                append(run_dir / 'metrics.jsonl', value)
                write_json(run_dir / f'val_predictions_{step}.json', predictions)
                if report['episode_macro_joint_correct'] > best:
                    best = report['episode_macro_joint_correct']
                    base.save_best_checkpoint(run_dir, state, value, filter_)
                tqdm.tqdm.write(json.dumps(value, ensure_ascii=False))
    finally:
        manager.wait_until_finished()
        manager.close()


if __name__ == '__main__':
    main()
