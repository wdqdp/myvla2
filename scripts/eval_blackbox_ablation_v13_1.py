#!/usr/bin/env python3
"""Evaluate one fixed V13.1 run; never selects checkpoints using the test split."""
# ruff: noqa: E402
import argparse
import dataclasses
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('OPENPI_DATA_HOME', '/home/qxh/.cache/openpi')
os.environ.setdefault('HF_HOME', str(PROJECT / '.cache/huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', str(PROJECT / '.cache/huggingface/datasets'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', default='best', help='best (val-selected), or a saved numeric step')
    p.add_argument('--split', choices=('val', 'test'), default='test')
    extra = p.parse_args()
    from openpi.training import sharding
    from scripts.train_blackbox_ablation_v13_1 import components, validate_stage_a
    from tactile_vla.vla.blackbox_ablation_v13_1_data import read_json, write_json
    from tactile_vla.vla.blackbox_ablation_v13_1_runtime import evaluate, init_state, make_loader, restore_overlay
    run = extra.run_dir.resolve()
    saved = read_json(run / 'config.json')
    args = argparse.Namespace(**saved)
    for k in ('index_file', 'stage_a_checkpoint', 'output_dir'):
        setattr(args, k, Path(getattr(args, k)))
    validate_stage_a(args.stage_a_checkpoint)
    index, rows, config, tokenizer, grammar, norm = components(args)
    if index['training_data_hash'] != saved['training_data_hash']:
        raise ValueError('Evaluation data differs from training')
    mesh = sharding.make_mesh(args.fsdp_devices)
    state, _, _ = init_state(args, config, mesh)
    checkpoint = extra.checkpoint
    if checkpoint != 'best' and (not checkpoint.isdigit() or checkpoint != str(int(checkpoint))):
        p.error('--checkpoint must be best or a numeric step')
    folder = run / checkpoint
    path = folder / ('delta_params' if checkpoint == 'best' else 'params')
    state = dataclasses.replace(state, params=restore_overlay(state.params, path, include_rng=checkpoint != 'best'))
    selected = [r for r in rows if r['split'] == extra.split]
    loader = make_loader(selected, group=args.group, model_config=config, tokenizer=tokenizer,
        grammar=grammar, norm=norm, batch_size=args.batch_size, num_workers=args.num_workers, inference=True)
    report, predictions = evaluate(state, loader, grammar)
    output = run / f'{extra.split}_{checkpoint}'
    write_json(output.with_suffix('.metrics.json'), {'group': args.group, 'checkpoint': checkpoint,
               'split': extra.split, 'metrics': report})
    write_json(output.with_suffix('.predictions.json'), predictions)
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
