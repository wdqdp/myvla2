#!/usr/bin/env python3
"""Test V13.4 feedback-only models; same experiment-only qpos masking as training."""
# ruff: noqa: E402
import argparse
import dataclasses
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT / 'openpi/src')]


def validate_protocol(saved):
    from tactile_vla.vla import blackbox_ablation_v13_4_data as data
    if (saved.get('experiment_version') != data.SCHEMA
            or saved.get('feedback_input_protocol') != data.FEEDBACK_PROTOCOL
            or saved.get('qpos_input_protocol') != data.QPOS_PROTOCOL or saved.get('qpos_scope') != data.QPOS_SCOPE
            or saved.get('group') not in data.GROUPS):
        raise ValueError('Expected V13.4 feedback-only checkpoint with experiment-only qpos masking')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', default='best')
    p.add_argument('--split', choices=('val', 'test'), default='test')
    extra = p.parse_args()
    checkpoint = extra.checkpoint
    if checkpoint != 'best' and (not checkpoint.isdigit() or checkpoint != str(int(checkpoint))):
        p.error('--checkpoint must be best or a numeric step')
    from scripts.train_blackbox_ablation_v13_4 import common, components
    from openpi.training import sharding
    from tactile_vla.vla.artifacts import sha256_file
    from tactile_vla.vla.blackbox_ablation_v13_1_data import read_json, write_json
    from tactile_vla.vla.blackbox_ablation_v13_1_runtime import init_state, restore_overlay
    from tactile_vla.vla.blackbox_ablation_v13_4_runtime import evaluate, make_loader

    run = extra.run_dir.resolve()
    saved = read_json(run / 'config.json')
    validate_protocol(saved)
    args = argparse.Namespace(**saved)
    for k in ('index_file', 'stage_a_checkpoint', 'output_dir'):
        setattr(args, k, Path(getattr(args, k)))
    common.validate_stage_a(args.stage_a_checkpoint)
    if sha256_file(args.stage_a_checkpoint / 'params/_METADATA') != saved['stage_a_metadata_sha256']:
        raise ValueError('Stage A metadata changed')
    index, rows, config, tokenizer, grammar, norm = components(args)
    if index['training_data_hash'] != saved['training_data_hash']:
        raise ValueError('Evaluation data differs from training')
    state, _, _ = init_state(args, config, sharding.make_mesh(args.fsdp_devices))
    path = run / checkpoint / ('delta_params' if checkpoint == 'best' else 'params')
    state = dataclasses.replace(state, params=restore_overlay(state.params, path, include_rng=checkpoint != 'best'))
    selected = [r for r in rows if r['split'] == extra.split]
    loader = make_loader(selected, group=args.group, model_config=config, tokenizer=tokenizer,
        grammar=grammar, norm=norm, batch_size=args.batch_size, num_workers=args.num_workers, inference=True)
    report, predictions = evaluate(state, loader, grammar)
    output = run / f'{extra.split}_{checkpoint}'
    step = read_json(run / 'best/metrics.json')['step'] if checkpoint == 'best' else int(checkpoint)
    write_json(output.with_suffix('.metrics.json'), {k: saved[k] for k in (
        'group', 'experiment_version', 'feedback_input_protocol', 'qpos_input_protocol', 'qpos_scope')}
        | {'checkpoint': checkpoint, 'checkpoint_step': step, 'split': extra.split,
           'input_audit': loader.qpos_audit, 'metrics': report})
    write_json(output.with_suffix('.predictions.json'), predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
