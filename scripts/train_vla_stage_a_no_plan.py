#!/usr/bin/env python3
"""Train task-specific no-Plan Stage A using each source task's data policy."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / 'src'), str(PROJECT_ROOT / 'openpi/src')]
SOURCE_ROOT = Path('/data1/qxh/tac_vla_new/tac_data/demon_data')

from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v7_11_datasets import DirectActionDataset

TASKS = {
    'book': ('v9_7', 'scripts.train_vla_stage_a_book_v9_5'),
    'sponge': ('v10_3', 'scripts.train_vla_stage_a_sponge_v10_1'),
    'bottle': ('v11_4', 'scripts.train_vla_stage_a_bottle_v11_2'),
}


def prompt_identity(task):
    root = PROJECT_ROOT / 'src/tactile_vla/vla'
    return {'task': task, 'version': TASKS[task][0],
            'condition': 'v7_11_current_tactile_no_plan',
            'prompt_code_sha256': sha256_file(root / 'v7_11_data.py'),
            'dataset_code_sha256': sha256_file(root / 'v7_11_datasets.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--task', choices=TASKS, required=True)
    args, remaining = parser.parse_known_args()
    version, module_name = TASKS[args.task]
    source = importlib.import_module(module_name)
    source.configure_version()
    trainer = source.trainer if hasattr(source, 'trainer') else source.previous.trainer
    source_parse = trainer.parse_args
    source_config = trainer.checkpoint_config_payload
    source_resume = trainer.validate_v4_resume_config
    source_protocol = trainer.selected_stage_a_protocol
    run_name = f'pi05_{args.task}_{version}_stage_a_direct_tactile'
    output = SOURCE_ROOT / args.task / 'outputs' / 'stage_a_action'
    for flag, value in {'--run-name': run_name, '--output-dir': output, '--num-steps': 10000,
                        '--save-interval': 10000, '--keep-period': 10000}.items():
        if not any(v == flag or v.startswith(flag + '=') for v in remaining):
            remaining.extend([flag, str(value)])

    def parse():
        parsed = source_parse(remaining)
        if (parsed.num_steps, parsed.save_interval, parsed.keep_period) != (10000, 10000, 10000):
            raise ValueError('no-Plan Stage A requires 10000 steps and save/keep interval 10000')
        return parsed

    def protocol(parsed):
        name, policy = source_protocol(parsed)
        return name, policy | {'num_steps': 10000, 'save_interval': 10000, 'keep_period': 10000}

    def config(parsed, identity):
        return source_config(parsed, identity) | {'no_plan_stage_a': prompt_identity(args.task)}

    def resume(saved, parsed):
        if saved.get('no_plan_stage_a') != prompt_identity(args.task):
            raise ValueError('Cannot resume a different task or direct-tactile prompt version')
        return source_resume(saved, parsed)

    trainer.parse_args = parse
    trainer.TactileVLAFrameDataset = DirectActionDataset
    trainer.checkpoint_config_payload = config
    trainer.validate_v4_resume_config = resume
    trainer.selected_stage_a_protocol = protocol
    trainer.main()


if __name__ == '__main__':
    main()
