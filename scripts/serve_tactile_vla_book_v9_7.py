#!/usr/bin/env python3
"""Serve Book V9.7 direct-tactile actions and two classification heads."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src"),
                str(PROJECT_ROOT / "openpi/packages/openpi-client/src")]

import jax
import numpy as np
from flax import nnx
from openpi.models import gemma
from openpi.models import tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils, normalize
from scripts import serve_tactile_vla_policy_v3 as v3
from scripts import serve_tactile_vla_v7_7 as shared
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.openpi_bridge import build_action_output_transform, build_structured_inference_transform, build_transform
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v7_7_multitask_model import V77MultitaskModel
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt
from tactile_vla.vla.v7_11_data import DATA_PROFILE, PROMPT_PROFILE, phase_prompt_without_plan
from scripts.serve_tactile_vla_v7_11 import SerializedServer, restore_rng_keys
from tactile_vla.vla.book_v9_7_runtime import (
    BOOK_INSTRUCTION, DEFAULT_RUN, DEFAULT_NORM_DIR, SERVER_NAME,
    validate_training_config, validate_request,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_RUN / '2000')
    parser.add_argument('--config', type=Path, help='Original Book V9.7 training config.json')
    parser.add_argument('--norm-stats-dir', type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--num-inference-steps', type=int, default=10)
    parser.add_argument('--need-recovery-threshold', type=float, default=0.5)
    parser.add_argument('--adjustment-end-threshold', type=float, default=0.9)
    parser.add_argument('--precision', choices=('auto', 'bfloat16', 'float32'), default='auto')
    parser.add_argument('--validate-only', action='store_true', help='Inspect artifacts without loading model weights')
    parser.add_argument('--dry-run', action='store_true', help='Restore model, warm up and exit')
    args = parser.parse_args(argv)
    if args.validate_only and args.dry_run:
        parser.error('--validate-only and --dry-run cannot be combined')
    if not 1 <= args.port <= 65535 or args.num_inference_steps <= 0:
        parser.error('port and inference steps must be positive')
    for name in ('need_recovery_threshold', 'adjustment_end_threshold'):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f'{name} must be finite and in [0,1]')
    return args


def inspect_artifacts(args):
    path = args.checkpoint.expanduser().resolve()
    step = path.parent if path.name == 'full_params' else path
    if not step.name.isdecimal() or int(step.name) <= 0:
        raise ValueError('checkpoint must be a numeric step or its full_params directory')
    full, export_file = step / 'full_params', step / 'book_v9_7_export.json'
    candidates = [args.config.expanduser().resolve()] if args.config else [step.parent / 'config.json', step / 'config.json']
    config_file = next((p for p in candidates if p.is_file()), None)
    norm_file = args.norm_stats_dir.expanduser().resolve() / 'norm_stats.json'
    missing = [str(p) for p in (full / '_METADATA', export_file, norm_file) if not p.is_file()]
    if config_file is None:
        missing.append('original config.json: ' + ', '.join(str(p) for p in candidates))
    incomplete = list(full.rglob('*.incomplete')) if full.is_dir() else []
    if incomplete:
        missing.append(f'{len(incomplete)} unfinished files under {full}')
    if missing:
        raise FileNotFoundError('Book V9.7 artifacts incomplete:\n  ' + '\n  '.join(missing))
    config, export = json.loads(config_file.read_text()), json.loads(export_file.read_text())
    norm_sha = sha256_file(norm_file)
    validate_training_config(config, norm_sha)
    if export.get('step') != int(step.name) or export.get('default_deployment') != 'full_params':
        raise ValueError('Book V9.7 export step/format mismatch')
    if export.get('resume_supported') is not False:
        raise ValueError('Book V9.7 requires the inference-only full_params export')
    args.checkpoint, args.norm_stats_dir = full, norm_file.parent
    return config, {'checkpoint_step': int(step.name), 'checkpoint': str(full),
                    'norm_stats_sha256': norm_sha,
                    'training_data_hash': config['artifact_identity']['book_v9_7_training_data_hash'],
                    'config_file': str(config_file), 'export_file': str(export_file)}


class BookV97Policy(shared.V77Policy):
    def __init__(self, args, config, identity):
        args.reasoning_max_token_len = None
        model_config = v3._model_config(args, config)
        stats = normalize.load(args.norm_stats_dir)
        tok = tokenizer.PaligemmaTokenizer(int(config.get('reasoning_max_token_len', 320)))
        self._regular_transform = build_transform(model_config, norm_stats=stats, use_quantile_norm=True,
                                                  delta_action_dims=7)
        self._assessment_transform = build_structured_inference_transform(
            model_config, tokenizer=tok, max_len=model_config.max_token_len, norm_stats=stats, use_quantile_norm=True,
        )
        self._output_transform = build_action_output_transform(norm_stats=stats, use_quantile_norm=True,
                                                               delta_action_dims=7)

        def create():
            backbone = model_config.create(jax.random.key(0))
            return V77MultitaskModel(backbone, paligemma_width=gemma.get_config(model_config.paligemma_variant).width,
                                     need_hidden_dim=int(config.get('need_hidden_dim', 512)),
                                     need_dropout=float(config.get('need_dropout', 0.1)),
                                     rngs=nnx.Rngs(jax.random.key(1)))

        abstract = nnx.eval_shape(create)
        graph, state = nnx.split(abstract)
        reference = state.to_pure_dict()
        params = restore_rng_keys(reference, shared._restore_exported_full_params(args.checkpoint))
        at.check_pytree_equality(expected=reference, got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        self._model = nnx.merge(graph, state)
        self._model.eval()
        # This trainer exports no parameter digest. Compute a runtime identity
        # after Orbax restore; do not claim comparison against a training digest.
        identity = identity | {'full_params_sha256': parameter_tree_sha256(nnx.state(self._model))}
        self._sample_actions = nnx_utils.module_jit(self._model.backbone.sample_actions)
        self._need_logits = nnx_utils.module_jit(self._model.need_recovery_logits)
        self._adjustment_logits = nnx_utils.module_jit(self._model.adjustment_end_logits)
        self._rng = jax.random.key(2)
        self._num_inference_steps, self._output_action_dim = args.num_inference_steps, 7
        self._need_threshold, self._adjustment_threshold = args.need_recovery_threshold, args.adjustment_end_threshold
        self._metadata = {
            'name': SERVER_NAME, 'experiment_version': 'book_v9_7',
            'instruction': BOOK_INSTRUCTION, 'classification_qpos_policy': 'raw_no_gripper_remap',
            'tactile_label_fields': ['area', 'fx_state', 'fy_state', 'fz_state', 'fz_bias', 'rotation'],
            'tactile_label_schema': 'tactile_multifield_v4_masked',
            'requires_captioner': True, 'requires_tactile_topics': True, 'data_profile': DATA_PROFILE, 'phase_prompt_profile': PROMPT_PROFILE,
            'phase_prefill_protocol': 'classification_only', 'supports_streamed_phase_events': True,
            'supports_action_noise': True, 'requires_action_noise': True,
            'supports_failure_generation': False, 'supports_recovery_generation': False,
            'action_horizon': 30, 'action_dim': 32, 'output_action_dim': 7,
            'use_state_history': False, 'state_history_len': 0,
            'need_recovery_threshold': self._need_threshold, 'adjustment_end_threshold': self._adjustment_threshold,
            **identity,
        }

    @property
    def metadata(self):
        return self._metadata

    def infer_events(self, request):
        validate_request(request)
        mode = request.get('mode')
        started = time.monotonic()
        if mode in ('execution', 'action'):
            inputs = dict(request)
            noise = inputs.pop('action_noise', None)
            for key in ('noise_seed', 'noise_phase', 'noise_index'):
                inputs.pop(key, None)
            transformed = self._regular_transform({k: v for k, v in inputs.items() if k != 'mode'})
            observation = v3.TactileVLAPolicyV3._observation(transformed)
            raw, actions = self._actions_with_noise(transformed, observation, noise)
            yield {'raw_model_actions': raw, 'actions': actions,
                   'policy_timing': {'infer_ms': (time.monotonic() - started) * 1000}}
            return
        if mode != 'phase' or request.get('phase') not in ('execution', 'adjustment') or not request.get('request_id'):
            raise ValueError('Book V9.7 accepts only action or identified phase requests')
        transformed = self._assessment_transform({k: v for k, v in request.items() if k != 'mode'})
        observation = v3.TactileVLAPolicyV3._observation(transformed)
        phase = request['phase']
        logits = self._need_logits(observation) if phase == 'execution' else self._adjustment_logits(observation)
        threshold = self._need_threshold if phase == 'execution' else self._adjustment_threshold
        decision, probabilities = self._decision(logits, threshold)
        field = 'need_recovery' if phase == 'execution' else 'adjustment_end'
        yield {'event': 'phase_decision', 'request_id': request['request_id'], 'phase': phase,
               field: decision, f'{field}_probs': probabilities,
               'policy_timing': {'infer_ms': (time.monotonic() - started) * 1000}}


def warm_up(policy):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    base = {'observation/image': image, 'observation/wrist_image': image,
            'observation/state': np.zeros(7, dtype=np.float32)}
    prompt = phase_prompt_without_plan(build_phase_prompt(
        instruction=BOOK_INSTRUCTION, tactile_caption=(
            'Touch[area=none; Fx=near_zero; Fy=near_zero; Fz=near_zero; Fz_bias=balanced; rotation=none]'), recovery_plan='none',
        qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32),
    ))
    requests = [base | {'mode': 'execution', 'prompt': f'Mode: execution. Task: {BOOK_INSTRUCTION}',
                        'action_noise': np.zeros((30, 32), dtype=np.float32)}]
    requests += [base | {'mode': 'phase', 'phase': phase, 'request_id': f'warmup-{phase}', 'prompt': prompt}
                 for phase in ('execution', 'adjustment')]
    for request in requests:
        list(policy.infer_events(request))
        logging.info('Book V9.7 warm-up complete: %s', request.get('phase', 'action'))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)
    args = parse_args()
    config, identity = inspect_artifacts(args)
    logging.info('Book V9.7 checkpoint=%s need=%s end=%s', args.checkpoint,
                 args.need_recovery_threshold, args.adjustment_end_threshold)
    if args.validate_only:
        print(json.dumps(identity, indent=2, ensure_ascii=False))
        return
    policy = BookV97Policy(args, config, identity)
    warm_up(policy)
    if args.dry_run:
        print(json.dumps(policy.metadata, indent=2, default=str))
        return
    SerializedServer(policy, args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
