#!/usr/bin/env python3
"""Serve V7.11 direct-tactile actions and two classification heads."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
from pathlib import Path
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src"),
                str(PROJECT_ROOT / "openpi/packages/openpi-client/src")]

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import websockets
import websockets.frames
from openpi.models import gemma
from openpi.models import tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils, normalize
from openpi_client import msgpack_numpy
from scripts import serve_tactile_vla_policy_v3 as v3
from scripts import serve_tactile_vla_v7_7 as shared
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.openpi_bridge import build_action_output_transform, build_structured_inference_transform, build_transform
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v7_7_multitask_model import V77MultitaskModel
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt
from tactile_vla.vla.v7_11_data import DATA_PROFILE, PROMPT_PROFILE, phase_prompt_without_plan
from tactile_vla.vla.v7_11_stage_a import checkpoint_prompt_identity

ROOT = Path('/data1/qxh/tac_vla_new/tac_data/demon_data/black_box')
DEFAULT_RUN = ROOT / 'outputs/multitask_v7_11/pi05_rotation_v7_11_direct_tactile_h100'
DEFAULT_NORM = ROOT / 'outputs/rotation_v4_1_new_env/norm_stats'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_RUN / '2001')
    parser.add_argument('--config', type=Path, help='Original V7.11 training config.json')
    parser.add_argument('--norm-stats-dir', type=Path, default=DEFAULT_NORM)
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
    full, export_file = step / 'full_params', step / 'v7_11_export.json'
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
        raise FileNotFoundError('V7.11 artifacts incomplete:\n  ' + '\n  '.join(missing))
    config, export = json.loads(config_file.read_text()), json.loads(export_file.read_text())
    expected = {
        'experiment_version': 'v7_11', 'data_profile': DATA_PROFILE, 'prompt_profile': PROMPT_PROFILE,
        'task_cycle': ['action', 'adjustment', 'need'], 'phase_prefill_protocol': 'classification_only',
        'recovery_action_condition': 'current_tactile_no_plan_no_recovery_history',
        'phase_condition': 'current_tactile_h100_no_plan', 'use_state_history': False,
        'state_history_len': 0, 'history_hidden_dim': 0, 'action_horizon': 30, 'action_dim': 32,
        'state_history_dim': 7, 'state_history_fps': 30.0,
    }
    mismatch = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if mismatch:
        raise ValueError(f'V7.11 config mismatch: {mismatch}')
    if config.get('failure_grammar') != [] or config.get('recovery_grammar') != []:
        raise ValueError('V7.11 must use classification-only training without text grammars')
    stage_a = config.get('stage_a_initialization_identity', {}).get('prompt_identity', {})
    if stage_a != checkpoint_prompt_identity():
        raise ValueError('V7.11 Stage A prompt/data code identity does not match this workspace')
    if export.get('step') != int(step.name) or export.get('default_deployment') != 'full_params':
        raise ValueError('V7.11 export step/format mismatch')
    full_sha = export.get('exports', {}).get('full_params', {}).get('parameter_tree_sha256')
    if not isinstance(full_sha, str) or len(full_sha) != 64 or any(c not in '0123456789abcdef' for c in full_sha):
        raise ValueError('V7.11 export is missing a valid full_params SHA256')
    norm_sha = sha256_file(norm_file)
    artifact = config.get('artifact_identity', {})
    if artifact.get('v4_norm_stats_sha256') != norm_sha:
        raise ValueError('V7.11 norm stats do not match training identity')
    args.checkpoint, args.norm_stats_dir = full, norm_file.parent
    return config, {'checkpoint_step': int(step.name), 'checkpoint': str(full),
                    'full_params_sha256': full_sha, 'norm_stats_sha256': norm_sha,
                    'config_file': str(config_file), 'export_file': str(export_file)}


def restore_rng_keys(reference, restored):
    at.check_pytree_equality(expected=reference, got=restored)

    def convert(path, expected, value):
        dtype = getattr(expected, 'dtype', None)
        if dtype is None or not jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return value
        actual_dtype = getattr(value, 'dtype', None)
        if actual_dtype is None or not jax.dtypes.issubdtype(actual_dtype, jax.dtypes.prng_key):
            raw = jnp.asarray(value)
            if raw.dtype != jnp.uint32:
                raise ValueError(f'Invalid RNG key at {jax.tree_util.keystr(path)}')
            # The reference came from nnx.eval_shape, so it is a
            # ShapeDtypeStruct rather than a concrete key accepted by key_impl.
            impl = getattr(dtype, '_impl', None)
            if impl is None:
                raise ValueError(f'Cannot determine RNG implementation at {jax.tree_util.keystr(path)}')
            value = jax.random.wrap_key_data(raw, impl=impl.name)
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f'RNG key shape/type mismatch at {jax.tree_util.keystr(path)}')
        return value

    return jax.tree_util.tree_map_with_path(convert, reference, restored)


class V711Policy(shared.V77Policy):
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
        if parameter_tree_sha256(nnx.state(self._model)) != identity['full_params_sha256']:
            raise ValueError('Restored V7.11 parameter SHA256 differs from export')
        self._sample_actions = nnx_utils.module_jit(self._model.backbone.sample_actions)
        self._need_logits = nnx_utils.module_jit(self._model.need_recovery_logits)
        self._adjustment_logits = nnx_utils.module_jit(self._model.adjustment_end_logits)
        self._rng = jax.random.key(2)
        self._num_inference_steps, self._output_action_dim = args.num_inference_steps, 7
        self._need_threshold, self._adjustment_threshold = args.need_recovery_threshold, args.adjustment_end_threshold
        self._metadata = {
            'name': 'tactile_vla_v7_11', 'data_profile': DATA_PROFILE, 'phase_prompt_profile': PROMPT_PROFILE,
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
        mode = request.get('mode')
        started = time.monotonic()
        if mode in ('execution', 'action'):
            prompt = str(request.get('prompt', ''))
            if not (prompt.startswith('Mode: execution. Task: ') or
                    (prompt.startswith('Mode: adjustment. Task: ') and '\nTouch: ' in prompt)):
                raise ValueError('V7.11 action prompt must be execution or adjustment with current Touch')
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
            raise ValueError('V7.11 accepts only action or identified phase requests')
        prompt = str(request.get('prompt', ''))
        if 'Recovery plan:' in prompt or '\nTouch: ' not in prompt or '\nState history H100 sampled to 11 points: ' not in prompt:
            raise ValueError('V7.11 phase prompt must contain Touch and H100 without a plan')
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


class SerializedServer(shared.StreamingPolicyServer):
    def __init__(self, policy, host, port):
        super().__init__(policy, host, port)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='v711-model')

    @staticmethod
    def _next(events):
        try:
            return True, next(events)
        except StopIteration:
            return False, None

    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        loop = asyncio.get_running_loop()
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                events = self.policy.infer_events(request)
                while True:
                    exists, event = await loop.run_in_executor(self._executor, self._next, events)
                    if not exists:
                        break
                    await websocket.send(packer.pack(event))
            except websockets.ConnectionClosed:
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason='V7.11 inference error')
                raise

    def serve_forever(self):
        try:
            super().serve_forever()
        finally:
            self._executor.shutdown(wait=True)


def warm_up(policy):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    base = {'observation/image': image, 'observation/wrist_image': image,
            'observation/state': np.zeros(7, dtype=np.float32)}
    prompt = phase_prompt_without_plan(build_phase_prompt(
        instruction='dry run', tactile_caption='Touch[rotation=none]', recovery_plan='none',
        qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32),
    ))
    requests = [base | {'mode': 'execution', 'prompt': 'Mode: execution. Task: dry run.',
                        'action_noise': np.zeros((30, 32), dtype=np.float32)}]
    requests += [base | {'mode': 'phase', 'phase': phase, 'request_id': f'warmup-{phase}', 'prompt': prompt}
                 for phase in ('execution', 'adjustment')]
    for request in requests:
        list(policy.infer_events(request))
        logging.info('V7.11 warm-up complete: %s', request.get('phase', 'action'))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)
    args = parse_args()
    config, identity = inspect_artifacts(args)
    logging.info('V7.11 checkpoint=%s need=%s end=%s', args.checkpoint,
                 args.need_recovery_threshold, args.adjustment_end_threshold)
    if args.validate_only:
        print(json.dumps(identity, indent=2, ensure_ascii=False))
        return
    policy = V711Policy(args, config, identity)
    warm_up(policy)
    if args.dry_run:
        print(json.dumps(policy.metadata, indent=2, default=str))
        return
    SerializedServer(policy, args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
