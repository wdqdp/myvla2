#!/usr/bin/env python3
"""Serve V7.7 with streamed phase-decision/failure events."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
from pathlib import Path
import socket
import sys
import time
import traceback
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"),
                str(PROJECT_ROOT / "openpi/src"),
                str(PROJECT_ROOT / "openpi/packages/openpi-client/src")]

import jax
import jax.numpy as jnp
import numpy as np
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames
from openpi_client import msgpack_numpy
from openpi.shared import nnx_utils, normalize
from tactile_vla.vla.structured_generation import constrained_greedy_generate_from_prefill
from tactile_vla.vla.v7_7_multitask_data import DATA_PROFILE
from tactile_vla.vla.v7_7_multitask_model import V77MultitaskModel
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE
from scripts import serve_tactile_vla_policy_v3 as v3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="V7.7 numeric step/full_params directory")
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-action-dim", type=int, default=7)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--need-recovery-threshold", type=float)
    parser.add_argument("--adjustment-end-threshold", type=float)
    parser.add_argument("--reasoning-max-token-len", type=int)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"))
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class V77Policy(v3.TactileVLAPolicyV3):
    def __init__(self, *, args, config, model_config, norm_stats):
        if config.get("data_profile") != DATA_PROFILE:
            raise ValueError(f"checkpoint data_profile must be {DATA_PROFILE!r}")
        if config.get("prompt_profile") != PROMPT_PROFILE:
            raise ValueError(f"checkpoint prompt_profile must be {PROMPT_PROFILE!r}")
        if bool(config.get("use_state_history")) or int(config.get("state_history_len", -1)) != 0:
            raise ValueError("V7.7 deployment requires the no-history Action Expert")
        old = v3.StageBV3Model
        old_prompt_loader = v3.load_checkpoint_prompt_profile
        requested_need_threshold = args.need_recovery_threshold
        if args.need_recovery_threshold is None:
            args.need_recovery_threshold = float(config.get("thresholds", {}).get("need_recovery", 0.5))
        v3.StageBV3Model = V77MultitaskModel
        v3.load_checkpoint_prompt_profile = lambda _config: PROMPT_PROFILE
        try:
            super().__init__(args=args, config=config, model_config=model_config, norm_stats=norm_stats)
        finally:
            v3.StageBV3Model = old
            v3.load_checkpoint_prompt_profile = old_prompt_loader
        self._adjustment_logits = nnx_utils.module_jit(self._model.adjustment_end_logits)
        thresholds = config.get("thresholds", {})
        self._need_threshold = float(
            requested_need_threshold if requested_need_threshold is not None
            else thresholds.get("need_recovery", 0.5)
        )
        self._adjustment_threshold = float(
            args.adjustment_end_threshold if args.adjustment_end_threshold is not None
            else thresholds.get("adjustment_end", 0.5)
        )
        self._metadata.update({
            "name": "tactile_vla_v7_7", "supports_streamed_phase_events": True,
            "supports_action_noise": True, "requires_action_noise": True,
            "need_recovery_threshold": self._need_threshold,
            "adjustment_end_threshold": self._adjustment_threshold,
            "phase_prompt_profile": PROMPT_PROFILE,
        })

    def _actions_with_noise(self, transformed, observation, noise):
        noise = np.asarray(noise, dtype=np.float32)
        expected = (self._model.backbone.action_horizon, self._model.backbone.action_dim)
        if noise.shape != expected or not np.isfinite(noise).all():
            raise ValueError(f"V7.7 execution requires finite action_noise {expected}, got {noise.shape}")
        actions = self._sample_actions(
            self._rng, observation, num_steps=self._num_inference_steps,
            noise=jnp.asarray(noise)[None, ...],
        )
        restored = self._output_transform(
            {"state": transformed["state"], "actions": np.asarray(actions[0])}
        )
        raw = np.asarray(actions[0], dtype=np.float32)
        result = np.asarray(restored["actions"], dtype=np.float32)[:, :self._output_action_dim]
        if raw.shape != expected or result.shape != (expected[0], self._output_action_dim):
            raise ValueError("V7.7 action output shape mismatch")
        if not np.isfinite(raw).all() or not np.isfinite(result).all():
            raise ValueError("V7.7 action inference produced non-finite values")
        return raw, result

    @staticmethod
    def _decision(logits, threshold):
        probs = np.asarray(jax.nn.softmax(logits[0]), dtype=np.float32)
        return bool(float(probs[1]) >= threshold), probs.tolist()

    def infer_events(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        mode = str(request.get("mode", "execution"))
        if mode != "phase":
            if mode in {"action", "execution"}:
                started = time.monotonic()
                inputs = dict(request)
                noise = inputs.pop("action_noise", None)
                for key in ("noise_seed", "noise_phase", "noise_index"):
                    inputs.pop(key, None)
                transformed, observation = self._prepare(inputs, mode="execution")
                raw, actions = self._actions_with_noise(transformed, observation, noise)
                yield {
                    "raw_model_actions": raw,
                    "actions": actions,
                    "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000.0},
                }
                return
            yield super().infer(request)
            return
        phase = str(request.get("phase", ""))
        request_id = str(request.get("request_id", ""))
        if phase not in {"execution", "adjustment"} or not request_id:
            raise ValueError("phase requests require phase=execution|adjustment and request_id")
        started = time.monotonic()
        _, observation = self._prepare(request, mode="assessment")
        if phase == "adjustment":
            value, probs = self._decision(self._adjustment_logits(observation), self._adjustment_threshold)
            yield {
                "event": "phase_decision", "request_id": request_id, "phase": phase,
                "adjustment_end": value, "adjustment_end_probs": probs,
                "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000.0},
            }
            return
        compact = jnp.asarray(self._failure_grammar.compact_token_ids, dtype=jnp.int32)
        logits, first, cache, mask, position = self._assessment_prefill(observation, compact)
        value, probs = self._decision(logits, self._need_threshold)
        yield {
            "event": "phase_decision", "request_id": request_id, "phase": phase,
            "need_recovery": value, "need_recovery_probs": probs,
            "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000.0},
        }
        if not value:
            return
        decode_started = time.monotonic()
        failure = constrained_greedy_generate_from_prefill(
            self._model.backbone, self._failure_grammar, logits=first, kv_cache=cache,
            prefix_mask=mask, semantic_position=position,
            step_fn=lambda _backbone, *values: self._generation_step(*values),
        )
        yield {
            "event": "failure_reason", "request_id": request_id, "phase": phase,
            "failure_reason": failure,
            "policy_timing": {"decode_ms": (time.monotonic() - decode_started) * 1000.0},
        }


class StreamingPolicyServer:
    def __init__(self, policy, host, port):
        self.policy, self.host, self.port = policy, host, port

    async def _health(self, connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                for event in self.policy.infer_events(request):
                    await websocket.send(packer.pack(event))
            except websockets.ConnectionClosed:
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                      reason="V7.7 inference error")
                raise

    def serve_forever(self):
        async def run():
            async with ws_server.serve(self._handler, self.host, self.port, compression=None,
                                       max_size=None, process_request=self._health) as server:
                await server.serve_forever()
        asyncio.run(run())


def main():
    args = parse_args()
    config = v3._find_config(args.checkpoint)
    if (args.checkpoint / "full_params").is_dir():
        args.checkpoint = args.checkpoint / "full_params"
    model_config = v3._model_config(args, config)
    norm_stats = None if args.no_norm else normalize.load(args.norm_stats_dir)
    policy = V77Policy(args=args, config=config, model_config=model_config, norm_stats=norm_stats)
    if args.dry_run:
        print(json.dumps(policy.metadata, indent=2, default=str))
        return
    logging.info("Serving V7.7 on %s (%s):%d", socket.gethostname(), args.host, args.port)
    StreamingPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
