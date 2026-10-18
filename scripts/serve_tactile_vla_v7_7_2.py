#!/usr/bin/env python3
"""Serve V7.7.2 with startup compilation warmup and a dedicated model worker."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import serve_tactile_vla_v7_7 as server
import jax
import jax.numpy as jnp
import numpy as np
import websockets
from openpi_client import msgpack_numpy
from tactile_vla.common.labels_v4 import neutral_caption
from tactile_vla.common.labels import LABEL_SCHEMA_VERSION
from tactile_vla.vla.prompts import build_phase_prompt as action_prompt
from tactile_vla.vla.prompts import build_recovery_prompt
from tactile_vla.vla.v7_7_2_multitask_data import DATA_PROFILE
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt


def _warm_decode_lengths(policy, grammar, cache, mask, position, *, label):
    """Compile every reachable growing-cache length, independent of greedy output."""
    sequence = max((grammar.sequence_for_text(text) for text in grammar.texts), key=len)
    compact = jnp.asarray(grammar.compact_token_ids, dtype=jnp.int32)
    # A completed output ends at EOS; runtime never feeds its last token back.
    for index, token in enumerate(sequence[:-1]):
        started = time.monotonic()
        logits, cache = policy._generation_step(
            jnp.asarray([token], dtype=jnp.int32),
            compact,
            cache,
            mask,
            position + index,
        )
        jax.block_until_ready((logits, cache))
        logging.info(
            "[WARMUP] %s decode %d/%d ready (%.2fs)", label, index + 1, len(sequence) - 1, time.monotonic() - started
        )
    return len(sequence) - 1


def warm_up(policy):
    """Warm each distinct compiled path once, plus its changing decode-cache shapes."""
    started = time.monotonic()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    common = {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/state": np.zeros(7, dtype=np.float32),
    }
    instruction = "Pick up and transfer the object stably."
    caption = neutral_caption(schema_version=LABEL_SCHEMA_VERSION)
    example_plan = "recovery_plan=move horizontally right moderately, move vertically none moderately."
    # Execution and adjustment share this function and the same padded shapes.
    step_started = time.monotonic()
    prompt = action_prompt(phase="execution", instruction=instruction, recovery_plan="", prompt_profile="phase_v2")
    events = list(
        policy.infer_events(
            common
            | {
                "mode": "execution",
                "prompt": prompt,
                "action_noise": np.zeros((30, 32), dtype=np.float32),
            }
        )
    )
    if len(events) != 1 or np.asarray(events[0]["actions"]).shape != (30, 7):
        raise ValueError("V7.7.2 action warmup returned an invalid shape")
    action_seconds = time.monotonic() - step_started
    logging.info("[WARMUP] shared execution/adjustment action ready (%.2fs)", action_seconds)

    phase_prompt = build_phase_prompt(
        instruction=instruction,
        tactile_caption=caption,
        recovery_plan="",
        qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32),
    )
    phase_request = common | {"mode": "phase", "prompt": phase_prompt}
    _, observation = policy._prepare(phase_request, mode="assessment")
    compact = jnp.asarray(policy._failure_grammar.compact_token_ids, dtype=jnp.int32)
    step_started = time.monotonic()
    need_logits, logits, cache, mask, position = policy._assessment_prefill(observation, compact)
    jax.block_until_ready((need_logits, logits, cache))
    policy._decision(need_logits, policy._need_threshold)
    logging.info("[WARMUP] need/failure shared prefill ready (%.2fs)", time.monotonic() - step_started)
    failure_steps = _warm_decode_lengths(policy, policy._failure_grammar, cache, mask, position, label="failure")
    del need_logits, logits, cache, mask, position

    step_started = time.monotonic()
    adjusted_prompt = build_phase_prompt(
        instruction=instruction,
        tactile_caption=caption,
        recovery_plan=example_plan,
        qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32),
    )
    adjustment = list(
        policy.infer_events(
            common
            | {
                "mode": "phase",
                "phase": "adjustment",
                "request_id": "warmup-adjustment",
                "prompt": adjusted_prompt,
            }
        )
    )
    if len(adjustment) != 1 or adjustment[0].get("event") != "phase_decision":
        raise ValueError("V7.7.2 adjustment head warmup returned an invalid event")
    logging.info("[WARMUP] adjustment_end ready (%.2fs)", time.monotonic() - step_started)

    # All memory lengths are padded to the same reasoning buffer: one prefill
    # suffices. Decode-cache lengths still change and each needs compilation.
    memory = [{"recovery_plan": "initial plan", "failure_reason": "failure_reason=rotate right,grasp appropriate."}]
    prompt = build_recovery_prompt(
        instruction=instruction,
        failed_tactile_caption=caption,
        failure_recovery_memory=memory,
        prompt_profile="minimal_v1",
    )
    _, observation = policy._prepare(common | {"prompt": prompt}, mode="reasoning")
    step_started = time.monotonic()
    compact = jnp.asarray(policy._plan_grammar.compact_token_ids, dtype=jnp.int32)
    logits, cache, mask, position = policy._generation_prefill(observation, compact)
    jax.block_until_ready((logits, cache))
    logging.info("[WARMUP] plan prefill ready (%.2fs)", time.monotonic() - step_started)
    plan_steps = _warm_decode_lengths(policy, policy._plan_grammar, cache, mask, position, label="plan")
    del logits, cache, mask, position
    return {
        "seconds": time.monotonic() - started,
        "action_seconds": action_seconds,
        "failure_decode_steps": failure_steps,
        "plan_decode_steps": plan_steps,
    }


def _next_event(iterator):
    # StopIteration cannot be propagated through an asyncio Future.
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


class WorkerStreamingPolicyServer(server.StreamingPolicyServer):
    """Keep sockets responsive and send a need decision before decoding failure."""

    def __init__(self, policy, host, port):
        super().__init__(policy, host, port)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v7-7-2-model")

    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        loop = asyncio.get_running_loop()
        iterator = None
        try:
            while True:
                request = msgpack_numpy.unpackb(await websocket.recv())
                iterator = iter(self.policy.infer_events(request))
                while True:
                    finished, event = await loop.run_in_executor(self._executor, _next_event, iterator)
                    if finished:
                        break
                    await websocket.send(packer.pack(event))
                iterator = None
        except websockets.ConnectionClosed:
            return
        except Exception:
            logging.exception("V7.7.2 inference error")
            await websocket.send(traceback.format_exc())
            await websocket.close(code=1011, reason="V7.7.2 inference error")
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                await loop.run_in_executor(self._executor, iterator.close)

    def serve_forever(self):
        try:
            super().serve_forever()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    server.DATA_PROFILE = DATA_PROFILE
    args = server.parse_args()
    config = server.v3._find_config(args.checkpoint)
    if (args.checkpoint / "full_params").is_dir():
        args.checkpoint = args.checkpoint / "full_params"
    policy = server.V77Policy(
        args=args,
        config=config,
        model_config=server.v3._model_config(args, config),
        norm_stats=None if args.no_norm else server.normalize.load(args.norm_stats_dir),
    )
    logging.info("[WARMUP] Starting V7.7.2 compilation before accepting robot connections")
    report = warm_up(policy)
    policy._metadata.update(
        {
            "startup_warmup_complete": True,
            "startup_warmup": report,
            "worker_policy": "single_model_worker_stream_events_before_continuation",
        }
    )
    logging.info("[WARMUP] All V7.7.2 paths ready: %s", json.dumps(report))
    if args.dry_run:
        print(json.dumps(policy.metadata, indent=2, default=str))
        return
    logging.info("V7.7.2 server listening on %s:%d", args.host, args.port)
    WorkerStreamingPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
