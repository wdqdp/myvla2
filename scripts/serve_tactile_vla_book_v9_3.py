#!/usr/bin/env python3
"""Serve Book V9.3 five tasks with streamed decisions and serialized worker inference."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src"),
                str(PROJECT_ROOT / "openpi/packages/openpi-client/src")]

import numpy as np
import websockets
from openpi_client import msgpack_numpy
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.book_v9_3_multitask_data import DATA_PROFILE
from tactile_vla.vla.book_v9_3_runtime import (
    BOOK_INSTRUCTION, DEFAULT_NORM_DIR, DEFAULT_RUN, HISTORY_OFFSETS, MEMORY_POLICY,
    SERVER_NAME, resolve_thresholds, validate_training_config,
)
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt, build_phase_prompt as action_prompt
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt
from scripts import serve_tactile_vla_v7_7 as base


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "4000")
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument("--thresholds-file", type=Path, help="Optional calibrated thresholds; omitted by default")
    parser.add_argument("--need-recovery-threshold", type=float,
                        help="Override need_recovery threshold; defaults to 0.5 without a thresholds file")
    parser.add_argument("--adjustment-end-threshold", type=float,
                        help="Override adjustment_end threshold; defaults to 0.5 without a thresholds file")
    parser.add_argument("--captioner-checkpoint-sha256", required=True,
                        help="SHA256 of the captioner used by the robot (runtime identity, not training provenance)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--validate-only", action="store_true", help="Validate artifacts without restoring weights")
    parser.add_argument("--dry-run", action="store_true", help="Restore weights and warm up all five tasks, then exit")
    args = parser.parse_args(argv)
    args.output_action_dim = 7
    args.reasoning_max_token_len = None
    args.no_norm = False
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive")
    sha = args.captioner_checkpoint_sha256
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        parser.error("--captioner-checkpoint-sha256 must be a lowercase 64-character SHA256")
    return args


def inspect_artifacts(args):
    path = args.checkpoint.expanduser().resolve()
    step_dir = path.parent if path.name == "full_params" else path
    full_params = step_dir / "full_params"
    if not full_params.is_dir():
        raise FileNotFoundError(full_params)
    config = base.v3._find_config(step_dir)
    norm_sha = sha256_file(args.norm_stats_dir / "norm_stats.json")
    validate_training_config(config, norm_sha)
    export_path = step_dir / "book_v9_3_export.json"
    export = json.loads(export_path.read_text())
    step = int(step_dir.name)
    full_sha = export["exports"]["full_params"]["parameter_tree_sha256"]
    if export.get("step") != step or export.get("default_deployment") != "full_params":
        raise ValueError("Book V9.3 export step/format mismatch")
    if not isinstance(full_sha, str) or len(full_sha) != 64:
        raise ValueError("Book V9.3 full_params export digest is missing")
    args.checkpoint = full_params
    return config, {
        "checkpoint_step": step, "full_params_sha256": full_sha, "norm_stats_sha256": norm_sha,
        "training_data_hash": config["artifact_identity"]["book_v9_3_training_data_hash"],
        "export_metadata": str(export_path),
    }


def resolve_deployment(args, identity):
    calibration = json.loads(args.thresholds_file.read_text()) if args.thresholds_file else None
    thresholds, overrides = resolve_thresholds(
        calibration=calibration, step=identity["checkpoint_step"],
        full_params_sha=identity["full_params_sha256"], norm_sha=identity["norm_stats_sha256"],
        training_data_hash=identity["training_data_hash"], need_override=args.need_recovery_threshold,
        adjustment_override=args.adjustment_end_threshold,
    )
    args.need_recovery_threshold = thresholds["need_recovery"]
    args.adjustment_end_threshold = thresholds["adjustment_end"]
    return {
        "thresholds": thresholds, "threshold_manual_overrides": overrides,
        "thresholds_status": (
            "explicit_manual_override" if any(overrides.values())
            else "calibrated_on_book_val" if calibration is not None else "default_0_5"
        ),
        "thresholds_file": str(args.thresholds_file.resolve()) if args.thresholds_file else None,
        "accepted_for_robot": False,
    }


class BookV93Policy(base.V77Policy):
    def __init__(self, *, args, config, model_config, norm_stats, identity, deployment):
        old_profile = base.DATA_PROFILE
        base.DATA_PROFILE = DATA_PROFILE
        try:
            super().__init__(args=args, config=config, model_config=model_config, norm_stats=norm_stats)
        finally:
            base.DATA_PROFILE = old_profile
        actual_sha = parameter_tree_sha256(base.v3.nnx.state(self._model))
        if actual_sha != identity["full_params_sha256"]:
            raise ValueError("Restored Book V9.3 parameter-tree SHA256 differs from export")
        self._metadata.update(identity | deployment | {
            "name": SERVER_NAME, "instruction": BOOK_INSTRUCTION, "action_prompt_profile": "phase_v2",
            "qpos_h100_sample_offsets": HISTORY_OFFSETS, "max_memory_pairs": 1,
            "memory_policy": MEMORY_POLICY, "max_supported_attempts": None,
            "classification_qpos_policy": "raw_no_gripper_remap",
            "episode_start_padding": "left_pad_episode_frame_0", "captioner_window_size": 30,
            "captioner_checkpoint_sha256": args.captioner_checkpoint_sha256,
            "captioner_identity_scope": "runtime_client_server_only_training_provenance_unverified",
            "worker_policy": "single_model_worker_stream_events_before_continuation",
        })

    def infer_events(self, request):
        prompt = str(request.get("prompt", ""))
        if BOOK_INSTRUCTION not in prompt:
            raise ValueError("Book V9.3 requires its original book instruction")
        if any(key in request for key in ("observation/state_history", "observation/state_history_mask")):
            raise ValueError("Book V9.3 must not receive continuous state_history arrays")
        mode = request.get("mode", "execution")
        if mode == "phase":
            if not prompt.startswith("Mode: phase\n") or "State history H100 sampled to 11 points: " not in prompt:
                raise ValueError("Book V9.3 requires the V7.7 H100 phase prompt")
        elif mode == "reasoning":
            if not prompt.startswith("Mode: reasoning.") or "Failure-recovery memory: " not in prompt:
                raise ValueError("Book V9.3 requires the minimal_v1 plan prompt")
            memory_text = prompt.split("Failure-recovery memory: ", 1)[1]
            if " | " in memory_text or memory_text.count("failure_reason=") != 1:
                raise ValueError("Book V9.3 plan requests require exactly one failed-plan/reason pair")
        elif mode not in {"action", "execution"}:
            raise ValueError(f"Unsupported Book V9.3 inference mode: {mode}")
        yield from super().infer_events(request)


def load_policy(args, config, identity, deployment):
    model_config = base.v3._model_config(args, config)
    return BookV93Policy(
        args=args, config=config, model_config=model_config,
        norm_stats=base.normalize.load(args.norm_stats_dir), identity=identity, deployment=deployment,
    )


def warm_up(policy):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    common = {"observation/image": image, "observation/wrist_image": image,
              "observation/state": np.zeros(7, dtype=np.float32)}
    caption = "Touch[area=none; Fx=near_zero; Fy=near_zero; Fz=near_zero; rotation=none]"
    phase_prompt = build_phase_prompt(instruction=BOOK_INSTRUCTION, tactile_caption=caption,
                                     recovery_plan="", qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32))
    execution_prompt = action_prompt(phase="execution", instruction=BOOK_INSTRUCTION,
                                     recovery_plan="", prompt_profile="phase_v2")
    # Force the conditional failure branch for warm-up only, without changing runtime metadata.
    threshold = policy._need_threshold
    try:
        policy._need_threshold = 0.0
        execution = list(policy.infer_events(common | {"mode": "phase", "phase": "execution",
                                                     "request_id": "warmup-need", "prompt": phase_prompt}))
    finally:
        policy._need_threshold = threshold
    adjustment = list(policy.infer_events(common | {"mode": "phase", "phase": "adjustment",
                                                   "request_id": "warmup-adjustment", "prompt": phase_prompt}))
    plan_prompt = build_recovery_prompt(
        instruction=BOOK_INSTRUCTION, failed_tactile_caption=caption,
        failure_recovery_memory=[{"recovery_plan": "initial plan",
                                  "failure_reason": "failure_reason=rotate right,grasp appropriate."}],
        prompt_profile=MINIMAL_PROMPT_PROFILE,
    )
    plan = list(policy.infer_events(common | {"mode": "reasoning", "prompt": plan_prompt}))
    action = list(policy.infer_events(common | {"mode": "execution", "prompt": execution_prompt,
                                               "action_noise": np.zeros((30, 32), dtype=np.float32)}))
    if [event.get("event") for event in execution] != ["phase_decision", "failure_reason"]:
        raise ValueError("Book V9.3 warm-up did not exercise shared need/failure events")
    return {"action_shape": list(action[0]["actions"].shape), "execution": execution,
            "adjustment": adjustment, "plan": plan}


def next_event(iterator):
    # StopIteration cannot cross an asyncio Future boundary.
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


class WorkerStreamingPolicyServer(base.StreamingPolicyServer):
    """Never run synchronous JAX inference on the WebSocket event loop.

    A single worker protects the shared NNX model; each generator continuation
    is submitted only after its previous event has been sent to the client.
    """

    def __init__(self, policy, host, port):
        super().__init__(policy, host, port)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="book-v9-3-model")

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
                    finished, event = await loop.run_in_executor(self._executor, next_event, iterator)
                    if finished:
                        break
                    await websocket.send(packer.pack(event))
                iterator = None
        except websockets.ConnectionClosed:
            return
        except Exception:
            logging.exception("Book V9.3 inference error")
            await websocket.send(traceback.format_exc())
            await websocket.close(code=1011, reason="Book V9.3 inference error")
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                await loop.run_in_executor(self._executor, iterator.close)

    def serve_forever(self):
        try:
            super().serve_forever()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    args = parse_args()
    config, identity = inspect_artifacts(args)
    deployment = resolve_deployment(args, identity)
    logging.info("Book V9.3 runtime thresholds: need_recovery=%s adjustment_end=%s source=%s",
                 args.need_recovery_threshold, args.adjustment_end_threshold, deployment["thresholds_status"])
    if args.validate_only:
        print(json.dumps(identity | deployment, indent=2, ensure_ascii=False))
        return
    policy = load_policy(args, config, identity, deployment)
    logging.info("Book V9.3 five-task warm-up: %s", json.dumps(warm_up(policy), default=str, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(policy.metadata, indent=2, default=str, ensure_ascii=False))
        return
    logging.info("Serving Book V9.3 at %s:%d", args.host, args.port)
    WorkerStreamingPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
