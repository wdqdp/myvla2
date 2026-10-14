#!/usr/bin/env python3
"""Serve Bottle V11.1 five tasks with streamed decisions and serialized worker inference."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "openpi/src"),
    str(PROJECT_ROOT / "openpi/packages/openpi-client/src"),
]

import numpy as np
import websockets
from openpi_client import msgpack_numpy

from scripts import serve_tactile_vla_v7_7 as base
from tactile_vla.common.labels_v4 import neutral_caption
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.bottle_v11_1_runtime import (
    BOTTLE_INSTRUCTION,
    DEFAULT_CAPTIONER_SHA256,
    DEFAULT_CHECKPOINT,
    DEFAULT_NORM_DIR,
    HISTORY_OFFSETS,
    MEMORY_POLICY,
    FAILURE_TARGET,
    PLAN_TARGET,
    V112_VERSION,
    V113_VERSION,
    deployment_policy_fields,
    deployment_phase_profile,
    deployment_runtime_history_policy,
    deployment_version,
    resolve_thresholds,
    validate_tactile_caption,
    validate_training_config,
    validate_server_metadata,
)
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt
from tactile_vla.vla.prompts import build_phase_prompt as action_prompt
from tactile_vla.vla.bottle_v11_1_grammar import bottle_failure_reasons, bottle_recovery_plans
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt
from tactile_vla.vla import bottle_v11_3_prompts as visual_prompts


def parse_args(argv=None, *, default_checkpoint=DEFAULT_CHECKPOINT, description=__doc__, captioner_enabled=True):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--config", type=Path, help="Explicit training config.json if stored separately from weights")
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument("--thresholds-file", type=Path, help="Optional calibrated thresholds; omitted by default")
    parser.add_argument(
        "--need-recovery-threshold",
        type=float,
        help="Override need_recovery threshold; defaults to 0.5 without a thresholds file",
    )
    parser.add_argument(
        "--adjustment-end-threshold",
        type=float,
        help="Override adjustment_end threshold; defaults to 0.5 without a thresholds file",
    )
    if captioner_enabled:
        parser.add_argument(
            "--captioner-checkpoint-sha256",
            default=DEFAULT_CAPTIONER_SHA256,
            help="Robot captioner SHA256; must match the training captioner identity",
        )
    else:
        parser.set_defaults(captioner_checkpoint_sha256=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--validate-only", action="store_true", help="Validate artifacts without restoring weights")
    validation.add_argument(
        "--dry-run", action="store_true", help="Restore weights and warm up all five tasks, then exit"
    )
    args = parser.parse_args(argv)
    args.output_action_dim = 7
    args.reasoning_max_token_len = None
    args.no_norm = False
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive")
    sha = args.captioner_checkpoint_sha256
    if captioner_enabled and (len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha)):
        parser.error("--captioner-checkpoint-sha256 must be a lowercase 64-character SHA256")
    return args


def inspect_artifacts(args):
    path = args.checkpoint.expanduser().resolve()
    step_dir = path.parent if path.name == "full_params" else path
    full_params = step_dir / "full_params"
    if not full_params.is_dir():
        raise FileNotFoundError(full_params)
    config = json.loads(args.config.expanduser().read_text()) if args.config else base.v3._find_config(step_dir)
    args.norm_stats_dir = args.norm_stats_dir.expanduser().resolve()
    norm_sha = sha256_file(args.norm_stats_dir / "norm_stats.json")
    validate_training_config(config, norm_sha)
    version = deployment_version(config["data_profile"])
    if version == V113_VERSION and args.captioner_checkpoint_sha256 is not None:
        raise ValueError("Bottle V11.3 has no runtime captioner; use the V11.3 server entry")
    if version != V113_VERSION and args.captioner_checkpoint_sha256 != config["captioner_identity"]["checkpoint_sha256"]:
        raise ValueError("Bottle V11.1 runtime captioner SHA256 differs from training")
    export_path = step_dir / f"{version}_export.json"
    export = json.loads(export_path.read_text())
    if export.get("experiment_version") != version or export.get("data_profile") != config["data_profile"]:
        raise ValueError(f"{version} export version/profile differs from training config")
    policy_fields = deployment_policy_fields(version)
    identity_fields = (
        "source_scope",
        "stage_a_initialization_identity",
        "boundary_audit_sha256",
        "training_data_hash",
        "training_target_coverage",
    )
    if version in {V112_VERSION, V113_VERSION}:
        identity_fields += ("action_target_policy", "idle_gap_audit_sha256")
    if version == V113_VERSION:
        identity_fields += ("source_multitask_training_data_hash", "source_multitask_index_sha256")
    checked_fields = (*policy_fields, *identity_fields)
    mismatches = {key: (export.get(key), config[key]) for key in checked_fields if export.get(key) != config[key]}
    for key in (
        "prompt_profile",
        "failure_grammar",
        "recovery_grammar",
        "captioner_identity",
        "phase_prefill_protocol",
    ):
        if export.get(key) != config[key]:
            mismatches[key] = (export.get(key), config[key])
    if mismatches:
        raise ValueError(f"{version} export policies differ from training config: {mismatches}")
    step = int(step_dir.name)
    full_sha = export["exports"]["full_params"]["parameter_tree_sha256"]
    if export.get("step") != step or export.get("default_deployment") != "full_params":
        raise ValueError("Bottle V11.1 export step/format mismatch")
    if not isinstance(full_sha, str) or re.fullmatch(r"[0-9a-f]{64}", full_sha) is None:
        raise ValueError("Bottle V11.1 full_params export digest is missing")
    args.checkpoint = full_params
    return config, {
        "checkpoint": str(full_params),
        "norm_stats_dir": str(args.norm_stats_dir),
        "checkpoint_step": step,
        "full_params_sha256": full_sha,
        "norm_stats_sha256": norm_sha,
        "training_data_hash": config["artifact_identity"][f"{version}_training_data_hash"],
        "export_metadata": str(export_path),
        "data_profile": config["data_profile"],
        "experiment_version": version,
        **({"adjustment_history_runtime_policy": deployment_runtime_history_policy(version)} if version in {V112_VERSION, V113_VERSION} else {}),
        **{key: config[key] for key in policy_fields},
        **{key: config[key] for key in identity_fields},
    }


def resolve_deployment(args, identity):
    calibration = json.loads(args.thresholds_file.read_text()) if args.thresholds_file else None
    thresholds, overrides = resolve_thresholds(
        calibration=calibration,
        step=identity["checkpoint_step"],
        full_params_sha=identity["full_params_sha256"],
        norm_sha=identity["norm_stats_sha256"],
        training_data_hash=identity["training_data_hash"],
        need_override=args.need_recovery_threshold,
        adjustment_override=args.adjustment_end_threshold,
        data_profile=identity["data_profile"],
    )
    args.need_recovery_threshold = thresholds["need_recovery"]
    args.adjustment_end_threshold = thresholds["adjustment_end"]
    return {
        "thresholds": thresholds,
        "threshold_manual_overrides": overrides,
        "thresholds_status": (
            "explicit_manual_override"
            if any(overrides.values())
            else "calibrated_on_bottle_val"
            if calibration is not None
            else "default_0_5"
        ),
        "thresholds_file": str(args.thresholds_file.resolve()) if args.thresholds_file else None,
    }


class BottleV111Policy(base.V77Policy):
    def __init__(self, *, args, config, model_config, norm_stats, identity, deployment):
        old_profile = base.DATA_PROFILE
        base.DATA_PROFILE = config["data_profile"]
        try:
            super().__init__(
                args=args, config=config, model_config=model_config, norm_stats=norm_stats,
                prompt_profile=deployment_phase_profile(identity["experiment_version"]),
            )
        finally:
            base.DATA_PROFILE = old_profile
        actual_sha = parameter_tree_sha256(base.v3.nnx.state(self._model))
        if actual_sha != identity["full_params_sha256"]:
            raise ValueError("Restored Bottle V11.1 parameter-tree SHA256 differs from export")
        self._metadata.update(
            identity
            | deployment
            | {
                "name": f"tactile_vla_{identity['experiment_version']}",
                "instruction": BOTTLE_INSTRUCTION,
                "action_prompt_profile": "phase_v2",
                "qpos_h100_sample_offsets": HISTORY_OFFSETS,
                "max_memory_pairs": 4,
                "memory_policy": MEMORY_POLICY,
                "max_supported_attempts": None,
                "classification_qpos_policy": "raw_no_gripper_remap",
                "episode_start_padding": "left_pad_episode_frame_0",
                "captioner_window_size": 30,
                "captioner_checkpoint_sha256": args.captioner_checkpoint_sha256,
                "captioner_identity": config["captioner_identity"],
                "captioner_identity_scope": config.get("captioner_identity_scope", "training_and_runtime_client_server"),
                "requires_captioner": True,
                "requires_tactile_topics": True,
                "worker_policy": "single_model_worker_stream_events_before_continuation",
            }
        )
        if identity["experiment_version"] in {V112_VERSION, V113_VERSION}:
            self._metadata["adjustment_history_runtime_policy"] = deployment_runtime_history_policy(identity["experiment_version"])
        if identity["experiment_version"] == V113_VERSION:
            self._metadata.update(captioner_window_size=0, requires_captioner=False, requires_tactile_topics=False)
        validate_server_metadata(self._metadata)

    def infer_events(self, request):
        prompt = str(request.get("prompt", ""))
        if BOTTLE_INSTRUCTION not in prompt:
            raise ValueError("Bottle V11.1 requires its original bottle instruction")
        if any(key in request for key in ("observation/state_history", "observation/state_history_mask")):
            raise ValueError("Bottle V11.1 must not receive continuous state_history arrays")
        mode = request.get("mode", "execution")
        visual_only = self._metadata["experiment_version"] == V113_VERSION
        if visual_only:
            visual_prompts.validate_no_touch(prompt)
        if mode == "phase":
            if self._metadata["experiment_version"] in {V112_VERSION, V113_VERSION}:
                if request.get("phase") not in {"execution", "adjustment"} or request.get("history_clock") != "raw_30Hz_frames":
                    raise ValueError("Bottle V11.2 phase request has the wrong history clock")
            if not prompt.startswith("Mode: phase\n") or "State history H100 sampled to 11 points: " not in prompt:
                raise ValueError("Bottle V11.1 requires the V7.7 H100 phase prompt")
            if not visual_only:
                touch_lines = [line[7:] for line in prompt.splitlines() if line.startswith("Touch: ")]
                if len(touch_lines) != 1:
                    raise ValueError("Bottle phase prompt requires exactly one Touch line")
                validate_tactile_caption(touch_lines[0])
        elif mode == "reasoning":
            if not prompt.startswith("Mode: reasoning.") or "Failure-recovery memory: " not in prompt:
                raise ValueError("Bottle V11.1 requires the minimal_v1 plan prompt")
            memory_text = prompt.split("Failure-recovery memory: ", 1)[1]
            pairs = memory_text.split(" | ")
            if not 1 <= len(pairs) <= 4:
                raise ValueError("Bottle V11.1 plan requests require 1–4 failed-plan/reason pairs")
            for index, pair in enumerate(pairs):
                plan, separator, reason = pair.partition("; ")
                if not separator or reason not in bottle_failure_reasons():
                    raise ValueError("Bottle V11.1 plan memory contains an invalid failure pair")
                if (index == 0 and plan != "recovery_plan=initial plan") or (
                    index > 0 and plan not in bottle_recovery_plans()
                ):
                    raise ValueError("Bottle V11.1 plan memory must preserve the initial pair and executed plans")
            if not visual_only:
                captions = re.findall(r"Touch\[[^\]]*\]", prompt.split("Failure-recovery memory: ", 1)[0])
                if len(captions) != 1:
                    raise ValueError("Bottle plan requires exactly one current Touch caption")
                validate_tactile_caption(captions[0])
        elif mode not in {"action", "execution"}:
            raise ValueError(f"Unsupported Bottle V11.1 inference mode: {mode}")
        inputs = dict(request)
        inputs.pop("history_clock", None)
        yield from super().infer_events(inputs)


def load_policy(args, config, identity, deployment):
    model_config = base.v3._model_config(args, config)
    return BottleV111Policy(
        args=args,
        config=config,
        model_config=model_config,
        norm_stats=base.normalize.load(args.norm_stats_dir),
        identity=identity,
        deployment=deployment,
    )


def warm_up(policy):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    common = {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/state": np.zeros(7, dtype=np.float32),
    }
    visual_only = policy.metadata["experiment_version"] == V113_VERSION
    caption = "" if visual_only else neutral_caption()
    phase_args = {
        "instruction": BOTTLE_INSTRUCTION, "recovery_plan": "",
        "qpos_h100_11_discrete": np.zeros((11, 7), dtype=np.int32),
    }
    phase_prompt = (
        visual_prompts.build_phase_prompt(**phase_args)
        if visual_only else build_phase_prompt(**phase_args, tactile_caption=caption)
    )
    execution_prompt = action_prompt(
        phase="execution", instruction=BOTTLE_INSTRUCTION, recovery_plan="", prompt_profile="phase_v2"
    )
    # Force the conditional failure branch for warm-up only, without changing runtime metadata.
    threshold = policy._need_threshold
    clocks = policy.metadata["experiment_version"] in {V112_VERSION, V113_VERSION}
    try:
        policy._need_threshold = 0.0
        execution = list(
            policy.infer_events(
                common | {"mode": "phase", "phase": "execution", "request_id": "warmup-need", "prompt": phase_prompt}
                | ({"history_clock": "raw_30Hz_frames"} if clocks else {})
            )
        )
    finally:
        policy._need_threshold = threshold
    adjustment = list(
        policy.infer_events(
            common | {"mode": "phase", "phase": "adjustment", "request_id": "warmup-adjustment", "prompt": phase_prompt}
            | ({"history_clock": "raw_30Hz_frames"} if clocks else {})
        )
    )
    plan = []
    memory = [{"recovery_plan": "initial plan", "failure_reason": FAILURE_TARGET}]
    for length in range(1, 5):
        plan_prompt = (
            visual_prompts.build_recovery_prompt(instruction=BOTTLE_INSTRUCTION, failure_recovery_memory=memory)
            if visual_only else build_recovery_prompt(
                instruction=BOTTLE_INSTRUCTION, failed_tactile_caption=caption,
                failure_recovery_memory=memory, prompt_profile=MINIMAL_PROMPT_PROFILE,
            )
        )
        plan.append(
            {
                "memory_length": length,
                "events": list(policy.infer_events(common | {"mode": "reasoning", "prompt": plan_prompt})),
            }
        )
        memory.append({"recovery_plan": PLAN_TARGET, "failure_reason": FAILURE_TARGET})
    action = list(
        policy.infer_events(
            common
            | {"mode": "execution", "prompt": execution_prompt, "action_noise": np.zeros((30, 32), dtype=np.float32)}
        )
    )
    if [event.get("event") for event in execution] != ["phase_decision", "failure_reason"]:
        raise ValueError("Bottle V11.1 warm-up did not exercise shared need/failure events")
    return {
        "action_shape": list(action[0]["actions"].shape),
        "execution": execution,
        "adjustment": adjustment,
        "plan": plan,
    }


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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bottle-v11-1-model")

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
            logging.exception("Bottle V11.1 inference error")
            await websocket.send(traceback.format_exc())
            await websocket.close(code=1011, reason="Bottle V11.1 inference error")
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                await loop.run_in_executor(self._executor, iterator.close)

    def serve_forever(self):
        try:
            super().serve_forever()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)


def main(*, argv=None, expected_version=None, argument_parser=parse_args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    args = argument_parser(argv)
    config, identity = inspect_artifacts(args)
    if expected_version is not None and identity["experiment_version"] != expected_version:
        raise ValueError(f"Deployment entry requires experiment_version={expected_version}")
    deployment = resolve_deployment(args, identity)
    label = identity["experiment_version"]
    logging.info(
        "%s checkpoint=%s step=%s norm_stats_dir=%s Stage_A_step=%s captioner_sha=%s",
        label,
        identity["checkpoint"],
        identity["checkpoint_step"],
        identity["norm_stats_dir"],
        identity["stage_a_initialization_identity"]["step"],
        args.captioner_checkpoint_sha256,
    )
    logging.info(
        "%s runtime thresholds: need_recovery=%s adjustment_end=%s source=%s",
        label,
        args.need_recovery_threshold,
        args.adjustment_end_threshold,
        deployment["thresholds_status"],
    )
    if args.validate_only:
        print(json.dumps(identity | deployment, indent=2, ensure_ascii=False))
        return
    policy = load_policy(args, config, identity, deployment)
    logging.info("Starting %s five-task warm-up", label)
    logging.info("%s five-task warm-up: %s", label, json.dumps(warm_up(policy), default=str, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(policy.metadata, indent=2, default=str, ensure_ascii=False))
        return
    logging.info("Serving %s at %s:%d", label, args.host, args.port)
    WorkerStreamingPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
