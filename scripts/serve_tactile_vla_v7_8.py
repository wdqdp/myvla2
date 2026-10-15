#!/usr/bin/env python3
"""Serve V7.8/2000 visual-state five-task inference with streamed phase events."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import numpy as np

from scripts import serve_tactile_vla_v7_7 as shared
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.prompts import build_phase_prompt as action_prompt
from tactile_vla.vla.structured_text import legal_failure_reasons, legal_recovery_plans
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v7_8_phase_prompt import (
    PROMPT_PROFILE,
    build_phase_prompt,
    build_recovery_prompt,
    validate_no_touch,
)
from tactile_vla.vla.v7_8_runtime import (
    DATA_PROFILE,
    DEFAULT_CHECKPOINT,
    DEFAULT_NORM_DIR,
    EXPORT_FIELDS,
    HISTORY_OFFSETS,
    INSTRUCTION,
    SERVER_NAME,
    require_sha,
    threshold,
    validate_server_metadata,
    validate_training_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Run, numeric step or full_params directory"
    )
    parser.add_argument("--config", type=Path, help="Explicit training config.json if stored separately")
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--need-recovery-threshold", type=float, default=0.5)
    parser.add_argument("--adjustment-end-threshold", type=float, default=0.5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validate-only", action="store_true", help="Check config/export/norm without loading weights")
    group.add_argument("--dry-run", action="store_true", help="Load weights and warm up all five tasks, then exit")
    args = parser.parse_args(argv)
    if args.num_inference_steps <= 0 or not 1 <= args.port <= 65535:
        parser.error("inference steps must be positive and port must be in [1,65535]")
    for name in ("need_recovery_threshold", "adjustment_end_threshold"):
        try:
            setattr(args, name, threshold(getattr(args, name), name))
        except ValueError as exc:
            parser.error(str(exc))
    args.output_action_dim, args.reasoning_max_token_len, args.no_norm = 7, None, False
    return args


def inspect_artifacts(args):
    path = args.checkpoint.expanduser().resolve()
    if (path / "2000/full_params").is_dir():
        path = path / "2000"
    step = path.parent if path.name == "full_params" else path
    if not step.name.isdecimal():
        raise ValueError("V7.8 requires a numeric checkpoint step or run containing 2000/full_params")
    full_params = step / "full_params"
    if not (full_params / "_METADATA").is_file():
        raise FileNotFoundError(full_params / "_METADATA")
    config = json.loads(args.config.expanduser().read_text()) if args.config else shared.v3._find_config(step)
    args.norm_stats_dir = args.norm_stats_dir.expanduser().resolve()
    norm_sha = sha256_file(args.norm_stats_dir / "norm_stats.json")
    validate_training_config(config, norm_sha)
    export_file = step / "v7_8_export.json"
    export = json.loads(export_file.read_text())
    mismatch = {key: (export.get(key), config.get(key)) for key in EXPORT_FIELDS if export.get(key) != config.get(key)}
    if mismatch:
        raise ValueError(f"V7.8 export differs from training config: {mismatch}")
    if export.get("step") != int(step.name) or export.get("default_deployment") != "full_params":
        raise ValueError("V7.8 export step/format mismatch")
    full_sha = require_sha(export.get("exports", {}).get("full_params", {}).get("parameter_tree_sha256"), "full_params")
    args.checkpoint = full_params
    return config, {
        **{key: config[key] for key in EXPORT_FIELDS},
        "checkpoint": str(full_params),
        "checkpoint_step": int(step.name),
        "export_metadata": str(export_file),
        "norm_stats_dir": str(args.norm_stats_dir),
        "norm_stats_sha256": norm_sha,
        "full_params_sha256": full_sha,
    }


class V78Policy(shared.V77Policy):
    def __init__(self, *, args, config, identity):
        super().__init__(
            args=args,
            config=config,
            model_config=shared.v3._model_config(args, config),
            norm_stats=shared.normalize.load(args.norm_stats_dir),
            data_profile=DATA_PROFILE,
            prompt_profile=PROMPT_PROFILE,
        )
        actual_sha = parameter_tree_sha256(shared.v3.nnx.state(self._model))
        if actual_sha != identity["full_params_sha256"]:
            raise ValueError("V7.8 restored parameter-tree SHA256 differs from export")
        self._metadata.update(
            identity
            | {
                "name": SERVER_NAME,
                "instruction": INSTRUCTION,
                "action_prompt_profile": "phase_v2",
                "requires_captioner": False,
                "requires_tactile_topics": False,
                "qpos_h100_sample_offsets": HISTORY_OFFSETS,
                "history_frame_filter": "none",
                "max_memory_pairs": 5,
                "max_supported_attempts": None,
                "memory_policy": "latest_five_real_failure_plan_pairs",
                "worker_policy": "single_model_worker_stream_events_before_continuation",
            }
        )
        validate_server_metadata(self._metadata)

    def infer_events(self, request):
        prompt = str(request.get("prompt", ""))
        validate_no_touch(prompt)
        if f"Task: {INSTRUCTION}" not in prompt:
            raise ValueError("V7.8 requires its original task instruction")
        if any(key in request for key in ("observation/state_history", "observation/state_history_mask")):
            raise ValueError("V7.8 must not receive continuous state_history arrays")
        mode = request.get("mode", "execution")
        if mode == "phase":
            if not prompt.startswith("Mode: phase\n") or "State history H100 sampled to 11 points: " not in prompt:
                raise ValueError("V7.8 requires its visual H100 phase prompt")
        elif mode == "reasoning":
            if not prompt.startswith("Mode: reasoning.") or "Failure-recovery memory: " not in prompt:
                raise ValueError("V7.8 requires its visual recovery memory prompt")
            pairs = prompt.split("Failure-recovery memory: ", 1)[1].split(" | ")
            if not 1 <= len(pairs) <= 5:
                raise ValueError("V7.8 supports 1–5 real failure/plan pairs")
            for index, pair in enumerate(pairs):
                plan, separator, reason = pair.partition("; ")
                if not separator or reason not in legal_failure_reasons():
                    raise ValueError("Invalid V7.8 failure memory")
                if plan not in legal_recovery_plans() and not (index == 0 and plan == "recovery_plan=initial plan"):
                    raise ValueError("Invalid V7.8 executed plan memory")
        elif mode not in {"action", "execution"}:
            raise ValueError(f"Unsupported V7.8 mode: {mode}")
        yield from super().infer_events(request)


def warm_up(policy):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/state": np.zeros(7, dtype=np.float32),
    }
    phase = build_phase_prompt(
        instruction=INSTRUCTION, recovery_plan="", qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32)
    )
    old_threshold = policy._need_threshold
    try:
        policy._need_threshold = 0.0
        need = list(
            policy.infer_events(
                observation | {"mode": "phase", "phase": "execution", "request_id": "warmup-need", "prompt": phase}
            )
        )
    finally:
        policy._need_threshold = old_threshold
    if [event.get("event") for event in need] != ["phase_decision", "failure_reason"]:
        raise RuntimeError("V7.8 warm-up did not exercise need/failure shared prefill")
    adjustment = list(
        policy.infer_events(
            observation | {"mode": "phase", "phase": "adjustment", "request_id": "warmup-adjustment", "prompt": phase}
        )
    )
    reason = "failure_reason=rotate right,grasp appropriate."
    plan = "recovery_plan=move horizontally right moderately, move vertically none moderately."
    memory = []
    for length in range(1, 6):
        memory.append({"failure_reason": reason, "recovery_plan": "initial plan" if length == 1 else plan})
        prompt = build_recovery_prompt(instruction=INSTRUCTION, failure_recovery_memory=memory)
        list(policy.infer_events(observation | {"mode": "reasoning", "prompt": prompt}))
        logging.info("V7.8 plan warm-up L%d complete", length)
    shapes = {}
    for name in ("execution", "adjustment"):
        prompt = action_prompt(
            phase=name,
            instruction=INSTRUCTION,
            recovery_plan=plan if name == "adjustment" else "",
            prompt_profile="phase_v2",
        )
        response = list(
            policy.infer_events(
                observation
                | {"mode": "execution", "prompt": prompt, "action_noise": np.zeros((30, 32), dtype=np.float32)}
            )
        )
        shapes[name] = list(response[0]["actions"].shape)
    return {"action_shapes": shapes, "need": need, "adjustment": adjustment, "memory_lengths": [1, 2, 3, 4, 5]}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    args = parse_args()
    config, identity = inspect_artifacts(args)
    logging.info(
        "V7.8 checkpoint=%s step=%s need=%s adjustment=%s",
        args.checkpoint,
        identity["checkpoint_step"],
        args.need_recovery_threshold,
        args.adjustment_end_threshold,
    )
    if args.validate_only:
        print(
            json.dumps(
                identity
                | {
                    "thresholds": {
                        "need_recovery": args.need_recovery_threshold,
                        "adjustment_end": args.adjustment_end_threshold,
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    policy = V78Policy(args=args, config=config, identity=identity)
    logging.info("V7.8 warm-up: %s", json.dumps(warm_up(policy), ensure_ascii=False, default=str))
    if args.dry_run:
        print(json.dumps(policy.metadata, ensure_ascii=False, default=str, indent=2))
        return
    logging.info("Serving V7.8 at %s:%s", args.host, args.port)
    shared.StreamingPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
