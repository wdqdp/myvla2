#!/usr/bin/env python3
"""Serve joint V12.1 weights for either blackbox or book with asynchronous phase events."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import jax
import numpy as np

from scripts import serve_tactile_vla_v7_7 as base
from scripts.serve_tactile_vla_book_v9_5 import WorkerStreamingPolicyServer
from tactile_vla.common.labels_v4 import neutral_caption
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.prompts import build_phase_prompt as action_prompt
from tactile_vla.vla.prompts import build_recovery_prompt
from tactile_vla.vla.structured_text import legal_failure_reasons, legal_recovery_plans
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v12_1_deployment import (
    DATA_PROFILE,
    DEFAULT_CHECKPOINT,
    DEFAULT_NORM_DIR,
    TASKS,
    deployment_fields,
    runtime_caption,
    threshold,
    validate_tactile_caption,
    validate_training_config,
)
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt

_RESTORE_FULL_PARAMS = base._restore_exported_full_params


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Run directory (selects 1000), numeric step, or full_params",
    )
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--need-recovery-threshold", type=float, default=0.5)
    parser.add_argument("--adjustment-end-threshold", type=float, default=0.5)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--validate-only", action="store_true", help="Inspect config/export/norm without loading weights"
    )
    group.add_argument("--dry-run", action="store_true", help="Restore and warm up the five tasks, then exit")
    args = parser.parse_args(argv)
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive")
    try:
        threshold(args.need_recovery_threshold)
        threshold(args.adjustment_end_threshold)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_action_dim, args.reasoning_max_token_len, args.no_norm = 7, None, False
    return args


def inspect_artifacts(args):
    path = args.checkpoint.expanduser().resolve()
    if path.name == "full_params":
        path = path.parent
    elif (path / "config.json").is_file() and not (path / "full_params").is_dir():
        path = path / "1000"
    if not path.name.isdecimal():
        raise ValueError(f"V12.1 deployment requires a numeric checkpoint step, got {path}")
    full_params = path / "full_params"
    if not full_params.is_dir():
        raise FileNotFoundError(full_params)
    config = base.v3._find_config(path)
    validate_training_config(config)
    args.norm_stats_dir = args.norm_stats_dir.expanduser().resolve()
    norm_file = args.norm_stats_dir / "norm_stats.json"
    if not norm_file.is_file():
        raise FileNotFoundError(
            f"V12.1 shared norm_stats is missing: {norm_file}. Copy the training artifact with SHA256 "
            f"{config['artifact_identity']['norm_stats_sha256']}; do not substitute blackbox/book source norms."
        )
    norm_sha = sha256_file(norm_file)
    validate_training_config(config, norm_sha)
    export_path = path / "v12_1_export.json"
    export = json.loads(export_path.read_text())
    if export.get("step") != int(path.name) or export.get("default_deployment") != "full_params":
        raise ValueError("V12.1 export step/format mismatch")
    digest = export.get("exports", {}).get("full_params", {}).get("parameter_tree_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
        raise ValueError("V12.1 full_params parameter-tree digest is missing")
    # Export paths can refer to the training machine; resolve weights relative to the local step.
    args.checkpoint = full_params
    return config, deployment_fields(args.task) | {
        "checkpoint_step": int(path.name),
        "full_params_sha256": digest,
        "norm_stats_sha256": norm_sha,
        "training_data_hash": config["training_data_hash"],
        "export_metadata": str(export_path),
        "need_recovery_threshold": args.need_recovery_threshold,
        "adjustment_end_threshold": args.adjustment_end_threshold,
    }


def _restore_full_params(path, **kwargs):
    restored = _RESTORE_FULL_PARAMS(path, **kwargs)

    def restore_rng(key_path, value):
        names = [getattr(key, "key", None) for key in key_path]
        if names[-1:] == ["key"] and "rngs" in names:
            if getattr(value, "dtype", None) == np.dtype("uint32") and value.shape == (2,):
                return jax.random.wrap_key_data(value, impl="threefry2x32")
        return value

    # Orbax may serialize NNX typed keys as uint32[2]; retain the full shape check in the base loader.
    return jax.tree_util.tree_map_with_path(restore_rng, restored)


class V121Policy(base.V77Policy):
    def __init__(self, *, args, config, identity):
        # The complete export replaces every state leaf; a random initialized copy is unnecessary.
        args.abstract_model_init = True
        old_profile, old_restore = base.DATA_PROFILE, base._restore_exported_full_params
        base.DATA_PROFILE, base._restore_exported_full_params = DATA_PROFILE, _restore_full_params
        try:
            super().__init__(
                args=args,
                config=config,
                model_config=base.v3._model_config(args, config),
                norm_stats=base.normalize.load(args.norm_stats_dir),
            )
        finally:
            base.DATA_PROFILE, base._restore_exported_full_params = old_profile, old_restore
        if parameter_tree_sha256(base.v3.nnx.state(self._model)) != identity["full_params_sha256"]:
            raise ValueError("Restored V12.1 parameter-tree SHA256 differs from export")
        self.task = args.task
        self._metadata.update(identity)

    def infer_events(self, request):
        mode = request.get("mode", "execution")
        prompt = str(request.get("prompt", ""))
        instruction = TASKS[self.task]["instruction"]
        if any(key in request for key in ("observation/state_history", "observation/state_history_mask")):
            raise ValueError("V12.1 does not accept dense state_history arrays")
        if mode == "phase":
            lines = prompt.splitlines()
            if len(lines) != 5 or lines[:2] != ["Mode: phase", f"Task: {instruction}"]:
                raise ValueError("V12.1 requires the selected task's V7.7 phase prompt")
            if not lines[2].startswith("Touch: ") or not lines[3].startswith("Recovery plan: "):
                raise ValueError("V12.1 phase Touch/plan fields are missing")
            validate_tactile_caption(lines[2][7:], self.task)
            heading = "State history H100 sampled to 11 points: "
            if not lines[4].startswith(heading):
                raise ValueError("V12.1 phase H100 text is missing")
            values = np.asarray(json.loads(lines[4][len(heading) :]))
            if values.shape != (11, 7) or values.dtype.kind not in "iu" or np.any((values < -1) | (values > 255)):
                raise ValueError("V12.1 requires discrete H100 [11,7] values in [-1,255], without clipping")
        elif mode == "reasoning":
            prefix = f"Mode: reasoning. Task: {instruction} "
            head, separator, text = prompt.partition("Failure-recovery memory: ")
            captions = re.findall(r"Touch\[[^\]]*\]", head)
            if not prompt.startswith(prefix) or not separator or len(captions) != 1:
                raise ValueError("V12.1 requires the selected task's minimal_v1 plan prompt")
            validate_tactile_caption(captions[0], self.task)
            pairs = text.split(" | ")
            if not 1 <= len(pairs) <= 4:
                raise ValueError("V12.1 plan requires 1–4 real failure-plan pairs")
            for index, pair in enumerate(pairs):
                plan, separator, reason = pair.partition("; ")
                if not separator or reason not in legal_failure_reasons():
                    raise ValueError("Invalid V12.1 failure memory")
                if (index == 0 and plan != "recovery_plan=initial plan") or (
                    index > 0 and plan not in legal_recovery_plans()
                ):
                    raise ValueError("V12.1 memory must preserve the initial pair and actual executed plans")
        elif mode in {"execution", "action"}:
            if prompt != action_prompt(
                phase="execution", instruction=instruction, recovery_plan="", prompt_profile="phase_v2"
            ):
                prefix = f"Mode: adjustment. Task: {instruction}\nPut the object back, and follow this recovery plan: "
                if not prompt.startswith(prefix) or prompt[len(prefix) :] not in legal_recovery_plans():
                    raise ValueError("V12.1 action requires the selected task's phase_v2 prompt")
        else:
            raise ValueError(f"Unsupported V12.1 mode: {mode}")
        yield from super().infer_events(request)


def warm_up(policy):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    common = {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/state": np.zeros(7, np.float32),
    }
    spec = TASKS[policy.task]
    schema = spec["schema_version"]
    caption = runtime_caption(neutral_caption(schema_version=schema), policy.task)
    prompt = build_phase_prompt(
        instruction=spec["instruction"],
        tactile_caption=caption,
        recovery_plan="",
        qpos_h100_11_discrete=np.zeros((11, 7), np.int32),
    )
    old_threshold = policy._need_threshold
    try:
        policy._need_threshold = 0.0
        execution = list(
            policy.infer_events(
                common | {"mode": "phase", "phase": "execution", "request_id": "warmup-need", "prompt": prompt}
            )
        )
    finally:
        policy._need_threshold = old_threshold
    adjustment = list(
        policy.infer_events(
            common | {"mode": "phase", "phase": "adjustment", "request_id": "warmup-adjustment", "prompt": prompt}
        )
    )
    memory = [{"recovery_plan": "initial plan", "failure_reason": "failure_reason=rotate right,grasp appropriate."}]
    plan_prompt = build_recovery_prompt(
        instruction=spec["instruction"],
        failed_tactile_caption=caption,
        failure_recovery_memory=memory,
        prompt_profile="minimal_v1",
    )
    plan = list(policy.infer_events(common | {"mode": "reasoning", "prompt": plan_prompt}))
    actions = list(
        policy.infer_events(
            common
            | {
                "mode": "execution",
                "action_noise": np.zeros((30, 32), np.float32),
                "prompt": action_prompt(
                    phase="execution", instruction=spec["instruction"], recovery_plan="", prompt_profile="phase_v2"
                ),
            }
        )
    )
    if [event.get("event") for event in execution] != ["phase_decision", "failure_reason"]:
        raise ValueError("V12.1 warmup did not exercise the shared need/failure branch")
    return {
        "action_shape": list(actions[0]["actions"].shape),
        "execution": execution,
        "adjustment": adjustment,
        "plan": plan,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    args = parse_args()
    config, identity = inspect_artifacts(args)
    if args.validate_only:
        print(json.dumps(identity, indent=2, ensure_ascii=False))
        return
    logging.info("Restoring V12.1 task=%s checkpoint=%s", args.task, args.checkpoint)
    policy = V121Policy(args=args, config=config, identity=identity)
    logging.info("Warming up V12.1 action, need/failure, adjustment and plan")
    logging.info("Warmup: %s", warm_up(policy))
    if args.dry_run:
        print(json.dumps(policy.metadata, indent=2, default=str, ensure_ascii=False))
        return
    logging.info("V12.1 server listening on %s:%d task=%s", args.host, args.port, args.task)
    WorkerStreamingPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
