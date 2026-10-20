#!/usr/bin/env python3
"""Serve V7.10 H1/H31/H61 full_params with the streamed V7.7 phase protocol."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
from functools import partial
import json
import logging
import math
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import serve_tactile_vla_v7_7 as server
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v7_10_multitask_data import COUNTERFACTUAL_POLICY, ROOT, VERSION_TAG, data_profile
from tactile_vla.vla.v7_10_phase_prompt import (
    WINDOW_FRAMES, build_phase_prompt, history_policy, history_prefix, prompt_profile,
)

DEFAULT_NORM_DIR = ROOT / "outputs/rotation_v4_1_new_env/norm_stats"
DEFAULT_MODEL_ROOT = ROOT / "outputs/multitask_v7_10"
EXPORT_FIELDS = (
    "experiment_version", "history_window", "history_policy", "data_profile", "prompt_profile",
    "counterfactual_policy", "training_data_hash", "source_multitask_training_data_hash",
    "source_multitask_index_sha256", "adjustment_sampling_policy", "adjustment_label_policy",
    "need_successful_recovery_start", "stage_a_initialization_identity", "phase_prefill_protocol",
    "training_target_coverage",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_ROOT,
                        help="Model root, run directory, numeric step or full_params directory")
    parser.add_argument("--checkpoint-step", type=int,
                        help="Step to select from a run/root; otherwise select its latest complete export")
    parser.add_argument("--config", type=Path, help="Original training config.json if stored separately")
    parser.add_argument("--history-window", choices=tuple(WINDOW_FRAMES),
                        help="Optional assertion; otherwise read the window from the checkpoint config")
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--need-recovery-threshold", type=float, help="Override checkpoint threshold (fallback: 0.5)")
    parser.add_argument("--adjustment-end-threshold", type=float, help="Override checkpoint threshold (fallback: 0.5)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validate-only", action="store_true", help="Inspect metadata/norm without restoring weights")
    group.add_argument("--dry-run", action="store_true", help="Restore weights and warm up, then exit")
    args = parser.parse_args(argv)
    if args.num_inference_steps <= 0 or not 1 <= args.port <= 65535:
        parser.error("inference steps must be positive and port must be in [1,65535]")
    if args.checkpoint_step is not None and args.checkpoint_step <= 0:
        parser.error("--checkpoint-step must be positive")
    args.output_action_dim, args.reasoning_max_token_len, args.no_norm = 7, None, False
    return args


def resolve_checkpoint(args):
    """Resolve one run/step without guessing between different history models."""
    path = args.checkpoint.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"V7.10 checkpoint directory is missing: {path}")
    if path.name == "full_params":
        path = path.parent
    if path.name.isdecimal():
        if args.checkpoint_step is not None and int(path.name) != args.checkpoint_step:
            raise ValueError("--checkpoint-step differs from the explicitly selected checkpoint path")
        return path
    # A run may keep config.json separately via --config. Its numeric step
    # directories still distinguish it from the root containing several runs.
    is_run = (path / "config.json").is_file() or any(
        p.is_dir() and p.name.isdecimal() for p in path.iterdir()
    )
    if not is_run:
        runs = []
        for child in sorted(path.iterdir()):
            config_path = child / "config.json"
            if not child.is_dir() or not config_path.is_file():
                continue
            config = json.loads(config_path.read_text())
            if config.get("experiment_version") != VERSION_TAG:
                continue
            if args.history_window is None or config.get("history_window") == args.history_window:
                runs.append(child)
        if len(runs) != 1:
            found = ", ".join(p.name for p in runs) or "none"
            raise ValueError(
                f"V7.10 root must select exactly one run, found: {found}. "
                "Specify --history-window 0s|1s|2s, or pass the exact run directory."
            )
        path = runs[0]
    if args.checkpoint_step is not None:
        step = path / str(args.checkpoint_step)
        if not step.is_dir():
            raise FileNotFoundError(f"Requested V7.10 step is missing: {step}")
        return step
    complete = [p for p in path.iterdir() if p.is_dir() and p.name.isdecimal()
                and (p / "full_params/_METADATA").is_file() and (p / "v7_10_export.json").is_file()]
    if not complete:
        raise FileNotFoundError(f"No complete V7.10 full_params export under {path}")
    return max(complete, key=lambda p: int(p.name))


def inspect_artifacts(args):
    step = resolve_checkpoint(args)
    full_params, export_file = step / "full_params", step / "v7_10_export.json"
    candidates = [args.config.expanduser().resolve()] if args.config else [
        step / "config.json", step.parent / "config.json",
    ]
    config_file = next((p for p in candidates if p.is_file()), None)
    args.norm_stats_dir = args.norm_stats_dir.expanduser().resolve()
    norm_file = args.norm_stats_dir / "norm_stats.json"
    missing = [str(p) for p in (full_params / "_METADATA", export_file, norm_file) if not p.is_file()]
    if config_file is None:
        missing.append("original V7.10 config.json; looked in: " + ", ".join(map(str, candidates)))
    if missing:
        raise FileNotFoundError("V7.10 deployment artifacts are incomplete:\n  " + "\n  ".join(missing))
    config, export = json.loads(config_file.read_text()), json.loads(export_file.read_text())
    window = config.get("history_window")
    if args.history_window is not None and args.history_window != window:
        raise ValueError(f"Requested window {args.history_window!r} differs from checkpoint {window!r}")
    policy = history_policy(window)
    required = {
        "experiment_version": VERSION_TAG, "data_profile": data_profile(window),
        "prompt_profile": prompt_profile(window), "history_policy": policy,
        "qpos_history_frames": WINDOW_FRAMES[window], "counterfactual_policy": COUNTERFACTUAL_POLICY,
        "adjustment_sampling_policy": "ratio_1_to_2",
        "action_horizon": 30, "action_dim": 32, "max_token_len": 512, "reasoning_max_token_len": 320,
        "use_state_history": False, "state_history_len": 0, "state_history_dim": 7, "history_hidden_dim": 0,
        "state_history_fps": 30.0, "phase_prefill_protocol": "need_failure_shared_kv_v1",
    }
    mismatch = {k: (config.get(k), v) for k, v in required.items() if config.get(k) != v}
    if mismatch:
        raise ValueError(f"V7.10 training protocol mismatch: {mismatch}")
    mismatch = {k: (export.get(k), config.get(k)) for k in EXPORT_FIELDS
                if k not in config or k not in export or export[k] != config[k]}
    if mismatch:
        raise ValueError(f"V7.10 export/config mismatch: {mismatch}")
    if export.get("step") != int(step.name) or export.get("default_deployment") != "full_params":
        raise ValueError("V7.10 export step/format mismatch")
    full_sha = export.get("exports", {}).get("full_params", {}).get("parameter_tree_sha256")
    if not isinstance(full_sha, str) or re.fullmatch(r"[0-9a-f]{64}", full_sha) is None:
        raise ValueError("V7.10 full_params parameter-tree SHA256 is missing or invalid")
    norm_sha = sha256_file(norm_file)
    artifact = config.get("artifact_identity", {})
    if artifact.get("v4_norm_stats_sha256") != norm_sha:
        raise ValueError("V7.10 norm_stats differs from its training identity")
    if artifact.get("v7_10_training_data_hash") != config["training_data_hash"]:
        raise ValueError("V7.10 training data identity mismatch")
    thresholds = config.get("thresholds", {})
    for name in ("need_recovery", "adjustment_end"):
        attr = f"{name}_threshold"
        value = getattr(args, attr)
        value = float(thresholds.get(name, 0.5) if value is None else value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{attr} must be finite and in [0,1]")
        setattr(args, attr, value)
    args.checkpoint, args.history_window = full_params, window
    return config, {
        **{k: config[k] for k in EXPORT_FIELDS},
        "checkpoint": str(full_params), "checkpoint_step": int(step.name),
        "config_file": str(config_file), "export_metadata": str(export_file),
        "norm_stats_sha256": norm_sha, "full_params_sha256": full_sha,
        "requires_captioner": True, "requires_tactile_topics": True,
        "need_recovery_threshold": args.need_recovery_threshold,
        "adjustment_end_threshold": args.adjustment_end_threshold,
    }


class V710Policy(server.V77Policy):
    def __init__(self, *, args, config, identity):
        self._history_window = config["history_window"]
        super().__init__(
            args=args, config=config, model_config=server.v3._model_config(args, config),
            norm_stats=server.normalize.load(args.norm_stats_dir),
            data_profile=data_profile(self._history_window), prompt_profile=prompt_profile(self._history_window),
        )
        if parameter_tree_sha256(server.v3.nnx.state(self._model)) != identity["full_params_sha256"]:
            raise ValueError("V7.10 restored parameter-tree SHA256 differs from export")
        self._metadata.update(identity)

    def infer_events(self, request):
        if request.get("mode") == "phase":
            lines = str(request.get("prompt", "")).splitlines()
            if not lines or not lines[-1].startswith(history_prefix(self._history_window)):
                raise ValueError(f"V7.10 phase request must use trained H{WINDOW_FRAMES[self._history_window]} text")
        yield from super().infer_events(request)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    args = parse_args()
    config, identity = inspect_artifacts(args)
    logging.info("V7.10 checkpoint=%s history=%s", args.checkpoint, args.history_window)
    logging.info("V7.10 thresholds: need_recovery=%s adjustment_end=%s",
                 args.need_recovery_threshold, args.adjustment_end_threshold)
    if args.validate_only:
        print(json.dumps(identity, ensure_ascii=False, indent=2))
        return
    policy = V710Policy(args=args, config=config, identity=identity)
    original_prompt_builder = server.build_phase_prompt
    try:
        server.build_phase_prompt = partial(build_phase_prompt, window=args.history_window)
        server.warm_up_v77(policy)
    finally:
        server.build_phase_prompt = original_prompt_builder
    if args.dry_run:
        print(json.dumps(policy.metadata, indent=2, default=str))
        return
    logging.info("Serving V7.10 on %s:%d", args.host, args.port)
    server.StreamingPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
