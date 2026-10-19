#!/usr/bin/env python3
"""Serve downloaded V7.9 full_params with the tactile V7.7 async protocol."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
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
from tactile_vla.vla.v7_9_multitask_data import (
    ADJUSTMENT_SAMPLING_POLICY,
    COUNTERFACTUAL_POLICY,
    DATA_PROFILE,
    PROMPT_PROFILE,
    ROOT,
    RUN_NAME,
    VERSION_TAG,
)

DEFAULT_RUN = ROOT / "outputs/multitask_v7_9" / RUN_NAME
DEFAULT_CHECKPOINT = DEFAULT_RUN / "2000/full_params"
DEFAULT_NORM_DIR = ROOT / "outputs/rotation_v4_1_new_env/norm_stats"
EXPORT_FIELDS = (
    "experiment_version", "data_profile", "prompt_profile", "counterfactual_policy",
    "training_data_hash", "source_multitask_training_data_hash", "source_multitask_index_sha256",
    "history_policy", "adjustment_sampling_policy", "adjustment_label_policy",
    "need_successful_recovery_start", "stage_a_initialization_identity",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="full_params, numeric step, run, or multitask_v7_9 directory; run defaults to step 2000")
    parser.add_argument("--config", type=Path, help="Original V7.9 training config.json if stored separately")
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--need-recovery-threshold", type=float, default=0.5)
    parser.add_argument("--adjustment-end-threshold", type=float, default=0.5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validate-only", action="store_true", help="Inspect config/export/norm without restoring weights")
    group.add_argument("--dry-run", action="store_true", help="Restore weights and run shared warm-up, then exit")
    args = parser.parse_args(argv)
    if args.num_inference_steps <= 0 or not 1 <= args.port <= 65535:
        parser.error("inference steps must be positive and port must be in [1,65535]")
    for key in ("need_recovery_threshold", "adjustment_end_threshold"):
        value = getattr(args, key)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            parser.error(f"{key} must be finite and in [0,1]")
    args.output_action_dim, args.reasoning_max_token_len, args.no_norm = 7, None, False
    return args


def inspect_artifacts(args):
    path = args.checkpoint.expanduser().resolve()
    if path.name == "multitask_v7_9":
        path = path / RUN_NAME
    if path.name == RUN_NAME:
        path = path / "2000"
    step = path.parent if path.name == "full_params" else path
    if not step.name.isdecimal():
        raise ValueError("V7.9 requires a numeric step/full_params, run, or multitask_v7_9 directory")
    full_params = step / "full_params"
    export_file = step / "v7_9_export.json"
    candidates = [args.config.expanduser().resolve()] if args.config else [
        step / "config.json", step.parent / "config.json",
    ]
    config_file = next((p for p in candidates if p.is_file()), None)
    args.norm_stats_dir = args.norm_stats_dir.expanduser().resolve()
    norm_file = args.norm_stats_dir / "norm_stats.json"
    missing = [str(p) for p in (full_params / "_METADATA", export_file, norm_file) if not p.is_file()]
    if config_file is None:
        missing.append("original V7.9 training config.json (looked in: " + ", ".join(map(str, candidates)) + ")")
    if missing:
        raise FileNotFoundError("V7.9 deployment artifacts are incomplete; sync these files first:\n  " + "\n  ".join(missing))
    config = json.loads(config_file.read_text())
    export = json.loads(export_file.read_text())
    required = {
        "experiment_version": VERSION_TAG, "data_profile": DATA_PROFILE, "prompt_profile": PROMPT_PROFILE,
        "counterfactual_policy": COUNTERFACTUAL_POLICY, "adjustment_sampling_policy": ADJUSTMENT_SAMPLING_POLICY,
        "action_horizon": 30, "action_dim": 32, "max_token_len": 512, "reasoning_max_token_len": 320,
        "use_state_history": False, "state_history_len": 0, "state_history_dim": 7, "history_hidden_dim": 0,
        "phase_prefill_protocol": "need_failure_shared_kv_v1",
    }
    mismatches = {key: (config.get(key), value) for key, value in required.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"V7.9 training config mismatch: {mismatches}")
    mismatches = {key: (export.get(key), config.get(key)) for key in EXPORT_FIELDS
                  if key not in config or key not in export or export[key] != config[key]}
    if mismatches:
        raise ValueError(f"V7.9 export/config identity mismatch: {mismatches}")
    if export.get("step") != int(step.name) or export.get("default_deployment") != "full_params":
        raise ValueError("V7.9 export step/format mismatch")
    full_sha = export.get("exports", {}).get("full_params", {}).get("parameter_tree_sha256")
    if not isinstance(full_sha, str) or re.fullmatch(r"[0-9a-f]{64}", full_sha) is None:
        raise ValueError("V7.9 full_params parameter-tree SHA256 is missing or invalid")
    artifact = config.get("artifact_identity", {})
    norm_sha = sha256_file(norm_file)
    if artifact.get("v4_norm_stats_sha256") != norm_sha:
        raise ValueError("V7.9 norm_stats differs from the original training artifact identity")
    if artifact.get("v7_9_training_data_hash") != config["training_data_hash"]:
        raise ValueError("V7.9 training_data_hash differs from its artifact identity")
    args.checkpoint = full_params
    return config, {
        **{key: config[key] for key in EXPORT_FIELDS},
        "checkpoint": str(full_params), "checkpoint_step": int(step.name),
        "config_file": str(config_file), "export_metadata": str(export_file),
        "norm_stats_sha256": norm_sha, "full_params_sha256": full_sha,
        "requires_captioner": True, "requires_tactile_topics": True,
    }


class V79Policy(server.V77Policy):
    def __init__(self, *, args, config, identity):
        super().__init__(
            args=args, config=config, model_config=server.v3._model_config(args, config),
            norm_stats=server.normalize.load(args.norm_stats_dir),
            data_profile=DATA_PROFILE, prompt_profile=PROMPT_PROFILE,
        )
        actual_sha = parameter_tree_sha256(server.v3.nnx.state(self._model))
        if actual_sha != identity["full_params_sha256"]:
            raise ValueError("V7.9 restored parameter-tree SHA256 differs from export")
        # Keep the shared wire-protocol name; data_profile identifies V7.9.
        self._metadata.update(identity)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    args = parse_args()
    config, identity = inspect_artifacts(args)
    logging.info("V7.9 checkpoint=%s config=%s", args.checkpoint, identity["config_file"])
    logging.info("V7.9 thresholds: need_recovery=%s adjustment_end=%s",
                 args.need_recovery_threshold, args.adjustment_end_threshold)
    if args.validate_only:
        print(json.dumps(identity, indent=2, ensure_ascii=False))
        return
    policy = V79Policy(args=args, config=config, identity=identity)
    server.warm_up_v77(policy)
    if args.dry_run:
        print(json.dumps(policy.metadata, indent=2, default=str))
        return
    logging.info("Serving V7.9 on %s:%d", args.host, args.port)
    server.StreamingPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
