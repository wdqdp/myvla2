#!/usr/bin/env python3
"""V7.10 H1/H31/H61 async inference with b/fault/plan/adjustment raw recording."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

import websockets.sync.client

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import agilex_inference_v7_7_2_asyn_record_adjustment as recording
import agilex_inference_v7_7_asyn as shared
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v7_10_multitask_data import COUNTERFACTUAL_POLICY
from tactile_vla.vla.v7_10_multitask_data import ROOT
from tactile_vla.vla.v7_10_multitask_data import data_profile
from tactile_vla.vla.v7_10_phase_prompt import WINDOW_FRAMES
from tactile_vla.vla.v7_10_phase_prompt import build_phase_prompt
from tactile_vla.vla.v7_10_phase_prompt import history_policy
from tactile_vla.vla.v7_10_phase_prompt import prompt_profile


def read_server_metadata(args):
    """Read only the handshake, before choosing a versioned history profile."""
    with websockets.sync.client.connect(
        f"ws://{args.host}:{args.port}", compression=None, max_size=None,
        open_timeout=args.request_timeout_seconds, close_timeout=args.request_timeout_seconds,
    ) as connection:
        metadata = shared.msgpack_numpy.unpackb(connection.recv(timeout=args.request_timeout_seconds))
    if not isinstance(metadata, dict) or metadata.get("experiment_version") != "v7_10":
        raise ValueError("Expected a V7.10 server handshake")
    return metadata


def main(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    window_parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    window_parser.add_argument("--history-window", choices=("auto", *WINDOW_FRAMES), default="auto",
                               help="Read the server's trained history window, or assert a specific window")
    window_parser.add_argument("--expected-checkpoint-step", type=int,
                               help="Optional assertion of the server's checkpoint step")
    custom, remaining = window_parser.parse_known_args(supplied)
    if any(arg in {"-h", "--help"} for arg in supplied):
        window_parser.print_help()
    original_validate, original_prompt = shared.validate_server_metadata, shared.build_phase_prompt
    original_profile = shared.PROMPT_PROFILE
    original_data, original_norm, original_logs = shared.DATA_PROFILE, shared.DEFAULT_NORM_STATS, shared.DEFAULT_LOG_ROOT

    try:
        shared.DEFAULT_NORM_STATS = ROOT / "outputs/rotation_v4_1_new_env/norm_stats/norm_stats.json"
        shared.DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/v7_10_adjustment_measurements"
        if not any(arg.split("=", 1)[0] == "--port" for arg in remaining):
            remaining.extend(["--port", "8001"])
        args, parser = shared.get_arguments(remaining)
        args.history_window_request = custom.history_window
        args.expected_checkpoint_step = custom.expected_checkpoint_step
        if args.state_history_fps != 30.0:
            parser.error("V7.10 runtime history requires the 30 Hz grid")
        if args.expected_checkpoint_step is not None and args.expected_checkpoint_step <= 0:
            parser.error("--expected-checkpoint-step must be positive")
        shared.runtime.apply_yaml_defaults(args, parser)
        shared.validate_args(args, parser)
        metadata = read_server_metadata(args)
        window = metadata.get("history_window") if custom.history_window == "auto" else custom.history_window
        policy = history_policy(window)
        args.history_window = window
        args.resolved_checkpoint = metadata.get("checkpoint")
        args.resolved_checkpoint_step = metadata.get("checkpoint_step")
        args.resolved_training_data_hash = metadata.get("training_data_hash")
        args.resolved_full_params_sha256 = metadata.get("full_params_sha256")
        if not isinstance(args.resolved_checkpoint_step, int) or args.resolved_checkpoint_step <= 0:
            raise ValueError("V7.10 server must expose a positive checkpoint_step")
        for key in ("training_data_hash", "full_params_sha256"):
            value = metadata.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"V7.10 server must expose {key}")
        default_log_root = PROJECT_ROOT / f"outputs/runtime/v7_10_{window}_adjustment_measurements"
        if not any(arg.split("=", 1)[0] == "--log-dir" for arg in remaining):
            args.log_dir = default_log_root
        shared.DATA_PROFILE, shared.PROMPT_PROFILE = data_profile(window), prompt_profile(window)

        def validate_metadata(metadata):
            original_validate(metadata)
            expected = {
                "experiment_version": "v7_10", "history_window": window, "history_policy": policy,
                "prompt_profile": prompt_profile(window), "counterfactual_policy": COUNTERFACTUAL_POLICY,
                "adjustment_sampling_policy": "ratio_1_to_2",
                "requires_captioner": True, "requires_tactile_topics": True,
                "norm_stats_sha256": sha256_file(args.norm_stats_file),
                "checkpoint_step": args.resolved_checkpoint_step,
                "training_data_hash": args.resolved_training_data_hash,
                "full_params_sha256": args.resolved_full_params_sha256,
            }
            if args.expected_checkpoint_step is not None:
                expected["checkpoint_step"] = args.expected_checkpoint_step
            mismatch = {k: (metadata.get(k), v) for k, v in expected.items() if metadata.get(k) != v}
            if mismatch:
                raise ValueError(f"V7.10 client/server deployment mismatch: {mismatch}")

        shared.validate_server_metadata = validate_metadata
        validate_metadata(metadata)
        print(
            f"[V7.10 MODEL] window={window} H{WINDOW_FRAMES[window]} "
            f"step={args.resolved_checkpoint_step} checkpoint={args.resolved_checkpoint}", flush=True,
        )
        shared.build_phase_prompt = partial(build_phase_prompt, window=window)
        recording.main(
            data_profile=data_profile(window),
            default_norm_stats=ROOT / "outputs/rotation_v4_1_new_env/norm_stats/norm_stats.json",
            default_log_root=default_log_root,
            argument_parser=lambda: (args, parser),
            deployment_label=f"V7.10 {window} H{WINDOW_FRAMES[window]}",
            phase_history_window_frames=WINDOW_FRAMES[window],
            phase_history_training_policy=policy,
            record_lift=True,
        )
    finally:
        shared.PROMPT_PROFILE = original_profile
        shared.DATA_PROFILE, shared.DEFAULT_NORM_STATS, shared.DEFAULT_LOG_ROOT = original_data, original_norm, original_logs
        shared.build_phase_prompt = original_prompt
        shared.validate_server_metadata = original_validate


if __name__ == "__main__":
    main()
