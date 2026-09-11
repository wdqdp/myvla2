#!/usr/bin/env python3
"""Asynchronous single-arm V7.5 adjustment-end inference with raw H100 qpos."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
import signal
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

import agilex_inference_forced_phase_anlation as v52
import agilex_inference_forced_phase_anlation_5_3 as v53
import agilex_inference_forced_phase_anlation_5_3_asyn as base
import agilex_inference_tactile_vla_sync_single as runtime
import numpy as np
from tactile_vla.vla.v7_4_adjustment_data import ROTATION_PHASE_V7_4_ADJUSTMENT
from tactile_vla.vla.v7_4_adjustment_data import V7_4_EXPERIMENT_KIND
from tactile_vla.vla.v7_5_adjustment_end_data import DATA_PROFILE
from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_PROMPT_PROFILE
from tactile_vla.vla.v7_5_runtime_history import RUNTIME_PAST_QPOS_FRAMES
from tactile_vla.vla.v7_5_runtime_history import RUNTIME_SAMPLE_OFFSETS
from tactile_vla.vla.v7_5_runtime_history import build_runtime_adjustment_end_prompt

DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/forced_phase_ablation_v7_5_async"
DEFAULT_NORM_STATS = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/rotation_v4/norm_stats/norm_stats.json")


@dataclass(frozen=True)
class AsyncV75AdjustmentEndResult:
    generation: int
    phase_index: int
    captured_step: int
    prompt: str
    qpos_h100_11_discrete: list[list[int]]
    feedback_qpos_h100: list[np.ndarray]
    feedback_timestamps_h100: list[float]
    current_qpos: np.ndarray
    synchronized_timestamps: dict[str, float]
    tactile_caption: str
    adjustment_end: bool
    probabilities: np.ndarray
    submitted_monotonic: float
    finished_monotonic: float
    client_infer_ms: float
    server_infer_ms: float | None


def validate_server_metadata(args, metadata: dict[str, Any]) -> None:
    expected = {
        "supports_action_noise": True,
        "requires_action_noise": True,
        "supports_adjustment_end": True,
        "prompt_profile": "phase_v2",
        "data_profile": ROTATION_PHASE_V7_4_ADJUSTMENT,
        "experiment_kind": V7_4_EXPERIMENT_KIND,
        "stage_a_protocol": "v7_4_no_state_history",
        "phase_change_prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
        "qpos_history_frames": 100,
        "qpos_history_includes_current": True,
        "qpos_sampled_frames": 11,
        "qpos_h100_sample_offsets": list(RUNTIME_SAMPLE_OFFSETS),
        "runtime_history_policy": "uniform_h100_no_ga_idle_compression",
        "captioner_window_size": 30,
        "action_horizon": 30,
        "action_dim": 32,
        "output_action_dim": 7,
        "state_history_len": 0,
        "state_history_dim": 7,
        "use_state_history": False,
        "adjustment_end_data_profile": DATA_PROFILE,
    }
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise ValueError(f"V7.5 client/server metadata mismatch: {mismatch}")
    if not 0.0 <= float(metadata.get("adjustment_end_threshold", -1.0)) <= 1.0:
        raise ValueError("V7.5 server adjustment_end_threshold is invalid")
    if metadata.get("adjustment_end_experimental_override", False) and not getattr(args, "allow_experimental_adjustment_end", False):
        raise ValueError("Pass --allow-experimental-adjustment-end to acknowledge a threshold override")
    if float(metadata.get("phase_change_timeout_seconds", -1.0)) != args.phase_change_timeout_seconds:
        raise ValueError("V7.5 client/server phase-change timeout mismatch")
    if metadata.get("captioner_checkpoint_sha256") != v53.sha256_file(args.captioner_checkpoint):
        raise ValueError("Client captioner differs from the V7.5 classifier training captioner")


def _run_async_adjustment_end_once(
    *, args, policy, operator, captioner, stats, generation, phase_index, captured_step,
    feedback_qpos_h30, feedback_timestamps, submitted_monotonic,
) -> AsyncV75AdjustmentEndResult:
    """Use 99 post-action feedback points plus a synchronized current point p."""

    if len(feedback_qpos_h30) != RUNTIME_PAST_QPOS_FRAMES or len(feedback_timestamps) != RUNTIME_PAST_QPOS_FRAMES:
        raise ValueError("V7.5 asynchronous adjustment_end requires exactly 99 past qpos frames")
    if any(left >= right for left, right in pairwise(feedback_timestamps)):
        raise ValueError("V7.5 past qpos timestamps are not strictly increasing")
    observation, synchronized_timestamps = v53._capture_classification_observation(
        args, operator, captioner, after_timestamp=feedback_timestamps[-1]
    )
    prompt, discrete, h100 = build_runtime_adjustment_end_prompt(
        instruction=args.instruction, tactile_caption=observation.tactile_caption,
        recovery_plan=args.forced_recovery_plan, past_qpos_h99=np.stack(feedback_qpos_h30),
        current_qpos=observation.qpos, stats=stats,
    )
    payload = runtime.build_payload(
        mode="adjustment_end", img_front_bgr=observation.img_front,
        img_left_bgr=observation.img_left, qpos=observation.qpos,
        state_history=None, state_history_mask=None, prompt=prompt,
    )
    started = time.perf_counter()
    try:
        response = policy.infer_with_timeout(payload, timeout=args.phase_change_timeout_seconds)
    except Exception as exc:
        raise v53.FailClosedError("asynchronous V7.5 adjustment_end request failed") from exc
    adjustment_end, probabilities = base._validate_adjustment_end_response(response)
    server_infer_ms = response.get("policy_timing", {}).get("infer_ms")
    return AsyncV75AdjustmentEndResult(
        generation=generation, phase_index=phase_index, captured_step=captured_step, prompt=prompt,
        qpos_h100_11_discrete=discrete.tolist(), feedback_qpos_h100=[row.copy() for row in h100],
        feedback_timestamps_h100=[*feedback_timestamps, float(observation.timestamp)],
        current_qpos=observation.qpos, synchronized_timestamps=synchronized_timestamps,
        tactile_caption=observation.tactile_caption, adjustment_end=adjustment_end,
        probabilities=probabilities, submitted_monotonic=submitted_monotonic,
        finished_monotonic=time.monotonic(), client_infer_ms=(time.perf_counter() - started) * 1000.0,
        server_infer_ms=float(server_infer_ms) if server_infer_ms is not None else None,
    )


def _report_async_adjustment_end(*, args, logger, result, handled_step, previous_submit_monotonic) -> None:
    submit_interval_ms = (
        (result.submitted_monotonic - previous_submit_monotonic) * 1000.0
        if previous_submit_monotonic is not None else None
    )
    lag_steps = max(0, handled_step - result.captured_step)
    print(f"[V7.5 ADJUSTMENT_END async] result={result.adjustment_end} probs={result.probabilities.tolist()} lag_steps={lag_steps}")
    logger.record({
        "event": "v7_5_adjustment_end_async_inference", "generation": result.generation,
        "phase_index": result.phase_index, "captured_step": result.captured_step,
        "handled_step": handled_step, "lag_steps": lag_steps, "adjustment_end": result.adjustment_end,
        "adjustment_end_probs": result.probabilities, "prompt": result.prompt,
        "qpos_h100_11_discrete": result.qpos_h100_11_discrete,
        "feedback_qpos_h100": result.feedback_qpos_h100,
        "feedback_timestamps_h100": result.feedback_timestamps_h100,
        "current_qpos": result.current_qpos, "synchronized_timestamps": result.synchronized_timestamps,
        "tactile_caption": result.tactile_caption, "target_rate_hz": args.adjustment_end_rate_hz,
        "submit_interval_ms": submit_interval_ms,
        "actual_submit_rate_hz": 1000.0 / submit_interval_ms if submit_interval_ms and submit_interval_ms > 0 else None,
        "client_infer_ms": result.client_infer_ms, "server_infer_ms": result.server_infer_ms,
        "total_async_ms": (result.finished_monotonic - result.submitted_monotonic) * 1000.0,
    })


def _configure_base() -> None:
    # The base controller owns safe action publication.  Override only its
    # classifier protocol in this process; V5.3/V7 programs are unaffected.
    v53.V7_DATA_PROFILE = ROTATION_PHASE_V7_4_ADJUSTMENT
    base.v53.V7_DATA_PROFILE = ROTATION_PHASE_V7_4_ADJUSTMENT
    v53.DEFAULT_NORM_STATS = DEFAULT_NORM_STATS
    base.DEFAULT_LOG_ROOT = DEFAULT_LOG_ROOT
    base.__doc__ = __doc__
    base.ASYNC_QPOS_WINDOW_SIZE = RUNTIME_PAST_QPOS_FRAMES
    base.v53.validate_server_metadata = validate_server_metadata
    base._run_async_adjustment_end_once = _run_async_adjustment_end_once
    base._report_async_adjustment_end = _report_async_adjustment_end


def main() -> None:
    _configure_base()
    if "--expected-data-profile" not in sys.argv and not any(value.startswith("--expected-data-profile=") for value in sys.argv[1:]):
        sys.argv.extend(["--expected-data-profile", ROTATION_PHASE_V7_4_ADJUSTMENT])
    args, parser = base.get_arguments()
    runtime.apply_yaml_defaults(args, parser)
    base.validate_args(args, parser)
    if not sys.stdin.isatty():
        parser.error("V7.5 asynchronous manual inference requires an interactive TTY")
    runtime.shutdown_event.clear()
    signal.signal(signal.SIGINT, runtime._on_sigint)
    captioner = runtime.load_captioner(args)
    if captioner is None:
        parser.error("V7.5 requires a tactile captioner")
    action_policy = v53.TimeoutWebsocketPolicy(args.host, args.port)
    classification_policy = v53.TimeoutWebsocketPolicy(args.host, args.port)
    operator = runtime.RosOperator(args)
    v53._install_tactile_arrival_timestamps(operator)
    trial_id = args.trial_id
    if trial_id and (args.log_dir.resolve() / trial_id).exists():
        trial_id = f"{trial_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = v52.TrialLogger(args.log_dir, trial_id=trial_id)
    print(f"Trial log directory: {logger.directory}")
    try:
        if not args.start_immediately:
            input("Press enter to start EXECUTION")
        with runtime.KeyboardPoller() as keyboard:
            base.run_v5_3_async(args, operator, action_policy, classification_policy, captioner, keyboard, logger)
    except Exception as exc:
        logger.record({"event": "fail_closed_safety_stop", "error": repr(exc)})
        print(f"FAIL-CLOSED safety stop: {exc}. No more actions will be published; press q to exit.")
    finally:
        base._close_policy(classification_policy)
        base._close_policy(action_policy)


if __name__ == "__main__":
    main()
