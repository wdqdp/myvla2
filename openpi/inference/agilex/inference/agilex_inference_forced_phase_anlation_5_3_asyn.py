#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Continuous V5.3 phase inference with asynchronous adjustment-end checks."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
import signal
import sys
import time
from typing import Any, Literal

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

import agilex_inference_forced_phase_anlation as v52
import agilex_inference_forced_phase_anlation_5_3 as v53
import agilex_inference_tactile_vla_sync_single as runtime
import numpy as np
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles
from tactile_vla.vla.v5_3_phase_change import build_adjustment_end_prompt

DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/forced_phase_ablation_v5_3_async"
ASYNC_QPOS_WINDOW_SIZE = 30
DEFAULT_ADJUSTMENT_END_RATE_HZ = 7.0
Phase = Literal["execution", "adjustment"]


@dataclass(frozen=True)
class AsyncAdjustmentEndResult:
    generation: int
    phase_index: int
    captured_step: int
    prompt: str
    qpos_h10_discrete: list[list[int]]
    feedback_qpos_h30: list[np.ndarray]
    feedback_timestamps: list[float]
    current_qpos: np.ndarray
    synchronized_timestamps: dict[str, float]
    tactile_caption: str
    adjustment_end: bool
    probabilities: np.ndarray
    submitted_monotonic: float
    finished_monotonic: float
    client_infer_ms: float
    server_infer_ms: float | None


def should_submit_adjustment_end(
    *,
    phase: Phase,
    feedback_count: int,
    request_active: bool,
    now_monotonic: float,
    last_submit_monotonic: float | None,
    target_rate_hz: float,
) -> bool:
    """Rate-limit one in-flight classifier after a complete rolling H30 exists."""

    if phase != "adjustment" or feedback_count < ASYNC_QPOS_WINDOW_SIZE or request_active:
        return False
    if target_rate_hz <= 0.0:
        raise ValueError("target_rate_hz must be positive")
    if last_submit_monotonic is None:
        return True
    interval = 1.0 / target_rate_hz
    return now_monotonic - last_submit_monotonic >= interval - 1e-9


def _validate_adjustment_end_response(response: dict[str, Any]) -> tuple[bool, np.ndarray]:
    if not isinstance(response.get("adjustment_end"), bool):
        raise ValueError("Server did not return a boolean adjustment_end")
    probabilities = np.asarray(response.get("adjustment_end_probs"), dtype=np.float32)
    if (
        probabilities.shape != (2,)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.isclose(float(probabilities.sum()), 1.0, atol=1e-4)
    ):
        raise ValueError("Server returned invalid adjustment_end_probs")
    return bool(response["adjustment_end"]), probabilities


def _run_async_adjustment_end_once(
    *,
    args: argparse.Namespace,
    policy: v53.TimeoutWebsocketPolicy,
    operator: Any,
    captioner: Any,
    stats: Any,
    generation: int,
    phase_index: int,
    captured_step: int,
    feedback_qpos_h30: list[np.ndarray],
    feedback_timestamps: list[float],
    submitted_monotonic: float,
) -> AsyncAdjustmentEndResult:
    if len(feedback_qpos_h30) != ASYNC_QPOS_WINDOW_SIZE:
        raise ValueError("Async adjustment_end requires exactly one rolling H30 qpos window")
    if len(feedback_timestamps) != ASYNC_QPOS_WINDOW_SIZE:
        raise ValueError("Async adjustment_end requires exactly 30 feedback timestamps")
    if any(left >= right for left, right in pairwise(feedback_timestamps)):
        raise ValueError("Async H30 feedback timestamps are not strictly increasing")

    observation, synchronized_timestamps = v53._capture_classification_observation(
        args,
        operator,
        captioner,
        after_timestamp=feedback_timestamps[-1],
    )
    prompt, discrete = build_adjustment_end_prompt(
        instruction=args.instruction,
        tactile_caption=observation.tactile_caption,
        recovery_plan=args.forced_recovery_plan,
        qpos_h30=np.stack(feedback_qpos_h30),
        stats=stats,
    )
    payload = runtime.build_payload(
        mode="adjustment_end",
        img_front_bgr=observation.img_front,
        img_left_bgr=observation.img_left,
        qpos=observation.qpos,
        state_history=observation.state_history,
        state_history_mask=observation.state_history_mask,
        prompt=prompt,
    )
    started = time.perf_counter()
    try:
        response = policy.infer_with_timeout(payload, timeout=args.phase_change_timeout_seconds)
    except Exception as exc:
        raise v53.FailClosedError("asynchronous adjustment_end request failed") from exc
    client_infer_ms = (time.perf_counter() - started) * 1000.0
    adjustment_end, probabilities = _validate_adjustment_end_response(response)
    finished_monotonic = time.monotonic()
    server_infer_ms = response.get("policy_timing", {}).get("infer_ms")
    if server_infer_ms is not None:
        server_infer_ms = float(server_infer_ms)
    return AsyncAdjustmentEndResult(
        generation=generation,
        phase_index=phase_index,
        captured_step=captured_step,
        prompt=prompt,
        qpos_h10_discrete=discrete,
        feedback_qpos_h30=feedback_qpos_h30,
        feedback_timestamps=feedback_timestamps,
        current_qpos=observation.qpos,
        synchronized_timestamps=synchronized_timestamps,
        tactile_caption=observation.tactile_caption,
        adjustment_end=adjustment_end,
        probabilities=probabilities,
        submitted_monotonic=submitted_monotonic,
        finished_monotonic=finished_monotonic,
        client_infer_ms=client_infer_ms,
        server_infer_ms=server_infer_ms,
    )


def _report_async_adjustment_end(
    *,
    args: argparse.Namespace,
    logger: v52.TrialLogger,
    result: AsyncAdjustmentEndResult,
    handled_step: int,
    previous_submit_monotonic: float | None,
) -> None:
    submit_interval_ms = (
        (result.submitted_monotonic - previous_submit_monotonic) * 1000.0
        if previous_submit_monotonic is not None
        else None
    )
    actual_rate_hz = (
        1000.0 / submit_interval_ms
        if submit_interval_ms is not None and submit_interval_ms > 0.0
        else None
    )
    lag_steps = max(0, handled_step - result.captured_step)
    rate_text = f" rate={actual_rate_hz:.2f}Hz" if actual_rate_hz is not None else ""
    print(
        "[ADJUSTMENT_END async] "
        f"result={result.adjustment_end} probs={result.probabilities.tolist()} "
        f"lag_steps={lag_steps}{rate_text}"
    )
    logger.record({
        "event": "adjustment_end_async_inference",
        "generation": result.generation,
        "phase_index": result.phase_index,
        "captured_step": result.captured_step,
        "handled_step": handled_step,
        "lag_steps": lag_steps,
        "adjustment_end": result.adjustment_end,
        "adjustment_end_probs": result.probabilities,
        "prompt": result.prompt,
        "qpos_h10_discrete": result.qpos_h10_discrete,
        "feedback_qpos_h30": result.feedback_qpos_h30,
        "feedback_timestamps": result.feedback_timestamps,
        "current_qpos": result.current_qpos,
        "synchronized_timestamps": result.synchronized_timestamps,
        "tactile_caption": result.tactile_caption,
        "target_rate_hz": args.adjustment_end_rate_hz,
        "submit_interval_ms": submit_interval_ms,
        "actual_submit_rate_hz": actual_rate_hz,
        "client_infer_ms": result.client_infer_ms,
        "server_infer_ms": result.server_infer_ms,
        "total_async_ms": (result.finished_monotonic - result.submitted_monotonic) * 1000.0,
    })


def _poll_key(args: argparse.Namespace, keyboard: Any, *, phase: Phase) -> str | None:
    key = keyboard.get_key()
    if key is None:
        return None
    if key == args.quit_key:
        return "quit"
    if key == " " and phase == "execution":
        return "trigger"
    return None


def _enter_adjustment(
    *,
    args: argparse.Namespace,
    operator: Any,
    captioner: Any,
    logger: v52.TrialLogger,
    published_steps: int,
    discarded_raw_actions: int,
) -> v52.FrozenObservation:
    """Switch prompts using a live observation without touching the H60 buffer."""

    observation = v52._capture_observation(
        args,
        operator,
        captioner,
        reset_history=False,
        after_timestamp=time.time(),
    )
    image_paths = logger.save_trigger_images(observation.img_front, observation.img_left)
    logger.record({
        "event": "forced_recovery_trigger",
        "previous_phase": "execution",
        "phase": "adjustment",
        "published_steps": published_steps,
        "discarded_raw_actions": discarded_raw_actions,
        "rotation_direction": args.rotation_direction,
        "preset_failure_reason": args.forced_failure_reason,
        "forced_recovery_plan": args.forced_recovery_plan,
        "tactile_caption": observation.tactile_caption,
        "observation_timestamp": observation.timestamp,
        "switch_qpos": observation.qpos,
        "reset_state_history": False,
        "live_state_history_valid_frames": int(observation.state_history_mask.sum()),
        **image_paths,
    })
    print(
        "ADJUSTMENT started: discarded the remaining EXECUTION chunk; "
        "live H60 history was not paused or reset; "
        f"rotation_direction={args.rotation_direction} plan={args.forced_recovery_plan}."
    )
    return observation


def run_v5_3_async(
    args: argparse.Namespace,
    operator: Any,
    action_policy: v53.TimeoutWebsocketPolicy,
    classification_policy: v53.TimeoutWebsocketPolicy,
    captioner: Any,
    keyboard: Any,
    logger: v52.TrialLogger,
) -> None:
    action_metadata = action_policy.get_server_metadata()
    classification_metadata = classification_policy.get_server_metadata()
    v53.validate_server_metadata(args, action_metadata)
    v53.validate_server_metadata(args, classification_metadata)
    if action_metadata != classification_metadata:
        raise ValueError("Action and asynchronous classification connections expose different metadata")
    stats = load_state_quantiles(args.norm_stats_file)
    logger.record({
        "event": "run_start",
        "server_metadata": action_metadata,
        "args": vars(args),
        "continuous_live_state_history": True,
        "async_adjustment_end": True,
    })
    print(
        "V5.3 async controls: SPACE=execution to adjustment, q=quit; chunks continue automatically. "
        f"adjustment_end target rate={args.adjustment_end_rate_hz:g}Hz, "
        f"server threshold={action_metadata['adjustment_end_threshold']}."
    )
    if action_metadata.get("adjustment_end_experimental_override", False):
        print("WARNING: EXPERIMENTAL adjustment_end deployment is active.")

    phase: Phase = "execution"
    phase_indices = {"execution": 0, "adjustment": 0}
    generation = 0
    pending_observation: v52.FrozenObservation | None = None
    pre_action: np.ndarray | None = None
    published_steps = 0
    feedback_window: deque[tuple[np.ndarray, float]] = deque(maxlen=ASYNC_QPOS_WINDOW_SIZE)
    previous_reported_submit: float | None = None
    last_submit: float | None = None

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="v5-3-adjustment-end") as executor:
        future: Future[AsyncAdjustmentEndResult] | None = None

        def consume_finished_request() -> bool:
            """Return True only when this result changed adjustment to execution."""

            nonlocal future, generation, last_submit, phase, previous_reported_submit
            if future is None or not future.done():
                return False
            result = future.result()
            future = None
            if result.generation != generation or phase != "adjustment":
                logger.record({
                    "event": "adjustment_end_async_stale_result",
                    "result_generation": result.generation,
                    "active_generation": generation,
                    "active_phase": phase,
                })
                return False
            _report_async_adjustment_end(
                args=args,
                logger=logger,
                result=result,
                handled_step=published_steps,
                previous_submit_monotonic=previous_reported_submit,
            )
            previous_reported_submit = result.submitted_monotonic
            if not result.adjustment_end:
                return False
            logger.record({
                "event": "phase_transition",
                "previous_phase": "adjustment",
                "phase": "execution",
                "trigger": "asynchronous_adjustment_end",
                "captured_step": result.captured_step,
                "handled_step": published_steps,
                "reset_state_history": False,
            })
            print("[PHASE] adjustment_end=true; switching to EXECUTION before the next action.")
            phase = "execution"
            generation += 1
            feedback_window.clear()
            last_submit = None
            return True

        def maybe_submit_request(*, phase_index: int) -> None:
            nonlocal future, last_submit
            now = time.monotonic()
            if not should_submit_adjustment_end(
                phase=phase,
                feedback_count=len(feedback_window),
                request_active=future is not None,
                now_monotonic=now,
                last_submit_monotonic=last_submit,
                target_rate_hz=args.adjustment_end_rate_hz,
            ):
                return
            qpos_h30 = [qpos.copy() for qpos, _ in feedback_window]
            timestamps = [timestamp for _, timestamp in feedback_window]
            future = executor.submit(
                _run_async_adjustment_end_once,
                args=args,
                policy=classification_policy,
                operator=operator,
                captioner=captioner,
                stats=stats,
                generation=generation,
                phase_index=phase_index,
                captured_step=published_steps,
                feedback_qpos_h30=qpos_h30,
                feedback_timestamps=timestamps,
                submitted_monotonic=now,
            )
            last_submit = now
            logger.record({
                "event": "adjustment_end_async_submit",
                "generation": generation,
                "phase_index": phase_index,
                "captured_step": published_steps,
                "feedback_timestamp_first": timestamps[0],
                "feedback_timestamp_last": timestamps[-1],
                "target_rate_hz": args.adjustment_end_rate_hz,
            })

        while published_steps < args.max_publish_step and not operator.is_shutdown():
            consume_finished_request()
            signal_value = _poll_key(args, keyboard, phase=phase)
            if signal_value == "quit":
                runtime.shutdown_event.set()
                logger.record({"event": "operator_quit", "phase": phase, "published_steps": published_steps})
                return
            if signal_value == "trigger":
                pending_observation = _enter_adjustment(
                    args=args,
                    operator=operator,
                    captioner=captioner,
                    logger=logger,
                    published_steps=published_steps,
                    discarded_raw_actions=0,
                )
                phase = "adjustment"
                generation += 1
                feedback_window.clear()
                last_submit = None
                previous_reported_submit = None
                pre_action = pending_observation.qpos.copy()

            observation = pending_observation or v52._capture_observation(
                args,
                operator,
                captioner,
                reset_history=False,
            )
            pending_observation = None
            if pre_action is None:
                pre_action = observation.qpos.copy()
            requested_phase = phase
            phase_index = phase_indices[requested_phase]
            _, actions = v52._request_action_chunk(
                args=args,
                policy=action_policy,
                logger=logger,
                observation=observation,
                phase=requested_phase,
                phase_index=phase_index,
            )
            phase_indices[requested_phase] += 1

            if consume_finished_request() and requested_phase == "adjustment":
                logger.record({
                    "event": "execution_chunk",
                    "phase": requested_phase,
                    "phase_index": phase_index,
                    "completed_raw_actions": 0,
                    "discarded_raw_actions": len(actions),
                    "control_signal": "adjustment_end",
                })
                continue

            limit = min(args.chunk_size, args.max_publish_step - published_steps, len(actions))
            complete = 0
            control: str | None = None
            control_rate = operator.rate(args.publish_rate)
            for action_index, action in enumerate(actions[:limit]):
                if consume_finished_request():
                    control = "adjustment_end"
                    break
                _, before_timestamp = v52._latest_feedback(operator)
                pre_action, _, raw_control = v52._publish_raw_action(
                    args=args,
                    operator=operator,
                    keyboard=keyboard,
                    logger=logger,
                    phase=requested_phase,
                    phase_index=phase_index,
                    action_index=action_index,
                    raw_action=action,
                    pre_action=pre_action,
                    control_rate=control_rate,
                )
                if raw_control is not None:
                    control = raw_control.kind
                    break
                feedback, timestamp = v53._wait_feedback_after(args, operator, before_timestamp)
                complete += 1
                published_steps += 1
                if phase == "adjustment":
                    feedback_window.append((feedback, timestamp))

                if consume_finished_request():
                    control = "adjustment_end"
                    break
                if phase == "adjustment":
                    maybe_submit_request(phase_index=phase_index)

            logger.record({
                "event": "execution_chunk",
                "phase": requested_phase,
                "phase_index": phase_index,
                "completed_raw_actions": complete,
                "discarded_raw_actions": max(0, limit - complete),
                "control_signal": control,
            })
            if control == "quit":
                runtime.shutdown_event.set()
                return
            if control == "trigger":
                if requested_phase != "execution":
                    raise AssertionError("SPACE is valid only in execution")
                pending_observation = _enter_adjustment(
                    args=args,
                    operator=operator,
                    captioner=captioner,
                    logger=logger,
                    published_steps=published_steps,
                    discarded_raw_actions=max(0, limit - complete),
                )
                phase = "adjustment"
                generation += 1
                feedback_window.clear()
                last_submit = None
                previous_reported_submit = None
                pre_action = pending_observation.qpos.copy()
            elif control == "adjustment_end":
                if phase != "execution":
                    raise AssertionError("adjustment_end must transition to execution")

        logger.record({"event": "max_publish_step_reached", "published_steps": published_steps})


def get_arguments() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--instruction", default=v52.DEFAULT_INSTRUCTION)
    parser.add_argument("--rotation-direction", choices=v52.ROTATION_DIRECTIONS)
    parser.add_argument("--rotation-magnitude", choices=("moderately", "slightly"), default="moderately")
    parser.add_argument("--forced-failure-reason")
    parser.add_argument("--forced-recovery-plan")
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--norm-stats-file", type=Path, default=v53.DEFAULT_NORM_STATS)
    parser.add_argument("--phase-change-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--adjustment-end-rate-hz", type=float, default=DEFAULT_ADJUSTMENT_END_RATE_HZ)
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--observation-poll-rate", type=int, default=200)
    parser.add_argument("--state-history-len", type=int, default=60)
    parser.add_argument("--state-history-fps", type=float, default=30.0)
    parser.add_argument("--state-history-max-gap-seconds", type=float, default=0.02)
    parser.add_argument("--img_front_topic", default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", default="/camera_l/color/image_raw")
    parser.add_argument("--puppet_arm_cmd_topic", default="/master/joint_right")
    parser.add_argument("--puppet_arm_topic", default="/puppet/joint_right")
    parser.add_argument("--tactile_left_force_topic", default="/xense/OG001251/force")
    parser.add_argument("--tactile_right_force_topic", default="/xense/OG000991/force")
    parser.add_argument("--tactile_left_mesh_3d_topic", default="/xense/OG001251/mesh_3d")
    parser.add_argument("--tactile_right_mesh_3d_topic", default="/xense/OG000991/mesh_3d")
    parser.add_argument("--tactile_left_mesh_3d_flow_topic", default="/xense/OG001251/mesh_3d_flow")
    parser.add_argument("--tactile_right_mesh_3d_flow_topic", default="/xense/OG000991/mesh_3d_flow")
    parser.add_argument("--tactile_window_size", type=int, default=30)
    parser.add_argument("--captioner_checkpoint", type=Path, default=v53.DEFAULT_CAPTIONER)
    parser.add_argument("--captioner_device", default="auto")
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--quit-key", default="q")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--replay-attempt-dir", type=Path)
    parser.add_argument("--replay-start-index", type=int, default=0)
    parser.add_argument("--replay-step-stride", type=int, default=1)
    parser.add_argument("--replay-max-frames", type=int)
    parser.add_argument("--use_actions_interpolation", action="store_true")
    parser.add_argument("--arm_steps_length", nargs=7, type=float, default=[0.01] * 6 + [0.2])
    parser.add_argument("--gripper_offset", type=float, default=0.001)
    parser.add_argument("--gripper-min", dest="gripper_min", type=float, required=True)
    parser.add_argument(
        "--allow-experimental-adjustment-end",
        action="store_true",
        help="Acknowledge a non-official checkpoint or manual server threshold override",
    )
    return parser.parse_args(), parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    v53.validate_args(args, parser)
    if args.publish_rate <= 0:
        parser.error("--publish_rate must be positive")
    if args.observation_poll_rate <= 0:
        parser.error("--observation-poll-rate must be positive")
    if not 0.0 < args.adjustment_end_rate_hz <= args.publish_rate:
        parser.error("--adjustment-end-rate-hz must be positive and at most --publish_rate")
    if args.replay_attempt_dir is not None:
        parser.error("The asynchronous real-time classifier does not support replay mode")


def _close_policy(policy: v53.TimeoutWebsocketPolicy) -> None:
    close = getattr(getattr(policy, "_ws", None), "close", None)
    if callable(close):
        close()


def main() -> None:
    args, parser = get_arguments()
    runtime.apply_yaml_defaults(args, parser)
    validate_args(args, parser)
    if not sys.stdin.isatty():
        parser.error("V5.3 asynchronous manual inference requires an interactive TTY")
    runtime.shutdown_event.clear()
    signal.signal(signal.SIGINT, runtime._on_sigint)
    captioner = runtime.load_captioner(args)
    if captioner is None:
        parser.error("V5.3 does not allow --no-captioner")
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
            try:
                run_v5_3_async(
                    args,
                    operator,
                    action_policy,
                    classification_policy,
                    captioner,
                    keyboard,
                    logger,
                )
            except Exception as exc:
                logger.record({"event": "fail_closed_safety_stop", "error": repr(exc)})
                print(
                    f"FAIL-CLOSED safety stop: {exc}. "
                    "No more actions will be published; press q to exit."
                )
                rate = operator.rate(args.observation_poll_rate)
                while not operator.is_shutdown() and not runtime.shutdown_event.is_set():
                    if keyboard.get_key() == args.quit_key:
                        runtime.shutdown_event.set()
                        logger.record({"event": "fail_closed_operator_quit"})
                        break
                    rate.sleep()
    except KeyboardInterrupt:
        runtime.shutdown_event.set()
        logger.record({"event": "keyboard_interrupt"})
    finally:
        _close_policy(classification_policy)
        _close_policy(action_policy)


if __name__ == "__main__":
    main()
