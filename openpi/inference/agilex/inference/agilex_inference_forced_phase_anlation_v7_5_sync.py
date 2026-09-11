#!/usr/bin/env python3
"""Synchronous single-arm V7.5 adjustment-end inference with raw H100 qpos."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import signal
import sys
import time
from typing import Literal

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

import agilex_inference_forced_phase_anlation as v52
import agilex_inference_forced_phase_anlation_5_3 as v53
import agilex_inference_forced_phase_anlation_5_3_asyn as async_base
import agilex_inference_forced_phase_anlation_v7_5_asyn as v75_async
import agilex_inference_tactile_vla_sync_single as runtime
import numpy as np
from tactile_vla.vla.v7_4_adjustment_data import ROTATION_PHASE_V7_4_ADJUSTMENT
from tactile_vla.vla.v7_5_adjustment_end_data import load_state_quantiles
from tactile_vla.vla.v7_5_runtime_history import RUNTIME_PAST_QPOS_FRAMES

DEFAULT_CAPTIONER = Path("/data1/outputs/tactile_captioner/tcn_v3_w30_rotation_head/best.pt")
DEFAULT_NORM_STATS = Path(
    "/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/rotation_v4/norm_stats/norm_stats.json"
)
DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/forced_phase_ablation_v7_5_sync"
Phase = Literal["execution", "adjustment"]


def should_classify_adjustment_end(*, phase: Phase, completed_raw_actions: int, feedback_count: int) -> bool:
    """Classify only after a complete adjustment chunk with a full past H99."""

    return (
        phase == "adjustment" and int(completed_raw_actions) == 30 and int(feedback_count) == RUNTIME_PAST_QPOS_FRAMES
    )


def _classify_adjustment_end_sync(
    *,
    args: argparse.Namespace,
    policy: v53.TimeoutWebsocketPolicy,
    operator,
    captioner,
    stats,
    logger: v52.TrialLogger,
    phase_index: int,
    published_steps: int,
    feedback_window: deque[tuple[np.ndarray, float]],
) -> bool:
    """Block action generation until one V7.5 H100 classification finishes."""

    if len(feedback_window) != RUNTIME_PAST_QPOS_FRAMES:
        raise ValueError("V7.5 synchronous classification requires exactly 99 past qpos frames")
    qpos = [value.copy() for value, _ in feedback_window]
    timestamps = [float(timestamp) for _, timestamp in feedback_window]
    submitted = time.monotonic()
    try:
        result = v75_async._run_async_adjustment_end_once(
            args=args,
            policy=policy,
            operator=operator,
            captioner=captioner,
            stats=stats,
            generation=0,
            phase_index=phase_index,
            captured_step=published_steps,
            feedback_qpos_h30=qpos,
            feedback_timestamps=timestamps,
            submitted_monotonic=submitted,
        )
    except Exception as exc:
        raise v53.FailClosedError("synchronous V7.5 adjustment_end request failed") from exc
    logger.record(
        {
            "event": "v7_5_adjustment_end_sync_inference",
            "phase_index": phase_index,
            "captured_step": published_steps,
            "adjustment_end": result.adjustment_end,
            "adjustment_end_probs": result.probabilities,
            "prompt": result.prompt,
            "qpos_h100_11_discrete": result.qpos_h100_11_discrete,
            "feedback_qpos_h100": result.feedback_qpos_h100,
            "feedback_timestamps_h100": result.feedback_timestamps_h100,
            "current_qpos": result.current_qpos,
            "synchronized_timestamps": result.synchronized_timestamps,
            "tactile_caption": result.tactile_caption,
            "client_infer_ms": result.client_infer_ms,
            "server_infer_ms": result.server_infer_ms,
            "total_sync_ms": (result.finished_monotonic - submitted) * 1000.0,
        }
    )
    print(f"[V7.5 ADJUSTMENT_END sync] result={result.adjustment_end} probs={result.probabilities.tolist()}")
    return bool(result.adjustment_end)


def run_v7_5_sync(args, operator, policy, captioner, keyboard, logger) -> None:
    metadata = policy.get_server_metadata()
    v75_async.validate_server_metadata(args, metadata)
    args.use_state_history = False
    stats = load_state_quantiles(args.norm_stats_file)
    logger.record(
        {
            "event": "run_start",
            "server_metadata": metadata,
            "args": vars(args),
            "sync_adjustment_end": True,
            "qpos_past_frames_required": RUNTIME_PAST_QPOS_FRAMES,
        }
    )
    print(
        "V7.5 sync controls: SPACE=execution to adjustment, q=quit; chunks continue "
        "automatically. After H99 is available, every complete adjustment chunk is "
        "followed by one blocking adjustment_end request. "
        f"Server threshold={metadata['adjustment_end_threshold']}."
    )
    if metadata.get("adjustment_end_experimental_override", False):
        print("WARNING: EXPERIMENTAL manual adjustment_end threshold is active.")

    phase: Phase = "execution"
    phase_indices = {"execution": 0, "adjustment": 0}
    pending_observation: v52.FrozenObservation | None = None
    pre_action: np.ndarray | None = None
    published_steps = 0
    feedback_window: deque[tuple[np.ndarray, float]] = deque(maxlen=RUNTIME_PAST_QPOS_FRAMES)

    while published_steps < args.max_publish_step and not operator.is_shutdown():
        key = async_base._poll_key(args, keyboard, phase=phase)
        if key == "quit":
            runtime.shutdown_event.set()
            return
        if key == "trigger":
            pending_observation = async_base._enter_adjustment(
                args=args,
                operator=operator,
                captioner=captioner,
                logger=logger,
                published_steps=published_steps,
                discarded_raw_actions=0,
            )
            phase = "adjustment"
            feedback_window.clear()
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
            policy=policy,
            logger=logger,
            observation=observation,
            phase=requested_phase,
            phase_index=phase_index,
        )
        phase_indices[requested_phase] += 1
        limit = min(args.chunk_size, args.max_publish_step - published_steps, len(actions))
        complete = 0
        control: str | None = None
        control_rate = operator.rate(args.publish_rate)

        for action_index, action in enumerate(actions[:limit]):
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
            if requested_phase == "adjustment":
                feedback_window.append((feedback.copy(), float(timestamp)))

        logger.record(
            {
                "event": "execution_chunk",
                "phase": requested_phase,
                "phase_index": phase_index,
                "completed_raw_actions": complete,
                "discarded_raw_actions": max(0, limit - complete),
                "control_signal": control,
            }
        )
        if control == "quit":
            runtime.shutdown_event.set()
            return
        if control == "trigger":
            if requested_phase != "execution":
                raise AssertionError("SPACE is valid only in execution")
            pending_observation = async_base._enter_adjustment(
                args=args,
                operator=operator,
                captioner=captioner,
                logger=logger,
                published_steps=published_steps,
                discarded_raw_actions=max(0, limit - complete),
            )
            phase = "adjustment"
            feedback_window.clear()
            pre_action = pending_observation.qpos.copy()
            continue

        if requested_phase != "adjustment":
            continue
        if complete != args.chunk_size:
            logger.record(
                {
                    "event": "v7_5_adjustment_end_sync_skipped_incomplete_chunk",
                    "phase_index": phase_index,
                    "completed_raw_actions": complete,
                }
            )
            continue
        if not should_classify_adjustment_end(
            phase=requested_phase,
            completed_raw_actions=complete,
            feedback_count=len(feedback_window),
        ):
            logger.record(
                {
                    "event": "v7_5_adjustment_end_sync_waiting_for_h99",
                    "phase_index": phase_index,
                    "feedback_count": len(feedback_window),
                    "required_feedback_count": RUNTIME_PAST_QPOS_FRAMES,
                }
            )
            print(
                "[V7.5 ADJUSTMENT_END sync] waiting for H99: "
                f"{len(feedback_window)}/{RUNTIME_PAST_QPOS_FRAMES}; continuing adjustment."
            )
            continue

        ended = _classify_adjustment_end_sync(
            args=args,
            policy=policy,
            operator=operator,
            captioner=captioner,
            stats=stats,
            logger=logger,
            phase_index=phase_index,
            published_steps=published_steps,
            feedback_window=feedback_window,
        )
        if ended:
            logger.record(
                {
                    "event": "phase_transition",
                    "previous_phase": "adjustment",
                    "phase": "execution",
                    "trigger": "synchronous_adjustment_end",
                    "phase_index": phase_index,
                    "published_steps": published_steps,
                    "reset_state_history": False,
                }
            )
            print("[PHASE] adjustment_end=true; switching to EXECUTION before next chunk.")
            phase = "execution"
            feedback_window.clear()

    logger.record({"event": "max_publish_step_reached", "phase": phase, "published_steps": published_steps})


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
    parser.add_argument(
        "--expected-data-profile",
        choices=(ROTATION_PHASE_V7_4_ADJUSTMENT,),
        default=ROTATION_PHASE_V7_4_ADJUSTMENT,
    )
    parser.add_argument("--norm-stats-file", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--phase-change-timeout-seconds", type=float, default=10.0)
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
    parser.add_argument("--captioner_checkpoint", type=Path, default=DEFAULT_CAPTIONER)
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
        help="Acknowledge a manual server threshold override",
    )
    return parser.parse_args(), parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    v53.V7_DATA_PROFILE = ROTATION_PHASE_V7_4_ADJUSTMENT
    v53.validate_args(args, parser)
    if args.publish_rate <= 0 or args.observation_poll_rate <= 0:
        parser.error("publish and observation polling rates must be positive")
    if args.replay_attempt_dir is not None:
        parser.error("V7.5 synchronous real-time inference does not support replay mode")


def main() -> None:
    args, parser = get_arguments()
    runtime.apply_yaml_defaults(args, parser)
    validate_args(args, parser)
    if not sys.stdin.isatty():
        parser.error("V7.5 synchronous manual inference requires an interactive TTY")
    runtime.shutdown_event.clear()
    signal.signal(signal.SIGINT, runtime._on_sigint)
    captioner = runtime.load_captioner(args)
    if captioner is None:
        parser.error("V7.5 requires a tactile captioner")
    policy = v53.TimeoutWebsocketPolicy(args.host, args.port)
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
                run_v7_5_sync(args, operator, policy, captioner, keyboard, logger)
            except Exception as exc:
                logger.record({"event": "fail_closed_safety_stop", "error": repr(exc)})
                print(f"FAIL-CLOSED safety stop: {exc}. No more actions will be published; press q to exit.")
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
        async_base._close_policy(policy)


if __name__ == "__main__":
    main()
