#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""V5.3 manual recovery trial with automatic adjustment-end transition."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

import argparse
from dataclasses import replace
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
import agilex_inference_tactile_vla_sync_single as runtime
import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.structured_text import legal_failure_reasons
from tactile_vla.vla.structured_text import legal_recovery_plans
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v5_3_phase_change import QPOS_SAMPLE_OFFSETS
from tactile_vla.vla.v5_3_phase_change import build_adjustment_end_prompt

DEFAULT_CAPTIONER = Path("/data1/outputs/tactile_captioner/tcn_v3_w30_rotation_head/best.pt")
DEFAULT_NORM_STATS = Path("/data1/outputs/vla/assets/tactile_vla_rotation_v4/norm_stats.json")
DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/forced_phase_ablation_v5_3"
ACTION_NOISE_SHAPE = (30, 32)
SYNC_TOLERANCE_SECONDS = 0.050
Phase = Literal["execution", "adjustment"]


class FailClosedError(RuntimeError):
    pass


class TimeoutWebsocketPolicy(websocket_client_policy.WebsocketClientPolicy):
    def infer_with_timeout(self, observation: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv(timeout=float(timeout))
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)


def validate_server_metadata(args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    expected = {
        "supports_action_noise": True,
        "requires_action_noise": True,
        "supports_adjustment_end": True,
        "prompt_profile": "phase_v2",
        "data_profile": "rotation_phase_v5_adjustment_v2",
        "experiment_kind": "phase_prompt_h30_terminal_hold",
        "phase_change_prompt_profile": "phase_change_v1",
        "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
        "qpos_h30_sample_offsets": list(QPOS_SAMPLE_OFFSETS),
        "qpos_bin_count": 256,
        "qpos_discretization_extra_clip": False,
        "captioner_window_size": 30,
        "action_horizon": 30,
        "action_dim": 32,
        "output_action_dim": 7,
        "state_history_len": 60,
        "state_history_dim": 7,
    }
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise ValueError(f"V5.3 client/server metadata mismatch: {mismatch}")
    threshold = float(metadata.get("adjustment_end_threshold", -1.0))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Server adjustment_end_threshold is invalid")
    if metadata.get("adjustment_end_experimental_override", False) and not getattr(
        args, "allow_experimental_adjustment_end", False
    ):
        raise ValueError(
            "Server uses a non-official adjustment_end robot probe; "
            "pass --allow-experimental-adjustment-end to acknowledge it"
        )
    timeout = float(metadata.get("phase_change_timeout_seconds", -1.0))
    if timeout != 10.0 or args.phase_change_timeout_seconds != timeout:
        raise ValueError("V5.3 phase-change timeout must be exactly 10 seconds")
    if metadata.get("captioner_checkpoint_sha256") != sha256_file(args.captioner_checkpoint):
        raise ValueError("Client captioner differs from the classifier training captioner")


def _install_tactile_arrival_timestamps(operator: Any) -> None:
    """Stamp headerless Float32MultiArray frames at ROS callback receipt time."""
    tactile = getattr(operator, "tactile", None)
    if tactile is None or not hasattr(tactile, "buffer") or not hasattr(operator, "rospy"):
        return
    buffer = tactile.buffer
    original = buffer.push_topic

    def stamped_push_topic(key, data, *, layout=None, timestamp=None):
        if timestamp is None:
            timestamp = float(operator.rospy.Time.now().to_sec())
        return original(key, data, layout=layout, timestamp=timestamp)

    buffer.push_topic = stamped_push_topic


def _latest_tactile_timestamp(operator: Any) -> float | None:
    tactile = getattr(operator, "tactile", None)
    if tactile is None:
        return None
    buffer = getattr(tactile, "buffer", tactile)
    frames = getattr(buffer, "_frames", None)
    lock = getattr(buffer, "_lock", None)
    if frames is None:
        return None
    if lock is None:
        frame = frames[-1] if frames else None
    else:
        with lock:
            frame = frames[-1] if frames else None
    timestamp = getattr(frame, "timestamp", None) if frame is not None else None
    return float(timestamp) if timestamp is not None else None


def _capture_classification_observation(
    args: argparse.Namespace,
    operator: Any,
    captioner: Any,
    *,
    after_timestamp: float,
) -> tuple[v52.FrozenObservation, dict[str, float]]:
    """Wait for front/left/qpos/tactile that are new and mutually synchronized."""
    rate = operator.rate(args.observation_poll_rate)
    started = time.monotonic()
    while not operator.is_shutdown() and not runtime.shutdown_event.is_set():
        if time.monotonic() - started > args.phase_change_timeout_seconds:
            raise FailClosedError("Timed out waiting for synchronized classification observation")
        if isinstance(operator, runtime.ReplayOperator):
            front, left, joint = runtime.get_ros_observation(args, operator, after_timestamp=after_timestamp)
            timestamp = v52._joint_timestamp(joint)
            stamps = {"front": timestamp, "left": timestamp, "qpos": timestamp}
        else:
            if not operator.img_front_deque or not operator.img_left_deque or not operator.puppet_arm_deque:
                rate.sleep()
                continue
            front_msg = operator.img_front_deque[-1]
            left_msg = operator.img_left_deque[-1]
            joint = operator.puppet_arm_deque[-1]
            stamps = {
                "front": float(front_msg.header.stamp.to_sec()),
                "left": float(left_msg.header.stamp.to_sec()),
                "qpos": float(joint.header.stamp.to_sec()),
            }
            if min(stamps.values()) <= after_timestamp or max(stamps.values()) - min(stamps.values()) > SYNC_TOLERANCE_SECONDS:
                rate.sleep()
                continue
            front = operator.bridge.imgmsg_to_cv2(front_msg, "passthrough")
            left = operator.bridge.imgmsg_to_cv2(left_msg, "passthrough")
        tactile_timestamp = _latest_tactile_timestamp(operator)
        if (
            tactile_timestamp is None
            or tactile_timestamp <= after_timestamp
            or abs(tactile_timestamp - stamps["qpos"]) > SYNC_TOLERANCE_SECONDS
        ):
            rate.sleep()
            continue
        if captioner is None or operator.tactile is None or not operator.tactile.ready:
            raise RuntimeError("V5.3 classification requires a ready 30-frame tactile captioner")
        qpos = np.asarray(joint.position, dtype=np.float32)
        history, mask = operator.get_state_history(joint)
        observation = v52.FrozenObservation(
            img_front=np.asarray(front).copy(),
            img_left=np.asarray(left).copy(),
            qpos=qpos,
            timestamp=stamps["qpos"],
            state_history=np.asarray(history, dtype=np.float32),
            state_history_mask=np.asarray(mask, dtype=np.bool_),
            tactile_caption=runtime.current_tactile_caption(operator, captioner),
        )
        return observation, {**stamps, "tactile": tactile_timestamp}
    raise RuntimeError("Stopped while waiting for synchronized V5.3 classification observation")


def _wait_feedback_after(args, operator, after_timestamp: float | None) -> tuple[np.ndarray, float]:
    rate = operator.rate(args.observation_poll_rate)
    started = time.monotonic()
    while not operator.is_shutdown() and not runtime.shutdown_event.is_set():
        if time.monotonic() - started > args.phase_change_timeout_seconds:
            raise FailClosedError("Timed out waiting for post-action qpos feedback")
        if isinstance(operator, runtime.ReplayOperator):
            result = operator.get_frame(after_timestamp=after_timestamp)
            if result:
                _, _, joint = result
                qpos = np.asarray(joint.position, dtype=np.float32)
                timestamp = v52._joint_timestamp(joint)
                if timestamp is not None and (after_timestamp is None or timestamp > after_timestamp):
                    return qpos, float(timestamp)
        qpos, timestamp = v52._latest_feedback(operator)
        if (
            qpos is not None
            and timestamp is not None
            and np.isfinite(timestamp)
            and (after_timestamp is None or timestamp > after_timestamp)
        ):
            qpos = np.asarray(qpos, dtype=np.float32)
            if qpos.shape == (7,) and np.isfinite(qpos).all():
                return qpos, float(timestamp)
        rate.sleep()
    raise RuntimeError("Stopped while waiting for post-action qpos feedback")


def _classify_adjustment_end(
    *,
    args: argparse.Namespace,
    policy: TimeoutWebsocketPolicy,
    operator: Any,
    captioner: Any,
    logger: v52.TrialLogger,
    feedback_qpos_h30: list[np.ndarray],
    feedback_timestamps: list[float],
    stats,
    phase_index: int,
) -> tuple[bool, v52.FrozenObservation]:
    if len(feedback_qpos_h30) != 30 or len(feedback_timestamps) != 30:
        raise ValueError("adjustment_end is called only after a complete H30 chunk")
    if any(left >= right for left, right in pairwise(feedback_timestamps)):
        raise ValueError("H30 feedback timestamps are not strictly increasing")
    observation, synchronized_timestamps = _capture_classification_observation(
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
        logger.record({"event": "adjustment_end_fail_closed", "error": repr(exc), "phase_index": phase_index})
        raise FailClosedError("adjustment_end request failed; fail-closed safety stop") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
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
    result = bool(response["adjustment_end"])
    logger.record({
        "event": "adjustment_end_inference",
        "phase_index": phase_index,
        "adjustment_end": result,
        "adjustment_end_probs": probabilities,
        "prompt": prompt,
        "qpos_h10_discrete": discrete,
        "feedback_qpos_h30": feedback_qpos_h30,
        "feedback_timestamps": feedback_timestamps,
        "current_qpos": observation.qpos,
        "synchronized_timestamps": synchronized_timestamps,
        "client_infer_ms": elapsed_ms,
        "server_infer_ms": response.get("policy_timing", {}).get("infer_ms"),
    })
    print(f"[ADJUSTMENT_END] result={result} probs={probabilities.tolist()}")
    return result, observation


def _poll_key(args, keyboard, *, phase: Phase, paused: bool):
    key = keyboard.get_key()
    if key is None:
        return None
    if key == args.quit_key:
        return "quit"
    if key == " " and phase == "execution":
        return "trigger"
    if key == args.continue_key and paused:
        return "continue"
    return None


def should_request_adjustment_end(*, phase: Phase, completed_raw_actions: int) -> bool:
    return phase == "adjustment" and int(completed_raw_actions) == 30


def _wait_for_s(args, operator, captioner, keyboard, logger, phase, phase_index, locked_history):
    v52._pause_state_history(operator)
    logger.record({"event": "chunk_pause", "phase": phase, "phase_index": phase_index})
    print(f"[{phase.upper()}] chunk complete; history locked. Press s for next chunk, SPACE in execution, q to quit.")
    rate = operator.rate(args.observation_poll_rate)
    while not operator.is_shutdown() and not runtime.shutdown_event.is_set():
        signal_value = _poll_key(args, keyboard, phase=phase, paused=True)
        if signal_value is None:
            rate.sleep()
            continue
        if signal_value == "continue":
            requested = time.time()
            live = v52._capture_observation(args, operator, captioner, reset_history=False, after_timestamp=requested)
            live = replace(
                live,
                state_history=locked_history.state_history.copy(),
                state_history_mask=locked_history.state_history_mask.copy(),
            )
            v52._resume_state_history(args, operator, live)
            return "continue", live
        return signal_value, None
    return "quit", None


def run_v5_3(args, operator, policy, captioner, keyboard, logger):
    metadata = policy.get_server_metadata()
    validate_server_metadata(args, metadata)
    stats = load_state_quantiles(args.norm_stats_file)
    logger.record({"event": "run_start", "server_metadata": metadata, "args": vars(args)})
    print(
        "V5.3 controls: s=next chunk, SPACE=execution to adjustment, q=quit. "
        f"Server threshold={metadata['adjustment_end_threshold']}; client never re-thresholds."
    )
    if metadata.get("adjustment_end_experimental_override", False):
        print(
            "WARNING: EXPERIMENTAL adjustment_end deployment (manual threshold or "
            "non-official probe). Each phase change still pauses before the next chunk."
        )
    phase: Phase = "execution"
    phase_indices = {"execution": 0, "adjustment": 0}
    pending = None
    pre_action = None
    published_steps = 0
    while published_steps < args.max_publish_step and not operator.is_shutdown():
        signal_value = _poll_key(args, keyboard, phase=phase, paused=False)
        if signal_value == "quit":
            runtime.shutdown_event.set()
            return
        if signal_value == "trigger":
            pending = v52._start_forced_recovery(
                args=args,
                operator=operator,
                captioner=captioner,
                logger=logger,
                published_steps=published_steps,
                discarded_raw_actions=0,
            )
            phase = "adjustment"
            pre_action = pending.qpos.copy()

        observation = pending or v52._capture_observation(args, operator, captioner, reset_history=False)
        pending = None
        if pre_action is None:
            pre_action = observation.qpos.copy()
        phase_index = phase_indices[phase]
        _, actions = v52._request_action_chunk(
            args=args,
            policy=policy,
            logger=logger,
            observation=observation,
            phase=phase,
            phase_index=phase_index,
        )
        phase_indices[phase] += 1
        limit = min(args.chunk_size, args.max_publish_step - published_steps, len(actions))
        complete = 0
        feedback_qpos_h30: list[np.ndarray] = []
        feedback_timestamps: list[float] = []
        control_rate = operator.rate(args.publish_rate)
        control = None
        for action_index, action in enumerate(actions[:limit]):
            _, before_timestamp = v52._latest_feedback(operator)
            pre_action, _, raw_control = v52._publish_raw_action(
                args=args,
                operator=operator,
                keyboard=keyboard,
                logger=logger,
                phase=phase,
                phase_index=phase_index,
                action_index=action_index,
                raw_action=action,
                pre_action=pre_action,
                control_rate=control_rate,
            )
            if raw_control is not None:
                control = raw_control.kind
                break
            feedback, timestamp = _wait_feedback_after(args, operator, before_timestamp)
            feedback_qpos_h30.append(feedback)
            feedback_timestamps.append(timestamp)
            complete += 1
            published_steps += 1

        logger.record({
            "event": "execution_chunk",
            "phase": phase,
            "phase_index": phase_index,
            "completed_raw_actions": complete,
            "discarded_raw_actions": max(0, limit - complete),
            "control_signal": control,
        })
        if control == "quit":
            runtime.shutdown_event.set()
            return
        if control == "trigger":
            if phase != "execution":
                raise AssertionError("SPACE is valid only in execution")
            pending = v52._start_forced_recovery(
                args=args,
                operator=operator,
                captioner=captioner,
                logger=logger,
                published_steps=published_steps,
                discarded_raw_actions=max(0, limit - complete),
            )
            phase = "adjustment"
            pre_action = pending.qpos.copy()
            continue

        if should_request_adjustment_end(phase=phase, completed_raw_actions=complete):
            ended, classification_observation = _classify_adjustment_end(
                args=args,
                policy=policy,
                operator=operator,
                captioner=captioner,
                logger=logger,
                feedback_qpos_h30=feedback_qpos_h30,
                feedback_timestamps=feedback_timestamps,
                stats=stats,
                phase_index=phase_index,
            )
            if ended:
                logger.record({
                    "event": "phase_transition",
                    "previous_phase": "adjustment",
                    "phase": "execution",
                    "reset_state_history": False,
                })
                phase = "execution"
        elif phase == "adjustment":
            logger.record({"event": "adjustment_end_skipped_incomplete_chunk", "completed_raw_actions": complete})

        if published_steps >= args.max_publish_step:
            break
        completed_at = time.time()
        locked = v52._capture_delayed_history_snapshot(
            args,
            operator,
            captioner,
            chunk_completed_timestamp=completed_at,
        )
        pause_signal, live = _wait_for_s(
            args, operator, captioner, keyboard, logger, phase, phase_index, locked
        )
        if pause_signal == "quit":
            runtime.shutdown_event.set()
            return
        if pause_signal == "trigger":
            pending = v52._start_forced_recovery(
                args=args,
                operator=operator,
                captioner=captioner,
                logger=logger,
                published_steps=published_steps,
                discarded_raw_actions=0,
                locked_observation=locked,
            )
            phase = "adjustment"
            pre_action = pending.qpos.copy()
        elif pause_signal == "continue":
            pending = live
            pre_action = live.qpos.copy()
        else:
            raise AssertionError(f"Unexpected pause signal {pause_signal}")


def get_arguments():
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
    parser.add_argument("--norm-stats-file", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--phase-change-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--observation-poll-rate", type=int, default=200)
    parser.add_argument("--history-freeze-delay-seconds", type=float, default=1.0)
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
    parser.add_argument("--continue-key", default="s")
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


def validate_args(args, parser):
    if args.rotation_direction is not None:
        expected_failure, expected_plan = v52.rotation_targets(args.rotation_direction, args.rotation_magnitude)
        if args.forced_failure_reason not in {None, expected_failure} or args.forced_recovery_plan not in {None, expected_plan}:
            parser.error("explicit failure/plan conflicts with --rotation-direction")
        args.forced_failure_reason, args.forced_recovery_plan = expected_failure, expected_plan
    if args.forced_failure_reason not in legal_failure_reasons():
        parser.error("provide --rotation-direction or a legal --forced-failure-reason")
    if args.forced_recovery_plan not in legal_recovery_plans():
        parser.error("provide --rotation-direction or a legal --forced-recovery-plan")
    if args.chunk_size != 30:
        parser.error("V5.3 requires --chunk_size=30 so every adjustment check owns one complete H30")
    if args.state_history_len != 60 or args.tactile_window_size != 30:
        parser.error("V5.3 requires state history 60 and tactile window 30")
    if args.phase_change_timeout_seconds != 10.0:
        parser.error("V5.3 phase-change timeout is fixed at 10 seconds")
    if not args.captioner_checkpoint.is_file() or not args.norm_stats_file.is_file():
        parser.error("captioner checkpoint and norm stats file must exist")
    if not 0 <= args.gripper_min <= 0.08:
        parser.error("--gripper-min must be in [0,0.08]")
    args.prompt_profile = "phase_v2"
    args.no_captioner = False


def main():
    args, parser = get_arguments()
    runtime.apply_yaml_defaults(args, parser)
    validate_args(args, parser)
    if not sys.stdin.isatty():
        parser.error("V5.3 manual inference requires an interactive TTY")
    runtime.shutdown_event.clear()
    signal.signal(signal.SIGINT, runtime._on_sigint)
    captioner = runtime.load_captioner(args)
    if captioner is None:
        parser.error("V5.3 does not allow --no-captioner")
    policy = TimeoutWebsocketPolicy(args.host, args.port)
    operator = runtime.ReplayOperator(args) if args.replay_attempt_dir else runtime.RosOperator(args)
    _install_tactile_arrival_timestamps(operator)
    trial_id = args.trial_id
    if trial_id and (args.log_dir.resolve() / trial_id).exists():
        trial_id = f"{trial_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = v52.TrialLogger(args.log_dir, trial_id=trial_id)
    print(f"Trial log directory: {logger.directory}")
    try:
        if not args.start_immediately and args.replay_attempt_dir is None:
            input("Press enter to start EXECUTION")
        with runtime.KeyboardPoller() as keyboard:
            try:
                run_v5_3(args, operator, policy, captioner, keyboard, logger)
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
        if isinstance(operator, runtime.ReplayOperator):
            operator.close()


if __name__ == "__main__":
    main()
