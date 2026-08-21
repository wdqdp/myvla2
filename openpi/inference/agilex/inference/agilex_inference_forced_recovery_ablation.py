#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Manual NORMAL -> forced-RECOVERY action-only ablation for tactile VLA."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
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

import agilex_inference_tactile_vla_sync_single as runtime
import cv2
import numpy as np
from openpi_client import websocket_client_policy
from tactile_vla.vla.prompts import build_execution_prompt
from tactile_vla.vla.prompts import resolve_prompt_profile
from tactile_vla.vla.structured_text import failure_reason_text
from tactile_vla.vla.structured_text import legal_failure_reasons
from tactile_vla.vla.structured_text import legal_recovery_plans
from tactile_vla.vla.structured_text import recovery_plan_text

DEFAULT_INSTRUCTION = "Pick up and transfer the object stably."
DEFAULT_CAPTIONER = Path("/data1/outputs/tactile_captioner/tcn_v3_w30_rotation_head/best.pt")
DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs" / "runtime" / "forced_recovery_ablation"
ACTION_NOISE_SHAPE = (30, 32)
ROTATION_DIRECTIONS = ("right", "left", "front", "back")
Phase = Literal["normal", "recovery"]


def moderately_rotation_targets(direction: str) -> tuple[str, str]:
    if direction not in ROTATION_DIRECTIONS:
        raise ValueError(
            f"Unsupported rotation direction={direction!r}; expected one of {ROTATION_DIRECTIONS}"
        )
    return (
        failure_reason_text(direction, "appropriate"),
        recovery_plan_text(direction, "moderately", "none", "moderately"),
    )


def deterministic_action_noise(seed: int, *, phase: Phase, index: int) -> np.ndarray:
    if index < 0:
        raise ValueError(f"Noise index must be non-negative, got {index}")
    phase_id = 0 if phase == "normal" else 1
    seed_sequence = np.random.SeedSequence([int(seed), phase_id, int(index)])
    noise = np.random.default_rng(seed_sequence).standard_normal(ACTION_NOISE_SHAPE)
    return noise.astype(np.float32)


def noise_sha256(noise: np.ndarray) -> str:
    value = np.asarray(noise, dtype=np.float32)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


class TrialLogger:
    def __init__(self, root: Path, *, trial_id: str | None) -> None:
        if trial_id is not None:
            safe_id = trial_id.strip()
            if not safe_id or Path(safe_id).name != safe_id:
                raise ValueError("--trial-id must be one non-empty path component")
            directory_name = safe_id
        else:
            directory_name = datetime.now().astimezone().strftime("trial_%Y%m%d_%H%M%S_%f")
        self.directory = root.resolve() / directory_name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events_path = self.directory / "events.jsonl"

    def record(self, event: dict[str, Any]) -> None:
        payload = {"wall_time": time.time(), "monotonic_time": time.monotonic(), **event}
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_jsonable(payload), ensure_ascii=False, separators=(",", ":")) + "\n")

    def save_trigger_images(self, front_bgr: np.ndarray, left_bgr: np.ndarray) -> dict[str, str]:
        paths = {
            "front_image": self.directory / "trigger_front.png",
            "left_image": self.directory / "trigger_left.png",
        }
        if not cv2.imwrite(str(paths["front_image"]), np.asarray(front_bgr)):
            raise OSError(f"Failed to save {paths['front_image']}")
        if not cv2.imwrite(str(paths["left_image"]), np.asarray(left_bgr)):
            raise OSError(f"Failed to save {paths['left_image']}")
        return {key: str(path) for key, path in paths.items()}


@dataclass(frozen=True)
class FrozenObservation:
    img_front: np.ndarray
    img_left: np.ndarray
    qpos: np.ndarray
    timestamp: float | None
    state_history: np.ndarray
    state_history_mask: np.ndarray
    tactile_caption: str


@dataclass(frozen=True)
class ControlSignal:
    kind: Literal["trigger", "success", "quit"]
    key: str


def _joint_timestamp(joint_state: Any) -> float | None:
    header = getattr(joint_state, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    timestamp = getattr(joint_state, "timestamp", None)
    return float(timestamp) if timestamp is not None else None


def _latest_feedback(operator: Any) -> tuple[np.ndarray | None, float | None]:
    joint_deque = getattr(operator, "puppet_arm_deque", None)
    if joint_deque:
        message = joint_deque[-1]
        return np.asarray(message.position, dtype=np.float32), _joint_timestamp(message)
    getter = getattr(operator, "get_latest_joint_state", None)
    if getter is not None:
        message = getter()
        if message is not None:
            return np.asarray(message.position, dtype=np.float32), _joint_timestamp(message)
    return None, None


def _poll_control_key(args: argparse.Namespace, keyboard: Any, *, phase: Phase) -> ControlSignal | None:
    key = keyboard.get_key()
    if key is None:
        return None
    if key == args.quit_key:
        return ControlSignal("quit", key)
    if key == args.success_key:
        return ControlSignal("success", key)
    if key == " " and phase == "normal":
        return ControlSignal("trigger", key)
    return None


def validate_server_metadata(args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    if not bool(metadata.get("action_only_ablation", False)):
        raise ValueError("Connected server is not the action-only forced-recovery ablation server")
    expected_data_profile = getattr(args, "expected_data_profile", None)
    server_data_profile = str(metadata.get("data_profile", "legacy"))
    if expected_data_profile is not None and server_data_profile != expected_data_profile:
        raise ValueError(
            "Client/server data profile mismatch: "
            f"expected={expected_data_profile!r}, server={server_data_profile!r}"
        )
    if not bool(metadata.get("supports_action_noise", False)):
        raise ValueError("Ablation server does not advertise supports_action_noise=true")
    noise_shape = tuple(int(value) for value in metadata.get("action_noise_shape", ()))
    if noise_shape != ACTION_NOISE_SHAPE:
        raise ValueError(f"Server action_noise_shape={noise_shape}, expected {ACTION_NOISE_SHAPE}")
    if int(metadata.get("action_horizon", 0)) != ACTION_NOISE_SHAPE[0]:
        raise ValueError(f"Server action_horizon must be 30, got {metadata.get('action_horizon')}")
    if int(metadata.get("action_dim", 0)) != ACTION_NOISE_SHAPE[1]:
        raise ValueError(f"Server action_dim must be 32, got {metadata.get('action_dim')}")
    if int(metadata.get("output_action_dim", 0)) != 7:
        raise ValueError(f"Server output_action_dim must be 7, got {metadata.get('output_action_dim')}")
    if not bool(metadata.get("use_state_history", False)):
        raise ValueError("Ablation requires a checkpoint with state history enabled")
    server_history = (
        int(metadata.get("state_history_len", 0)),
        int(metadata.get("state_history_dim", 0)),
    )
    if server_history != (args.state_history_len, 7):
        raise ValueError(
            f"Client/server state-history mismatch: client=[{args.state_history_len},7], "
            f"server={list(server_history)}"
        )
    server_fps = float(metadata.get("state_history_fps", 0.0))
    if not np.isclose(server_fps, args.state_history_fps):
        raise ValueError(
            f"Client state_history_fps={args.state_history_fps} does not match server={server_fps}"
        )
    if args.chunk_size > ACTION_NOISE_SHAPE[0]:
        raise ValueError(f"--chunk_size must be at most 30, got {args.chunk_size}")


def _capture_observation(
    args: argparse.Namespace,
    operator: Any,
    captioner: Any,
    *,
    reset_history: bool,
) -> FrozenObservation:
    img_front, img_left, puppet_arm = runtime.get_ros_observation(args, operator)
    qpos = np.asarray(puppet_arm.position, dtype=np.float32)
    if qpos.shape != (7,) or not np.isfinite(qpos).all():
        raise ValueError(f"Expected finite qpos [7], got {qpos.shape}")
    tactile_caption = runtime.current_tactile_caption(operator, captioner)
    if reset_history:
        operator.reset_state_history()
    state_history, state_history_mask = operator.get_state_history(puppet_arm)
    return FrozenObservation(
        img_front=np.asarray(img_front).copy(),
        img_left=np.asarray(img_left).copy(),
        qpos=qpos,
        timestamp=_joint_timestamp(puppet_arm),
        state_history=np.asarray(state_history, dtype=np.float32),
        state_history_mask=np.asarray(state_history_mask, dtype=np.bool_),
        tactile_caption=str(tactile_caption),
    )


def _start_forced_recovery(
    *,
    args: argparse.Namespace,
    operator: Any,
    captioner: Any,
    logger: TrialLogger,
    published_steps: int,
    discarded_raw_actions: int,
) -> FrozenObservation:
    # Capture only after the old NORMAL chunk has been stopped. Resetting the
    # history before snapshotting makes this qpos the sole valid new-attempt sample.
    frozen = _capture_observation(
        args,
        operator,
        captioner,
        reset_history=True,
    )
    image_paths = logger.save_trigger_images(frozen.img_front, frozen.img_left)
    logger.record(
        {
            "event": "forced_recovery_trigger",
            "previous_phase": "normal",
            "phase": "recovery",
            "published_steps": published_steps,
            "discarded_raw_actions": discarded_raw_actions,
            "rotation_direction": args.rotation_direction,
            "preset_failure_reason": args.forced_failure_reason,
            "forced_recovery_plan": args.forced_recovery_plan,
            "tactile_caption": frozen.tactile_caption,
            "observation_timestamp": frozen.timestamp,
            "switch_qpos": frozen.qpos,
            "reset_state_history": frozen.state_history,
            "reset_state_history_mask": frozen.state_history_mask,
            **image_paths,
        }
    )
    print(
        "RECOVERY started: discarded the remaining NORMAL chunk; "
        f"rotation_direction={args.rotation_direction} plan={args.forced_recovery_plan}"
    )
    return frozen


def _request_action_chunk(
    *,
    args: argparse.Namespace,
    policy: Any,
    logger: TrialLogger,
    observation: FrozenObservation,
    phase: Phase,
    phase_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    recovery_plan = "" if phase == "normal" else args.forced_recovery_plan
    prompt = build_execution_prompt(
        instruction=args.instruction,
        tactile_caption=observation.tactile_caption,
        input_recovery_plan=recovery_plan,
        prompt_profile=args.prompt_profile,
    )
    if args.forced_failure_reason in prompt:
        raise AssertionError("Preset failure_reason must never enter the action prompt")
    noise = deterministic_action_noise(args.noise_seed, phase=phase, index=phase_index)
    payload = runtime.build_payload(
        mode="execution",
        img_front_bgr=observation.img_front,
        img_left_bgr=observation.img_left,
        qpos=observation.qpos,
        state_history=observation.state_history,
        state_history_mask=observation.state_history_mask,
        prompt=prompt,
    )
    payload.update(
        {
            "action_noise": noise,
            "noise_seed": args.noise_seed,
            "noise_phase": phase,
            "noise_index": phase_index,
        }
    )
    started = time.perf_counter()
    response = policy.infer(payload)
    client_infer_ms = (time.perf_counter() - started) * 1000.0
    raw_model_actions = np.asarray(response.get("raw_model_actions"), dtype=np.float32)
    actions = np.asarray(response.get("actions"), dtype=np.float32)
    if raw_model_actions.shape != ACTION_NOISE_SHAPE or not np.isfinite(raw_model_actions).all():
        raise ValueError(f"Expected finite raw_model_actions [30,32], got {raw_model_actions.shape}")
    if actions.shape != (ACTION_NOISE_SHAPE[0], 7) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite transformed actions [30,7], got {actions.shape}")
    logger.record(
        {
            "event": "action_chunk_inference",
            "phase": phase,
            "phase_index": phase_index,
            "prompt": prompt,
            "rotation_direction": args.rotation_direction,
            "preset_failure_reason": args.forced_failure_reason,
            "forced_recovery_plan": args.forced_recovery_plan,
            "tactile_caption": observation.tactile_caption,
            "observation_timestamp": observation.timestamp,
            "qpos": observation.qpos,
            "state_history": observation.state_history,
            "state_history_mask": observation.state_history_mask,
            "state_history_valid_frames": int(observation.state_history_mask.sum()),
            "noise_seed": args.noise_seed,
            "noise_phase": phase,
            "noise_index": phase_index,
            "noise_sha256": noise_sha256(noise),
            "action_noise": noise,
            "raw_model_actions": raw_model_actions,
            "transformed_actions": actions,
            "client_infer_ms": client_infer_ms,
            "server_infer_ms": response.get("policy_timing", {}).get("infer_ms"),
        }
    )
    print(f"[{phase.upper()}] chunk={phase_index} prompt={prompt}")
    return raw_model_actions, actions


def _publish_raw_action(
    *,
    args: argparse.Namespace,
    operator: Any,
    keyboard: Any,
    logger: TrialLogger,
    phase: Phase,
    phase_index: int,
    action_index: int,
    raw_action: np.ndarray,
    pre_action: np.ndarray,
    control_rate: Any,
) -> tuple[np.ndarray, int, ControlSignal | None]:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
        raise ValueError(f"Expected finite raw action [7], got {raw_action.shape}")
    interpolated = (
        runtime.interpolate_action(args, pre_action, raw_action)
        if args.use_actions_interpolation
        else raw_action[None, :]
    )
    published_count = 0
    for interpolation_index, action in enumerate(interpolated):
        control = _poll_control_key(args, keyboard, phase=phase)
        if control is not None:
            return pre_action, published_count, control
        if runtime.shutdown_event.is_set() or operator.is_shutdown():
            return pre_action, published_count, ControlSignal("quit", "shutdown")

        published_action = np.asarray(action, dtype=np.float32).copy()
        published_action[6] = max(0.0, float(published_action[6]) - args.gripper_offset)
        if args.no_publish:
            runtime.published_actions_history.append(published_action.copy())
            command_timestamp = None
        else:
            command_timestamp = operator.puppet_arm_publish(published_action)
            runtime.published_actions_history.append(published_action.copy())
        published_count += 1
        control_rate.sleep()
        feedback_qpos, feedback_timestamp = _latest_feedback(operator)
        logger.record(
            {
                "event": "action_publish",
                "phase": phase,
                "phase_index": phase_index,
                "action_index": action_index,
                "interpolation_index": interpolation_index,
                "interpolation_count": len(interpolated),
                "raw_action": raw_action,
                "published_action": published_action,
                "command_timestamp": command_timestamp,
                "feedback_qpos": feedback_qpos,
                "feedback_timestamp": feedback_timestamp,
            }
        )

    return raw_action.copy(), published_count, None


def _handle_terminal_signal(
    signal_value: ControlSignal,
    *,
    logger: TrialLogger,
    phase: Phase,
    published_steps: int,
) -> bool:
    if signal_value.kind == "success":
        print("Operator confirmed success")
        logger.record({"event": "operator_success", "phase": phase, "published_steps": published_steps})
        return True
    if signal_value.kind == "quit":
        print("Operator requested safety stop")
        runtime.shutdown_event.set()
        logger.record(
            {
                "event": "operator_quit",
                "phase": phase,
                "published_steps": published_steps,
                "key": signal_value.key,
            }
        )
        return True
    return False


def run_ablation(
    args: argparse.Namespace,
    operator: Any,
    policy: Any,
    captioner: Any,
    keyboard: Any,
    logger: TrialLogger,
) -> None:
    metadata = policy.get_server_metadata()
    validate_server_metadata(args, metadata)
    args.prompt_profile = resolve_prompt_profile(metadata.get("prompt_profile"))
    logger.record(
        {
            "event": "run_start",
            "server_metadata": metadata,
            "checkpoint_kind": metadata["checkpoint_kind"],
            "checkpoint": metadata["checkpoint"],
            "args": vars(args),
            "rotation_direction": args.rotation_direction,
            "preset_failure_reason": args.forced_failure_reason,
            "forced_recovery_plan": args.forced_recovery_plan,
            "noise_seed": args.noise_seed,
        }
    )
    print(
        "Connected to action-only server: "
        f"kind={metadata['checkpoint_kind']} checkpoint={metadata['checkpoint']}"
    )
    print("NORMAL: press SPACE to discard the current chunk and start forced RECOVERY; s=success, q=quit")

    phase: Phase = "normal"
    phase_indices: dict[Phase, int] = {"normal": 0, "recovery": 0}
    pending_observation: FrozenObservation | None = None
    pre_action: np.ndarray | None = None
    published_steps = 0

    while published_steps < args.max_publish_step and not operator.is_shutdown():
        # Catch a key pressed during the pause between chunks before capturing
        # or inferring another NORMAL chunk.
        boundary_signal = _poll_control_key(args, keyboard, phase=phase)
        if boundary_signal is not None:
            if _handle_terminal_signal(
                boundary_signal,
                logger=logger,
                phase=phase,
                published_steps=published_steps,
            ):
                return
            if boundary_signal.kind != "trigger" or phase != "normal":
                raise AssertionError(f"Unexpected boundary signal {boundary_signal} in phase={phase}")
            pending_observation = _start_forced_recovery(
                args=args,
                operator=operator,
                captioner=captioner,
                logger=logger,
                published_steps=published_steps,
                discarded_raw_actions=0,
            )
            phase = "recovery"
            pre_action = pending_observation.qpos.copy()

        observation = pending_observation or _capture_observation(
            args,
            operator,
            captioner,
            reset_history=False,
        )
        pending_observation = None
        if pre_action is None:
            pre_action = observation.qpos.copy()
        phase_index = phase_indices[phase]
        _, actions = _request_action_chunk(
            args=args,
            policy=policy,
            logger=logger,
            observation=observation,
            phase=phase,
            phase_index=phase_index,
        )
        phase_indices[phase] += 1

        limit = min(args.chunk_size, args.max_publish_step - published_steps, actions.shape[0])
        completed_actions = 0
        published_commands = 0
        control_signal: ControlSignal | None = None
        control_rate = operator.rate(args.publish_rate)
        for action_index, raw_action in enumerate(actions[:limit]):
            pre_action, command_count, control_signal = _publish_raw_action(
                args=args,
                operator=operator,
                keyboard=keyboard,
                logger=logger,
                phase=phase,
                phase_index=phase_index,
                action_index=action_index,
                raw_action=raw_action,
                pre_action=pre_action,
                control_rate=control_rate,
            )
            published_commands += command_count
            if control_signal is not None:
                break
            completed_actions += 1
            published_steps += 1
            print(f"[{phase.upper()}] Published raw action step {published_steps}")

        logger.record(
            {
                "event": "execution_chunk",
                "phase": phase,
                "phase_index": phase_index,
                "completed_raw_actions": completed_actions,
                "published_commands": published_commands,
                "discarded_raw_actions": max(0, limit - completed_actions),
                "control_signal": control_signal.kind if control_signal is not None else None,
            }
        )

        if control_signal is None:
            continue
        if _handle_terminal_signal(
            control_signal,
            logger=logger,
            phase=phase,
            published_steps=published_steps,
        ):
            return
        if control_signal.kind != "trigger" or phase != "normal":
            raise AssertionError(f"Unexpected control signal {control_signal} in phase={phase}")

        # No old chunk action is published after this point. Capture one fresh,
        # synchronized observation and use it unchanged for the first recovery request.
        frozen = _start_forced_recovery(
            args=args,
            operator=operator,
            captioner=captioner,
            logger=logger,
            published_steps=published_steps,
            discarded_raw_actions=max(0, limit - completed_actions),
        )
        phase = "recovery"
        pre_action = frozen.qpos.copy()
        pending_observation = frozen

    logger.record({"event": "run_limit_reached", "phase": phase, "published_steps": published_steps})
    print(f"Stopped after max_publish_step={args.max_publish_step}")


def get_arguments() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--expected-data-profile",
        help="Refuse to run if server metadata does not advertise this exact data profile.",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--rotation-direction",
        choices=ROTATION_DIRECTIONS,
        help=(
            "Force a moderately horizontal recovery in this direction and automatically "
            "set the matching V3 failure reason and recovery plan"
        ),
    )
    parser.add_argument(
        "--forced-failure-reason",
        help="Explicit V3 failure reason (legacy alternative to --rotation-direction)",
    )
    parser.add_argument(
        "--forced-recovery-plan",
        help="Explicit V3 recovery plan (legacy alternative to --rotation-direction)",
    )
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--observation-poll-rate", type=int, default=200)
    parser.add_argument("--state-history-len", type=int, default=60)
    parser.add_argument("--state-history-fps", type=float, default=runtime.DEFAULT_STATE_HISTORY_FPS)
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
    parser.add_argument("--no-captioner", action="store_true")
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--success-key", default="s")
    parser.add_argument("--quit-key", default="q")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--replay-attempt-dir", type=Path)
    parser.add_argument("--replay-start-index", type=int, default=0)
    parser.add_argument("--replay-step-stride", type=int, default=1)
    parser.add_argument("--replay-max-frames", type=int)
    parser.add_argument("--use_actions_interpolation", action="store_true")
    parser.add_argument(
        "--arm_steps_length",
        nargs=7,
        type=float,
        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
    )
    parser.add_argument("--gripper_offset", type=float, default=0.001)
    args = parser.parse_args()
    return args, parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    direction = getattr(args, "rotation_direction", None)
    if direction is not None:
        expected_failure, expected_plan = moderately_rotation_targets(direction)
        if args.forced_failure_reason not in {None, expected_failure}:
            parser.error(
                "--forced-failure-reason conflicts with --rotation-direction; "
                f"expected {expected_failure!r}"
            )
        if args.forced_recovery_plan not in {None, expected_plan}:
            parser.error(
                "--forced-recovery-plan conflicts with --rotation-direction; "
                f"expected {expected_plan!r}"
            )
        args.forced_failure_reason = expected_failure
        args.forced_recovery_plan = expected_plan
    elif args.forced_failure_reason is None or args.forced_recovery_plan is None:
        parser.error(
            "set --rotation-direction, or provide both --forced-failure-reason "
            "and --forced-recovery-plan"
        )

    if args.forced_failure_reason not in legal_failure_reasons():
        parser.error("--forced-failure-reason is outside the fixed V3 failure grammar")
    if args.forced_recovery_plan not in legal_recovery_plans():
        parser.error("--forced-recovery-plan is outside the fixed V3 recovery grammar")
    if args.max_publish_step <= 0:
        parser.error("--max_publish_step must be positive")
    if args.noise_seed < 0:
        parser.error("--noise-seed must be non-negative")
    if not 1 <= args.chunk_size <= ACTION_NOISE_SHAPE[0]:
        parser.error("--chunk_size must be in [1, 30]")
    if args.publish_rate <= 0 or args.observation_poll_rate <= 0:
        parser.error("publish and observation polling rates must be positive")
    if args.state_history_len != 60:
        parser.error("--state-history-len must be 60 for this V3 ablation")
    if args.state_history_fps <= 0 or args.state_history_max_gap_seconds <= 0:
        parser.error("state-history fps and max gap must be positive")
    if len(args.success_key) != 1 or len(args.quit_key) != 1:
        parser.error("success/quit keys must each be one character")
    if args.success_key == args.quit_key or " " in {args.success_key, args.quit_key}:
        parser.error("success/quit keys must be distinct and cannot be SPACE")


def main() -> None:
    args, parser = get_arguments()
    runtime.apply_yaml_defaults(args, parser)
    validate_args(args, parser)
    if not sys.stdin.isatty():
        parser.error("This manual ablation requires an interactive TTY for SPACE/s/q controls")
    runtime.shutdown_event.clear()
    signal.signal(signal.SIGINT, runtime._on_sigint)  # noqa: SLF001

    captioner = runtime.load_captioner(args)
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    operator = runtime.ReplayOperator(args) if args.replay_attempt_dir is not None else runtime.RosOperator(args)
    logger = TrialLogger(args.log_dir, trial_id=args.trial_id)
    print(f"Trial log directory: {logger.directory}")

    try:
        if not args.start_immediately and args.replay_attempt_dir is None:
            input("Press enter to start NORMAL execution")
        with runtime.KeyboardPoller() as keyboard:
            run_ablation(args, operator, policy, captioner, keyboard, logger)
    except KeyboardInterrupt:
        runtime.shutdown_event.set()
        logger.record({"event": "keyboard_interrupt"})
    finally:
        if isinstance(operator, runtime.ReplayOperator):
            operator.close()


if __name__ == "__main__":
    main()
