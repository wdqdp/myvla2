#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Agilex single-arm closed-loop inference for the tactile VLA policy."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import select
import signal
import sys
import termios
import threading
import time
import tty
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from tactile_vla.captioner.predictor import TactileCaptionerPredictor
from tactile_vla.runtime.state_history import DEFAULT_STATE_HISTORY_FPS
from tactile_vla.runtime.state_history import StateHistoryBuffer
from tactile_vla.runtime.state_history import pad_state_history
from tactile_vla.runtime.tactile_buffer import DEFAULT_TACTILE_CAPTION
from tactile_vla.runtime.tactile_buffer import RosTactileBuffer
from tactile_vla.runtime.tactile_buffer import TactileTopics
from tactile_vla.runtime.tactile_buffer import TactileWindowBuffer
from tactile_vla.vla.prompts import build_execution_prompt
from tactile_vla.vla.prompts import build_reasoning_prompt

DEFAULT_INSTRUCTION = "Pick up and transfer the object stably."
DEFAULT_CAPTIONER = PROJECT_ROOT / "outputs" / "tactile_captioner" / "tcn_v1_balanced" / "best.pt"

shutdown_event = threading.Event()
published_actions_history: list[np.ndarray] = []


def _on_sigint(signum, frame):
    shutdown_event.set()
    try:
        import rospy

        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def interpolate_action(args: argparse.Namespace, prev_action: np.ndarray, cur_action: np.ndarray) -> np.ndarray:
    steps = np.asarray(args.arm_steps_length, dtype=float)
    diff = np.abs(cur_action - prev_action)
    step = int(np.max(np.ceil(diff / steps).astype(int)))
    if step <= 1:
        return cur_action[np.newaxis, :]
    return np.linspace(prev_action, cur_action, step + 1)[1:]


def _decode_hdf5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        print(f"PyYAML is not available; using CLI topic defaults instead of {path}")
        return {}
    if not path.exists():
        return {}
    with path.open() as file:
        return yaml.safe_load(file) or {}


def _name_topic_map(section: dict[str, Any]) -> dict[str, str]:
    names = section.get("names", []) or []
    topics = section.get("topics", []) or []
    return {str(name): str(topic) for name, topic in zip(names, topics, strict=False)}


def apply_yaml_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.config_path is None:
        return
    config = _read_yaml(args.config_path)
    data_info = config.get("dataInfo", {})

    color_topics = _name_topic_map(data_info.get("camera", {}).get("color", {}))
    arm_topics = _name_topic_map(data_info.get("arm", {}).get("jointState", {}))
    tactile = data_info.get("tactile", {})
    force_topics = _name_topic_map(tactile.get("force", {}))
    mesh_topics = _name_topic_map(tactile.get("mesh_3d", {}))
    flow_topics = _name_topic_map(tactile.get("mesh_3d_flow", {}))

    yaml_defaults = {
        "img_front_topic": color_topics.get("front"),
        "img_left_topic": color_topics.get("left"),
        "puppet_arm_topic": arm_topics.get("puppetRight"),
        "puppet_arm_cmd_topic": arm_topics.get("masterRight"),
        "tactile_left_force_topic": force_topics.get("left"),
        "tactile_right_force_topic": force_topics.get("right"),
        "tactile_left_mesh_3d_topic": mesh_topics.get("left"),
        "tactile_right_mesh_3d_topic": mesh_topics.get("right"),
        "tactile_left_mesh_3d_flow_topic": flow_topics.get("left"),
        "tactile_right_mesh_3d_flow_topic": flow_topics.get("right"),
    }
    for name, value in yaml_defaults.items():
        if value and getattr(args, name) == parser.get_default(name):
            setattr(args, name, value)


def make_tactile_topics(args: argparse.Namespace) -> TactileTopics:
    return TactileTopics(
        left_force=args.tactile_left_force_topic,
        right_force=args.tactile_right_force_topic,
        left_mesh_3d=args.tactile_left_mesh_3d_topic,
        right_mesh_3d=args.tactile_right_mesh_3d_topic,
        left_mesh_3d_flow=args.tactile_left_mesh_3d_flow_topic,
        right_mesh_3d_flow=args.tactile_right_mesh_3d_flow_topic,
    )


def prepare_rgb(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_tools.resize_with_pad(image.astype(np.uint8), 224, 224)


class KeyboardPoller:
    def __init__(self) -> None:
        self._enabled = sys.stdin.isatty()
        self._old_settings = None

    def __enter__(self):
        if self._enabled:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> str | None:
        if not self._enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return None
        return sys.stdin.read(1)


class NoopRate:
    def sleep(self) -> None:
        pass


class RosOperator:
    def __init__(self, args: argparse.Namespace) -> None:
        from cv_bridge import CvBridge
        import rospy
        from sensor_msgs.msg import Image
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Header

        self.args = args
        self.rospy = rospy
        self.Image = Image
        self.JointState = JointState
        self.Header = Header
        self.bridge = CvBridge()
        self.img_front_deque = deque()
        self.img_left_deque = deque()
        self.puppet_arm_deque = deque()
        self.state_history = StateHistoryBuffer(
            history_len=args.state_history_len,
            state_dim=7,
            history_fps=args.state_history_fps,
            max_sample_gap_seconds=args.state_history_max_gap_seconds,
        )
        self.puppet_arm_publisher = None
        self.tactile = None
        self._init_ros()

    def _init_ros(self) -> None:
        self.rospy.init_node("tactile_vla_single_arm_inference", anonymous=True)
        self.rospy.Subscriber(
            self.args.img_front_topic,
            self.Image,
            self.img_front_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.rospy.Subscriber(
            self.args.img_left_topic,
            self.Image,
            self.img_left_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.rospy.Subscriber(
            self.args.puppet_arm_topic,
            self.JointState,
            self.puppet_arm_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.puppet_arm_publisher = self.rospy.Publisher(
            self.args.puppet_arm_cmd_topic,
            self.JointState,
            queue_size=10,
        )
        self.tactile = RosTactileBuffer(make_tactile_topics(self.args), window_size=self.args.tactile_window_size)

    def rate(self, hz: int):
        return self.rospy.Rate(hz)

    def is_shutdown(self) -> bool:
        return self.rospy.is_shutdown() or shutdown_event.is_set()

    def puppet_arm_publish(self, joints: np.ndarray) -> None:
        joint_state_msg = self.JointState()
        joint_state_msg.header = self.Header()
        joint_state_msg.header.stamp = self.rospy.Time.now()
        joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_state_msg.position = joints.tolist()
        self.puppet_arm_publisher.publish(joint_state_msg)

    def get_frame(self):
        if len(self.img_front_deque) == 0 or len(self.img_left_deque) == 0 or len(self.puppet_arm_deque) == 0:
            return False

        frame_time = min(
            [
                self.img_front_deque[-1].header.stamp.to_sec(),
                self.img_left_deque[-1].header.stamp.to_sec(),
                self.puppet_arm_deque[-1].header.stamp.to_sec(),
            ]
        )
        if self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.puppet_arm_deque[-1].header.stamp.to_sec() < frame_time:
            return False

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), "passthrough")

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), "passthrough")

        while self.puppet_arm_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_deque.popleft()
        puppet_arm = self.puppet_arm_deque.popleft()
        return img_front, img_left, puppet_arm

    def img_front_callback(self, msg) -> None:
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_callback(self, msg) -> None:
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def puppet_arm_callback(self, msg) -> None:
        if len(self.puppet_arm_deque) >= 2000:
            self.puppet_arm_deque.popleft()
        self.puppet_arm_deque.append(msg)
        self.state_history.push(msg.header.stamp.to_sec(), np.asarray(msg.position, dtype=np.float32))

    def reset_state_history(self) -> None:
        self.state_history.clear()

    def get_state_history(self, puppet_arm) -> tuple[np.ndarray, np.ndarray]:
        return self.state_history.snapshot(
            current_timestamp=puppet_arm.header.stamp.to_sec(),
            current_state=np.asarray(puppet_arm.position, dtype=np.float32),
        )


@dataclass
class ReplayJointState:
    position: np.ndarray
    timestamp: float
    index: int


class ReplayOperator:
    def __init__(self, args: argparse.Namespace) -> None:
        import h5py

        self.args = args
        self.attempt_dir = args.replay_attempt_dir
        self.file = h5py.File(self.attempt_dir / "data.hdf5", "r")
        self.size = int(self.file["size"][()])
        self.index = int(args.replay_start_index)
        if args.replay_max_frames is not None:
            self.stop = min(self.size, self.index + int(args.replay_max_frames))
        else:
            self.stop = self.size
        self.tactile = TactileWindowBuffer(window_size=args.tactile_window_size)

    def close(self) -> None:
        self.file.close()

    def rate(self, hz: int):
        return NoopRate()

    def is_shutdown(self) -> bool:
        return shutdown_event.is_set() or self.index >= self.stop

    def puppet_arm_publish(self, joints: np.ndarray) -> None:
        published_actions_history.append(np.asarray(joints, dtype=float))

    def _read_image(self, key: str, index: int) -> np.ndarray:
        relative = Path(_decode_hdf5_string(self.file[key][index]))
        image = cv2.imread(str(self.attempt_dir / relative), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Replay image not found: {self.attempt_dir / relative}")
        return image

    def get_frame(self):
        if self.index >= self.stop:
            return False
        index = self.index
        self.index += max(1, int(self.args.replay_step_stride))
        self.tactile.push_raw(
            left_force=self.file["tactile/force/left"][index],
            right_force=self.file["tactile/force/right"][index],
            left_mesh_3d=self.file["tactile/mesh_3d/left"][index],
            right_mesh_3d=self.file["tactile/mesh_3d/right"][index],
            left_mesh_3d_flow=self.file["tactile/mesh_3d_flow/left"][index],
            right_mesh_3d_flow=self.file["tactile/mesh_3d_flow/right"][index],
            timestamp=float(self.file["timestamp"][index]),
        )
        return (
            self._read_image("camera/color/front", index),
            self._read_image("camera/color/left", index),
            ReplayJointState(
                np.asarray(self.file["arm/jointStatePosition/puppetRight"][index], dtype=float),
                float(self.file["timestamp"][index]),
                index,
            ),
        )

    def reset_state_history(self) -> None:
        # One replay file is one attempt/episode, so its frame index already defines the boundary.
        pass

    def get_state_history(self, puppet_arm: ReplayJointState) -> tuple[np.ndarray, np.ndarray]:
        start = max(0, puppet_arm.index - self.args.state_history_len + 1)
        states = np.asarray(
            self.file["arm/jointStatePosition/puppetRight"][start : puppet_arm.index + 1],
            dtype=np.float32,
        )
        return pad_state_history(states, history_len=self.args.state_history_len, state_dim=7)


class MockPolicy:
    def get_server_metadata(self) -> dict[str, Any]:
        return {
            "name": "mock_tactile_vla_policy",
            "action_horizon": 30,
            "output_action_dim": 7,
            "use_state_history": True,
            "state_history_len": 60,
            "state_history_dim": 7,
            "state_history_fps": DEFAULT_STATE_HISTORY_FPS,
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("mode") == "reasoning":
            return {"recovery_plan": "Regrasp left.", "recovery_plan_probs": [1.0, 0.0, 0.0, 0.0]}
        state = np.asarray(payload["observation/state"], dtype=np.float32)
        actions = np.repeat(state[np.newaxis, :], 30, axis=0)
        return {
            "actions": actions,
            "need_recovery": False,
            "need_recovery_probs": [1.0, 0.0],
            "failure_reason": "tilted left",
            "failure_reason_probs": [1.0, 0.0, 0.0, 0.0],
        }


def get_ros_observation(args: argparse.Namespace, operator: RosOperator | ReplayOperator):
    rate = operator.rate(args.publish_rate)
    warned = False
    while not operator.is_shutdown():
        result = operator.get_frame()
        if result:
            return result
        if not warned:
            print("sync fail when get_ros_observation")
            warned = True
        rate.sleep()
    raise StopIteration("operator shutdown while waiting for observation")


def build_payload(
    *,
    mode: str,
    img_front_bgr: np.ndarray,
    img_left_bgr: np.ndarray,
    qpos: np.ndarray,
    state_history: np.ndarray,
    state_history_mask: np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape[0] != 7:
        raise ValueError(f"Expected puppetRight qpos dim 7, got {qpos.shape}")
    state_history = np.asarray(state_history, dtype=np.float32)
    state_history_mask = np.asarray(state_history_mask, dtype=np.bool_)
    if state_history.shape != (state_history_mask.shape[0], 7):
        raise ValueError(
            f"Expected state_history [T,7] matching mask [T], got {state_history.shape} and {state_history_mask.shape}"
        )
    return {
        "mode": mode,
        "observation/image": prepare_rgb(img_front_bgr),
        "observation/wrist_image": prepare_rgb(img_left_bgr),
        "observation/state": qpos,
        "observation/state_history": state_history,
        "observation/state_history_mask": state_history_mask,
        "prompt": prompt,
    }


def load_captioner(args: argparse.Namespace):
    if args.no_captioner:
        print("Captioner disabled explicitly; all tactile captions will use the no-rotation default")
        return None
    if not args.captioner_checkpoint.is_file():
        raise FileNotFoundError(
            f"Captioner checkpoint not found: {args.captioner_checkpoint}. "
            "Pass --captioner_checkpoint with a valid best.pt, or use --no-captioner explicitly."
        )
    return TactileCaptionerPredictor(args.captioner_checkpoint, device=args.captioner_device)


def current_tactile_caption(operator: RosOperator | ReplayOperator, captioner) -> str:
    if operator.tactile is None:
        return DEFAULT_TACTILE_CAPTION
    return operator.tactile.caption(captioner)


def tactile_ignore_status(
    *,
    attempt_id: int,
    published_step: int,
    ignore_seconds: float,
    publish_rate: int,
) -> tuple[bool, int]:
    """Return whether tactile is ignored and how many base action steps remain."""
    if attempt_id <= 1:
        return False, 0
    ignore_steps = int(round(ignore_seconds * publish_rate))
    remaining_steps = max(0, ignore_steps - published_step)
    return remaining_steps > 0, remaining_steps


def append_runtime_log(args: argparse.Namespace, event: dict[str, Any]) -> None:
    if args.memory_log is None:
        return
    args.memory_log.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": time.time(), **event}
    with args.memory_log.open("a") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def publish_action_chunk(
    *,
    args: argparse.Namespace,
    operator: RosOperator | ReplayOperator,
    actions: np.ndarray,
    pre_action: np.ndarray,
    max_steps: int,
) -> tuple[np.ndarray, int]:
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions with shape [T,7], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Refusing to publish actions containing NaN or Inf")
    rate = operator.rate(args.publish_rate)
    published_steps = 0
    last_raw_action = pre_action.copy()
    limit = min(args.chunk_size, int(max_steps), actions.shape[0])
    for raw_action in actions[:limit]:
        if shutdown_event.is_set() or operator.is_shutdown():
            break
        if args.use_actions_interpolation:
            interp_actions = interpolate_action(args, pre_action, raw_action)
        else:
            interp_actions = raw_action[np.newaxis, :]

        for act in interp_actions:
            publish_action = np.asarray(act, dtype=float).copy()
            publish_action[6] = max(0.0, publish_action[6] - args.gripper_offset)
            if args.no_publish:
                published_actions_history.append(publish_action)
            else:
                operator.puppet_arm_publish(publish_action)
                published_actions_history.append(publish_action)
            rate.sleep()

        pre_action = np.asarray(raw_action, dtype=float).copy()
        last_raw_action = pre_action
        published_steps += 1
        print("Published Step", published_steps)
    return last_raw_action, published_steps


def run_reasoning(
    *,
    args: argparse.Namespace,
    policy,
    img_front: np.ndarray,
    img_left: np.ndarray,
    qpos: np.ndarray,
    state_history: np.ndarray,
    state_history_mask: np.ndarray,
    tactile_caption: str,
    memory: list[dict[str, Any]],
    failed_attempt_id: int,
) -> str:
    prompt = build_reasoning_prompt(
        instruction=args.instruction,
        failed_tactile_caption=tactile_caption,
        failure_recovery_memory=memory,
        case_id=args.case_id,
        failed_attempt_id=failed_attempt_id,
    )
    payload = build_payload(
        mode="reasoning",
        img_front_bgr=img_front,
        img_left_bgr=img_left,
        qpos=qpos,
        state_history=state_history,
        state_history_mask=state_history_mask,
        prompt=prompt,
    )
    response = policy.infer(payload)
    recovery_plan = str(response["recovery_plan"])
    append_runtime_log(
        args,
        {
            "event": "reasoning",
            "failed_attempt_id": failed_attempt_id,
            "recovery_plan": recovery_plan,
            "recovery_plan_probs": response.get("recovery_plan_probs"),
        },
    )
    print(f"Reasoning recovery_plan: {recovery_plan}")
    return recovery_plan


def run_closed_loop(args: argparse.Namespace, operator: RosOperator | ReplayOperator, policy, captioner) -> None:
    server_metadata = policy.get_server_metadata()
    print(f"Server metadata: {server_metadata}")
    server_action_horizon = int(server_metadata.get("action_horizon", 0))
    if server_action_horizon <= 0:
        raise ValueError(f"Server metadata has invalid action_horizon: {server_action_horizon}")
    if args.chunk_size > server_action_horizon:
        raise ValueError(
            f"Requested chunk_size={args.chunk_size}, but the server only returns "
            f"action_horizon={server_action_horizon}. Use matching H50/H30 artifacts or reduce --chunk_size."
        )
    if not bool(server_metadata.get("use_state_history", False)):
        raise ValueError("This V2 inference client requires a server with use_state_history=true")
    server_history_len = int(server_metadata.get("state_history_len", 0))
    server_history_dim = int(server_metadata.get("state_history_dim", 0))
    server_history_fps = float(server_metadata.get("state_history_fps", 0.0))
    if (server_history_len, server_history_dim) != (args.state_history_len, 7):
        raise ValueError(
            "Client/server state-history shape mismatch: "
            f"client=[{args.state_history_len},7], server=[{server_history_len},{server_history_dim}]"
        )
    if not np.isclose(server_history_fps, args.state_history_fps):
        raise ValueError(
            f"Client state_history_fps={args.state_history_fps} does not match "
            f"server state_history_fps={server_history_fps}"
        )
    history_span_seconds = (args.state_history_len - 1) / args.state_history_fps
    print(
        "State history resampling: "
        f"{args.state_history_len} frames at {args.state_history_fps:g} Hz "
        f"({history_span_seconds:.3f} s), nearest ROS qpos within "
        f"{args.state_history_max_gap_seconds * 1000.0:.1f} ms"
    )
    if not args.start_immediately and args.replay_attempt_dir is None:
        input("Press enter to continue")

    pre_action: np.ndarray | None = None
    input_recovery_plan = ""
    memory: list[dict[str, Any]] = []

    with KeyboardPoller() as keyboard:
        for attempt_id in range(1, args.max_attempts + 1):
            operator.reset_state_history()
            print(f"Starting attempt {attempt_id} recovery_plan={input_recovery_plan or 'none'}")
            step = 0
            if attempt_id > 1 and args.recovery_tactile_ignore_seconds > 0:
                ignore_steps = int(round(args.recovery_tactile_ignore_seconds * args.publish_rate))
                print(
                    "Ignoring tactile caption during homing for "
                    f"{ignore_steps} published steps ({args.recovery_tactile_ignore_seconds:.3f} s)"
                )
            while step < args.max_publish_step and not operator.is_shutdown():
                img_front, img_left, puppet_arm = get_ros_observation(args, operator)
                qpos = np.asarray(puppet_arm.position, dtype=float)
                state_history, state_history_mask = operator.get_state_history(puppet_arm)
                if pre_action is None:
                    pre_action = qpos.copy()

                tactile_ignored, tactile_ignore_remaining_steps = tactile_ignore_status(
                    attempt_id=attempt_id,
                    published_step=step,
                    ignore_seconds=args.recovery_tactile_ignore_seconds,
                    publish_rate=args.publish_rate,
                )
                if tactile_ignored:
                    tactile_caption = DEFAULT_TACTILE_CAPTION
                else:
                    tactile_caption = current_tactile_caption(operator, captioner)
                prompt = build_execution_prompt(
                    instruction=args.instruction,
                    tactile_caption=tactile_caption,
                    input_recovery_plan=input_recovery_plan,
                    case_id=args.case_id,
                    attempt_id=attempt_id,
                )
                payload = build_payload(
                    mode="execution",
                    img_front_bgr=img_front,
                    img_left_bgr=img_left,
                    qpos=qpos,
                    state_history=state_history,
                    state_history_mask=state_history_mask,
                    prompt=prompt,
                )
                start = time.time()
                response = policy.infer(payload)
                print(f"Model input prompt: {prompt}")
                print(f"State history valid frames: {int(state_history_mask.sum())}/{state_history_mask.shape[0]}")
                print(f"Model inference time: {(time.time() - start) * 1000:.3f} ms")

                need_recovery = bool(response.get("need_recovery", False))
                if need_recovery:
                    failure_reason = str(response.get("failure_reason", "unknown"))
                    memory_plan = "initial grasp" if attempt_id == 1 else input_recovery_plan
                    entry = {
                        "attempt_id": attempt_id,
                        "recovery_plan": memory_plan,
                        "failure_reason": failure_reason,
                    }
                    memory.append(entry)
                    append_runtime_log(
                        args,
                        {
                            "event": "need_recovery",
                            "attempt_id": attempt_id,
                            "memory_entry": entry,
                            "need_recovery_probs": response.get("need_recovery_probs"),
                            "failure_reason_probs": response.get("failure_reason_probs"),
                            "tactile_caption": tactile_caption,
                            "tactile_ignored": tactile_ignored,
                            "tactile_ignore_remaining_steps": tactile_ignore_remaining_steps,
                            "state_history_valid_frames": int(state_history_mask.sum()),
                        },
                    )
                    print(f"need_recovery=true failure_reason={failure_reason}; stopping this chunk")
                    if attempt_id >= args.max_attempts:
                        print("Reached max attempts after recovery trigger")
                        return
                    input_recovery_plan = run_reasoning(
                        args=args,
                        policy=policy,
                        img_front=img_front,
                        img_left=img_left,
                        qpos=qpos,
                        state_history=state_history,
                        state_history_mask=state_history_mask,
                        tactile_caption=tactile_caption,
                        memory=memory,
                        failed_attempt_id=attempt_id,
                    )
                    break

                actions = np.asarray(response["actions"], dtype=float)
                pre_action, published = publish_action_chunk(
                    args=args,
                    operator=operator,
                    actions=actions,
                    pre_action=pre_action,
                    max_steps=args.max_publish_step - step,
                )
                step += published
                append_runtime_log(
                    args,
                    {
                        "event": "execution_chunk",
                        "attempt_id": attempt_id,
                        "step": step,
                        "tactile_caption": tactile_caption,
                        "tactile_ignored": tactile_ignored,
                        "tactile_ignore_remaining_steps": max(
                            0, tactile_ignore_remaining_steps - published
                        ),
                        "need_recovery_probs": response.get("need_recovery_probs"),
                        "state_history_valid_frames": int(state_history_mask.sum()),
                    },
                )

                key = keyboard.get_key()
                if key == args.quit_key:
                    print("Operator requested safety stop")
                    shutdown_event.set()
                    append_runtime_log(args, {"event": "operator_quit", "attempt_id": attempt_id, "step": step})
                    return
                if key == args.success_key:
                    print("Operator confirmed success")
                    append_runtime_log(args, {"event": "operator_success", "attempt_id": attempt_id, "step": step})
                    return
            else:
                print(f"Attempt {attempt_id} ended without recovery trigger")
                return


def get_arguments() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Optional YAML topic override file. By default all runtime topics are hard-coded in this script.",
    )
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--case_id", type=str, default="runtime")
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--puppet_arm_cmd_topic", type=str, default="/master/joint_right")
    parser.add_argument("--puppet_arm_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--tactile_left_force_topic", type=str, default="/xense/OG001251/force")
    parser.add_argument("--tactile_right_force_topic", type=str, default="/xense/OG000991/force")
    parser.add_argument("--tactile_left_mesh_3d_topic", type=str, default="/xense/OG001251/mesh_3d")
    parser.add_argument("--tactile_right_mesh_3d_topic", type=str, default="/xense/OG000991/mesh_3d")
    parser.add_argument("--tactile_left_mesh_3d_flow_topic", type=str, default="/xense/OG001251/mesh_3d_flow")
    parser.add_argument("--tactile_right_mesh_3d_flow_topic", type=str, default="/xense/OG000991/mesh_3d_flow")
    parser.add_argument("--tactile_window_size", type=int, default=30)
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--state-history-len", type=int, default=60)
    parser.add_argument(
        "--state-history-fps",
        type=float,
        default=DEFAULT_STATE_HISTORY_FPS,
        help="Timestamp grid frequency for the model state history; independent of ROS topic and action publish rates.",
    )
    parser.add_argument(
        "--state-history-max-gap-seconds",
        type=float,
        default=0.02,
        help="Mask a history point when its nearest ROS qpos is farther away than this threshold.",
    )
    parser.add_argument(
        "--arm_steps_length",
        nargs=7,
        type=float,
        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
    )
    parser.add_argument("--use_actions_interpolation", action="store_true", default=False)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mock-policy", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--replay-attempt-dir", type=Path)
    parser.add_argument("--replay-start-index", type=int, default=0)
    parser.add_argument("--replay-step-stride", type=int, default=1)
    parser.add_argument("--replay-max-frames", type=int)
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--success-key", type=str, default="s")
    parser.add_argument("--quit-key", type=str, default="q")
    parser.add_argument("--captioner_checkpoint", type=Path, default=DEFAULT_CAPTIONER)
    parser.add_argument("--captioner_device", type=str, default="auto")
    parser.add_argument("--no-captioner", action="store_true")
    parser.add_argument(
        "--recovery-tactile-ignore-seconds",
        type=float,
        default=5.0,
        help=(
            "For attempts after the first, force the no-rotation tactile caption for this many seconds "
            "of published base action steps while homing."
        ),
    )
    parser.add_argument("--gripper_offset", type=float, default=0.001)
    parser.add_argument("--memory_log", type=Path, default=PROJECT_ROOT / "outputs" / "runtime" / "tactile_vla_memory.jsonl")
    args = parser.parse_args()
    return args, parser


def main() -> None:
    args, parser = get_arguments()
    apply_yaml_defaults(args, parser)
    if args.publish_rate <= 0:
        parser.error("--publish_rate must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")
    if args.state_history_len <= 0:
        parser.error("--state-history-len must be positive")
    if args.state_history_fps <= 0:
        parser.error("--state-history-fps must be positive")
    if args.state_history_max_gap_seconds <= 0:
        parser.error("--state-history-max-gap-seconds must be positive")
    if args.recovery_tactile_ignore_seconds < 0:
        parser.error("--recovery-tactile-ignore-seconds must be non-negative")
    if args.seed is not None:
        set_seed(args.seed)
    signal.signal(signal.SIGINT, _on_sigint)

    captioner = load_captioner(args)
    if args.mock_policy:
        policy = MockPolicy()
    else:
        policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    operator: RosOperator | ReplayOperator
    if args.replay_attempt_dir is not None:
        operator = ReplayOperator(args)
    else:
        operator = RosOperator(args)

    try:
        run_closed_loop(args, operator, policy, captioner)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        if isinstance(operator, ReplayOperator):
            operator.close()


if __name__ == "__main__":
    main()
