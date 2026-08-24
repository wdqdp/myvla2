#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Agilex single-arm closed-loop inference for the tactile VLA policy."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
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
from tactile_vla.vla.prompts import build_assessment_prompt
from tactile_vla.vla.prompts import build_execution_prompt
from tactile_vla.vla.prompts import build_failure_prompt
from tactile_vla.vla.prompts import build_monitor_prompt
from tactile_vla.vla.prompts import build_reasoning_prompt
from tactile_vla.vla.prompts import MAX_MEMORY_PAIRS
from tactile_vla.vla.prompts import MAX_SUPPORTED_ATTEMPTS
from tactile_vla.vla.prompts import resolve_prompt_profile
from tactile_vla.vla.prompts import update_failure_recovery_memory

DEFAULT_INSTRUCTION = "Pick up and transfer the object stably."
DEFAULT_CAPTIONER = Path("/data1/outputs/tactile_captioner/tcn_v3_w30_multifield/best.pt")

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
        # The forced-recovery ablation can pause for an arbitrary amount of
        # operator inspection time between chunks.  Keep that wall-clock pause
        # out of the model history instead of filling the 30 Hz window with a
        # stationary robot.  Normal inference never enables this gate.
        self._state_history_gate_lock = threading.Lock()
        self._state_history_paused = False
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

    def puppet_arm_publish(self, joints: np.ndarray) -> float:
        joint_state_msg = self.JointState()
        joint_state_msg.header = self.Header()
        joint_state_msg.header.stamp = self.rospy.Time.now()
        joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_state_msg.position = joints.tolist()
        self.puppet_arm_publisher.publish(joint_state_msg)
        return float(joint_state_msg.header.stamp.to_sec())

    def get_frame(self, *, after_timestamp: float | None = None):
        if len(self.img_front_deque) == 0 or len(self.img_left_deque) == 0 or len(self.puppet_arm_deque) == 0:
            return False

        frame_time = min(
            [
                self.img_front_deque[-1].header.stamp.to_sec(),
                self.img_left_deque[-1].header.stamp.to_sec(),
                self.puppet_arm_deque[-1].header.stamp.to_sec(),
            ]
        )
        if after_timestamp is not None and frame_time <= after_timestamp:
            return False
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

    def get_latest_frame(self):
        """Return the newest available messages without consuming the synchronized queues."""
        if len(self.img_front_deque) == 0 or len(self.img_left_deque) == 0 or len(self.puppet_arm_deque) == 0:
            return False
        img_front_msg = self.img_front_deque[-1]
        img_left_msg = self.img_left_deque[-1]
        puppet_arm = self.puppet_arm_deque[-1]
        img_front = self.bridge.imgmsg_to_cv2(img_front_msg, "passthrough")
        img_left = self.bridge.imgmsg_to_cv2(img_left_msg, "passthrough")
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
        with self._state_history_gate_lock:
            if not self._state_history_paused:
                self.state_history.push(
                    msg.header.stamp.to_sec(),
                    np.asarray(msg.position, dtype=np.float32),
                )

    def reset_state_history(self) -> None:
        with self._state_history_gate_lock:
            self.state_history.clear()

    def pause_state_history(self) -> None:
        """Stop accepting ROS qpos callbacks into the model history."""

        with self._state_history_gate_lock:
            self._state_history_paused = True

    def resume_state_history(
        self,
        history: np.ndarray,
        mask: np.ndarray,
        *,
        current_timestamp: float,
    ) -> None:
        """Restore a frozen snapshot on a fresh time grid and resume callbacks."""

        history = np.asarray(history, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.bool_)
        expected_history_shape = (self.state_history.history_len, self.state_history.state_dim)
        if history.shape != expected_history_shape:
            raise ValueError(
                f"Expected frozen history {expected_history_shape}, got {history.shape}"
            )
        if mask.shape != (self.state_history.history_len,):
            raise ValueError(
                f"Expected frozen history mask [{self.state_history.history_len}], got {mask.shape}"
            )
        current_timestamp = float(current_timestamp)
        if not np.isfinite(current_timestamp):
            raise ValueError("Frozen history resume timestamp must be finite")
        target_timestamps = current_timestamp + (
            np.arange(self.state_history.history_len, dtype=np.float64)
            - (self.state_history.history_len - 1)
        ) / self.state_history.history_fps
        with self._state_history_gate_lock:
            self.state_history.clear()
            for timestamp, state, valid in zip(
                target_timestamps,
                history,
                mask,
                strict=True,
            ):
                if valid:
                    self.state_history.push(float(timestamp), state)
            self._state_history_paused = False

    def get_state_history(self, puppet_arm) -> tuple[np.ndarray, np.ndarray]:
        with self._state_history_gate_lock:
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

    def get_frame(self, *, after_timestamp: float | None = None):
        del after_timestamp
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

    def get_latest_frame(self):
        return self.get_frame()

    def reset_state_history(self) -> None:
        # One replay file is one attempt/episode, so its frame index already defines the boundary.
        pass

    def pause_state_history(self) -> None:
        # Replay frames advance only when get_frame is called, so waiting is
        # already history-neutral.
        pass

    def resume_state_history(
        self,
        history: np.ndarray,
        mask: np.ndarray,
        *,
        current_timestamp: float,
    ) -> None:
        del history, mask, current_timestamp

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
            "supports_step_monitor": True,
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("mode") == "reasoning":
            return {"recovery_plan": "Regrasp left.", "recovery_plan_probs": [1.0, 0.0, 0.0, 0.0]}
        status = {
            "need_recovery": False,
            "need_recovery_probs": [1.0, 0.0],
            "failure_reason": "tilted left",
            "failure_reason_probs": [1.0, 0.0, 0.0, 0.0],
            "policy_timing": {"infer_ms": 0.0},
        }
        if payload.get("mode") == "monitor":
            return status
        state = np.asarray(payload["observation/state"], dtype=np.float32)
        actions = np.repeat(state[np.newaxis, :], 30, axis=0)
        return {
            "actions": actions,
            **status,
        }


def get_ros_observation(
    args: argparse.Namespace,
    operator: RosOperator | ReplayOperator,
    *,
    after_timestamp: float | None = None,
):
    rate = operator.rate(args.observation_poll_rate)
    warned = False
    while not operator.is_shutdown():
        result = operator.get_frame(after_timestamp=after_timestamp)
        if result:
            return result
        if not warned:
            print("sync fail when get_ros_observation")
            warned = True
        rate.sleep()
    raise StopIteration("operator shutdown while waiting for observation")


def get_latest_observation(args: argparse.Namespace, operator: RosOperator | ReplayOperator):
    """Wait for any complete latest observation without requiring it to postdate a specific action."""
    rate = operator.rate(args.observation_poll_rate)
    while not operator.is_shutdown():
        result = operator.get_latest_frame()
        if result:
            return result
        rate.sleep()
    raise StopIteration("operator shutdown while waiting for the latest observation")


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
        print("Captioner disabled explicitly; all tactile captions will use the all-neutral default")
        return None
    if not args.captioner_checkpoint.is_file():
        raise FileNotFoundError(
            f"Captioner checkpoint not found: {args.captioner_checkpoint}. "
            "Pass --captioner_checkpoint with a valid best.pt, or use --no-captioner explicitly."
        )
    predictor = TactileCaptionerPredictor(args.captioner_checkpoint, device=args.captioner_device)
    if predictor.window_size != args.tactile_window_size:
        raise ValueError(
            f"Captioner checkpoint window_size={predictor.window_size} does not match "
            f"--tactile_window_size={args.tactile_window_size}"
        )
    return predictor


def current_tactile_caption(operator: RosOperator | ReplayOperator, captioner) -> str:
    if operator.tactile is None:
        return DEFAULT_TACTILE_CAPTION
    return operator.tactile.caption(captioner)


def recovery_monitor_ignore_status(
    *,
    attempt_id: int,
    published_step: int,
    ignore_seconds: float,
    publish_rate: int,
) -> tuple[bool, int]:
    """Return whether recovery assessment is suppressed and how many action steps remain."""
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


def publish_single_action(
    *,
    args: argparse.Namespace,
    operator: RosOperator | ReplayOperator,
    raw_action: np.ndarray,
    pre_action: np.ndarray,
    rate,
) -> tuple[np.ndarray, float | None]:
    raw_action = np.asarray(raw_action, dtype=float)
    if raw_action.shape != (7,):
        raise ValueError(f"Expected one action with shape [7], got {raw_action.shape}")
    if not np.isfinite(raw_action).all():
        raise ValueError("Refusing to publish an action containing NaN or Inf")
    if args.use_actions_interpolation:
        interp_actions = interpolate_action(args, pre_action, raw_action)
    else:
        interp_actions = raw_action[np.newaxis, :]

    publish_timestamp = None
    for index, act in enumerate(interp_actions):
        if shutdown_event.is_set() or operator.is_shutdown():
            break
        publish_action = np.asarray(act, dtype=float).copy()
        publish_action[6] = max(0.0, publish_action[6] - args.gripper_offset)
        if args.no_publish:
            published_actions_history.append(publish_action)
        else:
            publish_timestamp = operator.puppet_arm_publish(publish_action)
            published_actions_history.append(publish_action)
        if index + 1 < len(interp_actions):
            rate.sleep()

    return raw_action.copy(), publish_timestamp


@dataclass(frozen=True)
class AsyncMonitorResult:
    attempt_id: int
    captured_step: int
    img_front: np.ndarray
    img_left: np.ndarray
    qpos: np.ndarray
    state_history: np.ndarray
    state_history_mask: np.ndarray
    tactile_caption: str
    response: dict[str, Any]
    observation_wait_ms: float
    caption_ms: float
    client_monitor_ms: float
    total_monitor_ms: float
    finished_monotonic: float


def run_async_monitor_once(
    *,
    args: argparse.Namespace,
    operator: RosOperator | ReplayOperator,
    policy,
    captioner,
    attempt_id: int,
    captured_step: int,
    input_recovery_plan: str,
) -> AsyncMonitorResult:
    start = time.perf_counter()
    img_front, img_left, puppet_arm = get_latest_observation(args, operator)
    observation_wait_ms = (time.perf_counter() - start) * 1000.0
    qpos = np.asarray(puppet_arm.position, dtype=float)
    state_history, state_history_mask = operator.get_state_history(puppet_arm)
    caption_start = time.perf_counter()
    tactile_caption = current_tactile_caption(operator, captioner)
    caption_ms = (time.perf_counter() - caption_start) * 1000.0
    if getattr(args, "v3_shared_assessment", False):
        prompt = build_assessment_prompt(
            instruction=args.instruction,
            tactile_caption=tactile_caption,
            input_recovery_plan=input_recovery_plan,
            prompt_profile=args.prompt_profile,
        )
        mode = "assessment"
    elif getattr(args, "v3_autoregressive", False):
        prompt = build_monitor_prompt(
            instruction=args.instruction,
            tactile_caption=tactile_caption,
            input_recovery_plan=input_recovery_plan,
            prompt_profile=args.prompt_profile,
        )
        mode = "monitor"
    else:
        prompt = build_execution_prompt(
            instruction=args.instruction,
            tactile_caption=tactile_caption,
            input_recovery_plan=input_recovery_plan,
            case_id=args.case_id,
            attempt_id=attempt_id,
            prompt_profile=args.prompt_profile,
        )
        mode = "monitor"
    payload = build_payload(
        mode=mode,
        img_front_bgr=img_front,
        img_left_bgr=img_left,
        qpos=qpos,
        state_history=state_history,
        state_history_mask=state_history_mask,
        prompt=prompt,
    )
    monitor_start = time.perf_counter()
    response = policy.infer(payload)
    client_monitor_ms = (time.perf_counter() - monitor_start) * 1000.0
    finished_monotonic = time.perf_counter()
    return AsyncMonitorResult(
        attempt_id=attempt_id,
        captured_step=captured_step,
        img_front=img_front,
        img_left=img_left,
        qpos=qpos,
        state_history=state_history,
        state_history_mask=state_history_mask,
        tactile_caption=tactile_caption,
        response=response,
        observation_wait_ms=observation_wait_ms,
        caption_ms=caption_ms,
        client_monitor_ms=client_monitor_ms,
        total_monitor_ms=(finished_monotonic - start) * 1000.0,
        finished_monotonic=finished_monotonic,
    )


def report_async_monitor_result(
    *,
    args: argparse.Namespace,
    result: AsyncMonitorResult,
    handled_step: int,
    previous_finished_monotonic: float | None,
) -> None:
    response = result.response
    need_recovery = bool(response.get("need_recovery", False))
    server_monitor_ms = response.get("policy_timing", {}).get("infer_ms")
    lag_steps = max(0, handled_step - result.captured_step)
    monitor_interval_ms = (
        (result.finished_monotonic - previous_finished_monotonic) * 1000.0
        if previous_finished_monotonic is not None
        else None
    )
    monitor_rate_hz = (
        1000.0 / monitor_interval_ms if monitor_interval_ms is not None and monitor_interval_ms > 0 else None
    )
    server_timing_text = f"server={float(server_monitor_ms):.3f} ms " if server_monitor_ms is not None else ""
    rate_text = f"monitor_rate={monitor_rate_hz:.3f} Hz " if monitor_rate_hz is not None else ""
    print(
        "Async monitor: "
        # f"captured_step={result.captured_step} handled_step={handled_step} lag_steps={lag_steps} "
        f"need_recovery={str(need_recovery).lower()} "
        # f"client={result.client_monitor_ms:.3f} ms "
        # f"{server_timing_text}"
        # f"caption={result.caption_ms:.3f} ms "
        # f"total={result.total_monitor_ms:.3f} ms "
        # f"{rate_text}"
    )
    append_runtime_log(
        args,
        {
            "event": "async_step_monitor",
            "attempt_id": result.attempt_id,
            "captured_step": result.captured_step,
            "handled_step": handled_step,
            "lag_steps": lag_steps,
            "need_recovery": need_recovery,
            "need_recovery_probs": response.get("need_recovery_probs"),
            "failure_reason": response.get("failure_reason"),
            "failure_reason_probs": response.get("failure_reason_probs"),
            "tactile_caption": result.tactile_caption,
            "state_history_valid_frames": int(result.state_history_mask.sum()),
            "observation_wait_ms": result.observation_wait_ms,
            "caption_ms": result.caption_ms,
            "client_monitor_ms": result.client_monitor_ms,
            "server_monitor_ms": server_monitor_ms,
            "total_monitor_ms": result.total_monitor_ms,
            "monitor_interval_ms": monitor_interval_ms,
            "monitor_rate_hz": monitor_rate_hz,
        },
    )


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
        prompt_profile=args.prompt_profile,
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


def run_failure_diagnosis(
    *,
    args: argparse.Namespace,
    policy,
    result: AsyncMonitorResult,
    input_recovery_plan: str,
) -> str:
    prompt = build_failure_prompt(
        instruction=args.instruction,
        tactile_caption=result.tactile_caption,
        input_recovery_plan=input_recovery_plan,
        prompt_profile=args.prompt_profile,
    )
    payload = build_payload(
        mode="failure",
        img_front_bgr=result.img_front,
        img_left_bgr=result.img_left,
        qpos=result.qpos,
        state_history=result.state_history,
        state_history_mask=result.state_history_mask,
        prompt=prompt,
    )
    started = time.perf_counter()
    response = policy.infer(payload)
    client_ms = (time.perf_counter() - started) * 1000.0
    failure_reason = str(response["failure_reason"])
    append_runtime_log(
        args,
        {
            "event": "failure_diagnosis",
            "attempt_id": result.attempt_id,
            "captured_step": result.captured_step,
            "failure_reason": failure_reason,
            "client_ms": client_ms,
            "server_ms": response.get("policy_timing", {}).get("infer_ms"),
        },
    )
    print(f"Failure diagnosis: {failure_reason}")
    return failure_reason


def run_closed_loop(args: argparse.Namespace, operator: RosOperator | ReplayOperator, policy, captioner) -> None:
    server_metadata = policy.get_server_metadata()
    print(f"Server metadata: {server_metadata}")
    expected_data_profile = getattr(args, "expected_data_profile", None)
    server_data_profile = str(server_metadata.get("data_profile", "legacy"))
    if expected_data_profile is not None and server_data_profile != expected_data_profile:
        raise ValueError(
            "Client/server data profile mismatch: "
            f"expected={expected_data_profile!r}, server={server_data_profile!r}"
        )
    args.prompt_profile = resolve_prompt_profile(server_metadata.get("prompt_profile"))
    print(f"Using checkpoint prompt profile: {args.prompt_profile}")
    server_max_memory_pairs = int(server_metadata.get("max_memory_pairs", MAX_MEMORY_PAIRS))
    server_max_attempts = int(server_metadata.get("max_supported_attempts", MAX_SUPPORTED_ATTEMPTS))
    if server_max_memory_pairs != MAX_MEMORY_PAIRS:
        raise ValueError(
            "Client/server recovery memory mismatch: "
            f"client={MAX_MEMORY_PAIRS}, server={server_max_memory_pairs}"
        )
    if server_max_attempts != MAX_SUPPORTED_ATTEMPTS:
        raise ValueError(
            "Client/server attempt limit mismatch: "
            f"client={MAX_SUPPORTED_ATTEMPTS}, server={server_max_attempts}"
        )
    if not 1 <= args.max_attempts <= MAX_SUPPORTED_ATTEMPTS:
        raise ValueError(
            f"Requested max_attempts={args.max_attempts}, but server supports at most "
            f"{MAX_SUPPORTED_ATTEMPTS} attempts"
        )
    args.v3_autoregressive = str(server_metadata.get("stage_b_version", "")).startswith("v3_")
    args.v3_shared_assessment = args.v3_autoregressive and bool(
        server_metadata.get("supports_shared_assessment", False)
    )
    if args.v3_autoregressive and not bool(server_metadata.get("supports_failure_generation", False)):
        raise ValueError("V3 server does not advertise supports_failure_generation=true")
    server_action_horizon = int(server_metadata.get("action_horizon", 0))
    if server_action_horizon <= 0:
        raise ValueError(f"Server metadata has invalid action_horizon: {server_action_horizon}")
    if args.chunk_size > server_action_horizon:
        raise ValueError(
            f"Requested chunk_size={args.chunk_size}, but the server only returns "
            f"action_horizon={server_action_horizon}. Use matching H50/H30 artifacts or reduce --chunk_size."
        )
    if not bool(server_metadata.get("supports_step_monitor", False)):
        raise ValueError(
            "The server does not advertise supports_step_monitor=true. "
            "Restart it with the updated serve_tactile_vla_policy.py."
        )
    server_uses_history = bool(server_metadata.get("use_state_history", False))
    if args.v3_autoregressive:
        if server_uses_history:
            print("V3 action expert uses dense proprioceptive state history.")
        else:
            print(
                "V3 checkpoint uses only the current proprioceptive state; "
                "state-history payload is ignored by the server."
            )
        if args.v3_shared_assessment:
            print(
                "V3 shared assessment enabled: need_recovery and failure_reason reuse one "
                "VLM prefix/KV cache."
            )
        else:
            print(
                "V3 server has no shared-assessment capability; falling back to separate "
                "monitor and failure requests."
            )
    elif not server_uses_history:
        raise ValueError("This V2 inference path requires a server with use_state_history=true")

    if server_uses_history:
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
    print(
        "Asynchronous monitor enabled: cached chunk actions publish at "
        f"{args.publish_rate} Hz while one background VLM request continuously uses the latest observation."
    )
    print(f"Latest-observation polling rate: {args.observation_poll_rate} Hz")
    if args.use_actions_interpolation:
        print("Monitor results are checked after each raw chunk action, not after interpolation substeps.")
    if not args.start_immediately and args.replay_attempt_dir is None:
        input("Press enter to continue")

    pre_action: np.ndarray | None = None
    input_recovery_plan = ""
    memory: list[dict[str, Any]] = []
    previous_monitor_finished: float | None = None

    def record_monitor(result: AsyncMonitorResult, *, handled_step: int) -> bool:
        nonlocal previous_monitor_finished
        report_async_monitor_result(
            args=args,
            result=result,
            handled_step=handled_step,
            previous_finished_monotonic=previous_monitor_finished,
        )
        previous_monitor_finished = result.finished_monotonic
        return bool(result.response.get("need_recovery", False))

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tactile-vla-monitor") as monitor_executor:
        monitor_future: Future[AsyncMonitorResult] | None = None
        with KeyboardPoller() as keyboard:
            for attempt_id in range(1, args.max_attempts + 1):
                if monitor_future is not None:
                    raise RuntimeError("Async monitor leaked across attempt boundary")
                operator.reset_state_history()
                print(f"Starting attempt {attempt_id} recovery_plan={input_recovery_plan or 'none'}")
                step = 0
                recovery_result: AsyncMonitorResult | None = None
                if attempt_id > 1 and args.recovery_tactile_ignore_seconds > 0:
                    ignore_steps = int(round(args.recovery_tactile_ignore_seconds * args.publish_rate))
                    print(
                        "Suppressing recovery assessment during homing for "
                        f"{ignore_steps} published steps ({args.recovery_tactile_ignore_seconds:.3f} s); "
                        "action inference still uses the live tactile caption"
                    )

                while step < args.max_publish_step and not operator.is_shutdown():
                    # No monitor request is active at chunk boundaries, so this WebSocket
                    # call cannot race with the background monitor.
                    img_front, img_left, puppet_arm = get_ros_observation(args, operator)
                    qpos = np.asarray(puppet_arm.position, dtype=float)
                    state_history, state_history_mask = operator.get_state_history(puppet_arm)
                    if pre_action is None:
                        pre_action = qpos.copy()

                    recovery_monitor_ignored, recovery_monitor_ignore_remaining_steps = (
                        recovery_monitor_ignore_status(
                            attempt_id=attempt_id,
                            published_step=step,
                            ignore_seconds=args.recovery_tactile_ignore_seconds,
                            publish_rate=args.publish_rate,
                        )
                    )
                    tactile_caption = current_tactile_caption(operator, captioner)
                    prompt = build_execution_prompt(
                        instruction=args.instruction,
                        tactile_caption=tactile_caption,
                        input_recovery_plan=input_recovery_plan,
                        case_id=args.case_id,
                        attempt_id=attempt_id,
                        prompt_profile=args.prompt_profile,
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
                    start = time.perf_counter()
                    response = policy.infer(payload)
                    action_infer_ms = (time.perf_counter() - start) * 1000.0
                    print(f"Model input prompt: {prompt}")
                    print(f"attemp: {attempt_id}")
                    # print(f"State history valid frames: {int(state_history_mask.sum())}/{state_history_mask.shape[0]}")
                    # print(f"Action chunk inference time: {action_infer_ms:.3f} ms")
                    append_runtime_log(
                        args,
                        {
                            "event": "action_chunk_inference",
                            "attempt_id": attempt_id,
                            "step": step,
                            "client_infer_ms": action_infer_ms,
                            "server_infer_ms": response.get("policy_timing", {}).get("infer_ms"),
                            "tactile_caption": tactile_caption,
                            "recovery_monitor_ignored": recovery_monitor_ignored,
                            "recovery_monitor_ignore_remaining_steps": recovery_monitor_ignore_remaining_steps,
                            "state_history_valid_frames": int(state_history_mask.sum()),
                        },
                    )

                    actions = np.asarray(response["actions"], dtype=float)
                    if actions.ndim != 2 or actions.shape[1] != 7:
                        raise ValueError(f"Expected actions with shape [T,7], got {actions.shape}")
                    if not np.isfinite(actions).all():
                        raise ValueError("Refusing to publish actions containing NaN or Inf")
                    limit = min(args.chunk_size, args.max_publish_step - step, actions.shape[0])
                    if limit <= 0:
                        break

                    chunk_published = 0
                    control_rate = operator.rate(args.publish_rate)
                    for raw_action in actions[:limit]:
                        if shutdown_event.is_set() or operator.is_shutdown():
                            break

                        # A completed true result stops execution before another action is sent.
                        if monitor_future is not None and monitor_future.done():
                            result = monitor_future.result()
                            monitor_future = None
                            if record_monitor(result, handled_step=step):
                                recovery_result = result
                                break

                        pre_action, _ = publish_single_action(
                            args=args,
                            operator=operator,
                            raw_action=raw_action,
                            pre_action=pre_action,
                            rate=control_rate,
                        )
                        step += 1
                        chunk_published += 1
                        print(f"Published Step {step}")

                        key = keyboard.get_key()
                        if key == args.quit_key:
                            print("Operator requested safety stop")
                            shutdown_event.set()
                            append_runtime_log(args, {"event": "operator_quit", "attempt_id": attempt_id, "step": step})
                            return
                        if key == args.success_key:
                            print("Operator confirmed success")
                            append_runtime_log(
                                args,
                                {"event": "operator_success", "attempt_id": attempt_id, "step": step},
                            )
                            return

                        # Check again after publishing so a result that finished during this
                        # action is handled without waiting for another 30 Hz period.
                        if monitor_future is not None and monitor_future.done():
                            result = monitor_future.result()
                            monitor_future = None
                            if record_monitor(result, handled_step=step):
                                recovery_result = result
                                break

                        recovery_monitor_ignored, _ = recovery_monitor_ignore_status(
                            attempt_id=attempt_id,
                            published_step=step,
                            ignore_seconds=args.recovery_tactile_ignore_seconds,
                            publish_rate=args.publish_rate,
                        )
                        # Suppress the request itself rather than discarding a true result:
                        # shared V3 assessment would otherwise already generate failure_reason.
                        if monitor_future is None and not recovery_monitor_ignored:
                            monitor_future = monitor_executor.submit(
                                run_async_monitor_once,
                                args=args,
                                operator=operator,
                                policy=policy,
                                captioner=captioner,
                                attempt_id=attempt_id,
                                captured_step=step,
                                input_recovery_plan=input_recovery_plan,
                            )

                        control_rate.sleep()

                    # Drain the in-flight request before using the same WebSocket for the
                    # next chunk. This also prevents missing a true result at a chunk boundary.
                    if recovery_result is None and monitor_future is not None:
                        result = monitor_future.result()
                        monitor_future = None
                        if record_monitor(result, handled_step=step):
                            recovery_result = result

                    append_runtime_log(
                        args,
                        {
                            "event": "execution_chunk",
                            "attempt_id": attempt_id,
                            "step": step,
                            "published_actions": chunk_published,
                            "discarded_actions": max(0, limit - chunk_published),
                            "recovery_triggered": recovery_result is not None,
                        },
                    )
                    if recovery_result is not None:
                        break

                if recovery_result is None:
                    print(f"Attempt {attempt_id} ended without recovery trigger")
                    return

                if args.v3_shared_assessment:
                    failure_reason = str(
                        recovery_result.response.get("failure_reason", "")
                    ).strip()
                    if not failure_reason:
                        raise ValueError(
                            "V3 assessment returned need_recovery=true without failure_reason"
                        )
                    print(f"Shared assessment failure_reason: {failure_reason}")
                elif args.v3_autoregressive:
                    failure_reason = run_failure_diagnosis(
                        args=args,
                        policy=policy,
                        result=recovery_result,
                        input_recovery_plan=input_recovery_plan,
                    )
                else:
                    failure_reason = str(recovery_result.response.get("failure_reason", "unknown"))
                memory_plan = "initial plan" if attempt_id == 1 else input_recovery_plan
                entry = {
                    "recovery_plan": memory_plan,
                    "failure_reason": failure_reason,
                }
                will_reason = attempt_id < args.max_attempts
                if will_reason:
                    memory = update_failure_recovery_memory(
                        memory,
                        entry,
                        prompt_profile=args.prompt_profile,
                    )
                append_runtime_log(
                    args,
                    {
                        "event": "need_recovery",
                        "attempt_id": attempt_id,
                        "step": step,
                        "captured_step": recovery_result.captured_step,
                        "lag_steps": max(0, step - recovery_result.captured_step),
                        "memory_entry": entry,
                        "memory_pairs": len(memory),
                        "reasoning_skipped": not will_reason,
                        "need_recovery_probs": recovery_result.response.get("need_recovery_probs"),
                        "failure_reason_probs": recovery_result.response.get("failure_reason_probs"),
                        "tactile_caption": recovery_result.tactile_caption,
                    },
                )
                print(f"need_recovery=true failure_reason={failure_reason}; discarding the remaining chunk")
                if not will_reason:
                    print("Reached max attempts after recovery trigger")
                    return
                input_recovery_plan = run_reasoning(
                    args=args,
                    policy=policy,
                    img_front=recovery_result.img_front,
                    img_left=recovery_result.img_left,
                    qpos=recovery_result.qpos,
                    state_history=recovery_result.state_history,
                    state_history_mask=recovery_result.state_history_mask,
                    tactile_caption=recovery_result.tactile_caption,
                    memory=memory,
                    failed_attempt_id=attempt_id,
                )


def get_arguments() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Optional YAML topic override file. By default all runtime topics are hard-coded in this script.",
    )
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--max_attempts", type=int, default=5)
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
    parser.add_argument(
        "--observation-poll-rate",
        type=int,
        default=200,
        help="Polling frequency while waiting for a synchronized observation newer than the published action.",
    )
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
    parser.add_argument(
        "--expected-data-profile",
        help="Refuse to run if server metadata does not advertise this exact data profile.",
    )
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
        "--recovery-need-ignore-seconds",
        "--recovery-tactile-ignore-seconds",
        dest="recovery_tactile_ignore_seconds",
        type=float,
        default=5.0,
        help=(
            "For attempts after the first, suppress asynchronous need_recovery/failure assessment for "
            "this many seconds of published base action steps while homing. Action inference continues "
            "to receive the live tactile caption."
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
    if not 1 <= args.max_attempts <= 5:
        parser.error("--max_attempts must be in [1, 5]")
    if args.observation_poll_rate <= 0:
        parser.error("--observation-poll-rate must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")
    if args.state_history_len <= 0:
        parser.error("--state-history-len must be positive")
    if args.state_history_fps <= 0:
        parser.error("--state-history-fps must be positive")
    if args.state_history_max_gap_seconds <= 0:
        parser.error("--state-history-max-gap-seconds must be positive")
    if args.recovery_tactile_ignore_seconds < 0:
        parser.error("--recovery-need-ignore-seconds must be non-negative")
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
