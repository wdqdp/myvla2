#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Agilex single-arm synchronous inference.

Derived from agilex_inference_openpi_sync.py, but adapted for:
- single arm only
- 7D state/action
- front + right cameras only

This script supports two payload layouts:
- libero: for cobot_magic-style openpi training configs that expect
  observation/image, observation/wrist_image, observation/state.
- agilex: for agilex-style configs that expect state + images[top_head, hand_right, hand_left].
"""

import argparse
import signal
import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from openpi_client import image_tools, websocket_client_policy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header


CAMERA_NAMES = ["cam_high", "cam_right_wrist"]
observation_window = None

DEFAULT_PROMPT = "pick the orange"

published_actions_history = []  # list[np.ndarray(shape=(7,))]
shutdown_event = threading.Event()


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def interpolate_action(args, prev_action, cur_action):
    steps = np.array(args.arm_steps_length, dtype=float)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    return {
        "episode_len": args.max_publish_step,
        "state_dim": 7,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }


def get_ros_observation(args, ros_operator):
    rate = rospy.Rate(args.publish_rate)
    print_flag = True
    time_start = time.time()

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if time.time() - time_start > 0.01:
            print("Get Frame Time is too long", time.time() - time_start, "s")
        if not result:
            if print_flag:
                print("sync fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        img_front, img_right, puppet_arm = result
        return img_front, img_right, puppet_arm


def update_observation_window(args, config, ros_operator):
    def jpeg_mapping(img):
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
        observation_window.append(
            {
                "qpos": None,
                "images": {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                },
            }
        )

    img_front, img_right, puppet_arm = get_ros_observation(args, ros_operator)
    img_front = jpeg_mapping(img_front)
    img_right = jpeg_mapping(img_right)
    qpos = np.array(puppet_arm.position, dtype=float)
    if qpos.shape[0] != config["state_dim"]:
        raise ValueError(f"Expected single-arm qpos dim {config['state_dim']}, got {qpos.shape[0]}")

    observation_window.append(
        {
            "qpos": qpos,
            "images": {
                config["camera_names"][0]: img_front,
                config["camera_names"][1]: img_right,
            },
        }
    )


def build_payload(latest_obs, args, config):
    image_arrs = [
        latest_obs["images"][config["camera_names"][0]],
        latest_obs["images"][config["camera_names"][1]],
    ]
    image_arrs = [
        image_tools.resize_with_pad(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 640, 480)
        for img in image_arrs
    ]
    image_arrs = image_tools.resize_with_pad(np.array(image_arrs), 224, 224)

    if args.payload_format == "libero":
        return {
            "observation/image": image_arrs[0],
            "observation/wrist_image": image_arrs[1],
            "observation/state": latest_obs["qpos"],
            "prompt": args.prompt,
        }

    payload_images = {
        "top_head": image_arrs[0].transpose(2, 0, 1),
        "hand_right": image_arrs[1].transpose(2, 0, 1),
    }

    if args.left_camera_fallback == "black":
        payload_images["hand_left"] = np.zeros_like(payload_images["hand_right"])
    elif args.left_camera_fallback == "copy_right":
        payload_images["hand_left"] = payload_images["hand_right"].copy()

    return {
        "state": latest_obs["qpos"],
        "images": payload_images,
        "prompt": args.prompt,
    }


def inference_fn(args, config, policy):
    global observation_window

    while True and not rospy.is_shutdown():
        payload = build_payload(observation_window[-1], args, config)
        time_start = time.time()
        actions = policy.infer(payload)["actions"]
        print(f"Model inference time: {(time.time() - time_start) * 1000:.3f} ms")
        return np.asarray(actions, dtype=float)


def model_inference(args, config, ros_operator):
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    input("Press enter to continue")

    try:
        update_observation_window(args, config, ros_operator)
        _ = policy.infer(build_payload(observation_window[-1], args, config))
    except Exception as e:
        rospy.logwarn(f"[startup_warmup] {e}")

    pre_action = observation_window[-1]["qpos"].copy()

    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)
            action_buffer = np.zeros((chunk_size, config["state_dim"]), dtype=float)

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                update_observation_window(args, config, ros_operator)

                if t % chunk_size == 0:
                    action_buffer = inference_fn(args, config, policy).copy()
                    if action_buffer.ndim != 2 or action_buffer.shape[0] == 0 or action_buffer.shape[1] != config["state_dim"]:
                        raise ValueError(
                            f"Expected actions with shape [T, {config['state_dim']}], got {action_buffer.shape}"
                        )
                    corrected_action_buffer = action_buffer.copy()

                raw_action = corrected_action_buffer[t % corrected_action_buffer.shape[0]]
                if args.use_actions_interpolation:
                    interp_actions = interpolate_action(args, pre_action, raw_action)
                else:
                    interp_actions = raw_action[np.newaxis, :]

                for act in interp_actions:
                    if args.ctrl_type != "joint":
                        raise ValueError("Only joint ctrl_type is supported in single-arm script")

                    publish_action = act.copy()
                    publish_action[6] = max(0.0, publish_action[6] - args.gripper_offset)
                    ros_operator.puppet_arm_publish(publish_action)
                    try:
                        published_actions_history.append(publish_action.astype(float))
                    except Exception:
                        pass
                    rate.sleep()

                t += 1
                print("Published Step", t)
                pre_action = raw_action.copy()

                if shutdown_event.is_set():
                    break


class RosOperator:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.img_front_deque = deque()
        self.img_right_deque = deque()
        self.puppet_arm_deque = deque()
        self.puppet_arm_publisher = None
        self.init_ros()

    def puppet_arm_publish(self, joints):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [
            "joint0",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        joint_state_msg.position = joints
        self.puppet_arm_publisher.publish(joint_state_msg)

    def get_frame(self):
        if len(self.img_front_deque) == 0 or len(self.img_right_deque) == 0 or len(self.puppet_arm_deque) == 0:
            return False

        frame_time = min(
            [
                self.img_front_deque[-1].header.stamp.to_sec(),
                self.img_right_deque[-1].header.stamp.to_sec(),
                self.puppet_arm_deque[-1].header.stamp.to_sec(),
            ]
        )

        if self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.puppet_arm_deque[-1].header.stamp.to_sec() < frame_time:
            return False

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), "passthrough")

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), "passthrough")

        while self.puppet_arm_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_deque.popleft()
        puppet_arm = self.puppet_arm_deque.popleft()

        return img_front, img_right, puppet_arm

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def puppet_arm_callback(self, msg):
        if len(self.puppet_arm_deque) >= 2000:
            self.puppet_arm_deque.popleft()
        self.puppet_arm_deque.append(msg)

    def init_ros(self):
        rospy.init_node("joint_state_publisher_single", anonymous=True)
        rospy.Subscriber(
            self.args.img_front_topic,
            Image,
            self.img_front_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.img_right_topic,
            Image,
            self.img_right_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.puppet_arm_topic,
            JointState,
            self.puppet_arm_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.puppet_arm_publisher = rospy.Publisher(self.args.puppet_arm_cmd_topic, JointState, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_publish_step", type=int, default=10000, help="Maximum action publishing steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw", help="Front camera topic")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw", help="Right camera topic")
    parser.add_argument(
        "--puppet_arm_cmd_topic",
        type=str,
        default="/master/joint_right",
        help="Single-arm command topic",
    )
    parser.add_argument(
        "--puppet_arm_topic",
        type=str,
        default="/puppet/joint_right",
        help="Single-arm state topic",
    )
    parser.add_argument("--publish_rate", type=int, default=30, help="Action publish rate")
    parser.add_argument("--chunk_size", type=int, default=50, help="Action chunk size")
    parser.add_argument(
        "--arm_steps_length",
        nargs=7,
        type=float,
        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
        help="Maximum change allowed for each joint per interpolation substep",
    )
    parser.add_argument(
        "--use_actions_interpolation",
        action="store_true",
        default=False,
        help="Whether to interpolate large joint jumps",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Websocket server host")
    parser.add_argument("--port", type=int, default=8000, help="Websocket server port")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Language prompt sent to the policy server. Must match the training prompt format.",
    )
    parser.add_argument(
        "--payload_format",
        type=str,
        choices=["libero", "agilex"],
        default="libero",
        help="Inference payload layout expected by the server",
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["joint"],
        default="joint",
        help="Control type for the robot arm",
    )
    parser.add_argument(
        "--left_camera_fallback",
        type=str,
        choices=["none", "black", "copy_right"],
        default="black",
        help="Placeholder strategy when agilex-style backend expects hand_left",
    )
    parser.add_argument(
        "--gripper_offset",
        type=float,
        default=0.003,
        help="Offset subtracted from gripper command before publishing",
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    signal.signal(signal.SIGINT, _on_sigint)
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
