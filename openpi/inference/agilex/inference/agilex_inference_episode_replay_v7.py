#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay one recorded V7 episode through the no-history action policy.

Observations always come from the selected recorded HDF5 files.  With
``--execute`` the predicted actions are sent to the real puppet command topic;
they are never replaced by live camera or qpos feedback.  This is therefore an
open-loop robot replay and must only be used after the robot has been placed in
the corresponding recorded initial pose.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Literal

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

import cv2
import h5py
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.prompts import build_phase_prompt
from tactile_vla.vla.v5_adjustment_data import phase_for_rexecution_frame


DEFAULT_DATA_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs" / "runtime" / "v7_episode_replay"
ACTION_HORIZON = 30
ACTION_DIM = 32
OUTPUT_ACTION_DIM = 7
Phase = Literal["execution", "adjustment"]


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def prepare_rgb(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_tools.resize_with_pad(image.astype(np.uint8), 224, 224)


def phase_for_frame(*, attempt_id: int, frame_index: int, rexecution_frame: int | None) -> Phase:
    # V7 deliberately shares the two-phase interval definition with V5.2:
    # [0, R) adjustment and [R, end) execution.  V7 differs only in that R
    # comes from the native V4 attempt timing rather than an offline detector.
    return phase_for_rexecution_frame(attempt_id, frame_index, rexecution_frame)


def deterministic_action_noise(
    seed: int, *, episode_id: int, attempt_id: int, chunk_index: int, phase: Phase
) -> np.ndarray:
    phase_id = 0 if phase == "execution" else 1
    sequence = np.random.SeedSequence([seed, episode_id, attempt_id, chunk_index, phase_id])
    return np.random.default_rng(sequence).standard_normal((ACTION_HORIZON, ACTION_DIM)).astype(np.float32)


def gripper_command(raw: float, *, offset: float, minimum: float) -> tuple[float, float, bool]:
    after_offset = float(raw) - float(offset)
    published = max(float(minimum), after_offset)
    return after_offset, published, published > after_offset


@dataclass(frozen=True)
class AttemptSpec:
    attempt_id: int
    path: Path
    frame_count: int
    instruction: str
    recovery_plan: str
    rexecution_timestamp: float | None
    rexecution_frame: int | None


@dataclass(frozen=True)
class ReplayChunk:
    attempt_id: int
    frame_index: int
    chunk_index: int
    phase: Phase


def _scalar(handle: h5py.File, name: str) -> Any:
    if name not in handle:
        raise ValueError(f"{handle.filename} lacks required dataset {name!r}")
    return handle[name][()]


def native_reexecution_boundaries(data_root: Path) -> dict[tuple[int, int], int | None]:
    """Read the frame boundaries that were actually used to construct V7."""

    index = _read_json(data_root / "outputs" / "rotation_v4" / "v4_training_index.json")
    timing = index.get("attempt_timing")
    if not isinstance(timing, dict):
        raise ValueError("V4 training index lacks attempt_timing")
    result: dict[tuple[int, int], int | None] = {}
    for key, value in timing.items():
        if not isinstance(value, dict) or not key.startswith("episode") or "/attempt" not in key:
            raise ValueError(f"Invalid V4 attempt_timing row: {key!r}")
        episode_text, attempt_text = key.removeprefix("episode").split("/attempt", 1)
        result[(int(episode_text), int(attempt_text))] = value.get("rexecution_frame_index")
    return result


def load_attempt(
    path: Path,
    *,
    expected_episode: int,
    expected_attempt: int,
    native_reexecution_frame: int | None = None,
) -> AttemptSpec:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        episode_id = int(_scalar(handle, "meta/episode_id"))
        attempt_id = int(_scalar(handle, "meta/attempt_id"))
        if (episode_id, attempt_id) != (expected_episode, expected_attempt):
            raise ValueError(f"{path} metadata is episode{episode_id}/attempt{attempt_id}")
        if not bool(_scalar(handle, "meta/valid")):
            raise ValueError(f"{path} is not valid")
        frame_count = int(_scalar(handle, "size"))
        timestamps = np.asarray(handle["timestamp"], dtype=np.float64)
        qpos = np.asarray(handle["arm/jointStatePosition/puppetRight"], dtype=np.float32)
        if timestamps.shape != (frame_count,) or qpos.shape != (frame_count, OUTPUT_ACTION_DIM):
            raise ValueError(f"Invalid V7 observation shapes in {path}")
        if not np.isfinite(timestamps).all() or not np.isfinite(qpos).all():
            raise ValueError(f"Non-finite timestamp/qpos in {path}")
        if expected_attempt == 2:
            rexecution_timestamp = float(_scalar(handle, "meta/rexecution_timestamp"))
            if not np.isfinite(rexecution_timestamp):
                raise ValueError(f"attempt2 has no finite native rexecution_timestamp: {path}")
            if native_reexecution_frame is None or isinstance(native_reexecution_frame, bool):
                raise ValueError(f"V7 native frame is absent for {path}")
            rexecution_frame = int(native_reexecution_frame)
            if not 0 < rexecution_frame < frame_count:
                raise ValueError(
                    f"native rexecution frame maps outside attempt2: frame={rexecution_frame}, "
                    f"size={frame_count}, path={path}"
                )
            if timestamps[rexecution_frame] < rexecution_timestamp or timestamps[rexecution_frame - 1] >= rexecution_timestamp:
                raise ValueError(
                    f"V4 native frame and HDF5 rexecution_timestamp disagree in {path}: "
                    f"frame={rexecution_frame}, timestamp={rexecution_timestamp}"
                )
        else:
            rexecution_timestamp = None
            rexecution_frame = None
        instruction = _decode(_scalar(handle, "meta/instruction")).strip()
        if not instruction:
            raise ValueError(f"Empty instruction in {path}")
        return AttemptSpec(
            attempt_id=attempt_id,
            path=path,
            frame_count=frame_count,
            instruction=instruction,
            recovery_plan=_decode(_scalar(handle, "meta/input_recovery_plan")).strip(),
            rexecution_timestamp=rexecution_timestamp,
            rexecution_frame=rexecution_frame,
        )


def eligible_episode_ids(
    data_root: Path,
    *,
    minimum: int,
    maximum: int,
    boundaries: dict[tuple[int, int], int | None],
) -> list[int]:
    """List replayable episodes without imposing a train/val/test restriction."""

    result: list[int] = []
    for episode_id in range(minimum, maximum + 1):
        try:
            for attempt_id in (1, 2):
                load_attempt(
                    data_root / "hdf5" / f"episode{episode_id}" / f"attempt{attempt_id}" / "data.hdf5",
                    expected_episode=episode_id,
                    expected_attempt=attempt_id,
                    native_reexecution_frame=boundaries.get((episode_id, attempt_id)),
                )
        except (FileNotFoundError, ValueError):
            continue
        result.append(episode_id)
    return result


def build_chunks(attempts: list[AttemptSpec], *, chunk_size: int) -> list[ReplayChunk]:
    chunks: list[ReplayChunk] = []
    for attempt in attempts:
        # A replay request publishes a complete H30 output.  Do not issue a
        # request for a partial tail which never appeared as a normal H30 start.
        for frame_index in range(0, attempt.frame_count - ACTION_HORIZON + 1, chunk_size):
            chunks.append(
                ReplayChunk(
                    attempt_id=attempt.attempt_id,
                    frame_index=frame_index,
                    chunk_index=len(chunks),
                    phase=phase_for_frame(
                        attempt_id=attempt.attempt_id,
                        frame_index=frame_index,
                        rexecution_frame=attempt.rexecution_frame,
                    ),
                )
            )
    if not chunks:
        raise ValueError("Episode has no complete H30 replay chunks")
    return chunks


class ReplayLogger:
    def __init__(self, root: Path, trial_id: str | None) -> None:
        name = trial_id or datetime.now().astimezone().strftime("episode_replay_%Y%m%d_%H%M%S_%f")
        if not name or Path(name).name != name:
            raise ValueError("--trial-id must be one path component")
        self.directory = root.resolve() / name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "events.jsonl"

    def record(self, event: dict[str, Any]) -> None:
        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Path):
                return str(value)
            return value

        payload = {"time": time.time(), **event}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, default=convert, ensure_ascii=False) + "\n")


class RobotActionPublisher:
    def __init__(self, topic: str) -> None:
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Header

        self.rospy = rospy
        self.JointState = JointState
        self.Header = Header
        rospy.init_node("tactile_vla_v7_episode_replay", anonymous=True)
        self.publisher = rospy.Publisher(topic, JointState, queue_size=10)

    def publish(self, action: np.ndarray) -> float:
        message = self.JointState()
        message.header = self.Header()
        message.header.stamp = self.rospy.Time.now()
        message.name = [f"joint{index}" for index in range(OUTPUT_ACTION_DIM)]
        message.position = np.asarray(action, dtype=float).tolist()
        self.publisher.publish(message)
        return float(message.header.stamp.to_sec())


def _read_image(handle: h5py.File, attempt: AttemptSpec, camera: str, index: int) -> np.ndarray:
    relative = Path(_decode(handle[f"camera/color/{camera}"][index]))
    image = cv2.imread(str(attempt.path.parent / relative), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read recorded {camera} image: {attempt.path.parent / relative}")
    return image


def validate_server_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "data_profile": "rotation_phase_v7_adjustment",
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": "phase_prompt_h30_terminal_hold_native_reexecution",
        "stage_a_protocol": "v7_no_state_history",
        "checkpoint_kind": "stage-a",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"V7 server metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    if bool(metadata.get("use_state_history", True)) or int(metadata.get("state_history_len", -1)) != 0:
        raise ValueError("V7 replay refuses a server that accepts H60 history")
    if tuple(metadata.get("action_noise_shape", ())) != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError("Server does not require V7 action noise [30,32]")


def run_replay(args: argparse.Namespace, attempts: list[AttemptSpec], chunks: list[ReplayChunk], logger: ReplayLogger) -> None:
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    validate_server_metadata(policy.get_server_metadata())
    publisher = RobotActionPublisher(args.puppet_arm_cmd_topic) if args.execute else None
    if args.execute:
        input("Recorded observations will drive real-robot commands. Press Enter to start publishing: ")
    by_attempt = {attempt.attempt_id: attempt for attempt in attempts}
    handles = {attempt.attempt_id: h5py.File(attempt.path, "r") for attempt in attempts}
    try:
        for replay_index, chunk in enumerate(chunks[: args.max_chunks]):
            attempt = by_attempt[chunk.attempt_id]
            handle = handles[chunk.attempt_id]
            qpos = np.asarray(handle["arm/jointStatePosition/puppetRight"][chunk.frame_index], dtype=np.float32)
            prompt = build_phase_prompt(
                phase=chunk.phase,
                instruction=args.instruction or attempt.instruction,
                recovery_plan=attempt.recovery_plan if chunk.phase == "adjustment" else "",
                prompt_profile=PHASE_PROMPT_PROFILE_V2,
            )
            noise = deterministic_action_noise(
                args.noise_seed,
                episode_id=args.episode_id,
                attempt_id=chunk.attempt_id,
                chunk_index=chunk.chunk_index,
                phase=chunk.phase,
            )
            response = policy.infer(
                {
                    "mode": "execution",
                    "observation/image": prepare_rgb(_read_image(handle, attempt, "front", chunk.frame_index)),
                    "observation/wrist_image": prepare_rgb(_read_image(handle, attempt, "left", chunk.frame_index)),
                    "observation/state": qpos,
                    "prompt": prompt,
                    "action_noise": noise,
                    "noise_seed": args.noise_seed,
                    "noise_phase": chunk.phase,
                    "noise_index": chunk.chunk_index,
                }
            )
            actions = np.asarray(response.get("actions"), dtype=np.float32)
            raw_actions = np.asarray(response.get("raw_model_actions"), dtype=np.float32)
            if actions.shape != (ACTION_HORIZON, OUTPUT_ACTION_DIM) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid transformed action output: {actions.shape}")
            if raw_actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(raw_actions).all():
                raise ValueError(f"Invalid raw action output: {raw_actions.shape}")
            print(
                f"[{replay_index + 1}/{min(len(chunks), args.max_chunks)}] episode={args.episode_id} "
                f"attempt={chunk.attempt_id} frame={chunk.frame_index} phase={chunk.phase}",
                flush=True,
            )
            event: dict[str, Any] = {
                "event": "chunk",
                "episode_id": args.episode_id,
                "attempt_id": chunk.attempt_id,
                "frame_index": chunk.frame_index,
                "phase": chunk.phase,
                "prompt": prompt,
                "qpos": qpos,
                "action_noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                "raw_model_actions": raw_actions,
                "actions": actions,
                "server_infer_ms": response.get("policy_timing", {}).get("infer_ms"),
                "execute": args.execute,
            }
            published: list[np.ndarray] = []
            if publisher is not None:
                rate = publisher.rospy.Rate(args.publish_rate)
                for action in actions[: args.chunk_size]:
                    command = action.copy()
                    after_offset, command[6], floor_applied = gripper_command(
                        float(command[6]), offset=args.gripper_offset, minimum=args.gripper_min
                    )
                    publisher.publish(command)
                    published.append(command)
                    event.setdefault("gripper", []).append(
                        {"after_offset": after_offset, "published": float(command[6]), "floor_applied": floor_applied}
                    )
                    rate.sleep()
            event["published_actions"] = published
            logger.record(event)
    finally:
        for handle in handles.values():
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--list-eligible", action="store_true")
    parser.add_argument("--episode-min", type=int, default=30)
    parser.add_argument("--episode-max", type=int, default=210)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--instruction", help="Optional override; default is the recorded instruction.")
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--max-chunks", type=int, default=sys.maxsize)
    parser.add_argument("--execute", action="store_true", help="Publish predicted actions to the real robot.")
    parser.add_argument("--publish-rate", type=float, default=30.0)
    parser.add_argument("--puppet-arm-cmd-topic", default="/master/joint_right")
    parser.add_argument("--gripper-offset", type=float, default=0.001)
    parser.add_argument(
        "--gripper-min",
        type=float,
        help="Required with --execute; floor is applied after --gripper-offset.",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--trial-id")
    args = parser.parse_args()
    if args.list_eligible == (args.episode_id is not None):
        parser.error("choose exactly one of --list-eligible or --episode-id")
    if args.episode_min > args.episode_max:
        parser.error("--episode-min must not exceed --episode-max")
    if args.episode_id is not None and not args.episode_min <= args.episode_id <= args.episode_max:
        parser.error("--episode-id is outside the permitted episode range")
    if args.noise_seed < 0 or not 1 <= args.chunk_size <= ACTION_HORIZON or args.max_chunks <= 0:
        parser.error("invalid noise/chunk arguments")
    if args.publish_rate <= 0 or args.gripper_offset < 0:
        parser.error("invalid publish or gripper arguments")
    if args.execute and args.gripper_min is None:
        parser.error("--execute requires --gripper-min")
    if args.gripper_min is not None and not 0 <= args.gripper_min <= 0.08:
        parser.error("--gripper-min must be in [0, 0.08]")
    return args


def main() -> None:
    args = parse_args()
    boundaries = native_reexecution_boundaries(args.data_root)
    if args.list_eligible:
        print(
            json.dumps(
                eligible_episode_ids(
                    args.data_root,
                    minimum=args.episode_min,
                    maximum=args.episode_max,
                    boundaries=boundaries,
                )
            )
        )
        return
    assert args.episode_id is not None
    eligible = set(
        eligible_episode_ids(
            args.data_root,
            minimum=args.episode_min,
            maximum=args.episode_max,
            boundaries=boundaries,
        )
    )
    if args.episode_id not in eligible:
        raise ValueError(f"episode{args.episode_id} lacks a valid V7 replay pair (attempt1 and attempt2)")
    attempts = [
        load_attempt(
            args.data_root / "hdf5" / f"episode{args.episode_id}" / f"attempt{attempt_id}" / "data.hdf5",
            expected_episode=args.episode_id,
            expected_attempt=attempt_id,
            native_reexecution_frame=boundaries.get((args.episode_id, attempt_id)),
        )
        for attempt_id in (1, 2)
    ]
    chunks = build_chunks(attempts, chunk_size=args.chunk_size)
    logger = ReplayLogger(args.log_dir, args.trial_id)
    logger.record(
        {
            "event": "start",
            "data_root": args.data_root,
            "episode_id": args.episode_id,
            "chunk_count": len(chunks),
            "execute": args.execute,
            "attempt2_rexecution_timestamp": attempts[1].rexecution_timestamp,
            "attempt2_rexecution_frame": attempts[1].rexecution_frame,
        }
    )
    print(f"Replay log: {logger.directory}; chunks={len(chunks)}; execute={args.execute}", flush=True)
    run_replay(args, attempts, chunks, logger)


if __name__ == "__main__":
    main()
