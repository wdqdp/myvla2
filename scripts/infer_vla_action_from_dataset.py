#!/usr/bin/env python3
"""Run one offline H30 action inference from a training-dataset frame."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets")
)
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.environ.setdefault("USE_TF", "0")


DEFAULT_MAX_TIMESTAMP_ERROR = 0.02
DEFAULT_STATE_HISTORY_FPS = 30.0
ACTION_HORIZON = 30
OUTPUT_ACTION_DIM = 7
V7_7_2_DATA_PROFILE = "rotation_phase_v7_7_2_five_task_h100"
ADJUSTMENT_DIRECTIONS = ("left", "right", "front", "back")
ADJUSTMENT_DEGREES = ("slightly", "moderately")
DEFAULT_PIPER_URDF = (
    PROJECT_ROOT
    / "openpi/inference/agilex/Piper_ros_private-ros-noetic/"
    "src/piper_description/urdf/piper_description.urdf"
)


@dataclass(frozen=True)
class SelectedFrame:
    global_index: int
    lerobot_episode_index: int
    episode_id: int
    attempt_id: int
    frame_index: int
    ros_timestamp: float
    timestamp_error_seconds: float
    parquet_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Stage-A or V7.7.2 run directory, numbered step directory, or parameter directory.",
    )
    parser.add_argument(
        "--checkpoint-kind", choices=("auto", "stage-a", "stage-b", "v7-7-2"), default="auto"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="LeRobot dataset root; defaults to dataset_dir recorded in checkpoint config.json.",
    )
    parser.add_argument(
        "--norm-stats-dir",
        type=Path,
        help="Norm-stat directory; defaults to norm_stats_dir recorded in checkpoint config.json.",
    )
    parser.add_argument(
        "--episode",
        "--episode-id",
        dest="episode_id",
        type=int,
        required=True,
        help="Original raw-data episode id.",
    )
    parser.add_argument(
        "--attempt",
        "--attempt-id",
        dest="attempt_id",
        type=int,
        required=True,
        help="Attempt id within the episode.",
    )
    parser.add_argument("--timestamp", type=float, required=True, help="ROS timestamp in seconds.")
    parser.add_argument(
        "--max-timestamp-error",
        type=float,
        default=DEFAULT_MAX_TIMESTAMP_ERROR,
        help="Reject the nearest frame when its absolute timestamp error exceeds this value.",
    )
    parser.add_argument("--mode", choices=("execution", "adjustment"), required=True)
    parser.add_argument("--direction", choices=ADJUSTMENT_DIRECTIONS)
    parser.add_argument(
        "--degree",
        "--magnitude",
        dest="degree",
        choices=ADJUSTMENT_DEGREES,
        help="Adjustment magnitude.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"))
    parser.add_argument(
        "--piper-urdf",
        type=Path,
        default=DEFAULT_PIPER_URDF,
        help="Piper URDF used to convert the first six joint targets to gripper_base poses.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    """Accept a parameter/step path or select a checkpoint from a run."""

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if resolved.name in {"params", "full_params"} or any(
        (resolved / name).is_dir() for name in ("params", "full_params")
    ):
        return resolved
    best_metrics = resolved / "best" / "metrics.json"
    if best_metrics.is_file():
        best_step = json.loads(best_metrics.read_text())["step"]
        best_checkpoint = resolved / str(best_step)
        if not (best_checkpoint / "full_params").is_dir():
            raise FileNotFoundError(f"Best checkpoint has no full_params export: {best_checkpoint}")
        return best_checkpoint
    numbered = sorted(
        (
            child
            for child in resolved.iterdir()
            if child.is_dir() and child.name.isdigit()
            and any((child / name).is_dir() for name in ("params", "full_params"))
        ),
        key=lambda child: int(child.name),
    )
    if not numbered:
        raise FileNotFoundError(
            f"{resolved} is neither a parameter/step directory nor a run containing numbered checkpoints"
        )
    return numbered[-1]


def find_checkpoint_config(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    candidates = [
        checkpoint / "config.json",
        checkpoint.parent / "config.json",
        checkpoint.parent.parent / "config.json",
    ]
    if checkpoint.name in {"params", "full_params"}:
        candidates.insert(0, checkpoint.parent / "config.json")
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            payload = json.loads(candidate.read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"Checkpoint config is not a JSON object: {candidate}")
            return candidate, payload
    raise FileNotFoundError(f"Cannot find config.json near checkpoint {checkpoint}")


def checkpoint_kind(requested_kind: str, config: dict[str, Any]) -> str:
    detected = (
        "v7-7-2" if config.get("data_profile") == V7_7_2_DATA_PROFILE
        else "stage-b" if str(config.get("checkpoint_format", "")).startswith("stage_b_v3_merged_full")
        else "stage-a"
    )
    if requested_kind != "auto" and requested_kind != detected:
        raise ValueError(f"Checkpoint is {detected}, but --checkpoint-kind={requested_kind}")
    return detected


def validate_prompt_arguments(mode: str, direction: str | None, degree: str | None) -> None:
    if mode == "adjustment":
        if direction is None or degree is None:
            raise ValueError("adjustment mode requires both --direction and --degree")
        return
    if direction is not None or degree is not None:
        raise ValueError("--direction/--degree are only valid with --mode adjustment")


def build_cli_prompt(
    *,
    mode: str,
    instruction: str,
    direction: str | None,
    degree: str | None,
    prompt_profile: str,
) -> tuple[str, str]:
    from tactile_vla.vla.prompts import build_phase_prompt
    from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE
    from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
    from tactile_vla.vla.prompts import resolve_prompt_profile
    from tactile_vla.vla.structured_text import recovery_plan_text

    validate_prompt_arguments(mode, direction, degree)
    resolved_profile = resolve_prompt_profile(prompt_profile)
    if resolved_profile not in {PHASE_PROMPT_PROFILE, PHASE_PROMPT_PROFILE_V2}:
        raise ValueError(
            "Mode-controlled action inference requires a phase prompt checkpoint; "
            f"got prompt_profile={resolved_profile!r}"
        )
    recovery_plan = (
        recovery_plan_text(str(direction), str(degree), "none", "moderately")
        if mode == "adjustment"
        else "none"
    )
    prompt = build_phase_prompt(
        phase=mode,
        instruction=instruction,
        recovery_plan=recovery_plan,
        prompt_profile=resolved_profile,
    )
    return prompt, recovery_plan


def locate_nearest_frame(
    dataset_dir: Path,
    *,
    episode_id: int,
    attempt_id: int,
    timestamp: float,
    max_timestamp_error: float,
) -> SelectedFrame:
    """Find the nearest ROS timestamp within one original episode/attempt."""

    if not np.isfinite(timestamp):
        raise ValueError("--timestamp must be finite")
    if max_timestamp_error < 0 or not np.isfinite(max_timestamp_error):
        raise ValueError("--max-timestamp-error must be finite and non-negative")

    # Keep PyArrow use before importing the JAX model stack. Some CUDA/PyArrow
    # combinations are sensitive to importing PyArrow after JAX initialization.
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    columns = (
        "index",
        "episode_index",
        "episode_id",
        "attempt_id",
        "frame_index",
        "ros_timestamp",
    )
    candidates: list[SelectedFrame] = []
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files under {dataset_dir / 'data'}")
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=list(columns))
        mask = pc.and_(
            pc.equal(table["episode_id"], episode_id),
            pc.equal(table["attempt_id"], attempt_id),
        )
        matched = table.filter(mask)
        if matched.num_rows == 0:
            continue
        values = matched.to_pydict()
        for offset in range(matched.num_rows):
            ros_timestamp = float(values["ros_timestamp"][offset])
            candidates.append(
                SelectedFrame(
                    global_index=int(values["index"][offset]),
                    lerobot_episode_index=int(values["episode_index"][offset]),
                    episode_id=int(values["episode_id"][offset]),
                    attempt_id=int(values["attempt_id"][offset]),
                    frame_index=int(values["frame_index"][offset]),
                    ros_timestamp=ros_timestamp,
                    timestamp_error_seconds=abs(ros_timestamp - timestamp),
                    parquet_path=str(parquet_path.resolve()),
                )
            )
    if not candidates:
        raise ValueError(
            f"No frames found for episode_id={episode_id}, attempt_id={attempt_id}"
        )
    # Prefer the older sample on an exact midpoint, matching runtime history
    # sampling behavior.
    selected = min(
        candidates,
        key=lambda row: (
            row.timestamp_error_seconds,
            row.ros_timestamp > timestamp,
            row.ros_timestamp,
        ),
    )
    if selected.timestamp_error_seconds > max_timestamp_error:
        raise ValueError(
            "Nearest frame exceeds --max-timestamp-error: "
            f"requested={timestamp:.9f}, nearest={selected.ros_timestamp:.9f}, "
            f"error={selected.timestamp_error_seconds:.9f}s, limit={max_timestamp_error:.9f}s"
        )
    return selected


def _scalar(value: Any) -> int | float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar metadata, got shape {array.shape}")
    return array.reshape(()).item()


def load_lerobot_observation(
    dataset_dir: Path,
    selected: SelectedFrame,
    *,
    use_state_history: bool,
    state_history_len: int,
    state_history_dim: int,
    state_history_fps: float,
    video_backend: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load both training images and build slave-arm history at 30 Hz."""

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if state_history_fps <= 0:
        raise ValueError(f"state_history_fps must be positive, got {state_history_fps}")
    if use_state_history and state_history_len <= 0:
        raise ValueError("History-enabled checkpoint must have state_history_len > 0")
    delta_timestamps: dict[str, list[float]] = {}
    if use_state_history:
        delta_timestamps["observation.state"] = [
            step / state_history_fps for step in range(-(state_history_len - 1), 1)
        ]
    dataset = LeRobotDataset(
        "tactile_vla",
        root=dataset_dir,
        delta_timestamps=delta_timestamps or None,
        download_videos=False,
        video_backend=video_backend,
    )
    item = dataset[selected.global_index]
    identity = {
        "global_index": int(_scalar(item["index"])),
        "lerobot_episode_index": int(_scalar(item["episode_index"])),
        "episode_id": int(_scalar(item["episode_id"])),
        "attempt_id": int(_scalar(item["attempt_id"])),
        "frame_index": int(_scalar(item["frame_index"])),
        # LeRobot converts numeric columns to Torch tensors and may cast this
        # epoch-sized float64 value to float32. Keep the Parquet value used for
        # selection as the authoritative timestamp.
        "ros_timestamp": selected.ros_timestamp,
    }
    expected_identity = {
        "global_index": selected.global_index,
        "lerobot_episode_index": selected.lerobot_episode_index,
        "episode_id": selected.episode_id,
        "attempt_id": selected.attempt_id,
        "frame_index": selected.frame_index,
    }
    for key, expected in expected_identity.items():
        actual = identity[key]
        if actual != expected:
            raise ValueError(f"LeRobot identity mismatch for {key}: {actual} != {expected}")

    state_values = np.asarray(item["observation.state"], dtype=np.float32)
    request: dict[str, Any] = {
        "observation/image": item["observation.images.front"],
        "observation/wrist_image": item["observation.images.left"],
        "observation/state": state_values[-1] if use_state_history else state_values,
    }
    history_summary: dict[str, Any] = {
        "used_by_model": use_state_history,
        "source": "observation.state (puppetRight/slave arm)",
        "fps": state_history_fps,
        "shape": None,
        "valid_count": 0,
        "padded_count": 0,
    }
    if use_state_history:
        history = state_values
        history_is_pad = np.asarray(item["observation.state_is_pad"], dtype=np.bool_)
        expected_history_shape = (state_history_len, state_history_dim)
        if history.shape != expected_history_shape:
            raise ValueError(
                f"Expected slave-arm history {expected_history_shape}, got {history.shape}"
            )
        if history_is_pad.shape != (state_history_len,):
            raise ValueError(
                f"Expected history padding mask {(state_history_len,)}, got {history_is_pad.shape}"
            )
        request["observation/state_history"] = history
        request["observation/state_history_mask"] = np.logical_not(history_is_pad)
        history_summary.update(
            {
                "shape": list(history.shape),
                "valid_count": int(np.count_nonzero(~history_is_pad)),
                "padded_count": int(np.count_nonzero(history_is_pad)),
            }
        )
    return request, {"identity": identity, "history": history_summary, "instruction": str(item["instruction"])}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def piper_fk_transforms(joint_positions: Any, chain: Any) -> np.ndarray:
    """Return base_link-to-gripper_base transforms for Piper arm qpos."""

    qpos = np.asarray(joint_positions, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < int(chain.revolute_joint_count):
        raise ValueError(
            "Piper joint positions must have shape "
            f"[T,>={chain.revolute_joint_count}], got {qpos.shape}"
        )
    if not np.isfinite(qpos).all():
        raise ValueError("Piper joint positions contain non-finite values")

    transforms = np.empty((len(qpos), 4, 4), dtype=np.float64)
    for frame_index, row in enumerate(qpos):
        transform = np.eye(4, dtype=np.float64)
        arm_index = 0
        for joint in chain.joints:
            transform = transform @ np.asarray(joint.origin, dtype=np.float64)
            if joint.axis is None:
                continue
            axis = np.asarray(joint.axis, dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
                raise ValueError(f"URDF joint {joint.name!r} has a zero rotation axis")
            x, y, z = axis / norm
            angle = float(row[arm_index])
            c, s = np.cos(angle), np.sin(angle)
            one = 1.0 - c
            rotation = np.array(
                [
                    [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
                    [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
                    [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
                ],
                dtype=np.float64,
            )
            joint_transform = np.eye(4, dtype=np.float64)
            joint_transform[:3, :3] = rotation
            transform = transform @ joint_transform
            arm_index += 1
        transforms[frame_index] = transform
    return transforms


def rotation_matrix_to_zyx_degrees(rotation: Any) -> dict[str, float]:
    """Decompose R = Rz(yaw) @ Ry(pitch) @ Rx(roll), returning degrees."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Rotation matrix must be finite [3,3], got {matrix.shape}")
    horizontal = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    pitch = float(np.arctan2(-matrix[2, 0], horizontal))
    if horizontal > 1e-9:
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
    else:
        # At the ZYX gimbal lock, set roll to zero and keep the observable yaw.
        yaw = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
        roll = 0.0
    yaw_deg, pitch_deg, roll_deg = np.rad2deg([yaw, pitch, roll])
    return {
        "z_yaw": _rounded(float(yaw_deg), digits=4),
        "y_pitch": _rounded(float(pitch_deg), digits=4),
        "x_roll": _rounded(float(roll_deg), digits=4),
    }


def _rounded(value: float, *, digits: int) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded


def _xyz_mapping(
    values: Any,
    *,
    scale: float = 1.0,
    digits: int = 6,
) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64).reshape(3) * scale
    return {
        "x": _rounded(vector[0], digits=digits),
        "y": _rounded(vector[1], digits=digits),
        "z": _rounded(vector[2], digits=digits),
    }


def _relative_pose_change(reference: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    # Translation is expressed along base_link axes. R_target @ R_reference.T
    # describes the target orientation change about those same fixed axes.
    relative_rotation = target[:3, :3] @ reference[:3, :3].T
    return {
        "translation_mm": _xyz_mapping(
            target[:3, 3] - reference[:3, 3], scale=1000.0, digits=3
        ),
        "rotation_zyx_deg": rotation_matrix_to_zyx_degrees(relative_rotation),
    }


def describe_h30_cartesian_changes(
    actions: Any,
    *,
    current_state: Any,
    chain: Any,
) -> dict[str, Any]:
    """Convert H30 absolute joint targets into human-readable Cartesian changes."""

    action_array = np.asarray(actions, dtype=np.float64)
    current = np.asarray(current_state, dtype=np.float64)
    if action_array.shape != (ACTION_HORIZON, OUTPUT_ACTION_DIM):
        raise ValueError(f"Expected H30 joint targets [30,7], got {action_array.shape}")
    if current.shape != (OUTPUT_ACTION_DIM,):
        raise ValueError(f"Expected current slave-arm state [7], got {current.shape}")
    if not np.isfinite(action_array).all() or not np.isfinite(current).all():
        raise ValueError("Joint targets/current state contain non-finite values")

    current_transform = piper_fk_transforms(current[None, :6], chain)[0]
    target_transforms = piper_fk_transforms(action_array[:, :6], chain)
    frames: list[dict[str, Any]] = []
    previous_transform = current_transform
    previous_gripper = float(current[6])
    for step, (target, target_joints) in enumerate(zip(target_transforms, action_array, strict=True)):
        frames.append(
            {
                "step": step,
                "time_offset_seconds": _rounded(
                    step / DEFAULT_STATE_HISTORY_FPS, digits=6
                ),
                "target_pose_in_base": {
                    "position_m": _xyz_mapping(target[:3, 3]),
                    "orientation_zyx_deg": rotation_matrix_to_zyx_degrees(target[:3, :3]),
                },
                "change_from_current": _relative_pose_change(current_transform, target),
                "change_from_previous": _relative_pose_change(previous_transform, target),
                "gripper": {
                    "target": _rounded(target_joints[6], digits=6),
                    "change_from_current": _rounded(
                        target_joints[6] - current[6], digits=6
                    ),
                    "change_from_previous": _rounded(
                        target_joints[6] - previous_gripper, digits=6
                    ),
                },
            }
        )
        previous_transform = target
        previous_gripper = float(target_joints[6])

    return {
        "coordinate_frame": f"{chain.base_link} fixed axes",
        "end_link": chain.end_link,
        "euler_convention": "ZYX: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        "translation_unit": "millimeter",
        "rotation_unit": "degree",
        "current_pose_in_base": {
            "position_m": _xyz_mapping(current_transform[:3, 3]),
            "orientation_zyx_deg": rotation_matrix_to_zyx_degrees(
                current_transform[:3, :3]
            ),
            "gripper": _rounded(current[6], digits=6),
        },
        "endpoint_change_from_current": {
            **_relative_pose_change(current_transform, target_transforms[-1]),
            "gripper": _rounded(action_array[-1, 6] - current[6], digits=6),
        },
        "frames": frames,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    validate_prompt_arguments(args.mode, args.direction, args.degree)
    checkpoint = resolve_checkpoint(args.checkpoint)
    config_path, config = find_checkpoint_config(checkpoint)
    kind = checkpoint_kind(args.checkpoint_kind, config)
    dataset_dir = (args.dataset_dir or Path(str(config.get("dataset_dir", "")))).expanduser().resolve()
    norm_stats_dir = (
        args.norm_stats_dir or Path(str(config.get("norm_stats_dir", "")))
    ).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not norm_stats_dir.is_dir():
        raise FileNotFoundError(f"Norm-stat directory not found: {norm_stats_dir}")

    selected = locate_nearest_frame(
        dataset_dir,
        episode_id=args.episode_id,
        attempt_id=args.attempt_id,
        timestamp=args.timestamp,
        max_timestamp_error=args.max_timestamp_error,
    )
    use_state_history = bool(config.get("use_state_history", False))
    state_history_len = int(config.get("state_history_len", 0))
    state_history_dim = int(config.get("state_history_dim", OUTPUT_ACTION_DIM))
    state_history_fps = float(config.get("state_history_fps", DEFAULT_STATE_HISTORY_FPS))
    request, observation_summary = load_lerobot_observation(
        dataset_dir,
        selected,
        use_state_history=use_state_history,
        state_history_len=state_history_len,
        state_history_dim=state_history_dim,
        state_history_fps=state_history_fps,
        video_backend=args.video_backend,
    )
    prompt, recovery_plan = build_cli_prompt(
        mode=args.mode,
        instruction=observation_summary["instruction"],
        direction=args.direction,
        degree=args.degree,
        # The V7.7.2 action stream was trained with the Stage-A phase_v2 prompt.
        prompt_profile=(
            "phase_v2" if kind == "v7-7-2" else str(config.get("prompt_profile", ""))
        ),
    )
    request["mode"] = "execution"
    request["prompt"] = prompt
    current_state = np.asarray(request["observation/state"], dtype=np.float32).copy()

    # Import the model-serving implementation only after the PyArrow frame
    # lookup, preserving the repository's CUDA/PyArrow import-order rule.
    from openpi.shared import normalize
    if kind == "v7-7-2":
        import serve_tactile_vla_v7_7 as multitask_server

        multitask_server.DATA_PROFILE = V7_7_2_DATA_PROFILE
        full_params = checkpoint if checkpoint.name == "full_params" else checkpoint / "full_params"
        if not full_params.is_dir():
            raise FileNotFoundError(f"V7.7.2 full_params export not found: {full_params}")
        policy_args = argparse.Namespace(
            checkpoint=full_params,
            norm_stats_dir=norm_stats_dir,
            num_inference_steps=args.num_inference_steps,
            precision=args.precision,
            output_action_dim=OUTPUT_ACTION_DIM,
            need_recovery_threshold=None,
            adjustment_end_threshold=None,
            reasoning_max_token_len=None,
            no_norm=False,
        )
        model_config = multitask_server.v3._model_config(policy_args, config)
        policy = multitask_server.V77Policy(
            args=policy_args,
            config=config,
            model_config=model_config,
            norm_stats=normalize.load(norm_stats_dir),
        )
    else:
        import serve_tactile_vla_action_ablation as action_server

        policy_args = argparse.Namespace(
            checkpoint_kind=kind,
            checkpoint=checkpoint,
            norm_stats_dir=norm_stats_dir,
            expected_data_profile=config.get("data_profile"),
            num_inference_steps=args.num_inference_steps,
            precision=args.precision,
        )
        action_server.validate_v4_norm_artifacts(policy_args, config)
        model_config = action_server._model_config(policy_args, config)
        policy = action_server.ActionOnlyAblationPolicy(
            args=policy_args,
            config_path=config_path,
            config=config,
            model_config=model_config,
            norm_stats=normalize.load(norm_stats_dir),
        )
    noise = np.random.default_rng(args.noise_seed).standard_normal(
        (model_config.action_horizon, model_config.action_dim), dtype=np.float32
    )
    request["action_noise"] = noise
    response = next(policy.infer_events(request)) if kind == "v7-7-2" else policy.infer(request)
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, OUTPUT_ACTION_DIM):
        raise ValueError(f"Expected output H30 action [30,7], got {actions.shape}")
    from tactile_vla.vla.v5_adjustment_data import load_piper_fk_chain

    fk_chain = load_piper_fk_chain(args.piper_urdf.expanduser().resolve())
    cartesian_changes = describe_h30_cartesian_changes(
        actions,
        current_state=current_state,
        chain=fk_chain,
    )

    output = {
        "checkpoint": str(checkpoint),
        "config_path": str(config_path),
        "data_profile": config.get("data_profile"),
        "prompt_profile": config.get("prompt_profile"),
        "selection": {
            "requested_timestamp": args.timestamp,
            **asdict(selected),
        },
        "mode": args.mode,
        "direction": args.direction,
        "degree": args.degree,
        "recovery_plan": recovery_plan,
        "prompt": prompt,
        "state_history": observation_summary["history"],
        "noise_seed": args.noise_seed,
        "action_representation": "absolute puppetRight joint-position targets after delta decoding",
        "h30_action_shape": list(actions.shape),
        "h30_action": actions,
        "h30_cartesian_changes": {
            "piper_urdf": str(fk_chain.source_path),
            "piper_urdf_sha256": fk_chain.source_sha256,
            **cartesian_changes,
        },
        "inference_timing": response.get("policy_timing", {}),
    }
    rendered = json.dumps(_json_ready(output), indent=2, ensure_ascii=False)
    if args.output_json is not None:
        destination = args.output_json.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        logging.info("Wrote H30 action result to %s", destination)
    print(rendered)


if __name__ == "__main__":
    main()
