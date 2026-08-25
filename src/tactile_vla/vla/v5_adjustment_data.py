"""V5.2 two-phase labels and boundary-isolated H30 targets on V4 data."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2, V2_ACTION_PHASES
from tactile_vla.vla.v4_data import SPLITS, V4Frame, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.v5_phase_data import ALLOWED_TASKS
from tactile_vla.vla.v5_phase_data import EXPECTED_ATTEMPT_COUNT, EXPECTED_EPISODE_COUNT
from tactile_vla.vla.v5_phase_data import PhaseBoundaryError
from tactile_vla.vla.v5_phase_data import _selected_stream_hash
from tactile_vla.vla.v5_phase_data import _true_runs
from tactile_vla.vla.v5_phase_data import detect_phase_boundaries
from tactile_vla.vla.v5_targets import apply_h30_terminal_hold


ROTATION_PHASE_V5_ADJUSTMENT_V2 = "rotation_phase_v5_adjustment_v2"
V2_EXPERIMENT_KIND = "phase_prompt_h30_terminal_hold"
V2_PHASE_BOUNDARY_SCHEMA = "tactile_vla_v5_adjustment_boundaries_v2"
V2_ACTION_PHASE_SCHEMA = "tactile_vla_v5_adjustment_action_manifest_v2"
V2_OVERRIDE_SCHEMA = "tactile_vla_v5_phase_boundary_overrides_v2"
V2_TRAINING_INDEX_SCHEMA = "tactile_vla_v5_adjustment_training_index_v2"
V2_SUMMARY_SCHEMA = "tactile_vla_v5_adjustment_summary_v2"
V2_TERMINAL_HOLD_SCHEMA = "tactile_vla_v5_h30_terminal_hold_v1"
V2_H30_CONTENT_SCHEMA = "tactile_vla_h30_float32_content_v1"
V2_EXPECTED_MODIFIED_CHUNKS = 11_832
V2_EXPECTED_MODIFIED_ACTION_STEPS = 177_480
V2Phase = Literal["execution", "adjustment"]

DEFAULT_PIPER_URDF = (
    Path(__file__).resolve().parents[3]
    / "openpi/inference/agilex/Piper_ros_private-ros-noetic/"
    "src/piper_description/urdf/piper_description.urdf"
)

_RAW_DATASETS = {
    "puppet_right": ("arm/jointStatePosition/puppetRight", "puppetRight"),
    "master_right": ("arm/jointStatePosition/masterRight", "masterRight"),
    "fz_left": ("tactile/force_resultant/left", "force_resultant/left"),
    "fz_right": ("tactile/force_resultant/right", "force_resultant/right"),
    "timestamp": ("timestamp",),
}


@dataclass(frozen=True)
class V2RawAttemptStreams:
    puppet_right: np.ndarray
    master_gripper: np.ndarray
    fz_left: np.ndarray
    fz_right: np.ndarray
    timestamp: np.ndarray
    dataset_names: dict[str, str]
    source_sha256: str

    @property
    def puppet_gripper(self) -> np.ndarray:
        return self.puppet_right[:, 6]


@dataclass(frozen=True)
class FKJoint:
    name: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray | None


@dataclass(frozen=True)
class PiperFKChain:
    source_path: Path
    source_sha256: str
    base_link: str
    end_link: str
    joints: tuple[FKJoint, ...]
    revolute_joint_count: int


@dataclass(frozen=True)
class DetectedRexecution:
    rexecution_frame: int
    rexecution_timestamp: float
    horizontal_endpoint_frame: int
    boundary_trigger: str
    horizontal_start_xy: tuple[float, float]
    horizontal_target_xy: tuple[float, float]
    horizontal_displacement_m: float
    endpoint_tolerance_m: float
    endpoint_stable_frames: int
    remaining_stability_fraction: float
    lift_height_m: float
    close_start_frame: int
    smoothing_frames: int


def _require_dataset(handle: h5py.File, logical_name: str) -> tuple[str, h5py.Dataset]:
    for name in _RAW_DATASETS[logical_name]:
        if name in handle:
            dataset = handle[name]
            if isinstance(dataset, h5py.Dataset):
                return name, dataset
            raise PhaseBoundaryError(f"Raw HDF5 object {name!r} is not a dataset")
    raise PhaseBoundaryError(
        f"Raw HDF5 lacks required {logical_name} dataset; tried {_RAW_DATASETS[logical_name]}"
    )


def load_v2_raw_attempt_streams(path: Path) -> V2RawAttemptStreams:
    """Load and hash every raw value used by V5.2 labeling."""

    if not path.is_file():
        raise FileNotFoundError(path)
    loaded: dict[str, tuple[str, np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        for logical_name in _RAW_DATASETS:
            dataset_name, dataset = _require_dataset(handle, logical_name)
            loaded[logical_name] = (dataset_name, np.asarray(dataset))

    puppet = loaded["puppet_right"][1]
    master = loaded["master_right"][1]
    left = loaded["fz_left"][1]
    right = loaded["fz_right"][1]
    timestamp = loaded["timestamp"][1]
    if puppet.ndim != 2 or puppet.shape[1] < 7:
        raise PhaseBoundaryError(f"puppetRight must have shape [T,>=7], got {puppet.shape}")
    if master.ndim != 2 or master.shape[1] < 7:
        raise PhaseBoundaryError(f"masterRight must have shape [T,>=7], got {master.shape}")
    if left.ndim != 2 or left.shape[1] < 3:
        raise PhaseBoundaryError(f"left force_resultant must have shape [T,>=3], got {left.shape}")
    if right.ndim != 2 or right.shape[1] < 3:
        raise PhaseBoundaryError(f"right force_resultant must have shape [T,>=3], got {right.shape}")
    if timestamp.ndim != 1:
        raise PhaseBoundaryError(f"timestamp must have shape [T], got {timestamp.shape}")
    lengths = {len(puppet), len(master), len(left), len(right), len(timestamp)}
    if len(lengths) != 1:
        raise PhaseBoundaryError(f"Raw V5.2 label streams have different lengths: {lengths}")

    selected = {
        "puppet_right": (loaded["puppet_right"][0], np.asarray(puppet[:, :7])),
        "master_gripper": (loaded["master_right"][0], np.asarray(master[:, 6])),
        "fz_left": (loaded["fz_left"][0], np.asarray(left[:, 2])),
        "fz_right": (loaded["fz_right"][0], np.asarray(right[:, 2])),
        "timestamp": (loaded["timestamp"][0], np.asarray(timestamp)),
    }
    arrays = {name: np.asarray(value[1], dtype=np.float64) for name, value in selected.items()}
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise PhaseBoundaryError(f"Raw V5.2 label stream {name} contains non-finite values")
    return V2RawAttemptStreams(
        puppet_right=arrays["puppet_right"],
        master_gripper=arrays["master_gripper"],
        fz_left=arrays["fz_left"],
        fz_right=arrays["fz_right"],
        timestamp=arrays["timestamp"],
        dataset_names={name: value[0] for name, value in selected.items()},
        source_sha256=_selected_stream_hash(selected),
    )


def _parse_vector(value: str | None, *, default: Sequence[float]) -> np.ndarray:
    values = list(default) if value is None else [float(part) for part in value.split()]
    if len(values) != 3 or not np.isfinite(values).all():
        raise PhaseBoundaryError(f"URDF vector must contain three finite values, got {value!r}")
    return np.asarray(values, dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise PhaseBoundaryError("URDF revolute joint has a zero axis")
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def load_piper_fk_chain(
    path: Path,
    *,
    base_link: str = "base_link",
    end_link: str = "gripper_base",
) -> PiperFKChain:
    """Parse the checked-in Piper URDF without adding a Pinocchio dependency."""

    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    by_child: dict[str, ET.Element] = {}
    for joint in root.findall("joint"):
        child_node = joint.find("child")
        if child_node is None or "link" not in child_node.attrib:
            raise PhaseBoundaryError("URDF joint lacks child link")
        child = child_node.attrib["link"]
        if child in by_child:
            raise PhaseBoundaryError(f"URDF has multiple parent joints for link {child!r}")
        by_child[child] = joint

    chain_nodes: list[ET.Element] = []
    cursor = end_link
    while cursor != base_link:
        joint = by_child.get(cursor)
        if joint is None:
            raise PhaseBoundaryError(f"URDF has no chain from {base_link!r} to {end_link!r}")
        chain_nodes.append(joint)
        parent_node = joint.find("parent")
        if parent_node is None or "link" not in parent_node.attrib:
            raise PhaseBoundaryError("URDF joint lacks parent link")
        cursor = parent_node.attrib["link"]
    chain_nodes.reverse()

    joints: list[FKJoint] = []
    revolute_count = 0
    for node in chain_nodes:
        joint_type = node.attrib.get("type", "")
        if joint_type not in {"fixed", "revolute", "continuous"}:
            raise PhaseBoundaryError(f"Unsupported Piper FK joint type {joint_type!r}")
        origin_node = node.find("origin")
        xyz = _parse_vector(
            origin_node.attrib.get("xyz") if origin_node is not None else None,
            default=(0.0, 0.0, 0.0),
        )
        rpy = _parse_vector(
            origin_node.attrib.get("rpy") if origin_node is not None else None,
            default=(0.0, 0.0, 0.0),
        )
        axis: np.ndarray | None = None
        if joint_type in {"revolute", "continuous"}:
            axis_node = node.find("axis")
            axis = _parse_vector(
                axis_node.attrib.get("xyz") if axis_node is not None else None,
                default=(1.0, 0.0, 0.0),
            )
            revolute_count += 1
        joints.append(
            FKJoint(
                name=node.attrib.get("name", "unnamed"),
                joint_type=joint_type,
                origin=_origin_matrix(xyz, rpy),
                axis=axis,
            )
        )
    if revolute_count != 6:
        raise PhaseBoundaryError(f"Piper FK chain must have six arm joints, got {revolute_count}")
    return PiperFKChain(
        source_path=path.resolve(),
        source_sha256=sha256_file(path),
        base_link=base_link,
        end_link=end_link,
        joints=tuple(joints),
        revolute_joint_count=revolute_count,
    )


def piper_fk_positions(joint_positions: Sequence[Sequence[float]], chain: PiperFKChain) -> np.ndarray:
    qpos = np.asarray(joint_positions, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < chain.revolute_joint_count:
        raise PhaseBoundaryError(
            f"Piper joint positions must have shape [T,>=6], got {qpos.shape}"
        )
    if not np.isfinite(qpos).all():
        raise PhaseBoundaryError("Piper joint positions contain non-finite values")
    output = np.empty((len(qpos), 3), dtype=np.float64)
    for frame_index, row in enumerate(qpos):
        transform = np.eye(4, dtype=np.float64)
        arm_index = 0
        for joint in chain.joints:
            transform = transform @ joint.origin
            if joint.axis is not None:
                transform = transform @ _axis_rotation(joint.axis, float(row[arm_index]))
                arm_index += 1
        output[frame_index] = transform[:3, 3]
    return output


def _moving_average(values: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0 or frames % 2 != 1:
        raise ValueError("Smoothing frame count must be a positive odd integer")
    pad = frames // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0)]
    )
    return (cumulative[frames:] - cumulative[:-frames]) / frames


def detect_rexecution_frame(
    *,
    puppet_right: Sequence[Sequence[float]],
    timestamp: Sequence[float],
    release_frame: int,
    regrasp_frame: int,
    fk_chain: PiperFKChain,
    smoothing_frames: int = 9,
    endpoint_stable_frames: int = 10,
) -> DetectedRexecution:
    """Detect the final stable horizontal target entry before re-execution."""

    qpos = np.asarray(puppet_right, dtype=np.float64)
    timestamps = np.asarray(timestamp, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 7 or timestamps.shape != (len(qpos),):
        raise PhaseBoundaryError("Rexecution detector requires aligned puppetRight[T,>=7] and timestamp[T]")
    if not 0 <= release_frame < regrasp_frame < len(qpos):
        raise PhaseBoundaryError(
            f"Rexecution search has invalid contact window {release_frame}..{regrasp_frame}"
        )
    if regrasp_frame - release_frame < endpoint_stable_frames + 20:
        raise PhaseBoundaryError("Release/regrasp window is too short for rexecution detection")

    positions = _moving_average(piper_fk_positions(qpos[:, :6], fk_chain), smoothing_frames)
    start_xy = np.median(
        positions[max(0, release_frame - 10) : release_frame + 11, :2], axis=0
    )
    target_xy = np.median(
        positions[max(release_frame, regrasp_frame - 25) : regrasp_frame + 1, :2],
        axis=0,
    )
    displacement = float(np.linalg.norm(target_xy - start_xy))
    if displacement < 0.010:
        raise PhaseBoundaryError(
            f"Horizontal recovery displacement {displacement:.6f} m is below 0.010 m"
        )
    tolerance = min(0.012, max(0.004, 0.10 * displacement))
    distance = np.linalg.norm(positions[:, :2] - target_xy, axis=1)

    candidate: int | None = None
    stability_fraction = 0.0
    search_end = regrasp_frame - endpoint_stable_frames
    for frame in range(release_frame + 10, search_end + 1):
        if float(np.max(distance[frame : frame + endpoint_stable_frames])) > tolerance:
            continue
        fraction = float(np.mean(distance[frame : regrasp_frame + 1] <= 2.0 * tolerance))
        if fraction >= 0.90:
            candidate = frame
            stability_fraction = fraction
            break
    if candidate is None:
        raise PhaseBoundaryError("No stable horizontal target entry before regrasp")

    gripper = qpos[:, 6]
    p5, p95 = np.percentile(gripper, (5.0, 95.0))
    span = float(p95 - p5)
    if span <= np.finfo(np.float64).eps * max(abs(float(p5)), abs(float(p95)), 1.0) * 100:
        raise PhaseBoundaryError("Puppet gripper cannot define relative closing progress")
    progress = np.clip((gripper - p5) / span, 0.0, 1.0)
    peak = release_frame + int(np.argmax(progress[release_frame : regrasp_frame + 1]))
    close_runs = [
        run
        for run in _true_runs(progress[peak : regrasp_frame + 1] < 0.85)
        if run[1] - run[0] + 1 >= 5
    ]
    close_start = peak + close_runs[0][0] if close_runs else regrasp_frame
    rexecution_frame = candidate
    boundary_trigger = "horizontal_endpoint_stable"
    if close_start < candidate:
        horizontal_progress = 1.0 - float(distance[close_start] / displacement)
        if distance[close_start] > 2.0 * tolerance or horizontal_progress < 0.80:
            raise PhaseBoundaryError(
                f"Closing starts at {close_start} before a safe horizontal endpoint at {candidate}; "
                f"distance={distance[close_start]:.6f}, progress={horizontal_progress:.3f}"
            )
        # Safety takes precedence over the last few millimeters of horizontal
        # convergence: the first real closing frame must belong to execution.
        rexecution_frame = close_start
        boundary_trigger = "closing_preempted_endpoint"
    release_z = float(positions[release_frame, 2])
    lift_height = float(
        np.max(positions[release_frame : rexecution_frame + 1, 2]) - release_z
    )
    if lift_height < 0.020:
        raise PhaseBoundaryError(f"Post-release lift {lift_height:.6f} m is below 0.020 m")
    if rexecution_frame >= regrasp_frame:
        raise PhaseBoundaryError("Rexecution frame must precede confirmed regrasp")

    return DetectedRexecution(
        rexecution_frame=rexecution_frame,
        rexecution_timestamp=float(timestamps[rexecution_frame]),
        horizontal_endpoint_frame=candidate,
        boundary_trigger=boundary_trigger,
        horizontal_start_xy=(float(start_xy[0]), float(start_xy[1])),
        horizontal_target_xy=(float(target_xy[0]), float(target_xy[1])),
        horizontal_displacement_m=displacement,
        endpoint_tolerance_m=float(tolerance),
        endpoint_stable_frames=endpoint_stable_frames,
        remaining_stability_fraction=stability_fraction,
        lift_height_m=lift_height,
        close_start_frame=int(close_start),
        smoothing_frames=smoothing_frames,
    )


def load_v2_phase_overrides(path: Path) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != V2_OVERRIDE_SCHEMA:
        raise ValueError("Unsupported V5.2 phase-boundary override schema")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 2:
        raise ValueError("V5.2 phase-boundary overrides require version=2")
    rows = payload.get("overrides")
    if not isinstance(rows, list):
        raise ValueError("V5.2 phase-boundary overrides must contain an overrides list")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    allowed_boundary_fields = ("release_frame", "regrasp_frame", "rexecution_frame")
    for row_number, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"Override row {row_number} is not an object")
        key = (int(raw["episode_id"]), int(raw["attempt_id"]))
        if key[1] != 2:
            raise ValueError(f"Only attempt2 may have V5.2 overrides, got {key}")
        if key in result:
            raise ValueError(f"Duplicate V5.2 phase-boundary override for {key}")
        present = [name for name in allowed_boundary_fields if name in raw]
        if not present:
            raise ValueError(f"Override {key} contains no boundary field")
        if ("release_frame" in raw) != ("regrasp_frame" in raw):
            raise ValueError(f"Override {key} must provide release_frame and regrasp_frame together")
        for name in present:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Override {key} {name} must be a frame integer or timestamp")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Override {key} {name} must be finite and non-negative")
        result[key] = dict(raw)
    return payload, result


def resolve_v2_phase_override(
    override: Mapping[str, Any],
    timestamps: Sequence[float],
    *,
    attempt_key: tuple[int, int],
) -> tuple[dict[str, int], dict[str, Any]]:
    attempt_timestamps = np.asarray(timestamps, dtype=np.float64)
    if attempt_timestamps.ndim != 1 or len(attempt_timestamps) == 0:
        raise ValueError(f"Override {attempt_key} cannot resolve against empty timestamps")
    if not np.isfinite(attempt_timestamps).all():
        raise ValueError(f"Override {attempt_key} cannot resolve against non-finite timestamps")

    resolved: dict[str, int] = {}
    audit: dict[str, Any] = {}
    for name in ("release_frame", "regrasp_frame", "rexecution_frame"):
        if name not in override:
            continue
        value = override[name]
        if isinstance(value, int) and 0 <= value < len(attempt_timestamps):
            frame = value
            audit[f"{name}_input_kind"] = "attempt_local_frame"
        else:
            requested = float(value)
            first = float(np.min(attempt_timestamps))
            last = float(np.max(attempt_timestamps))
            if not first <= requested <= last:
                raise ValueError(
                    f"Override {attempt_key} {name} timestamp {requested:.9f} "
                    f"is outside attempt range [{first:.9f}, {last:.9f}]"
                )
            frame = int(np.argmin(np.abs(attempt_timestamps - requested)))
            audit[f"{name}_input_kind"] = "approximate_timestamp"
            audit[f"{name}_timestamp_error_seconds"] = abs(
                float(attempt_timestamps[frame]) - requested
            )
        resolved[name] = frame
        audit[f"resolved_{name}"] = frame
        audit[f"resolved_{name}_timestamp"] = float(attempt_timestamps[frame])
    if "release_frame" in resolved and resolved["release_frame"] >= resolved["regrasp_frame"]:
        raise ValueError(f"Override {attempt_key} resolves to invalid release/regrasp order")
    return resolved, audit


def phase_for_rexecution_frame(
    attempt_id: int,
    frame_index: int,
    rexecution_frame: int | None,
) -> V2Phase:
    if attempt_id == 1:
        return "execution"
    if attempt_id != 2 or rexecution_frame is None:
        raise ValueError("V5.2 attempt2 phase lookup requires rexecution_frame")
    return "adjustment" if frame_index < rexecution_frame else "execution"


def _validate_attempt_alignment(
    meta: Mapping[str, Any],
    frames: Sequence[V4Frame],
    streams: V2RawAttemptStreams,
) -> None:
    expected_count = int(meta["frame_count"])
    ordered = sorted(frames, key=lambda frame: frame.frame_index)
    if len(ordered) != expected_count or len(streams.timestamp) != expected_count:
        raise PhaseBoundaryError(
            f"Attempt {(meta['episode_id'], meta['attempt_id'])} frame count mismatch"
        )
    if [frame.frame_index for frame in ordered] != list(range(expected_count)):
        raise PhaseBoundaryError("LeRobot frame_index does not exactly cover raw HDF5 frames")
    lerobot_timestamps = np.asarray([frame.ros_timestamp for frame in ordered], dtype=np.float64)
    if not np.allclose(lerobot_timestamps, streams.timestamp, rtol=0.0, atol=1e-9):
        mismatch = int(np.flatnonzero(np.abs(lerobot_timestamps - streams.timestamp) > 1e-9)[0])
        raise PhaseBoundaryError(f"LeRobot/raw timestamp mismatch at frame {mismatch}")


def _file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.expanduser().resolve()), "sha256": sha256_file(path)}


def _load_lerobot_actions(dataset_dir: Path, frame_count: int) -> list[np.ndarray]:
    import pyarrow.parquet as pq

    actions: list[np.ndarray | None] = [None] * frame_count
    for parquet_file in sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(parquet_file, columns=["index", "action"])
        payload = table.to_pydict()
        for global_index, value in zip(payload["index"], payload["action"], strict=True):
            index = int(global_index)
            if not 0 <= index < frame_count or actions[index] is not None:
                raise ValueError(f"Invalid or duplicate LeRobot action global index {index}")
            action = np.asarray(value, dtype=np.float32)
            if action.ndim != 1 or not np.isfinite(action).all():
                raise ValueError(f"Invalid LeRobot action at global index {index}: {action.shape}")
            actions[index] = action
    missing = [index for index, action in enumerate(actions) if action is None]
    if missing:
        raise ValueError(f"LeRobot action parquet lacks global indices: {missing[:20]}")
    return [action for action in actions if action is not None]


def compute_h30_content_identities(
    *,
    dataset_dir: Path,
    global_lookup: Mapping[int, V4Frame],
    action_rows: Sequence[Mapping[str, Any]],
    action_indices_identity: Mapping[str, Any],
    action_horizon: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Hash raw and terminal-held action chunks in manifest order."""

    actions = _load_lerobot_actions(dataset_dir, len(global_lookup))
    raw_digest = hashlib.sha256()
    effective_digest = hashlib.sha256()
    raw_digest.update(V2_H30_CONTENT_SCHEMA.encode())
    effective_digest.update(V2_H30_CONTENT_SCHEMA.encode())
    modified_chunks = 0
    modified_steps = 0
    action_dim: int | None = None
    for row in action_rows:
        global_index = int(row["global_index"])
        frame = global_lookup[global_index]
        chunk_values: list[np.ndarray] = []
        for offset in range(action_horizon):
            target_index = global_index + offset
            target_frame = global_lookup.get(target_index)
            if target_frame is None or target_frame.attempt_key != frame.attempt_key:
                raise ValueError(f"H30 chunk crosses an attempt at global index {global_index}")
            if target_frame.frame_index != frame.frame_index + offset:
                raise ValueError(f"H30 frame identity is discontinuous at global index {global_index}")
            chunk_values.append(actions[target_index])
        raw_chunk = np.ascontiguousarray(np.stack(chunk_values), dtype="<f4")
        action_dim = raw_chunk.shape[1] if action_dim is None else action_dim
        if raw_chunk.shape != (action_horizon, action_dim):
            raise ValueError(f"Inconsistent H30 shape at global index {global_index}")
        effective_chunk = np.ascontiguousarray(
            apply_h30_terminal_hold(
                raw_chunk,
                terminal_hold_from_offset=row.get("terminal_hold_from_offset"),
            ),
            dtype="<f4",
        )
        index_bytes = global_index.to_bytes(8, "little", signed=False)
        raw_digest.update(index_bytes)
        raw_digest.update(raw_chunk.tobytes(order="C"))
        effective_digest.update(index_bytes)
        effective_digest.update(effective_chunk.tobytes(order="C"))
        if row.get("terminal_hold_from_offset") is not None:
            modified_chunks += 1
            modified_steps += action_horizon - int(row["terminal_hold_from_offset"])
    if action_dim is None:
        raise ValueError("Cannot hash an empty H30 manifest")

    base = {
        "schema_version": V2_H30_CONTENT_SCHEMA,
        "dtype": "float32_le",
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "chunk_count": len(action_rows),
        "execution_indices": action_indices_identity,
    }
    raw_identity = {**base, "content_sha256": raw_digest.hexdigest()}
    raw_identity["sha256"] = sha256_json(raw_identity)
    effective_identity = {
        **base,
        "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
        "modified_chunk_count": modified_chunks,
        "modified_action_step_count": modified_steps,
        "content_sha256": effective_digest.hexdigest(),
    }
    effective_identity["sha256"] = sha256_json(effective_identity)
    if raw_identity["content_sha256"] == effective_identity["content_sha256"]:
        raise ValueError("V5.2 effective H30 content unexpectedly equals raw V4 H30 content")
    return raw_identity, effective_identity, {
        "modified_chunk_count": modified_chunks,
        "modified_action_step_count": modified_steps,
    }


def build_v5_adjustment_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    overrides_file: Path,
    v4_norm_stats_dir: Path,
    piper_urdf: Path = DEFAULT_PIPER_URDF,
    expected_episode_count: int | None = EXPECTED_EPISODE_COUNT,
    expected_attempt_count: int | None = EXPECTED_ATTEMPT_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build the complete V5.2 boundary, manifest, index and summary payloads."""

    v4_index = json.loads(v4_index_file.read_text())
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    source_files = v4_index["source_files"]
    selection_path = Path(str(source_files["selection"]["path"]))
    profile_path = Path(str(source_files["profile"]["path"]))
    split_path = Path(str(source_files["splits"]["path"]))
    selection = json.loads(selection_path.read_text())
    profile = json.loads(profile_path.read_text())
    splits = json.loads(split_path.read_text())
    selection_attempts = selection.get("attempts", [])
    profile_attempts = profile.get("attempts", [])
    episode_ids = {int(row["episode_id"]) for row in selection_attempts}
    if expected_episode_count is not None and len(episode_ids) != expected_episode_count:
        raise ValueError(f"V5.2 requires {expected_episode_count} V4 episodes, got {len(episode_ids)}")
    if expected_attempt_count is not None and len(selection_attempts) != expected_attempt_count:
        raise ValueError(f"V5.2 requires {expected_attempt_count} V4 attempts")
    selected_tasks = {str(row["task"]) for row in selection_attempts}
    if selected_tasks != ALLOWED_TASKS:
        raise ValueError(f"V5.2 selection has unexpected tasks: {sorted(selected_tasks)}")
    if any(
        str(row.get("task")) in {"no_grasp", "small_grasp"}
        or str(row.get("grasp_position", "")) == "small"
        for row in selection_attempts
    ):
        raise ValueError("V5.2 selection contains no_grasp/small_grasp samples")
    if v4_index.get("selection_hash") != selection.get("selection_hash"):
        raise ValueError("V5.2 selection hash differs from V4")

    split_ids = {
        split: [int(value) for value in splits["original_episode_ids"][split]]
        for split in SPLITS
    }
    profile_by_attempt = {
        (int(row["episode_id"]), int(row["attempt_id"])): row for row in profile_attempts
    }
    frames_by_attempt: dict[tuple[int, int], list[V4Frame]] = defaultdict(list)
    for frame in frames:
        frames_by_attempt[frame.attempt_key].append(frame)
    if set(frames_by_attempt) != set(profile_by_attempt):
        raise ValueError("V5.2 raw labeling attempt set differs from V4")

    override_payload, overrides = load_v2_phase_overrides(overrides_file)
    extra_overrides = sorted(set(overrides) - set(profile_by_attempt))
    if extra_overrides:
        raise ValueError(f"V5.2 overrides reference unselected attempts: {extra_overrides[:20]}")
    fk_chain = load_piper_fk_chain(piper_urdf)
    raw_root = Path(str(profile["raw_data_dir"]))
    boundary_rows: list[dict[str, Any]] = []
    raw_identities: dict[str, Any] = {}
    unresolved: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    contact_sources: Counter[str] = Counter()
    rexecution_sources: Counter[str] = Counter()
    rexecution_triggers: Counter[str] = Counter()

    for attempt_key in sorted(profile_by_attempt):
        meta = profile_by_attempt[attempt_key]
        raw_path = Path(str(meta["hdf5_path"]))
        if not raw_path.is_absolute():
            raw_path = raw_root / raw_path
        streams = load_v2_raw_attempt_streams(raw_path)
        attempt_frames = sorted(frames_by_attempt[attempt_key], key=lambda frame: frame.frame_index)
        _validate_attempt_alignment(meta, attempt_frames, streams)
        raw_identities[f"{attempt_key[0]}/{attempt_key[1]}"] = {
            "path": str(raw_path.resolve()),
            "sha256": streams.source_sha256,
            "dataset_names": streams.dataset_names,
            "frame_count": len(streams.timestamp),
        }
        common = {
            "schema_version": V2_PHASE_BOUNDARY_SCHEMA,
            "data_profile": ROTATION_PHASE_V5_ADJUSTMENT_V2,
            "prompt_profile": PHASE_PROMPT_PROFILE_V2,
            "experiment_kind": V2_EXPERIMENT_KIND,
            "episode_id": attempt_key[0],
            "attempt_id": attempt_key[1],
            "split": str(meta["split"]),
            "task": str(meta["task"]),
            "frame_count": len(attempt_frames),
            "raw_hdf5_path": str(raw_path.resolve()),
            "raw_label_source_sha256": streams.source_sha256,
            "raw_dataset_names": streams.dataset_names,
            "piper_urdf_sha256": fk_chain.source_sha256,
            "first_global_index": attempt_frames[0].global_index,
            "last_global_index": attempt_frames[-1].global_index,
        }
        if attempt_key[1] == 1:
            boundary_rows.append(
                {
                    **common,
                    "release_frame": None,
                    "regrasp_frame": None,
                    "rexecution_frame": None,
                    "contact_label_source": "attempt1_not_applicable",
                    "rexecution_label_source": "attempt1_execution",
                    "force_thresholds": None,
                    "automatic_contact_detection": None,
                    "automatic_contact_detection_error": None,
                    "automatic_rexecution_detection": None,
                    "automatic_rexecution_detection_error": None,
                    "override": None,
                }
            )
            contact_sources["attempt1_not_applicable"] += 1
            rexecution_sources["attempt1_execution"] += 1
            rexecution_triggers["attempt1_not_applicable"] += 1
            phase_counts["execution"] += len(attempt_frames)
            continue

        override = overrides.get(attempt_key)
        resolved_override: dict[str, int] = {}
        override_audit: dict[str, Any] = {}
        if override is not None:
            resolved_override, override_audit = resolve_v2_phase_override(
                override, streams.timestamp, attempt_key=attempt_key
            )

        detected_contact = None
        contact_error: str | None = None
        try:
            detected_contact = detect_phase_boundaries(
                puppet_gripper=streams.puppet_gripper,
                master_gripper=streams.master_gripper,
                fz_left=streams.fz_left,
                fz_right=streams.fz_right,
                tactile_captions=[frame.tactile_caption for frame in attempt_frames],
            )
        except PhaseBoundaryError as exc:
            contact_error = str(exc)
        if "release_frame" in resolved_override:
            release = resolved_override["release_frame"]
            regrasp = resolved_override["regrasp_frame"]
            contact_source = "override"
        elif detected_contact is not None:
            release = detected_contact.release_frame
            regrasp = detected_contact.regrasp_frame
            contact_source = "automatic"
        else:
            unresolved.append(
                {"episode_id": attempt_key[0], "attempt_id": 2, "error": contact_error}
            )
            continue
        if not 0 <= release < regrasp < len(attempt_frames):
            raise ValueError(f"V5.2 contact boundaries for {attempt_key} are outside the attempt")

        detected_rexecution = None
        rexecution_error: str | None = None
        try:
            detected_rexecution = detect_rexecution_frame(
                puppet_right=streams.puppet_right,
                timestamp=streams.timestamp,
                release_frame=release,
                regrasp_frame=regrasp,
                fk_chain=fk_chain,
            )
        except PhaseBoundaryError as exc:
            rexecution_error = str(exc)
        if "rexecution_frame" in resolved_override:
            rexecution = resolved_override["rexecution_frame"]
            rexecution_source = "override"
            rexecution_trigger = "override"
        elif detected_rexecution is not None:
            rexecution = detected_rexecution.rexecution_frame
            rexecution_source = "automatic_fk"
            rexecution_trigger = detected_rexecution.boundary_trigger
        else:
            unresolved.append(
                {"episode_id": attempt_key[0], "attempt_id": 2, "error": rexecution_error}
            )
            continue
        if not release < rexecution < regrasp:
            raise ValueError(
                f"V5.2 boundaries for {attempt_key} must satisfy release < rexecution < regrasp; "
                f"got {release} < {rexecution} < {regrasp}"
            )

        boundary_rows.append(
            {
                **common,
                "release_frame": release,
                "regrasp_frame": regrasp,
                "rexecution_frame": rexecution,
                "rexecution_timestamp": float(streams.timestamp[rexecution]),
                "contact_label_source": contact_source,
                "rexecution_label_source": rexecution_source,
                "force_thresholds": (
                    asdict(detected_contact.thresholds) if detected_contact is not None else None
                ),
                "automatic_contact_detection": (
                    {
                        key: value
                        for key, value in asdict(detected_contact).items()
                        if key not in {"thresholds", "release_frame", "regrasp_frame"}
                    }
                    if detected_contact is not None
                    else None
                ),
                "automatic_contact_detection_error": contact_error,
                "automatic_rexecution_detection": (
                    asdict(detected_rexecution) if detected_rexecution is not None else None
                ),
                "automatic_rexecution_detection_error": rexecution_error,
                "override": (
                    {**dict(override), **override_audit} if override is not None else None
                ),
            }
        )
        contact_sources[contact_source] += 1
        rexecution_sources[rexecution_source] += 1
        rexecution_triggers[rexecution_trigger] += 1
        phase_counts["adjustment"] += rexecution
        phase_counts["execution"] += len(attempt_frames) - rexecution

    if unresolved:
        preview = "; ".join(
            f"{row['episode_id']}/{row['attempt_id']}: {row['error']}" for row in unresolved[:20]
        )
        raise PhaseBoundaryError(
            f"{len(unresolved)} attempt2 rows require V5.2 overrides; {preview}"
        )
    boundary_by_attempt = {
        (int(row["episode_id"]), int(row["attempt_id"])): row for row in boundary_rows
    }
    if len(boundary_by_attempt) != len(profile_by_attempt):
        raise AssertionError("V5.2 boundary manifest does not cover every V4 attempt")

    action_rows: list[dict[str, Any]] = []
    split_entries: dict[str, Any] = {}
    all_v4_indices: list[int] = []
    horizon = int(v4_index["action_horizon"])
    for split in SPLITS:
        indices = [int(value) for value in v4_index["splits"][split]["execution_indices"]]
        manifest_rows: list[int] = []
        for global_index in indices:
            frame = global_lookup[global_index]
            boundary = boundary_by_attempt[frame.attempt_key]
            rexecution = boundary["rexecution_frame"]
            phase = phase_for_rexecution_frame(frame.attempt_id, frame.frame_index, rexecution)
            crosses = bool(
                phase == "adjustment"
                and rexecution is not None
                and frame.frame_index < int(rexecution) <= frame.frame_index + horizon - 1
            )
            hold_offset = int(rexecution) - frame.frame_index if crosses else None
            row_index = len(action_rows)
            action_rows.append(
                {
                    "schema_version": V2_ACTION_PHASE_SCHEMA,
                    "data_profile": ROTATION_PHASE_V5_ADJUSTMENT_V2,
                    "prompt_profile": PHASE_PROMPT_PROFILE_V2,
                    "experiment_kind": V2_EXPERIMENT_KIND,
                    "split": split,
                    "global_index": global_index,
                    "episode_id": frame.episode_id,
                    "attempt_id": frame.attempt_id,
                    "frame_index": frame.frame_index,
                    "phase": phase,
                    "rexecution_frame": rexecution,
                    "raw_chunk_phase_pure": not crosses,
                    "effective_chunk_phase_pure": True,
                    "chunk_phase_pure": not crosses,
                    "terminal_hold_from_offset": hold_offset,
                    "effective_h30_modified": crosses,
                    "chunk_end_frame": frame.frame_index + horizon - 1,
                    "action_horizon": horizon,
                }
            )
            manifest_rows.append(row_index)
        selected_rows = [action_rows[index] for index in manifest_rows]
        split_entries[split] = {
            "execution_indices": indices,
            "action_phase_manifest_row_indices": manifest_rows,
            "summary": {
                "action_count": len(indices),
                "phase_counts": dict(Counter(str(row["phase"]) for row in selected_rows)),
                "raw_chunk_phase_pure": sum(bool(row["raw_chunk_phase_pure"]) for row in selected_rows),
                "raw_chunk_crossing": sum(not bool(row["raw_chunk_phase_pure"]) for row in selected_rows),
                "effective_chunk_phase_pure": sum(
                    bool(row["effective_chunk_phase_pure"]) for row in selected_rows
                ),
                "terminal_hold_chunks": sum(bool(row["effective_h30_modified"]) for row in selected_rows),
            },
        }
        all_v4_indices.extend(indices)

    action_identity = v4_index["action_indices_identity"]
    if action_identity["all"] != {
        "count": len(all_v4_indices),
        "sha256": sha256_json(all_v4_indices),
    }:
        raise ValueError("V5.2 action starts differ from V4")
    raw_h30_identity, effective_h30_identity, modification_summary = compute_h30_content_identities(
        dataset_dir=dataset_dir,
        global_lookup=global_lookup,
        action_rows=action_rows,
        action_indices_identity=action_identity,
        action_horizon=horizon,
    )
    if modification_summary != {
        "modified_chunk_count": V2_EXPECTED_MODIFIED_CHUNKS,
        "modified_action_step_count": V2_EXPECTED_MODIFIED_ACTION_STEPS,
    }:
        raise ValueError(f"Unexpected V5.2 H30 modification counts: {modification_summary}")

    norm_summary_path = v4_norm_stats_dir / "summary.json"
    norm_stats_path = v4_norm_stats_dir / "norm_stats.json"
    if not norm_summary_path.is_file() or not norm_stats_path.is_file():
        raise FileNotFoundError("V5.2 must reuse persisted V4 norm stats")
    norm_summary = json.loads(norm_summary_path.read_text())
    norm_sha = sha256_file(norm_stats_path)
    if (
        norm_summary.get("norm_stats_sha256") != norm_sha
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity") != action_identity
    ):
        raise ValueError("V4 norm stats identity does not match V5.2 action starts")

    v4_h30_source_identity = {
        "action_horizon": horizon,
        "execution_indices": action_identity,
        "lerobot_identity": v4_index["lerobot_identity"],
        "lerobot_parquet_sha256": {
            name: value for name, value in sorted(source_files["lerobot_parquet"].items())
        },
    }
    v4_h30_source_identity["sha256"] = sha256_json(v4_h30_source_identity)
    index: dict[str, Any] = {
        "schema_version": V2_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V5_ADJUSTMENT_V2,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V2_EXPERIMENT_KIND,
        "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
        "data_config_hash": sha256_json(
            {
                "selection_hash": selection["selection_hash"],
                "v4_profile_config_hash": profile["profile_config_hash"],
                "prompt_profile": PHASE_PROMPT_PROFILE_V2,
                "experiment_kind": V2_EXPERIMENT_KIND,
                "piper_urdf_sha256": fk_chain.source_sha256,
                "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
            }
        ),
        "selection_hash": selection["selection_hash"],
        "v4_profile_config_hash": profile["profile_config_hash"],
        "dataset_dir": str(dataset_dir.resolve()),
        "action_horizon": horizon,
        "expected_episode_count": len(episode_ids),
        "expected_attempt_count": len(profile_by_attempt),
        "selected_tasks": sorted(selected_tasks),
        "split_episode_ids": split_ids,
        "splits": split_entries,
        "action_indices_identity": action_identity,
        "v4_action_indices_identity": v4_index["action_indices_identity"],
        "h30_target_identity": effective_h30_identity,
        "v4_h30_target_identity": raw_h30_identity,
        "v4_h30_source_identity": v4_h30_source_identity,
        "v4_lerobot_identity": v4_index["lerobot_identity"],
        "v4_training_data_hash": v4_index["training_data_hash"],
        "v4_norm_stats_sha256": norm_sha,
        "source_files": {
            "selection": _file_identity(selection_path),
            "profile": _file_identity(profile_path),
            "splits": _file_identity(split_path),
            "v4_training_index": _file_identity(v4_index_file),
            "phase_boundary_overrides": _file_identity(overrides_file),
            "piper_urdf": _file_identity(piper_urdf),
            "v4_norm_summary": _file_identity(norm_summary_path),
            "v4_norm_stats": _file_identity(norm_stats_path),
            "raw_label_sources": raw_identities,
        },
        "phase_boundaries_identity": {
            "count": len(boundary_rows),
            "content_sha256": sha256_json(boundary_rows),
        },
        "action_phase_manifest_identity": {
            "count": len(action_rows),
            "content_sha256": sha256_json(action_rows),
        },
    }
    summary = {
        "schema_version": V2_SUMMARY_SCHEMA,
        "data_profile": ROTATION_PHASE_V5_ADJUSTMENT_V2,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V2_EXPERIMENT_KIND,
        "selection_hash": selection["selection_hash"],
        "selected_episodes": len(episode_ids),
        "selected_attempts": len(profile_by_attempt),
        "selected_tasks": dict(Counter(str(row["task"]) for row in selection_attempts)),
        "attempt_ids": dict(Counter(str(row["attempt_id"]) for row in selection_attempts)),
        "no_grasp": 0,
        "small_grasp": 0,
        "phase_frame_counts": dict(phase_counts),
        "contact_label_sources": dict(contact_sources),
        "rexecution_label_sources": dict(rexecution_sources),
        "rexecution_boundary_triggers": dict(rexecution_triggers),
        "action_indices_identity": action_identity,
        "v4_norm_stats_sha256": norm_sha,
        "v4_h30_target_identity": raw_h30_identity,
        "effective_h30_target_identity": effective_h30_identity,
        "h30_modifications": modification_summary,
        "splits": {split: split_entries[split]["summary"] for split in SPLITS},
        "override_version": override_payload["version"],
        "override_count": len(overrides),
        "piper_urdf_sha256": fk_chain.source_sha256,
    }
    return boundary_rows, action_rows, index, summary


def _validate_file_identity(identity: Mapping[str, Any], *, context: str) -> Path:
    path = Path(str(identity.get("path", "")))
    expected = str(identity.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if len(expected) != 64 or actual != expected:
        raise ValueError(f"V5.2 {context} hash mismatch: expected={expected}, actual={actual}")
    return path


def _validate_h30_identity(identity: Any, *, context: str) -> None:
    if not isinstance(identity, Mapping):
        raise ValueError(f"V5.2 lacks {context}")
    stored = str(identity.get("sha256", ""))
    actual = sha256_json({key: value for key, value in identity.items() if key != "sha256"})
    if stored != actual or len(str(identity.get("content_sha256", ""))) != 64:
        raise ValueError(f"V5.2 {context} identity is invalid")


def validate_v5_adjustment_training_index(
    payload: Mapping[str, Any],
    *,
    index_path: Path | None = None,
    dataset_dir: Path | None = None,
    revalidate_raw_streams: bool = False,
    revalidate_h30_targets: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    expected_header = {
        "schema_version": V2_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V5_ADJUSTMENT_V2,
        "prompt_profile": PHASE_PROMPT_PROFILE_V2,
        "experiment_kind": V2_EXPERIMENT_KIND,
        "terminal_hold_schema": V2_TERMINAL_HOLD_SCHEMA,
    }
    for key, expected in expected_header.items():
        if payload.get(key) != expected:
            raise ValueError(f"V5.2 index {key}={payload.get(key)!r}, expected={expected!r}")
    stored_hash = str(payload.get("training_data_hash", ""))
    actual_hash = sha256_json({key: value for key, value in payload.items() if key != "training_data_hash"})
    if not stored_hash or stored_hash != actual_hash:
        raise ValueError("V5.2 training_data_hash does not match the index payload")
    if index_path is not None and not Path(index_path).is_file():
        raise FileNotFoundError(index_path)
    sources = payload.get("source_files")
    if not isinstance(sources, Mapping):
        raise ValueError("V5.2 index lacks source_files")
    required = {
        "selection", "profile", "splits", "v4_training_index",
        "phase_boundary_overrides", "phase_boundaries", "action_phase_manifest",
        "piper_urdf", "v4_norm_summary", "v4_norm_stats", "raw_label_sources",
    }
    missing = sorted(required - set(sources))
    if missing:
        raise ValueError(f"V5.2 index lacks source identities: {missing}")
    paths = {
        name: _validate_file_identity(sources[name], context=name)
        for name in required - {"raw_label_sources"}
    }
    v4_index = json.loads(paths["v4_training_index"].read_text())
    effective_dataset_dir = dataset_dir or Path(str(payload["dataset_dir"]))
    _, global_lookup = validate_v4_index_dataset(v4_index, effective_dataset_dir)
    if payload.get("selection_hash") != v4_index.get("selection_hash"):
        raise ValueError("V5.2/V4 selection hashes differ")
    if payload.get("v4_training_data_hash") != v4_index.get("training_data_hash"):
        raise ValueError("V5.2 references a different V4 index")
    if payload.get("action_indices_identity") != v4_index.get("action_indices_identity"):
        raise ValueError("V5.2/V4 action indices differ")
    for split in SPLITS:
        if payload["splits"][split]["execution_indices"] != v4_index["splits"][split]["execution_indices"]:
            raise ValueError(f"V5.2/V4 {split} action starts differ")

    raw_h30 = payload.get("v4_h30_target_identity")
    effective_h30 = payload.get("h30_target_identity")
    _validate_h30_identity(raw_h30, context="raw V4 H30")
    _validate_h30_identity(effective_h30, context="effective H30")
    if raw_h30["content_sha256"] == effective_h30["content_sha256"]:
        raise ValueError("V5.2 raw/effective H30 hashes must differ")
    if (
        int(effective_h30.get("modified_chunk_count", -1)) != V2_EXPECTED_MODIFIED_CHUNKS
        or int(effective_h30.get("modified_action_step_count", -1))
        != V2_EXPECTED_MODIFIED_ACTION_STEPS
    ):
        raise ValueError("V5.2 effective H30 modification counts are invalid")

    selection = json.loads(paths["selection"].read_text())
    profile = json.loads(paths["profile"].read_text())
    split_payload = json.loads(paths["splits"].read_text())
    if {str(row["task"]) for row in selection.get("attempts", [])} != ALLOWED_TASKS:
        raise ValueError("V5.2 selection contains unexpected tasks")
    if len(selection.get("selected_episode_ids", [])) != int(payload["expected_episode_count"]):
        raise ValueError("V5.2 selected episode count changed")
    if len(profile.get("attempts", [])) != int(payload["expected_attempt_count"]):
        raise ValueError("V5.2 selected attempt count changed")
    actual_split_ids = {
        split: [int(value) for value in split_payload["original_episode_ids"][split]]
        for split in SPLITS
    }
    if actual_split_ids != payload.get("split_episode_ids"):
        raise ValueError("V5.2 split differs from V4")

    boundary_rows = load_jsonl(paths["phase_boundaries"])
    action_rows = load_jsonl(paths["action_phase_manifest"])
    if (
        len(boundary_rows) != int(payload["phase_boundaries_identity"]["count"])
        or sha256_json(boundary_rows) != payload["phase_boundaries_identity"]["content_sha256"]
    ):
        raise ValueError("V5.2 phase boundary identity mismatch")
    if (
        len(action_rows) != int(payload["action_phase_manifest_identity"]["count"])
        or sha256_json(action_rows) != payload["action_phase_manifest_identity"]["content_sha256"]
    ):
        raise ValueError("V5.2 action manifest identity mismatch")
    attempt2_boundaries = [row for row in boundary_rows if int(row["attempt_id"]) == 2]
    if len(attempt2_boundaries) != 408 or any(row.get("rexecution_frame") is None for row in attempt2_boundaries):
        raise ValueError("V5.2 does not resolve all 408 attempt2 rexecution boundaries")

    action_lookup: dict[int, dict[str, Any]] = {}
    expected_actions: list[int] = []
    modified_chunks = 0
    modified_steps = 0
    for split in SPLITS:
        row_indices = payload["splits"][split]["action_phase_manifest_row_indices"]
        indices = payload["splits"][split]["execution_indices"]
        if len(row_indices) != len(indices):
            raise ValueError(f"V5.2 {split} manifest/action lengths differ")
        for row_index, global_index in zip(row_indices, indices, strict=True):
            row = action_rows[int(row_index)]
            if int(row["global_index"]) != int(global_index) or row["split"] != split:
                raise ValueError("V5.2 action manifest order mismatch")
            if row["phase"] not in V2_ACTION_PHASES:
                raise ValueError("V5.2 action row has invalid phase")
            frame = global_lookup[int(global_index)]
            if (
                int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"])
            ) != frame.key:
                raise ValueError("V5.2 action row differs from V4 frame identity")
            hold = row.get("terminal_hold_from_offset")
            if hold is not None:
                hold = int(hold)
                if row["phase"] != "adjustment" or not 1 <= hold < int(row["action_horizon"]):
                    raise ValueError("V5.2 terminal hold is attached to an invalid chunk")
                if int(row["frame_index"]) + hold != int(row["rexecution_frame"]):
                    raise ValueError("V5.2 terminal hold does not begin at rexecution_frame")
                modified_chunks += 1
                modified_steps += int(row["action_horizon"]) - hold
            if bool(row["effective_h30_modified"]) != (hold is not None):
                raise ValueError("V5.2 effective_h30_modified flag is inconsistent")
            if row.get("effective_chunk_phase_pure") is not True:
                raise ValueError("V5.2 effective H30 must be phase-pure")
            if int(global_index) in action_lookup:
                raise ValueError(f"Duplicate V5.2 global index {global_index}")
            action_lookup[int(global_index)] = row
            expected_actions.append(int(global_index))
    if (modified_chunks, modified_steps) != (
        V2_EXPECTED_MODIFIED_CHUNKS,
        V2_EXPECTED_MODIFIED_ACTION_STEPS,
    ):
        raise ValueError("V5.2 manifest terminal-hold totals are invalid")
    if payload["action_indices_identity"]["all"] != {
        "count": len(expected_actions),
        "sha256": sha256_json(expected_actions),
    }:
        raise ValueError("V5.2 action identity is invalid")

    norm_summary = json.loads(paths["v4_norm_summary"].read_text())
    norm_sha = sha256_file(paths["v4_norm_stats"])
    if (
        norm_sha != payload.get("v4_norm_stats_sha256")
        or norm_summary.get("norm_stats_sha256") != norm_sha
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity")
        != payload.get("action_indices_identity")
    ):
        raise ValueError("V5.2 did not preserve V4 norm stats")

    raw_sources = sources["raw_label_sources"]
    if not isinstance(raw_sources, Mapping) or len(raw_sources) != int(payload["expected_attempt_count"]):
        raise ValueError("V5.2 raw source identities do not cover every attempt")
    if revalidate_raw_streams:
        for key, identity in raw_sources.items():
            actual = load_v2_raw_attempt_streams(Path(str(identity["path"]))).source_sha256
            if actual != identity["sha256"]:
                raise ValueError(f"V5.2 raw label source hash mismatch for {key}")
    if revalidate_h30_targets:
        actual_raw, actual_effective, _ = compute_h30_content_identities(
            dataset_dir=effective_dataset_dir,
            global_lookup=global_lookup,
            action_rows=action_rows,
            action_indices_identity=payload["action_indices_identity"],
            action_horizon=int(payload["action_horizon"]),
        )
        if actual_raw != raw_h30 or actual_effective != effective_h30:
            raise ValueError("V5.2 persisted H30 content hashes do not match LeRobot/manifest")
    return action_rows, action_lookup
