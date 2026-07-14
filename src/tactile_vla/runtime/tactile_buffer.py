"""Runtime tactile buffering for closed-loop VLA inference."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import numpy as np


DEFAULT_TACTILE_CAPTION = "Tactile: no rotation."
GRID_SHAPE = (35, 20, 3)
WINDOW_SIZE = 30

_LEFT_FORCE = "left_force"
_RIGHT_FORCE = "right_force"
_LEFT_MESH = "left_mesh_3d"
_RIGHT_MESH = "right_mesh_3d"
_LEFT_FLOW = "left_mesh_3d_flow"
_RIGHT_FLOW = "right_mesh_3d_flow"
_REQUIRED_KEYS = frozenset({_LEFT_FORCE, _RIGHT_FORCE, _LEFT_MESH, _RIGHT_MESH, _LEFT_FLOW, _RIGHT_FLOW})


@dataclass(frozen=True)
class TactileTopics:
    """ROS topic names for the six raw tactile grids."""

    left_force: str = "/xense/OG001251/force"
    right_force: str = "/xense/OG000991/force"
    left_mesh_3d: str = "/xense/OG001251/mesh_3d"
    right_mesh_3d: str = "/xense/OG000991/mesh_3d"
    left_mesh_3d_flow: str = "/xense/OG001251/mesh_3d_flow"
    right_mesh_3d_flow: str = "/xense/OG000991/mesh_3d_flow"

    @classmethod
    def from_mapping(cls, values: Mapping[str, str] | None) -> "TactileTopics":
        if not values:
            return cls()
        return cls(
            left_force=values.get("left_force", cls.left_force),
            right_force=values.get("right_force", cls.right_force),
            left_mesh_3d=values.get("left_mesh_3d", cls.left_mesh_3d),
            right_mesh_3d=values.get("right_mesh_3d", cls.right_mesh_3d),
            left_mesh_3d_flow=values.get("left_mesh_3d_flow", cls.left_mesh_3d_flow),
            right_mesh_3d_flow=values.get("right_mesh_3d_flow", cls.right_mesh_3d_flow),
        )


@dataclass(frozen=True)
class TactileFrame:
    """One synchronized tactile frame in captioner input layout."""

    mesh_motion: np.ndarray
    force: np.ndarray
    timestamp: float | None = None


def _layout_shape(layout: Any) -> tuple[int, int, int] | None:
    dims = getattr(layout, "dim", None)
    if dims is None or len(dims) < 3:
        return None
    return tuple(int(getattr(dim, "size")) for dim in dims[:3])


def reshape_tactile_grid(data: Any, *, layout: Any = None, grid_shape: tuple[int, int, int] = GRID_SHAPE) -> np.ndarray:
    """Reshape a flat Float32MultiArray payload into ``[35, 20, 3]``."""

    shape = _layout_shape(layout) or grid_shape
    if shape != grid_shape:
        raise ValueError(f"Expected tactile layout dim={grid_shape}, got {shape}")
    array = np.asarray(data, dtype=np.float32)
    expected_size = int(np.prod(grid_shape))
    if array.size != expected_size:
        raise ValueError(f"Expected tactile grid with {expected_size} values, got {array.size}")
    return array.reshape(grid_shape)


def concat_tactile_frame(
    *,
    left_force: np.ndarray,
    right_force: np.ndarray,
    left_mesh_3d: np.ndarray,
    right_mesh_3d: np.ndarray,
    left_mesh_3d_flow: np.ndarray,
    right_mesh_3d_flow: np.ndarray,
) -> TactileFrame:
    """Build captioner inputs from raw left/right tactile grids."""

    force = np.concatenate([left_force, right_force], axis=-1).astype(np.float32, copy=False)
    mesh_motion = np.concatenate(
        [left_mesh_3d_flow, left_mesh_3d, right_mesh_3d_flow, right_mesh_3d],
        axis=-1,
    ).astype(np.float32, copy=False)
    if force.shape != (35, 20, 6):
        raise ValueError(f"Expected force shape [35,20,6], got {force.shape}")
    if mesh_motion.shape != (35, 20, 12):
        raise ValueError(f"Expected mesh_motion shape [35,20,12], got {mesh_motion.shape}")
    return TactileFrame(mesh_motion=mesh_motion, force=force)


class TactileWindowBuffer:
    """Maintain a recent tactile window from raw grid updates."""

    def __init__(self, *, window_size: int = WINDOW_SIZE, grid_shape: tuple[int, int, int] = GRID_SHAPE) -> None:
        self.window_size = int(window_size)
        self.grid_shape = grid_shape
        self._frames: deque[TactileFrame] = deque(maxlen=self.window_size)
        self._latest: dict[str, np.ndarray] = {}
        self._seen_since_frame: set[str] = set()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def ready(self) -> bool:
        return len(self) >= self.window_size

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._latest.clear()
            self._seen_since_frame.clear()

    def push_topic(self, key: str, data: Any, *, layout: Any = None, timestamp: float | None = None) -> bool:
        """Push one ROS topic update.

        Returns True only when this update completed a new six-topic tactile frame.
        """

        if key not in _REQUIRED_KEYS:
            raise KeyError(f"Unknown tactile key: {key}")
        grid = reshape_tactile_grid(data, layout=layout, grid_shape=self.grid_shape)
        with self._lock:
            self._latest[key] = grid
            self._seen_since_frame.add(key)
            if not _REQUIRED_KEYS.issubset(self._latest) or not _REQUIRED_KEYS.issubset(self._seen_since_frame):
                return False
            frame = self._frame_from_latest_locked(timestamp=timestamp)
            self._frames.append(frame)
            self._seen_since_frame.clear()
            return True

    def push_raw(
        self,
        *,
        left_force: Any,
        right_force: Any,
        left_mesh_3d: Any,
        right_mesh_3d: Any,
        left_mesh_3d_flow: Any,
        right_mesh_3d_flow: Any,
        timestamp: float | None = None,
    ) -> TactileFrame:
        """Append one complete tactile frame from raw arrays."""

        frame = concat_tactile_frame(
            left_force=np.asarray(left_force, dtype=np.float32).reshape(self.grid_shape),
            right_force=np.asarray(right_force, dtype=np.float32).reshape(self.grid_shape),
            left_mesh_3d=np.asarray(left_mesh_3d, dtype=np.float32).reshape(self.grid_shape),
            right_mesh_3d=np.asarray(right_mesh_3d, dtype=np.float32).reshape(self.grid_shape),
            left_mesh_3d_flow=np.asarray(left_mesh_3d_flow, dtype=np.float32).reshape(self.grid_shape),
            right_mesh_3d_flow=np.asarray(right_mesh_3d_flow, dtype=np.float32).reshape(self.grid_shape),
        )
        frame = TactileFrame(mesh_motion=frame.mesh_motion, force=frame.force, timestamp=timestamp)
        with self._lock:
            self._frames.append(frame)
        return frame

    def _frame_from_latest_locked(self, *, timestamp: float | None) -> TactileFrame:
        frame = concat_tactile_frame(
            left_force=self._latest[_LEFT_FORCE],
            right_force=self._latest[_RIGHT_FORCE],
            left_mesh_3d=self._latest[_LEFT_MESH],
            right_mesh_3d=self._latest[_RIGHT_MESH],
            left_mesh_3d_flow=self._latest[_LEFT_FLOW],
            right_mesh_3d_flow=self._latest[_RIGHT_FLOW],
        )
        return TactileFrame(mesh_motion=frame.mesh_motion, force=frame.force, timestamp=timestamp)

    def latest_window(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(mesh_motion, force)`` with shapes ``[30,35,20,12]`` and ``[30,35,20,6]``."""

        with self._lock:
            if len(self._frames) < self.window_size:
                return None
            frames = list(self._frames)[-self.window_size :]
        mesh_motion = np.stack([frame.mesh_motion for frame in frames], axis=0)
        force = np.stack([frame.force for frame in frames], axis=0)
        return mesh_motion, force

    def caption(self, predictor: Any | None) -> str:
        """Return the captioner result, or the default caption while the window is still warming up."""

        window = self.latest_window()
        if window is None or predictor is None:
            return DEFAULT_TACTILE_CAPTION
        mesh_motion, force = window
        return str(predictor.predict(mesh_motion, force).caption)


class RosTactileBuffer:
    """ROS subscriber wrapper around :class:`TactileWindowBuffer`."""

    def __init__(self, topics: TactileTopics | None = None, *, window_size: int = WINDOW_SIZE) -> None:
        import rospy
        from std_msgs.msg import Float32MultiArray

        self.buffer = TactileWindowBuffer(window_size=window_size)
        self.topics = topics or TactileTopics()
        topic_map = {
            _LEFT_FORCE: self.topics.left_force,
            _RIGHT_FORCE: self.topics.right_force,
            _LEFT_MESH: self.topics.left_mesh_3d,
            _RIGHT_MESH: self.topics.right_mesh_3d,
            _LEFT_FLOW: self.topics.left_mesh_3d_flow,
            _RIGHT_FLOW: self.topics.right_mesh_3d_flow,
        }
        self.subscribers = [
            rospy.Subscriber(topic, Float32MultiArray, self._make_callback(key), queue_size=1000, tcp_nodelay=True)
            for key, topic in topic_map.items()
        ]

    def _make_callback(self, key: str):
        def callback(msg) -> None:
            stamp = None
            try:
                stamp = msg.header.stamp.to_sec()
            except AttributeError:
                pass
            self.buffer.push_topic(key, msg.data, layout=msg.layout, timestamp=stamp)

        return callback

    @property
    def ready(self) -> bool:
        return self.buffer.ready

    def latest_window(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self.buffer.latest_window()

    def caption(self, predictor: Any | None) -> str:
        return self.buffer.caption(predictor)


def load_attempt_tactile_window(
    attempt_dir: str | Path,
    *,
    start: int = 0,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one tactile window from a recorded attempt directory."""

    import h5py

    attempt_dir = Path(attempt_dir)
    buffer = TactileWindowBuffer(window_size=window_size)
    with h5py.File(attempt_dir / "data.hdf5", "r") as file:
        size = int(file["size"][()])
        stop = min(size, int(start) + int(window_size))
        for index in range(int(start), stop):
            buffer.push_raw(
                left_force=file["tactile/force/left"][index],
                right_force=file["tactile/force/right"][index],
                left_mesh_3d=file["tactile/mesh_3d/left"][index],
                right_mesh_3d=file["tactile/mesh_3d/right"][index],
                left_mesh_3d_flow=file["tactile/mesh_3d_flow/left"][index],
                right_mesh_3d_flow=file["tactile/mesh_3d_flow/right"][index],
                timestamp=float(file["timestamp"][index]),
            )
    window = buffer.latest_window()
    if window is None:
        raise ValueError(f"Attempt {attempt_dir} has fewer than {window_size} tactile frames after start={start}")
    return window

