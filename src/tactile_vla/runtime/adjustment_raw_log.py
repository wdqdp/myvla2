"""Raw ROS feedback and publication boundaries for offline adjustment measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from queue import Full, Queue
import threading
import time

import numpy as np


class AdjustmentRawRecorder:
    """Write observations on a worker thread; never compute kinematics or distances."""

    def __init__(self, directory: Path, operator, *, urdf_path: Path):
        self.path = directory / "adjustment_raw.jsonl"
        self._queue = Queue(maxsize=8192)
        self._error = None
        self._closed = False
        self._enabled = threading.Event()
        self._sample_lock = threading.Lock()
        self._latest = None
        self._active = None
        self._segment_count = 0
        self._command_count = 0
        self._stop_reason = None
        self._urdf = {"source_path": str(urdf_path), "snapshot": None, "sha256": None}
        if urdf_path.is_file():
            contents = urdf_path.read_bytes()
            snapshot = directory / "adjustment_robot.urdf"
            snapshot.write_bytes(contents)
            self._urdf.update(snapshot=snapshot.name, sha256=hashlib.sha256(contents).hexdigest())
        self._file = self.path.open("x", encoding="utf-8")
        self._worker = threading.Thread(target=self._write_loop, name="adjustment-raw-writer", daemon=True)
        try:
            self._subscriber = operator.rospy.Subscriber(
                operator.args.puppet_arm_topic,
                operator.JointState,
                self._joint_callback,
                queue_size=1000,
                tcp_nodelay=True,
            )
        except Exception:
            self._file.close()
            raise
        self._worker.start()
        # Seed the cache without consuming the observation deque or waiting for another sample.
        try:
            self._joint_callback(operator.puppet_arm_deque[-1])
        except IndexError:
            pass

    def _write_loop(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._file.write(json.dumps(item, ensure_ascii=False, default=self._json_default) + "\n")
                self._file.flush()
        except Exception as exc:
            self._error = exc
        finally:
            self._file.close()

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    def _record(self, event, **fields):
        if self._error is not None:
            raise RuntimeError(f"Raw adjustment log write failed: {self.path}") from self._error
        payload = {"event": event, "wall_time": time.time(), "monotonic_time": time.monotonic(), **fields}
        try:
            self._queue.put_nowait(payload)
        except Full as exc:
            self._error = RuntimeError("Raw adjustment log queue is full; recording cannot keep up")
            raise self._error from exc

    def _joint_callback(self, message):
        sample = {
            "feedback_timestamp": float(message.header.stamp.to_sec()),
            "received_wall_time": time.time(),
            "received_monotonic_time": time.monotonic(),
            "joint_names": list(message.name),
            "position": list(message.position),
            "velocity": list(message.velocity),
            "effort": list(message.effort),
        }
        # Replace the entire snapshot; queued records never reference mutable ROS messages.
        with self._sample_lock:
            self._latest = sample
            if self._enabled.is_set():
                try:
                    self._record("joint_feedback", **sample)
                except RuntimeError:
                    # The control-thread publisher/finalizer reports the stored failure.
                    self._enabled.clear()

    def snapshot(self):
        with self._sample_lock:
            return self._snapshot_unlocked()

    def start(self, args, metadata):
        self._record(
            "run_start",
            schema_version="adjustment_raw_v1",
            args=vars(args).copy(),
            server_metadata=metadata,
            qpos_source=args.puppet_arm_topic,
            arm_position_unit="rad",
            gripper_position_unit="m",
            arm_position_indices=list(range(6)),
            urdf=self._urdf,
            fk_reference={"base_link": "base_link", "end_link": "gripper_base"},
            tcp_offset_recorded=False,
            no_publish=bool(args.no_publish),
            start_boundary="feedback_before_first_published_adjustment_command",
            end_boundary="feedback_before_first_published_following_execution_command",
        )
        with self._sample_lock:
            self._enabled.set()
            self._record("initial_feedback", **self._snapshot_unlocked())

    def _snapshot_unlocked(self):
        return {"sampled_monotonic_time": time.monotonic(), "feedback": self._latest}

    def on_publish(
        self,
        *,
        phase,
        attempt_id,
        recovery_plan,
        generation,
        phase_index,
        action_index,
        command,
        command_timestamp,
        before,
    ):
        self._command_count += 1
        publication = {
            "command_id": self._command_count,
            "phase": phase,
            "attempt_id": attempt_id,
            "recovery_plan": recovery_plan,
            "action_generation": generation,
            "phase_index": phase_index,
            "action_index": action_index,
            "published_action": np.asarray(command).copy(),
            "command_timestamp": command_timestamp,
            "before_publication": before,
        }
        if phase == "adjustment" and self._active is None:
            self._segment_count += 1
            self._active = {
                "segment_id": self._segment_count,
                "attempt_id": attempt_id,
                "recovery_plan": recovery_plan,
                "adjustment_end_received": False,
            }
            self._record("adjustment_start", **self._active, publication=publication)
        elif phase == "execution" and self._active is not None:
            status = "complete" if self._active["adjustment_end_received"] else "execution_without_end_decision"
            self._record("adjustment_end", **self._active, status=status, publication=publication)
            self._active = None
        self._record("action_publish", segment_id=self._active["segment_id"] if self._active else None, **publication)

    def on_feedback_after_action(self, qpos, timestamp):
        self._record(
            "feedback_after_action",
            command_id=self._command_count,
            feedback_qpos=np.asarray(qpos).copy(),
            feedback_timestamp=timestamp,
        )

    def on_adjustment_end(self, *, request_id, attempt_id, probs):
        if self._active is not None:
            self._active["adjustment_end_received"] = True
        self._record(
            "adjustment_end_decision",
            segment_id=self._active["segment_id"] if self._active else None,
            request_id=request_id,
            attempt_id=attempt_id,
            probs=probs,
            feedback=self.snapshot(),
        )

    def mark_stop(self, reason):
        self._stop_reason = reason
        self._record("stop_requested", reason=reason)

    def finish(self, reason):
        with self._sample_lock:
            self._enabled.clear()
        if reason == "run_returned" and self._stop_reason is not None:
            reason = self._stop_reason
        if self._active is not None:
            self._record("adjustment_incomplete", **self._active, reason=reason, last_feedback=self.snapshot())
            self._active = None
        self._record(
            "run_end",
            reason=reason,
            command_count=self._command_count,
            segment_count=self._segment_count,
            last_feedback=self.snapshot(),
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        with self._sample_lock:
            self._enabled.clear()
        self._subscriber.unregister()
        if self._worker.is_alive():
            try:
                self._queue.put(None, timeout=1.0)
            except Full as exc:
                raise RuntimeError("Cannot close raw adjustment log: writer queue is full") from exc
            self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            raise RuntimeError("Raw adjustment log writer did not finish within 5 seconds")
        if self._error is not None:
            raise RuntimeError(f"Raw adjustment log write failed: {self.path}") from self._error
