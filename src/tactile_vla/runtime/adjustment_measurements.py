"""Save phase times and raw qpos for offline adjustment distance calculations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Full, Queue
import threading
import time

import numpy as np

SCHEMA = "v7_7_2_adjustment_measurements_v1"


class AdjustmentMeasurements:
    """Small boundary records only; disk IO runs outside the control thread."""

    def __init__(self, directory: Path, *, urdf: Path, args, metadata):
        contents = urdf.read_bytes()
        snapshot = directory / "adjustment_robot.urdf"
        snapshot.write_bytes(contents)
        self.path = directory / "adjustment_measurements.jsonl"
        self._file = self.path.open("x", encoding="utf-8")
        self._queue = Queue(maxsize=256)
        self._error = None
        self._active = None
        self._count = 0
        self._plan = ""
        self._failure = ""
        self._stop_reason = None
        self._closed = False
        self._worker = threading.Thread(target=self._write_loop, name="adjustment-measurements", daemon=True)
        self._worker.start()
        self._record(
            "run_start",
            schema_version=SCHEMA,
            args=vars(args).copy(),
            server_metadata=metadata,
            no_publish=bool(args.no_publish),
            qpos_source=args.puppet_arm_topic,
            arm_indices=list(range(6)),
            arm_unit="rad",
            gripper_unit="m",
            urdf_snapshot=snapshot.name,
            urdf_sha256=hashlib.sha256(contents).hexdigest(),
            fk_reference={"base_link": "base_link", "end_link": "gripper_base"},
            tcp_offset_recorded=False,
        )

    @staticmethod
    def _default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    def _write_loop(self):
        try:
            while True:
                row = self._queue.get()
                if row is None:
                    break
                self._file.write(json.dumps(row, ensure_ascii=False, default=self._default) + "\n")
                self._file.flush()
        except Exception as exc:
            self._error = exc
        finally:
            self._file.close()

    def _record(self, event, **fields):
        if self._error is not None:
            raise RuntimeError(f"Failed to save adjustment measurements: {self.path}") from self._error
        row = {"event": event, "wall_time": time.time(), "monotonic_time": time.monotonic(), **fields}
        try:
            self._queue.put_nowait(row)
        except Full as exc:
            self._error = RuntimeError("Adjustment measurement queue is full")
            raise self._error from exc

    def observe(self, event, point):
        kind = event.get("event")
        if kind == "failure_and_plan":
            self._plan = event["recovery_plan"]
            self._failure = event["failure_reason"]
            self._record(
                "plan",
                recovery_plan=self._plan,
                failure_reason=self._failure,
                source_attempt_id=event.get("attempt_id"),
                point=point,
            )
        elif kind == "phase_transition":
            if event["phase"] == "adjustment":
                if self._active is not None:
                    self._record(
                        "adjustment_incomplete", **self._identity(), reason="new_adjustment_before_publication_end"
                    )
                self._count += 1
                self._active = {
                    "segment_id": self._count,
                    "attempt_id": event["attempt_id"],
                    "recovery_plan": self._plan,
                    "failure_reason": self._failure,
                    "phase_exit_recorded": False,
                    "first_adjustment_recorded": False,
                }
                self._record(
                    "adjustment_enter", **self._identity(), point=point, published_steps=event.get("published_steps")
                )
            elif event.get("previous_phase") == "adjustment" and self._active is not None:
                self._active["phase_exit_recorded"] = True
                self._record(
                    "adjustment_exit",
                    **self._identity(),
                    point=point,
                    trigger=event.get("trigger"),
                    published_steps=event.get("published_steps"),
                )
        elif kind in {
            "operator_quit",
            "operator_success",
            "fail_closed_safety_stop",
            "max_attempts_reached",
            "max_publish_step_reached",
        }:
            self._stop_reason = kind
            self._record("stop", reason=kind, error=event.get("error"), point=point)

    def _identity(self):
        return {key: self._active[key] for key in ("segment_id", "attempt_id", "recovery_plan", "failure_reason")}

    def decision(self, event, point):
        identity = self._identity() if self._active else {"segment_id": None, "attempt_id": event.get("attempt_id")}
        self._record(
            "adjustment_end_decision",
            **identity,
            request_id=event.get("request_id"),
            probs=event.get("adjustment_end_probs"),
            point=point,
        )

    def needs_publication(self, phase, attempt_id):
        if self._active is None or self._active["attempt_id"] != attempt_id:
            return False
        return (phase == "adjustment" and not self._active["first_adjustment_recorded"]) or (
            phase == "execution" and self._active["phase_exit_recorded"]
        )

    def publication(self, phase, attempt_id, before, after_monotonic):
        if self._active is None or self._active["attempt_id"] != attempt_id:
            return
        if phase == "adjustment" and not self._active["first_adjustment_recorded"]:
            self._active["first_adjustment_recorded"] = True
            self._record(
                "first_adjustment_command",
                **self._identity(),
                point=before,
                publication_confirmed_monotonic=after_monotonic,
            )
        elif phase == "execution" and self._active["phase_exit_recorded"]:
            self._record(
                "first_execution_command",
                **self._identity(),
                point=before,
                adjustment_command_recorded=self._active["first_adjustment_recorded"],
                publication_confirmed_monotonic=after_monotonic,
            )
            self._active = None

    def finish(self, reason, point):
        if reason == "run_returned":
            reason = self._stop_reason or reason
        if self._active is not None:
            self._record(
                "adjustment_incomplete",
                **self._identity(),
                reason=reason,
                phase_exit_recorded=self._active["phase_exit_recorded"],
                point=point,
            )
        self._record("run_end", reason=reason, segment_count=self._count, point=point)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._worker.is_alive():
            self._queue.put(None, timeout=1.0)
            self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            raise RuntimeError("Adjustment measurement writer did not finish")
        if self._error is not None:
            raise RuntimeError(f"Failed to save adjustment measurements: {self.path}") from self._error


def capture_point(operator):
    """Snapshot cached, unmodified ROS feedback without waiting for another frame."""
    wall = time.time()
    with operator.measurement_lock:
        feedback = operator.measurement_feedback
    return {
        "wall_time": wall,
        "utc_time": datetime.fromtimestamp(wall, timezone.utc).isoformat(),
        "monotonic_time": time.monotonic(),
        "ros_time": float(operator.rospy.Time.now().to_sec()),
        "feedback": feedback,
    }
