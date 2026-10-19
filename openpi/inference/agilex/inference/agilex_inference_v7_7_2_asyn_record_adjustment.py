#!/usr/bin/env python3
"""V7.7.2 async recording: pre-lift b marker, need=true height and adjustment FK data."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

from pathlib import Path
from queue import Empty
from queue import SimpleQueue
import signal
import sys
import threading
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import agilex_inference_v7_7_asyn as shared
from tactile_vla.runtime.adjustment_measurements import AdjustmentMeasurements
from tactile_vla.runtime.adjustment_measurements import capture_point
from tactile_vla.vla.v7_7_2_multitask_data import DATA_PROFILE

DEFAULT_URDF = (
    PROJECT_ROOT
    / "openpi/inference/agilex/Piper_ros_private-ros-noetic/src/piper_description/urdf/piper_description.urdf"
)


class RecordingRosOperator(shared.runtime.RosOperator):
    def __init__(self, args):
        self.measurement_lock = threading.Lock()
        self.measurement_feedback = None
        super().__init__(args)

    def puppet_arm_callback(self, message):
        sample = {
            "timestamp": float(message.header.stamp.to_sec()),
            "received_monotonic_time": time.monotonic(),
            "joint_names": list(message.name),
            "qpos": list(message.position),
            "velocity": list(message.velocity),
            "effort": list(message.effort),
        }
        with self.measurement_lock:
            self.measurement_feedback = sample
        super().puppet_arm_callback(message)


def recording_logger(recorder, operator, directory, trial_id):
    class MeasurementTrialLogger(shared.v52.TrialLogger):
        action_count = 0

        def record(self, event):
            if recorder[0] is not None and getattr(operator.args, "asynchronous_lift_measurements", False) and (
                event.get("event") == "action_publish"
            ):
                self.action_count += 1
                recorder[0].point(
                    "action_feedback", capture_point(operator),
                    phase=event["phase"], attempt_id=event["attempt_id"],
                    phase_index=event["phase_index"], action_index=event["action_index"],
                    published_steps=self.action_count,
                )
            if recorder[0] is not None and event.get("event") in {
                "failure_and_plan",
                "phase_transition",
                "operator_quit",
                "operator_success",
                "fail_closed_safety_stop",
                "max_attempts_reached",
                "max_publish_step_reached",
            }:
                # Capture before existing event-log disk IO. This call is immediately
                # after the control loop changes phase, without any settling delay.
                recorder[0].observe(event, capture_point(operator))
            super().record(event)

    return MeasurementTrialLogger(directory, trial_id=trial_id)


def recording_gate(base_gate, recorder, operator, *, no_publish, record_lift=False):
    class MeasurementGate(base_gate):
        def __init__(self, *args, **kwargs):
            self._execution_starts = set()
            super().__init__(*args, **kwargs)
            if record_lift:
                recorder.control_gate = self

        def handle_phase_event(self, event):
            applied = super().handle_phase_event(event)
            if applied and event.get("phase") == "adjustment" and event.get("event") == "phase_decision":
                recorder.decision(event, capture_point(operator))
            elif record_lift and applied and event.get("phase") == "execution" and (
                event.get("event") == "phase_decision"
            ):
                recorder.point(
                    "fault_detected", capture_point(operator), attempt_id=self.snapshot()[1],
                    request_id=event["request_id"], need_recovery_probs=event["need_recovery_probs"],
                )
            return applied

        def try_publish_action(self, action, *, generation, publish):
            phase, attempt_id, _, _, _ = self.snapshot()
            first_execution = record_lift and phase == "execution" and attempt_id not in self._execution_starts
            if no_publish or not (first_execution or recorder.needs_publication(phase, attempt_id)):
                return super().try_publish_action(action, generation=generation, publish=publish)

            def measured_publish(value):
                before = capture_point(operator)
                publish(value)
                recorder.publication(phase, attempt_id, before, time.monotonic())
                if first_execution:
                    self._execution_starts.add(attempt_id)
                    recorder.point("execution_start", before, attempt_id=attempt_id)
                    if len(self._execution_starts) == 1:
                        recorder.point("trial_baseline", before, attempt_id=attempt_id)

            return super().try_publish_action(action, generation=generation, publish=measured_publish)

    return MeasurementGate


class HeightMeasurementKeyboard:
    """Capture b during blocking inference; pass other keys to the async controller."""

    def __init__(self, recorder, operator):
        self._recorder = recorder
        self._operator = operator
        self._keyboard = shared.runtime.KeyboardPoller()
        self._keys = SimpleQueue()
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._poll, name="height-baseline-keys", daemon=True)

    def _poll(self):
        try:
            while not self._stop.is_set():
                key = self._keyboard.get_key()
                if key == "b":
                    gate = getattr(self._recorder, "control_gate", None)
                    phase, attempt = gate.snapshot()[:2] if gate is not None else ("execution", 1)
                    self._recorder.point(
                        "lift_baseline", capture_point(self._operator), phase=phase,
                        attempt_id=attempt, source="keyboard_b",
                    )
                    print("[LIFT BASELINE] saved raw qpos before lifting (b).", flush=True)
                elif key is not None:
                    self._keys.put(key)
                self._stop.wait(0.005)
        except Exception as exc:
            self._error = exc

    def __enter__(self):
        self._keyboard.__enter__()
        try:
            self._thread.start()
        except Exception:
            self._keyboard.__exit__(*sys.exc_info())
            raise
        return self

    def get_key(self):
        if self._error is not None:
            raise RuntimeError("Failed to record a pre-lift keyboard marker") from self._error
        try:
            return self._keys.get_nowait()
        except Empty:
            return None

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._keyboard.__exit__(exc_type, exc, tb)
        if self._thread.is_alive():
            raise RuntimeError("Height baseline keyboard thread did not stop")
        if self._error is not None and exc_type is None:
            raise RuntimeError("Failed to record a pre-lift keyboard marker") from self._error


def main(
    *,
    data_profile=DATA_PROFILE,
    default_norm_stats=None,
    default_log_root=None,
    argument_parser=None,
    deployment_label="V7.7.2",
    controller=None,
    measurements_type=AdjustmentMeasurements,
    gate_factory=None,
):
    # Versioned entries share the recording hooks and unchanged async controller.
    shared.DATA_PROFILE = data_profile
    shared.DEFAULT_NORM_STATS = (
        Path(default_norm_stats) if default_norm_stats is not None else Path(
            "/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/rotation_v4_1_new_env/norm_stats/norm_stats.json"
        )
    )
    shared.DEFAULT_LOG_ROOT = (
        Path(default_log_root) if default_log_root is not None
        else PROJECT_ROOT / "outputs/runtime/v7_7_2_adjustment_measurements"
    )
    args, parser = (argument_parser or shared.get_arguments)()
    shared.runtime.apply_yaml_defaults(args, parser)
    shared.validate_args(args, parser)
    args.asynchronous_lift_measurements = controller is None and data_profile == DATA_PROFILE
    if args.asynchronous_lift_measurements and "b" in {args.success_key, args.quit_key}:
        parser.error("Key b is reserved for marking the pre-lift height baseline")
    if not sys.stdin.isatty():
        parser.error(f"{deployment_label} real-robot inference requires an interactive TTY")
    if not DEFAULT_URDF.is_file():
        parser.error(f"Robot URDF is missing: {DEFAULT_URDF}")
    shared.runtime.shutdown_event.clear()
    signal.signal(signal.SIGINT, shared.runtime._on_sigint)
    captioner = shared.runtime.load_captioner(args)
    action_policy = shared.v53.TimeoutWebsocketPolicy(args.host, args.port)
    phase_client = None
    recorder = [None]
    operator = None
    logger = None
    old_gate = shared.V77AsyncControlGate
    reason = "run_returned"
    try:
        phase_client = shared.StreamingPhaseClient(args.host, args.port, timeout=args.request_timeout_seconds)
        operator = RecordingRosOperator(args)
        shared.v53._install_tactile_arrival_timestamps(operator)
        trial_id = args.trial_id
        if trial_id and (args.log_dir.resolve() / trial_id).exists():
            trial_id = f"{trial_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        logger = recording_logger(recorder, operator, args.log_dir, trial_id)
        recorder[0] = measurements_type(
            logger.directory,
            urdf=DEFAULT_URDF,
            args=args,
            metadata=action_policy.get_server_metadata(),
        )
        if gate_factory is not None:
            shared.V77AsyncControlGate = gate_factory(old_gate, recorder[0], operator, no_publish=args.no_publish)
        else:
            shared.V77AsyncControlGate = recording_gate(
                old_gate, recorder[0], operator, no_publish=args.no_publish,
                record_lift=args.asynchronous_lift_measurements,
            )
        print(f"Trial log directory: {logger.directory}")
        if not args.start_immediately:
            input(f"Press enter to start {deployment_label} EXECUTION with adjustment measurements")
        keyboard_context = (
            HeightMeasurementKeyboard(recorder[0], operator) if args.asynchronous_lift_measurements
            else shared.runtime.KeyboardPoller()
        )
        if args.asynchronous_lift_measurements:
            print("b=mark pre-lift baseline; heights are calculated offline from need=true keypoints.")
        with keyboard_context as keyboard:
            (controller or shared.run_v7_7_async)(
                args, operator, action_policy, phase_client, captioner, keyboard, logger
            )
    except BaseException as exc:
        reason = f"{type(exc).__name__}:{exc}"
        if operator is not None:
            feedback, _ = shared.v52._latest_feedback(operator)
            if feedback is not None and not args.no_publish:
                hold = feedback.copy()
                hold[6] = max(float(args.gripper_min), float(hold[6]))
                operator.puppet_arm_publish(hold)
        if logger is not None:
            try:
                logger.record({"event": "fail_closed_safety_stop", "error": repr(exc)})
            except Exception as log_error:
                print(f"Could not record inference exception: {log_error}", file=sys.stderr)
        raise
    finally:
        shared.V77AsyncControlGate = old_gate
        try:
            if recorder[0] is not None:
                try:
                    recorder[0].finish(reason, capture_point(operator))
                finally:
                    recorder[0].close()
        finally:
            if phase_client is not None:
                phase_client.close()
            close = getattr(getattr(action_policy, "_ws", None), "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()
