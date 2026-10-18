#!/usr/bin/env python3
"""V7.7.2 copy with raw adjustment phase/publication measurements for offline FK."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

from pathlib import Path
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
        def record(self, event):
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


def recording_gate(base_gate, recorder, operator, *, no_publish):
    class MeasurementGate(base_gate):
        def handle_phase_event(self, event):
            applied = super().handle_phase_event(event)
            if applied and event.get("phase") == "adjustment" and event.get("event") == "phase_decision":
                recorder.decision(event, capture_point(operator))
            return applied

        def try_publish_action(self, action, *, generation, publish):
            phase, attempt_id, _, _, _ = self.snapshot()
            if no_publish or not recorder.needs_publication(phase, attempt_id):
                return super().try_publish_action(action, generation=generation, publish=publish)

            def measured_publish(value):
                before = capture_point(operator)
                publish(value)
                recorder.publication(phase, attempt_id, before, time.monotonic())

            return super().try_publish_action(action, generation=generation, publish=measured_publish)

    return MeasurementGate


def main():
    # Same identities, norm stats, parser and controller as the original V7.7.2 entry.
    shared.DATA_PROFILE = DATA_PROFILE
    shared.DEFAULT_NORM_STATS = Path(
        "/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/rotation_v4_1_new_env/norm_stats/norm_stats.json"
    )
    shared.DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/v7_7_2_adjustment_measurements"
    args, parser = shared.get_arguments()
    shared.runtime.apply_yaml_defaults(args, parser)
    shared.validate_args(args, parser)
    if not sys.stdin.isatty():
        parser.error("V7.7.2 real-robot inference requires an interactive TTY")
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
        recorder[0] = AdjustmentMeasurements(
            logger.directory,
            urdf=DEFAULT_URDF,
            args=args,
            metadata=action_policy.get_server_metadata(),
        )
        shared.V77AsyncControlGate = recording_gate(old_gate, recorder[0], operator, no_publish=args.no_publish)
        print(f"Trial log directory: {logger.directory}")
        if not args.start_immediately:
            input("Press enter to start V7.7.2 EXECUTION with adjustment measurements")
        with shared.runtime.KeyboardPoller() as keyboard:
            shared.run_v7_7_async(args, operator, action_policy, phase_client, captioner, keyboard, logger)
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
