#!/usr/bin/env python3
"""V7.7.2: execute H30, assess once, then generate the next chunk; save raw FK data."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import agilex_inference_v7_7_2_asyn_record_adjustment as recording
from tactile_vla.runtime.adjustment_measurements import AdjustmentMeasurements
from tactile_vla.runtime.adjustment_measurements import capture_point
from tactile_vla.vla.v7_7_2_multitask_data import DATA_PROFILE

shared = recording.shared
DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/v7_7_2_sync_measurements"


class SyncMeasurements(AdjustmentMeasurements):
    def point(self, event, point, **fields):
        self._record(event, point=point, **fields)


def sync_recording_gate(base_gate, recorder, operator, *, no_publish):
    measured_gate = recording.recording_gate(base_gate, recorder, operator, no_publish=no_publish)

    class SyncGate(measured_gate):
        measurements = recorder

        def handle_phase_event(self, event):
            if shared.runtime.shutdown_event.is_set():
                return False
            with self._lock:
                if event.get("request_id") != self.state.accepted_phase_request_id or (
                    event.get("phase") != self.state.phase
                ):
                    raise ValueError("Synchronous phase response has a stale request or wrong phase")
            if event.get("event") == "phase_decision":
                key = "need_recovery" if event.get("phase") == "execution" else "adjustment_end"
                probs = np.asarray(event.get(key + "_probs"), dtype=np.float64)
                if not isinstance(event.get(key), bool | np.bool_) or probs.shape != (2,) or (
                    not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
                    or not np.isclose(probs.sum(), 1.0, atol=1e-4)
                ):
                    raise ValueError(f"Invalid synchronous {key} decision: {event}")
            applied = super().handle_phase_event(event)
            if applied and event.get("phase") == "execution" and event.get("event") == "phase_decision":
                # Record receipt of the first true need event, before failure/plan decoding.
                recorder.point(
                    "fault_detected", capture_point(operator), attempt_id=self.snapshot()[1],
                    request_id=event["request_id"], need_recovery_probs=event["need_recovery_probs"],
                )
            return applied

    return SyncGate


class OperatorStoppedError(Exception):
    pass


def run_sync(args, operator, action_policy, phase_client, captioner, keyboard, logger):
    if args.chunk_size != shared.ACTION_HORIZON:
        raise ValueError("Synchronous V7.7.2 requires --chunk_size=30, including YAML defaults")
    if "b" in {args.success_key, args.quit_key}:
        raise ValueError("Key b is reserved for the pre-lift height baseline")
    metadata = action_policy.get_server_metadata()
    shared.validate_server_metadata(metadata)
    if metadata != phase_client.metadata:
        raise ValueError("V7.7.2 action and phase connections expose different metadata")
    args.use_state_history = False
    stats = shared.load_state_quantiles(args.norm_stats_file)
    target = [None]

    def hold():
        value = target[0]
        if value is None:
            value, _ = shared.v52._latest_feedback(operator)
        if value is not None and not args.no_publish:
            command = np.asarray(value, dtype=np.float32).copy()
            command[6] = max(float(args.gripper_min), float(command[6]))
            operator.puppet_arm_publish(command)

    gate = shared.V77AsyncControlGate(hold=hold)
    measurements = gate.measurements
    published_steps = 0
    phase_indices = {"execution": 0, "adjustment": 0}
    recovery_plan, memory = "", []
    execution_starts = set()
    request_count = 0

    def poll_control():
        if operator.is_shutdown() or shared.runtime.shutdown_event.is_set():
            logger.record({"event": "operator_quit", "reason": "ROS_or_signal_shutdown",
                           "published_steps": published_steps})
            shared.runtime.shutdown_event.set()
            gate.stop()
            raise OperatorStoppedError
        key = keyboard.get_key()
        if key == "b":
            phase, attempt, _, _, _ = gate.snapshot()
            measurements.point("lift_baseline", capture_point(operator), phase=phase,
                               attempt_id=attempt, published_steps=published_steps, source="keyboard_b")
            print("[LIFT BASELINE] saved raw qpos before lifting (b).", flush=True)
            return
        if key in {args.quit_key, args.success_key}:
            phase, attempt, _, _, _ = gate.snapshot()
            logger.record({"event": "operator_success" if key == args.success_key else "operator_quit",
                           "phase": phase, "attempt_id": attempt, "published_steps": published_steps})
            gate.stop()
            shared.runtime.shutdown_event.set()
            raise OperatorStoppedError

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v7-7-2-sync")

    def request(function, **kwargs):
        # The sole worker keeps the terminal responsive; inference is fully serial.
        poll_control()
        hold()
        future = executor.submit(function, **kwargs)
        rate = operator.rate(args.publish_rate)
        while not future.done():
            poll_control()
            hold()
            rate.sleep()
        poll_control()
        return future.result()

    logger.record({"event": "run_start", "server_metadata": metadata, "args": vars(args),
                   "phase_assessment_policy": "once_after_complete_h30_in_both_phases",
                   "continuous_h100": True, "height_feedback_sampling": "one_raw_point_per_action_feedback"})
    print("V7.7.2 SYNC: H30 actions -> one phase assessment -> next chunk; "
          f"b=mark pre-lift baseline, {args.success_key}=success, {args.quit_key}=quit.", flush=True)
    try:
        if not shared.v53._wait_for_tactile_ready(args, operator, captioner, poll_control=poll_control):
            return
        operator.reset_state_history()
        initial = request(shared._capture_action_observation, args=args, operator=operator)
        if initial.timestamp is None:
            raise ValueError("Initial qpos timestamp is missing")
        operator.state_history.push(float(initial.timestamp), initial.qpos)
        target[0] = initial.qpos.copy()
        pending_observation = initial
        while not operator.is_shutdown():
            poll_control()
            if args.max_publish_step - published_steps < shared.ACTION_HORIZON:
                logger.record({"event": "max_publish_step_reached", "published_steps": published_steps,
                               "reason": "remaining_budget_less_than_complete_h30"})
                return
            phase, attempt, _, phase_generation, _ = gate.snapshot()
            observation = pending_observation or request(
                shared._capture_action_observation, args=args, operator=operator,
            )
            pending_observation = None
            target[0] = observation.qpos.copy()
            generation = gate.begin_action_request()
            phase_index = phase_indices[phase]
            actions = request(
                shared._request_action_chunk, args=args, policy=action_policy, observation=observation,
                phase=phase, phase_index=phase_index, recovery_plan=recovery_plan, logger=logger,
            )
            phase_indices[phase] += 1
            accepted = (
                gate.release_hold_with_fresh_actions(generation, actions) if gate.snapshot()[-1]
                else gate.accept_action_result(generation, actions)
            )
            if not accepted:
                raise ValueError("Synchronous action chunk has a stale generation")
            logger.record({"event": "chunk_start", "phase": phase, "attempt_id": attempt,
                           "phase_index": phase_index, "published_steps": published_steps})
            last_feedback_timestamp = None
            rate = operator.rate(args.publish_rate)
            for action_index, value in enumerate(actions):
                poll_control()
                raw = np.asarray(value, dtype=np.float32)
                command = raw.copy()
                offset, command[6], floor_applied = shared.v52.gripper_command_with_safety_floor(
                    float(command[6]), gripper_offset=args.gripper_offset, gripper_min=args.gripper_min,
                )
                _, before_timestamp = shared.v52._latest_feedback(operator)
                command_timestamp = [None]

                def publish(action, *, command_phase=phase, command_attempt=attempt,
                            stamp_slot=command_timestamp, step=published_steps):
                    first_execution = command_phase == "execution" and command_attempt not in execution_starts
                    before = capture_point(operator) if first_execution else None
                    if not args.no_publish:
                        stamp_slot[0] = operator.puppet_arm_publish(action)
                    shared.runtime.published_actions_history.append(np.asarray(action).copy())
                    if first_execution and not args.no_publish:
                        execution_starts.add(command_attempt)
                        measurements.point("execution_start", before, attempt_id=command_attempt,
                                           published_steps=step)
                        if len(execution_starts) == 1:
                            measurements.point("trial_baseline", before, attempt_id=command_attempt)

                if not gate.try_publish_action(command, generation=generation, publish=publish):
                    raise ValueError("Action publication was rejected during a synchronous chunk")
                rate.sleep()
                feedback, stamp = shared.v53._wait_feedback_after(args, operator, before_timestamp)
                target[0] = feedback.copy()
                published_steps += 1
                last_feedback_timestamp = stamp
                measurements.point(
                    "action_feedback", capture_point(operator), phase=phase, attempt_id=attempt,
                    phase_index=phase_index, action_index=action_index, published_steps=published_steps,
                )
                logger.record({"event": "action_publish", "phase": phase, "attempt_id": attempt,
                               "phase_index": phase_index, "action_index": action_index,
                               "raw_action": raw, "published_action": command,
                               "gripper_after_offset": offset, "gripper_min_applied": floor_applied,
                               "command_timestamp": command_timestamp[0],
                               "feedback_qpos": feedback, "feedback_timestamp": stamp})
            logger.record({"event": "execution_chunk", "phase": phase, "attempt_id": attempt,
                           "phase_index": phase_index, "completed_raw_actions": len(actions),
                           "discarded_raw_actions": 0, "published_steps": published_steps})
            poll_control()
            request_id = f"sync-{phase}-{attempt}-{request_count}"
            request_count += 1
            gate.begin_phase_request(request_id)
            logger.record({"event": "phase_assessment_submit", "phase": phase, "attempt_id": attempt,
                           "phase_index": phase_index, "request_id": request_id,
                           "captured_step": published_steps, "after_complete_chunk": True})
            result = request(
                shared._run_phase_assessment, args=args, client=phase_client, operator=operator,
                captioner=captioner, stats=stats, gate=gate, phase=phase, generation=phase_generation,
                attempt_id=attempt, request_id=request_id, captured_step=published_steps,
                after_timestamp=float(last_feedback_timestamp), recovery_plan=recovery_plan,
                submitted_monotonic=time.monotonic(),
            )
            log = shared._phase_result_log(result, handled_step=published_steps)
            log.update(synchronous=True, total_sync_ms=log.pop("total_async_ms"),
                       classification_observation_qpos=result.observation.qpos,
                       classification_observation_timestamp=result.observation.timestamp)
            logger.record(log)
            key = "need_recovery" if phase == "execution" else "adjustment_end"
            print(f"[SYNC {phase.upper()}] chunk={phase_index} {key}={result.triggered} "
                  f"probs={result.decision[key + '_probs']} caption={result.observation.tactile_caption}", flush=True)
            if not result.triggered:
                continue
            if phase == "adjustment":
                gate.switch_phase("execution")
                logger.record({"event": "phase_transition", "previous_phase": "adjustment",
                               "phase": "execution", "trigger": "adjustment_end",
                               "attempt_id": attempt, "published_steps": published_steps})
                continue
            failures = [e for e in result.events if e.get("event") == "failure_reason"]
            if len(failures) != 1 or failures[0].get("failure_reason") not in shared.legal_failure_reasons():
                raise ValueError("need_recovery=true must return exactly one legal failure_reason")
            failure_reason = failures[0]["failure_reason"]
            if attempt >= args.max_attempts:
                logger.record({"event": "max_attempts_reached", "attempt_id": attempt,
                               "failure_reason": failure_reason})
                return
            payload, updated_memory = shared.plan_payload_after_failure(
                instruction=args.instruction, tactile_caption=result.observation.tactile_caption,
                executed_recovery_plan=recovery_plan, failure_reason=failure_reason, memory=memory,
                image=shared.runtime.prepare_rgb(result.observation.img_front),
                wrist_image=shared.runtime.prepare_rgb(result.observation.img_left), qpos=result.observation.qpos,
            )
            response = request(
                action_policy.infer_with_timeout, observation=payload, timeout=args.request_timeout_seconds,
            )
            new_plan = str(response.get("recovery_plan", "")).strip()
            if new_plan not in shared.legal_recovery_plans():
                raise ValueError(f"Illegal recovery_plan: {new_plan!r}")
            recovery_plan, memory = new_plan, updated_memory
            logger.record({"event": "failure_and_plan", "attempt_id": attempt,
                           "failure_reason": failure_reason, "recovery_plan": recovery_plan,
                           "memory": memory, "plan_prompt": payload["prompt"]})
            gate.switch_phase("adjustment", increment_attempt=True)
            logger.record({"event": "phase_transition", "previous_phase": "execution",
                           "phase": "adjustment", "trigger": "need_recovery",
                           "attempt_id": attempt + 1, "published_steps": published_steps})
    except OperatorStoppedError:
        return
    finally:
        try:
            hold()
        finally:
            # Drain the sole bounded request before the outer recorder closes.
            executor.shutdown(wait=True, cancel_futures=True)


def get_arguments():
    original_doc = shared.__doc__
    shared.__doc__ = __doc__
    try:
        args, parser = shared.get_arguments()
    finally:
        shared.__doc__ = original_doc
    if args.chunk_size != shared.ACTION_HORIZON:
        parser.error("Synchronous V7.7.2 uses complete H30 chunks; --chunk_size must be 30")
    args.synchronous_phase_checks = True
    return args, parser


def main():
    recording.main(
        data_profile=DATA_PROFILE, default_log_root=DEFAULT_LOG_ROOT,
        argument_parser=get_arguments, deployment_label="V7.7.2 SYNC",
        controller=run_sync, measurements_type=SyncMeasurements, gate_factory=sync_recording_gate,
    )


if __name__ == "__main__":
    main()
