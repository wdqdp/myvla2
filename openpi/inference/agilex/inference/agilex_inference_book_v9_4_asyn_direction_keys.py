#!/usr/bin/env python3
"""Book V9.4 asynchronous action deployment with a/s/d/f recovery selection.

Execution transitions are manual; only adjustment_end is queried asynchronously.
The existing V9.4 five-task server supplies action and phase decisions.
"""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
from concurrent.futures import Future
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import agilex_inference_book_v9_4_asyn as base
from agilex_inference_book_v9_4_asyn import BookV94AsyncControlGate
from agilex_inference_book_v9_4_asyn import OperatorStopError
from agilex_inference_book_v9_4_asyn import PhaseAssessmentResult
from agilex_inference_book_v9_4_asyn import StreamingPhaseClient
from agilex_inference_book_v9_4_asyn import _capture_action_observation
from agilex_inference_book_v9_4_asyn import _phase_result_log
from agilex_inference_book_v9_4_asyn import _request_action_chunk
from agilex_inference_book_v9_4_asyn import _run_phase_assessment
from agilex_inference_book_v9_4_asyn import _wait_future_while_holding
from agilex_inference_book_v9_4_asyn import inference_executor
from agilex_inference_book_v9_4_asyn import validate_server_metadata
from tactile_vla.common.labels_v4 import LABEL_FIELDS
from tactile_vla.common.labels_v4 import LABEL_SCHEMA_VERSION
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles

runtime = base.runtime
v52 = base.v52
v53 = base.v53
DEFAULT_LOG_ROOT = base.PROJECT_ROOT / "outputs/runtime/book_v9_4_async_direction_keys"
DIRECTION_KEYS = {
    "a": ("left", "moderately"),
    "s": ("left", "slightly"),
    "d": ("right", "slightly"),
    "f": ("right", "moderately"),
}


class BookV94DirectionControlGate(BookV94AsyncControlGate):
    def enter_adjustment(self) -> bool:
        """Freeze the latest feedback and invalidate execution actions under one lock."""
        with self._lock:
            if self._terminal_stop or self.state.phase != "execution":
                return False
            self.state.stop_latched = True
            self.state.pending_actions.clear()
            self.state.action_generation += 1
            self._reset_hold()
            self._hold()
            self.state.switch_phase("adjustment", increment_attempt=True)
            return True


def run_book_v9_4_async_direction_keys(
    args: argparse.Namespace, operator: Any, action_policy: v53.TimeoutWebsocketPolicy,
    phase_client: StreamingPhaseClient, captioner: Any, keyboard: Any,
    logger: v52.TrialLogger,
) -> None:
    action_metadata = action_policy.get_server_metadata()
    validate_server_metadata(action_metadata)
    if action_metadata.get("instruction") != args.instruction:
        raise ValueError("Book instruction differs between client and server")
    if action_metadata.get("norm_stats_sha256") != sha256_file(args.norm_stats_file):
        raise ValueError("Book client/server norm stats differ")
    if action_metadata.get("captioner_checkpoint_sha256") != sha256_file(args.captioner_checkpoint):
        raise ValueError("Book client/server captioner identities differ")
    if (getattr(captioner, "schema_version", None) != LABEL_SCHEMA_VERSION
            or list(getattr(captioner, "label_fields", ())) != list(LABEL_FIELDS)
            or captioner.window_size != 30):
        raise ValueError("Book V9.4 requires the six-field V4 masked W30 captioner")
    if action_metadata != phase_client.metadata:
        raise ValueError("Book V9.4 action and phase connections expose different metadata")
    args.use_state_history = False
    stats = load_state_quantiles(args.norm_stats_file)
    recovery_plan = ""
    phase_indices = {"execution": 0, "adjustment": 0}
    published_steps = 0
    phase_request_count = 0
    last_phase_submit = {"execution": None, "adjustment": None}
    last_feedback_timestamp: float | None = None
    phase_future: Future[PhaseAssessmentResult] | None = None
    hold_target: list[np.ndarray | None] = [None]
    pending_observation: v52.FrozenObservation | None = None

    def hold() -> None:
        target = hold_target[0]
        if target is None:
            target, _ = v52._latest_feedback(operator)
            if target is None or np.asarray(target).shape != (7,) or not np.isfinite(target).all():
                raise RuntimeError("Cannot hold: no valid current ROS qpos feedback")
            hold_target[0] = np.asarray(target, dtype=np.float32).copy()
        command = np.asarray(target, dtype=np.float32).copy()
        command[6] = max(float(args.gripper_min), float(command[6]))
        if args.no_publish:
            return
        operator.puppet_arm_publish(command)

    def reset_hold() -> None:
        hold_target[0] = None

    gate = BookV94DirectionControlGate(hold=hold, reset_hold=reset_hold)
    logger.record({"event": "run_start", "server_metadata": action_metadata,
                   "args": vars(args), "continuous_h100": True, "control_mode": "direction_keys"})
    print(
        "Book V9.4 direction keys: a=left moderately, s=left slightly, "
        "d=right slightly, f=right moderately; q=stop; SPACE disabled. "
        f"adjustment_end={args.adjustment_end_rate_hz:g}Hz "
        f"threshold={float(action_metadata['adjustment_end_threshold']):g}", flush=True,
    )

    def poll_control() -> None:
        nonlocal recovery_plan, pending_observation
        key = keyboard.get_key()
        if key == args.quit_key:
            gate.stop()
            runtime.shutdown_event.set()
            logger.record({"event": "operator_quit", "phase": gate.snapshot()[0],
                           "published_steps": published_steps})
            raise OperatorStopError("operator_quit")
        if runtime.shutdown_event.is_set() or operator.is_shutdown():
            gate.stop()
            raise OperatorStopError("shutdown")
        if phase_future is not None and phase_future.done():
            error = phase_future.exception()
            if error is not None:
                raise error
        if key in DIRECTION_KEYS and gate.enter_adjustment():
            direction, magnitude = DIRECTION_KEYS[key]
            _, recovery_plan = v52.rotation_targets(direction, magnitude)
            pending_observation = None
            last_phase_submit["adjustment"] = None
            logger.record({
                "event": "phase_transition", "previous_phase": "execution", "phase": "adjustment",
                "trigger": "direction_key", "key": key, "direction": direction, "magnitude": magnitude,
                "recovery_plan": recovery_plan, "attempt_id": gate.snapshot()[1],
                "published_steps": published_steps,
            })
            print(f"[MANUAL_RECOVERY] key={key} plan={recovery_plan}; holding and switching to ADJUSTMENT.",
                  flush=True)

    with inference_executor(gate=gate, action_policy=action_policy, phase_client=phase_client) as executor:
        # This is the only H100 reset: phase and attempt transitions preserve it.
        operator.reset_state_history()
        # Action prompts contain no touch text; reserve the captioner for the
        # asynchronous phase worker to avoid serializing action generation on it.
        initial = _wait_future_while_holding(
            executor.submit(_capture_action_observation, args, operator),
            gate=gate, operator=operator, publish_rate=args.publish_rate, poll_control=poll_control,
        )
        if initial.timestamp is None:
            raise ValueError("initial ROS qpos does not have a timestamp")
        operator.state_history.push(float(initial.timestamp), initial.qpos)
        args.episode_start_timestamp = float(initial.timestamp)
        args.episode_start_qpos = initial.qpos.copy()
        last_feedback_timestamp = initial.timestamp
        pending_observation = initial if gate.snapshot()[0] == "execution" else None

        def maybe_submit_phase() -> None:
            nonlocal phase_future, phase_request_count
            phase, attempt_id, _, phase_generation, stopped = gate.snapshot()
            if phase != "adjustment" or stopped or phase_future is not None or last_feedback_timestamp is None:
                return
            rate_hz = args.adjustment_end_rate_hz
            now = time.monotonic()
            previous = last_phase_submit[phase]
            if previous is not None and now - previous < 1.0 / rate_hz:
                return
            request_id = f"{phase}-{attempt_id}-{phase_generation}-{phase_request_count}"
            phase_request_count += 1
            gate.begin_phase_request(request_id)
            last_phase_submit[phase] = now
            phase_future = executor.submit(
                _run_phase_assessment,
                args=args, client=phase_client, operator=operator,
                captioner=captioner, stats=stats, gate=gate,
                phase=phase, generation=phase_generation, attempt_id=attempt_id,
                request_id=request_id, captured_step=published_steps,
                after_timestamp=float(last_feedback_timestamp),
                recovery_plan=recovery_plan, submitted_monotonic=now,
            )
            logger.record({"event": "phase_assessment_submit", "request_id": request_id,
                           "phase": phase, "attempt_id": attempt_id,
                           "generation": phase_generation, "captured_step": published_steps})

        def finish_phase_result(*, wait: bool = False) -> tuple[bool, bool]:
            """Return (transitioned, terminal)."""
            nonlocal phase_future, pending_observation
            if phase_future is None:
                return False, False
            if not phase_future.done() and not wait:
                return False, False
            result = (_wait_future_while_holding(
                phase_future, gate=gate, operator=operator, publish_rate=args.publish_rate,
                poll_control=poll_control,
            ) if wait else phase_future.result())
            phase_future = None
            if result.phase != "adjustment":
                raise ValueError("Direction-key deployment only requests adjustment assessments")
            logger.record(_phase_result_log(result, handled_step=published_steps))
            adjustment_probs = np.asarray(
                result.decision.get("adjustment_end_probs"), dtype=np.float64,
            ).reshape(-1).tolist()
            print(
                f"[ADJUSTMENT_END] request_id={result.request_id} "
                f"caption={result.observation.tactile_caption} "
                f"result={result.triggered} probs(false,true)={adjustment_probs} "
                f"threshold={float(action_metadata['adjustment_end_threshold']):g}", flush=True,
            )
            active_phase, active_attempt, _, active_generation, _ = gate.snapshot()
            if (result.phase, result.attempt_id, result.generation) != (
                active_phase, active_attempt, active_generation
            ):
                logger.record({"event": "stale_phase_result", "request_id": result.request_id})
                return False, False
            if not result.triggered:
                return False, False
            print("[PHASE] adjustment_end=true; holding and switching to EXECUTION.")
            gate.switch_phase("execution")
            pending_observation = None
            logger.record({"event": "phase_transition", "previous_phase": "adjustment",
                           "phase": "execution", "trigger": "adjustment_end",
                           "attempt_id": active_attempt, "published_steps": published_steps})
            return True, False

        while published_steps < args.max_publish_step and not operator.is_shutdown():
            poll_control()
            _, _, _, _, stopped = gate.snapshot()
            if stopped and phase_future is not None:
                _, terminal = finish_phase_result(wait=True)
                if terminal:
                    return
                continue
            transitioned, terminal = finish_phase_result()
            if terminal:
                return
            if transitioned:
                continue

            phase, attempt_id, _, _, _ = gate.snapshot()
            observation = pending_observation
            if observation is None:
                observation = _wait_future_while_holding(
                    executor.submit(_capture_action_observation, args, operator),
                    gate=gate, operator=operator, publish_rate=args.publish_rate, poll_control=poll_control,
                )
            pending_observation = None
            if gate.snapshot()[0] != phase:
                # A direction key arrived while the old phase observation was captured.
                continue
            phase_index = phase_indices[phase]
            action_generation = gate.begin_action_request()
            action_future = executor.submit(
                _request_action_chunk, args=args, policy=action_policy, observation=observation,
                phase=phase, phase_index=phase_index, recovery_plan=recovery_plan, logger=logger,
            )
            actions = _wait_future_while_holding(
                action_future, gate=gate, operator=operator, publish_rate=args.publish_rate,
                poll_control=poll_control,
            )
            phase_indices[phase] += 1
            stopped_after_action = gate.snapshot()[-1]
            if stopped_after_action and phase_future is not None:
                logger.record({"event": "stale_action_chunk", "phase": phase,
                               "phase_index": phase_index, "discarded_actions": len(actions)})
                continue
            if stopped_after_action:
                if not gate.release_hold_with_fresh_actions(action_generation, actions):
                    logger.record({"event": "stale_action_chunk", "phase": phase,
                                   "phase_index": phase_index, "discarded_actions": len(actions)})
                    continue
            elif not gate.accept_action_result(action_generation, actions):
                logger.record({"event": "stale_action_chunk", "phase": phase,
                               "phase_index": phase_index, "discarded_actions": len(actions)})
                continue

            limit = min(args.chunk_size, args.max_publish_step - published_steps, len(actions))
            completed = 0
            control_rate = operator.rate(args.publish_rate)
            for action_index, action_value in enumerate(actions[:limit]):
                if phase_future is not None and phase_future.done():
                    transitioned, terminal = finish_phase_result()
                    if terminal:
                        return
                    if transitioned:
                        break
                if gate.snapshot()[-1]:
                    break
                poll_control()
                raw_action = np.asarray(action_value, dtype=np.float32)
                command = raw_action.copy()
                offset, command[6], floor_applied = v52.gripper_command_with_safety_floor(
                    float(command[6]), gripper_offset=args.gripper_offset,
                    gripper_min=args.gripper_min,
                )
                _, before_timestamp = v52._latest_feedback(operator)
                command_timestamp: list[float | None] = [None]

                def publish(value, timestamp_slot=command_timestamp):
                    if args.no_publish:
                        runtime.published_actions_history.append(np.asarray(value).copy())
                    else:
                        timestamp_slot[0] = operator.puppet_arm_publish(value)
                        runtime.published_actions_history.append(np.asarray(value).copy())

                if not gate.try_publish_action(command, generation=action_generation, publish=publish):
                    break
                control_rate.sleep()
                feedback, feedback_timestamp = _wait_future_while_holding(
                    executor.submit(v53._wait_feedback_after, args, operator, before_timestamp),
                    gate=gate, operator=operator, publish_rate=args.observation_poll_rate,
                    poll_control=poll_control,
                )
                last_feedback_timestamp = feedback_timestamp
                completed += 1
                published_steps += 1
                logger.record({
                    "event": "action_publish", "phase": phase,
                    "attempt_id": attempt_id, "phase_index": phase_index,
                    "action_index": action_index, "raw_action": raw_action,
                    "published_action": command, "gripper_after_offset": offset,
                    "gripper_min_applied": floor_applied,
                    "command_timestamp": command_timestamp[0],
                    "feedback_qpos": feedback, "feedback_timestamp": feedback_timestamp,
                })
                maybe_submit_phase()
                if gate.snapshot()[-1]:
                    break

            logger.record({
                "event": "execution_chunk", "phase": phase, "phase_index": phase_index,
                "completed_raw_actions": completed,
                "discarded_raw_actions": max(0, limit - completed),
                "stop_latched": gate.snapshot()[-1],
            })

        logger.record({"event": "max_publish_step_reached", "published_steps": published_steps})
def get_arguments(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = base.build_argument_parser(direction_keys=True)
    parser.description = __doc__
    parser.set_defaults(log_dir=DEFAULT_LOG_ROOT)
    return parser.parse_args(argv), parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    base.validate_args(args, parser)
    if args.quit_key != "q":
        parser.error("Direction-key deployment uses q to stop; --quit-key must be q")


def main() -> None:
    base.main(
        argument_parser=get_arguments, argument_validator=validate_args,
        run=run_book_v9_4_async_direction_keys, deployment_label="Book V9.4 direction keys",
    )


if __name__ == "__main__":
    main()
