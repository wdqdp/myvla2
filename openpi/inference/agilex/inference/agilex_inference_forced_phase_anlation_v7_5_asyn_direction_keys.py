#!/usr/bin/env python3
"""V7.6 async inference with keyboard-selected horizontal recovery plans."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

import agilex_inference_forced_phase_anlation as v52
import agilex_inference_forced_phase_anlation_v7_5_asyn as implementation
from tactile_vla.vla.v7_6_adjustment_end_data import DATA_PROFILE as V7_6_DATA_PROFILE

# Reuse the proven V7.5 online H100 implementation with the V7.6 model identity.
implementation.DATA_PROFILE = V7_6_DATA_PROFILE

Phase = Literal["execution", "adjustment"]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/forced_phase_ablation_v7_6_async_direction_keys"
DIRECTION_KEYS = {
    "a": ("left", "moderately"),
    "s": ("left", "slightly"),
    "d": ("right", "slightly"),
    "f": ("right", "moderately"),
}
FIXED_SELECTION_OPTIONS = (
    "--rotation-direction",
    "--rotation-magnitude",
    "--forced-failure-reason",
    "--forced-recovery-plan",
)
ROLLING_QPOS_BUFFER_FRAMES = 100


def _select_adjustment(args: argparse.Namespace, key: str) -> None:
    direction, magnitude = DIRECTION_KEYS[key]
    failure_reason, recovery_plan = v52.rotation_targets(direction, magnitude)
    args.rotation_direction = direction
    args.rotation_magnitude = magnitude
    args.forced_failure_reason = failure_reason
    args.forced_recovery_plan = recovery_plan
    args.adjustment_selection_key = key
    print(f"[ADJUSTMENT SELECT] key={key} direction={direction} magnitude={magnitude} plan={recovery_plan}")


def _poll_key(args: argparse.Namespace, keyboard: Any, *, phase: Phase) -> str | None:
    """Handle keys between chunks; direction keys replace the old SPACE trigger."""

    key = keyboard.get_key()
    if key is None:
        return None
    if key == args.quit_key:
        return "quit"
    if phase == "execution" and key in DIRECTION_KEYS:
        _select_adjustment(args, key)
        return "trigger"
    return None


def _poll_control_key(
    args: argparse.Namespace,
    keyboard: Any,
    *,
    phase: Phase,
    chunk_paused: bool = False,
) -> v52.ControlSignal | None:
    """Handle keys while actions are publishing; this client never pauses chunks."""

    del chunk_paused
    key = keyboard.get_key()
    if key is None:
        return None
    if key == args.quit_key:
        return v52.ControlSignal("quit", key)
    if phase == "execution" and key in DIRECTION_KEYS:
        _select_adjustment(args, key)
        return v52.ControlSignal("trigger", key)
    return None


def _keyboard_selection_validate_args(
    original_validate,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Run the shared validator with a temporary legal plan, then require a key."""

    if any(
        value == option or value.startswith(f"{option}=")
        for value in sys.argv[1:]
        for option in FIXED_SELECTION_OPTIONS
    ):
        parser.error("This client selects recovery with a/s/d/f; remove fixed rotation and recovery-plan options")
    placeholder_failure, placeholder_plan = v52.rotation_targets("right", "moderately")
    args.rotation_direction = "right"
    args.rotation_magnitude = "moderately"
    args.forced_failure_reason = placeholder_failure
    args.forced_recovery_plan = placeholder_plan
    original_validate(args, parser)
    args.rotation_direction = None
    args.rotation_magnitude = None
    args.forced_failure_reason = None
    args.forced_recovery_plan = None
    args.adjustment_selection_key = None


def run_v7_5_async_direction_keys(
    args: argparse.Namespace,
    operator: Any,
    action_policy: Any,
    classification_policy: Any,
    captioner: Any,
    keyboard: Any,
    logger: Any,
) -> None:
    """Run async V7.6 with one continuous H100 feedback window across phases."""

    action_metadata = action_policy.get_server_metadata()
    classification_metadata = classification_policy.get_server_metadata()
    implementation.validate_server_metadata(args, action_metadata)
    implementation.validate_server_metadata(args, classification_metadata)
    if action_metadata != classification_metadata:
        raise ValueError("Action and asynchronous classification connections expose different metadata")
    args.use_state_history = bool(action_metadata.get("use_state_history", False))
    stats = implementation.base.load_state_quantiles(args.norm_stats_file)
    logger.record(
        {
            "event": "run_start",
            "server_metadata": action_metadata,
            "args": vars(args),
            "continuous_live_state_history": True,
            "async_adjustment_end": True,
            "rolling_qpos_buffer_frames": ROLLING_QPOS_BUFFER_FRAMES,
            "rolling_qpos_buffer_phases": ["execution", "adjustment"],
            "rolling_qpos_buffer_reset_on_phase_transition": False,
        }
    )
    print(
        "V7.6 async direction controls: "
        "a=left moderately, s=left slightly, d=right slightly, "
        "f=right moderately, q=quit; SPACE is disabled. "
        f"adjustment_end target rate={args.adjustment_end_rate_hz:g}Hz, "
        f"server threshold={action_metadata['adjustment_end_threshold']}."
    )
    if action_metadata.get("adjustment_end_experimental_override", False):
        print("WARNING: EXPERIMENTAL adjustment_end deployment is active.")

    phase: Phase = "execution"
    phase_indices = {"execution": 0, "adjustment": 0}
    generation = 0
    pending_observation: v52.FrozenObservation | None = None
    pre_action: np.ndarray | None = None
    published_steps = 0
    feedback_window: deque[tuple[np.ndarray, float]] = deque(maxlen=ROLLING_QPOS_BUFFER_FRAMES)
    previous_reported_submit: float | None = None
    last_submit: float | None = None

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="v7-6-adjustment-end") as executor:
        future: Future[Any] | None = None

        def consume_finished_request() -> bool:
            """Return True only when this result changed adjustment to execution."""

            nonlocal future, generation, last_submit, phase, previous_reported_submit
            if future is None or not future.done():
                return False
            result = future.result()
            future = None
            if result.generation != generation or phase != "adjustment":
                logger.record(
                    {
                        "event": "adjustment_end_async_stale_result",
                        "result_generation": result.generation,
                        "active_generation": generation,
                        "active_phase": phase,
                    }
                )
                return False
            implementation._report_async_adjustment_end(
                args=args,
                logger=logger,
                result=result,
                handled_step=published_steps,
                previous_submit_monotonic=previous_reported_submit,
            )
            previous_reported_submit = result.submitted_monotonic
            if not result.adjustment_end:
                return False
            logger.record(
                {
                    "event": "phase_transition",
                    "previous_phase": "adjustment",
                    "phase": "execution",
                    "trigger": "asynchronous_adjustment_end",
                    "captured_step": result.captured_step,
                    "handled_step": published_steps,
                    "reset_state_history": False,
                    "reset_rolling_qpos_buffer": False,
                }
            )
            print("[PHASE] adjustment_end=true; switching to EXECUTION before the next action.")
            phase = "execution"
            generation += 1
            last_submit = None
            return True

        def maybe_submit_request(*, phase_index: int) -> None:
            nonlocal future, last_submit
            now = time.monotonic()
            if not implementation.base.should_submit_adjustment_end(
                phase=phase,
                feedback_count=len(feedback_window),
                request_active=future is not None,
                now_monotonic=now,
                last_submit_monotonic=last_submit,
                target_rate_hz=args.adjustment_end_rate_hz,
            ):
                return
            recent = list(feedback_window)[-implementation.RUNTIME_PAST_QPOS_FRAMES :]
            qpos_h99 = [qpos.copy() for qpos, _ in recent]
            timestamps = [timestamp for _, timestamp in recent]
            future = executor.submit(
                implementation._run_async_adjustment_end_once,
                args=args,
                policy=classification_policy,
                operator=operator,
                captioner=captioner,
                stats=stats,
                generation=generation,
                phase_index=phase_index,
                captured_step=published_steps,
                feedback_qpos_h30=qpos_h99,
                feedback_timestamps=timestamps,
                submitted_monotonic=now,
            )
            last_submit = now
            logger.record(
                {
                    "event": "adjustment_end_async_submit",
                    "generation": generation,
                    "phase_index": phase_index,
                    "captured_step": published_steps,
                    "feedback_timestamp_first": timestamps[0],
                    "feedback_timestamp_last": timestamps[-1],
                    "feedback_window_phases": ["execution", "adjustment"],
                    "target_rate_hz": args.adjustment_end_rate_hz,
                }
            )

        while published_steps < args.max_publish_step and not operator.is_shutdown():
            consume_finished_request()
            signal_value = _poll_key(args, keyboard, phase=phase)
            if signal_value == "quit":
                implementation.runtime.shutdown_event.set()
                logger.record({"event": "operator_quit", "phase": phase, "published_steps": published_steps})
                return
            if signal_value == "trigger":
                pending_observation = implementation.base._enter_adjustment(
                    args=args,
                    operator=operator,
                    captioner=captioner,
                    logger=logger,
                    published_steps=published_steps,
                    discarded_raw_actions=0,
                )
                phase = "adjustment"
                generation += 1
                last_submit = None
                previous_reported_submit = None
                pre_action = pending_observation.qpos.copy()

            observation = pending_observation or v52._capture_observation(
                args,
                operator,
                captioner,
                reset_history=False,
            )
            pending_observation = None
            if pre_action is None:
                pre_action = observation.qpos.copy()
            requested_phase = phase
            phase_index = phase_indices[requested_phase]
            _, actions = v52._request_action_chunk(
                args=args,
                policy=action_policy,
                logger=logger,
                observation=observation,
                phase=requested_phase,
                phase_index=phase_index,
            )
            phase_indices[requested_phase] += 1

            if consume_finished_request() and requested_phase == "adjustment":
                logger.record(
                    {
                        "event": "execution_chunk",
                        "phase": requested_phase,
                        "phase_index": phase_index,
                        "completed_raw_actions": 0,
                        "discarded_raw_actions": len(actions),
                        "control_signal": "adjustment_end",
                    }
                )
                continue

            limit = min(args.chunk_size, args.max_publish_step - published_steps, len(actions))
            complete = 0
            control: str | None = None
            control_rate = operator.rate(args.publish_rate)
            for action_index, action in enumerate(actions[:limit]):
                if consume_finished_request():
                    control = "adjustment_end"
                    break
                _, before_timestamp = v52._latest_feedback(operator)
                pre_action, _, raw_control = v52._publish_raw_action(
                    args=args,
                    operator=operator,
                    keyboard=keyboard,
                    logger=logger,
                    phase=requested_phase,
                    phase_index=phase_index,
                    action_index=action_index,
                    raw_action=action,
                    pre_action=pre_action,
                    control_rate=control_rate,
                )
                if raw_control is not None:
                    control = raw_control.kind
                    break
                feedback, timestamp = implementation.v53._wait_feedback_after(args, operator, before_timestamp)
                complete += 1
                published_steps += 1
                feedback_window.append((feedback.copy(), float(timestamp)))

                if consume_finished_request():
                    control = "adjustment_end"
                    break
                if phase == "adjustment":
                    maybe_submit_request(phase_index=phase_index)

            logger.record(
                {
                    "event": "execution_chunk",
                    "phase": requested_phase,
                    "phase_index": phase_index,
                    "completed_raw_actions": complete,
                    "discarded_raw_actions": max(0, limit - complete),
                    "control_signal": control,
                }
            )
            if control == "quit":
                implementation.runtime.shutdown_event.set()
                return
            if control == "trigger":
                if requested_phase != "execution":
                    raise AssertionError("direction recovery keys are valid only in execution")
                pending_observation = implementation.base._enter_adjustment(
                    args=args,
                    operator=operator,
                    captioner=captioner,
                    logger=logger,
                    published_steps=published_steps,
                    discarded_raw_actions=max(0, limit - complete),
                )
                phase = "adjustment"
                generation += 1
                last_submit = None
                previous_reported_submit = None
                pre_action = pending_observation.qpos.copy()
            elif control == "adjustment_end":
                if phase != "execution":
                    raise AssertionError("adjustment_end must transition to execution")

        logger.record({"event": "max_publish_step_reached", "published_steps": published_steps})


def main() -> None:
    original_validate = implementation.base.validate_args

    def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        _keyboard_selection_validate_args(original_validate, args, parser)

    implementation.DEFAULT_LOG_ROOT = DEFAULT_LOG_ROOT
    # V7.6 reuses the V7.5 online H100 construction, but the server identity is
    # the counterfactual V7.6 classifier profile.
    implementation.DATA_PROFILE = V7_6_DATA_PROFILE
    implementation.base._poll_key = _poll_key
    implementation.v52._poll_control_key = _poll_control_key
    implementation.base.validate_args = validate_args
    implementation.base.run_v5_3_async = run_v7_5_async_direction_keys
    print(
        "V7.6 async direction controls: "
        "a=left moderately, s=left slightly, d=right slightly, "
        "f=right moderately, q=quit; SPACE is disabled."
    )
    implementation.main()


if __name__ == "__main__":
    main()
