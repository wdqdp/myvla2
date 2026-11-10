#!/usr/bin/env python3
"""Book V9.7 asynchronous direct-tactile recovery, without recording.

Action generation and phase assessment use separate WebSocket connections.
The phase worker applies a true decision as soon as its first streamed event
arrives, so the current H30 chunk is stopped without waiting for failure or
plan text generation.
"""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, NamedTuple

import numpy as np
import websockets.sync.client

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(OPENPI_ROOT / "src"),
                str(PROJECT_ROOT / "openpi/packages/openpi-client/src")]

import agilex_inference_forced_phase_anlation as v52
import agilex_inference_forced_phase_anlation_5_3 as v53
import agilex_inference_tactile_vla_sync_single as runtime
from openpi_client import msgpack_numpy
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.book_v9_7_runtime import BOOK_INSTRUCTION
from tactile_vla.vla.book_v9_7_runtime import DEFAULT_CAPTIONER
from tactile_vla.vla.book_v9_7_runtime import DEFAULT_NORM_DIR
from tactile_vla.vla.book_v9_7_runtime import validate_captioner
from tactile_vla.vla.book_v9_7_runtime import validate_server_metadata as validate_book_metadata
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v5_3_phase_change import discretize_state_qpos
from tactile_vla.vla.v7_7_async_state import AsyncPhaseState
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt
from tactile_vla.vla.v7_11_data import direct_recovery_prompt
from tactile_vla.vla.v7_11_data import phase_prompt_without_plan

DEFAULT_NORM_STATS = DEFAULT_NORM_DIR / "norm_stats.json"
ACTION_PROMPT_PROFILE = "phase_v2"
ACTION_HORIZON = 30
ACTION_DIM = 32
_caption_capture_lock = threading.Lock()


def capture_tactile_observation(args, operator, captioner, *, after_timestamp):
    """Serialize the shared captioner and freeze a fresh observation for each request."""
    with _caption_capture_lock:
        return v53._capture_classification_observation(
            args, operator, captioner, after_timestamp=after_timestamp,
        )


class StreamingPhaseClient:
    """Single-decision WebSocket client used only by phase assessment."""

    def __init__(self, host: str, port: int, *, timeout: float = 30.0):
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None,
            open_timeout=timeout, close_timeout=timeout,
        )
        self._timeout = timeout
        self._packer = msgpack_numpy.Packer()
        self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=timeout))
        validate_book_metadata(self.metadata)
        if not self.metadata.get("supports_streamed_phase_events"):
            raise ValueError("phase server does not support streamed phase events")
        self._lock = threading.Lock()

    def events(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield the single classification decision under one lock."""
        with self._lock:
            self._ws.send(self._packer.pack(payload))
            decision = msgpack_numpy.unpackb(self._ws.recv(timeout=self._timeout))
            if decision.get("event") != "phase_decision":
                raise ValueError(f"expected phase_decision, got {decision.get('event')!r}")
            if decision.get("request_id") != payload.get("request_id"):
                raise ValueError("phase decision request_id mismatch")
            phase = payload.get("phase")
            if phase not in ("execution", "adjustment") or decision.get("phase") != phase:
                raise ValueError("phase decision phase mismatch")
            field = "need_recovery" if phase == "execution" else "adjustment_end"
            probs = np.asarray(decision.get(field + "_probs"), dtype=np.float64)
            if (probs.shape != (2,) or not np.isfinite(probs).all()
                    or np.any(probs < 0) or np.any(probs > 1) or not np.isclose(probs.sum(), 1)
                    or type(decision.get(field)) is not bool
                    or decision[field] != bool(probs[1] >= float(self.metadata[field + "_threshold"]))):
                raise ValueError("invalid phase decision probabilities or threshold result")
            yield decision

    def close(self):
        self._ws.close()


class BookV97AsyncControlGate:
    """Thread-safe bridge between assessment events and the action publication loop."""

    def __init__(self, *, hold: Callable[[], None]):
        self.state = AsyncPhaseState()
        self._hold = hold
        self._lock = threading.Lock()

    def handle_phase_event(self, event: dict[str, Any]) -> bool:
        with self._lock:
            if event.get("event") == "phase_decision":
                triggered = self.state.apply_phase_decision(event)
                if triggered:
                    # Called in the assessment worker as soon as the first WS
                    # event arrives; action publication also observes the latch.
                    self._hold()
                return triggered
            raise ValueError(f"unknown Book V9.7 event {event.get('event')!r}")

    def may_publish_action(self, generation: int) -> bool:
        with self._lock:
            if self.state.stop_latched or generation != self.state.action_generation:
                self._hold()
                return False
            return True

    def try_publish_action(self, action: Any, *, generation: int,
                           publish: Callable[[Any], None]) -> bool:
        """Atomically check the stop latch and issue one controller command."""
        with self._lock:
            if self.state.stop_latched or generation != self.state.action_generation:
                self._hold()
                return False
            publish(action)
            return True

    def begin_phase_request(self, request_id: str) -> tuple[int, str]:
        with self._lock:
            return self.state.begin_phase_request(request_id)

    def begin_action_request(self) -> int:
        with self._lock:
            return self.state.begin_action_request()

    def accept_action_result(self, generation: int, actions: Any) -> bool:
        with self._lock:
            return self.state.accept_action_result(generation, actions)

    def switch_phase(self, phase: str, *, increment_attempt: bool = False) -> None:
        with self._lock:
            self.state.switch_phase(phase, increment_attempt=increment_attempt)

    def release_hold_with_fresh_actions(self, generation: int, actions: Any) -> bool:
        with self._lock:
            return self.state.release_hold_with_fresh_actions(generation, actions)

    def snapshot(self) -> tuple[str, int, int, int, bool]:
        with self._lock:
            return (
                self.state.phase, self.state.attempt_id, self.state.action_generation,
                self.state.phase_generation, self.state.stop_latched,
            )

    def control_tick(self) -> None:
        with self._lock:
            if self.state.stop_latched:
                self._hold()

    def publish_chunk(self, actions, *, generation: int,
                      publish: Callable[[Any], None], before_each: Callable[[], None] | None = None) -> int:
        """Publish only while the generation is current and no stop is latched.

        ``before_each`` is where the ROS loop drains phase events.  Therefore a
        true decision received at any H30 position prevents that and every
        later action from reaching the publisher.
        """
        published = 0
        for action in actions:
            if before_each is not None:
                before_each()
            if not self.may_publish_action(generation):
                break
            publish(action)
            published += 1
        return published


class PhaseAssessmentResult(NamedTuple):
    request_id: str
    phase: str
    generation: int
    attempt_id: int
    captured_step: int
    observation: v52.FrozenObservation
    synchronized_timestamps: dict[str, float]
    prompt: str
    qpos_h100_11_discrete: list[list[int]]
    events: list[dict[str, Any]]
    submitted_monotonic: float
    finished_monotonic: float

    @property
    def decision(self) -> dict[str, Any]:
        return self.events[0]

    @property
    def triggered(self) -> bool:
        key = "need_recovery" if self.phase == "execution" else "adjustment_end"
        return bool(self.decision.get(key))


def validate_server_metadata(metadata: dict[str, Any]) -> None:
    validate_book_metadata(metadata)


def _run_phase_assessment(
    *, args: argparse.Namespace, client: StreamingPhaseClient, operator: Any,
    captioner: Any, stats: StateQuantileStats, gate: BookV97AsyncControlGate,
    phase: str, generation: int, attempt_id: int, request_id: str,
    captured_step: int, after_timestamp: float,
    recovery_plan: str, submitted_monotonic: float,
) -> PhaseAssessmentResult:
    observation, timestamps = capture_tactile_observation(
        args, operator, captioner, after_timestamp=after_timestamp,
    )
    dense_h100, _ = operator.state_history.snapshot(
        current_timestamp=float(observation.timestamp), current_state=observation.qpos,
    )
    if dense_h100.shape != (100, 7):
        raise ValueError(f"runtime H100 must be [100,7], got {dense_h100.shape}")
    discrete = discretize_state_qpos(dense_h100[np.linspace(0, 99, 11, dtype=np.int64)], stats)
    prompt = phase_prompt_without_plan(build_phase_prompt(
        instruction=args.instruction,
        tactile_caption=observation.tactile_caption,
        recovery_plan=recovery_plan,
        qpos_h100_11_discrete=discrete,
    ))
    payload = runtime.build_payload(
        mode="phase", img_front_bgr=observation.img_front,
        img_left_bgr=observation.img_left, qpos=observation.qpos,
        state_history=None, state_history_mask=None, prompt=prompt,
    )
    payload.update({
        "phase": phase, "request_id": request_id, "attempt_id": attempt_id,
        "generation": generation, "action_step": captured_step,
    })
    events = []
    for event in client.events(payload):
        gate.handle_phase_event(event)
        events.append(event)
    if not events:
        raise ValueError("Book V9.7 phase request returned no events")
    return PhaseAssessmentResult(
        request_id=request_id, phase=phase, generation=generation,
        attempt_id=attempt_id, captured_step=captured_step,
        observation=observation, synchronized_timestamps=timestamps,
        prompt=prompt, qpos_h100_11_discrete=discrete.tolist(), events=events,
        submitted_monotonic=submitted_monotonic,
        finished_monotonic=time.monotonic(),
    )


def _request_action_chunk(
    *, args: argparse.Namespace, policy: v53.TimeoutWebsocketPolicy,
    observation: v52.FrozenObservation, phase: str, phase_index: int,
    recovery_plan: str, logger: v52.TrialLogger,
) -> np.ndarray:
    prompt = (
        direct_recovery_prompt(args.instruction, observation.tactile_caption)
        if phase == "adjustment" else v52.build_phase_prompt(
            phase="execution", instruction=args.instruction, recovery_plan="",
            prompt_profile=ACTION_PROMPT_PROFILE,
        )
    )
    noise = v52.deterministic_action_noise(args.noise_seed, phase=phase, index=phase_index)
    payload = runtime.build_payload(
        mode="execution", img_front_bgr=observation.img_front,
        img_left_bgr=observation.img_left, qpos=observation.qpos,
        state_history=None, state_history_mask=None, prompt=prompt,
    )
    payload.update({
        "action_noise": noise, "noise_seed": args.noise_seed,
        "noise_phase": phase, "noise_index": phase_index,
    })
    started = time.perf_counter()
    response = policy.infer_with_timeout(payload, timeout=args.request_timeout_seconds)
    actions = np.asarray(response.get("actions"), dtype=np.float32)
    raw = np.asarray(response.get("raw_model_actions"), dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, 7) or not np.isfinite(actions).all():
        raise ValueError(f"Book V9.7 server returned invalid actions {actions.shape}")
    if raw.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(raw).all():
        raise ValueError(f"Book V9.7 server returned invalid raw_model_actions {raw.shape}")
    logger.record({
        "event": "action_chunk_inference", "phase": phase,
        "phase_index": phase_index, "prompt": prompt,
        "recovery_plan": recovery_plan, "tactile_caption": observation.tactile_caption,
        "observation_timestamp": observation.timestamp, "qpos": observation.qpos,
        "noise_seed": args.noise_seed, "noise_sha256": v52.noise_sha256(noise),
        "raw_model_actions": raw, "transformed_actions": actions,
        "client_infer_ms": (time.perf_counter() - started) * 1000.0,
        "server_infer_ms": response.get("policy_timing", {}).get("infer_ms"),
    })
    print(f"[{phase.upper()}] action chunk={phase_index}")
    return actions


def _wait_future_while_holding(
    future: Future, *, gate: BookV97AsyncControlGate, operator: Any, publish_rate: int,
):
    rate = operator.rate(publish_rate)
    while not future.done() and not operator.is_shutdown() and not runtime.shutdown_event.is_set():
        gate.control_tick()
        rate.sleep()
    if not future.done():
        raise RuntimeError("stopped while waiting for Book V9.7 inference")
    return future.result()


def _phase_result_log(result: PhaseAssessmentResult, *, handled_step: int) -> dict[str, Any]:
    decision = result.decision
    return {
        "event": "phase_assessment", "request_id": result.request_id,
        "phase": result.phase, "generation": result.generation,
        "attempt_id": result.attempt_id, "captured_step": result.captured_step,
        "handled_step": handled_step,
        "lag_steps": max(0, handled_step - result.captured_step),
        "prompt": result.prompt,
        "qpos_h100_11_discrete": result.qpos_h100_11_discrete,
        "tactile_caption": result.observation.tactile_caption,
        "synchronized_timestamps": result.synchronized_timestamps,
        "need_recovery": decision.get("need_recovery"),
        "need_recovery_probs": decision.get("need_recovery_probs"),
        "adjustment_end": decision.get("adjustment_end"),
        "adjustment_end_probs": decision.get("adjustment_end_probs"),
        "total_async_ms": (result.finished_monotonic - result.submitted_monotonic) * 1000.0,
    }


def run_book_v9_7_async(
    args: argparse.Namespace, operator: Any, action_policy: v53.TimeoutWebsocketPolicy,
    phase_client: StreamingPhaseClient, captioner: Any, keyboard: Any,
    logger: v52.TrialLogger,
) -> None:
    action_metadata = action_policy.get_server_metadata()
    validate_server_metadata(action_metadata)
    if args.instruction != BOOK_INSTRUCTION:
        raise ValueError("Book V9.7 requires its training instruction")
    validate_captioner(captioner)
    if action_metadata != phase_client.metadata:
        raise ValueError("Book V9.7 action and phase connections expose different metadata")
    args.use_state_history = False
    stats = load_state_quantiles(args.norm_stats_file)
    recovery_plan = ""
    phase_indices = {"execution": 0, "adjustment": 0}
    published_steps = 0
    phase_request_count = 0
    last_phase_submit = {"execution": None, "adjustment": None}
    last_feedback_timestamp: float | None = None
    hold_target: list[np.ndarray | None] = [None]

    def hold() -> None:
        target = hold_target[0]
        if target is None:
            target, _ = v52._latest_feedback(operator)
        if target is None:
            return
        command = np.asarray(target, dtype=np.float32).copy()
        command[6] = max(float(args.gripper_min), float(command[6]))
        if args.no_publish:
            return
        operator.puppet_arm_publish(command)

    gate = BookV97AsyncControlGate(hold=hold)
    logger.record({"event": "run_start", "server_metadata": action_metadata,
                   "args": vars(args), "continuous_h100": True})
    print(
        "Book V9.7 async started: need and adjustment checks run independently of action chunks; "
        f"need={args.need_recovery_rate_hz:g}Hz adjustment={args.adjustment_end_rate_hz:g}Hz. "
        f"{args.success_key}=success, {args.quit_key}=quit."
    )

    if action_metadata.get("norm_stats_sha256") != sha256_file(args.norm_stats_file):
        raise ValueError("Book V9.7 client/server norm stats differ")
    v53._wait_for_tactile_ready(args, operator, captioner)
    # This is the only H100 reset: phase and attempt transitions preserve it.
    operator.reset_state_history()
    # The first execution action is visual/state-only; adjustment actions use touch.
    initial = v52._capture_observation(args, operator, None, reset_history=False)
    if initial.timestamp is None:
        raise ValueError("initial ROS qpos does not have a timestamp")
    operator.state_history.push(float(initial.timestamp), initial.qpos)
    hold_target[0] = initial.qpos.copy()
    last_feedback_timestamp = initial.timestamp
    pending_observation: v52.FrozenObservation | None = initial

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="book-v9-7-inference") as executor:
        phase_future: Future[PhaseAssessmentResult] | None = None

        def maybe_submit_phase() -> None:
            nonlocal phase_future, phase_request_count
            phase, attempt_id, _, phase_generation, stopped = gate.snapshot()
            if stopped or phase_future is not None or last_feedback_timestamp is None:
                return
            rate_hz = (args.need_recovery_rate_hz if phase == "execution"
                       else args.adjustment_end_rate_hz)
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
            nonlocal phase_future, recovery_plan, pending_observation
            if phase_future is None:
                return False, False
            if not phase_future.done() and not wait:
                return False, False
            result = (_wait_future_while_holding(
                phase_future, gate=gate, operator=operator, publish_rate=args.publish_rate,
            ) if wait else phase_future.result())
            phase_future = None
            logger.record(_phase_result_log(result, handled_step=published_steps))
            field = "need_recovery_probs" if result.phase == "execution" else "adjustment_end_probs"
            print(f"[{result.phase.upper()}] touch={result.observation.tactile_caption} "
                  f"{field}={result.decision.get(field)}", flush=True)
            if result.phase == "adjustment":
                adjustment_probs = np.asarray(
                    result.decision.get("adjustment_end_probs"), dtype=np.float64,
                ).reshape(-1).tolist()
                print(
                    "[ADJUSTMENT_END] "
                    f"result={result.triggered} probs={adjustment_probs} "
                    f"threshold={float(action_metadata['adjustment_end_threshold']):g}"
                )
            active_phase, active_attempt, _, active_generation, _ = gate.snapshot()
            if (result.phase, result.attempt_id, result.generation) != (
                active_phase, active_attempt, active_generation
            ):
                logger.record({"event": "stale_phase_result", "request_id": result.request_id})
                return False, False
            if not result.triggered:
                return False, False
            if result.phase == "adjustment":
                print("[PHASE] adjustment_end=true; holding and switching to EXECUTION.")
                gate.switch_phase("execution")
                last_phase_submit["execution"] = None
                pending_observation = None
                logger.record({"event": "phase_transition", "previous_phase": "adjustment",
                               "phase": "execution", "trigger": "adjustment_end",
                               "attempt_id": active_attempt, "published_steps": published_steps})
                return True, False

            if active_attempt >= args.max_attempts:
                print("need_recovery=true at max_attempts; remaining stopped.")
                return False, True
            print("[PHASE] need_recovery=true; holding and switching to ADJUSTMENT.")
            gate.switch_phase("adjustment", increment_attempt=True)
            last_phase_submit["adjustment"] = None
            pending_observation = None
            logger.record({"event": "phase_transition", "previous_phase": "execution",
                           "phase": "adjustment", "trigger": "need_recovery",
                           "attempt_id": active_attempt + 1, "published_steps": published_steps})
            return True, False

        while published_steps < args.max_publish_step and not operator.is_shutdown():
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
            if phase == "adjustment":
                observation, _ = capture_tactile_observation(
                    args, operator, captioner, after_timestamp=float(last_feedback_timestamp),
                )
            else:
                observation = pending_observation or v52._capture_observation(
                    args, operator, None, reset_history=False,
                )
            pending_observation = None
            hold_target[0] = observation.qpos.copy()
            phase_index = phase_indices[phase]
            action_generation = gate.begin_action_request()
            actions = _request_action_chunk(
                args=args, policy=action_policy, observation=observation,
                phase=phase, phase_index=phase_index, recovery_plan=recovery_plan,
                logger=logger,
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
                key = keyboard.get_key()
                if key == args.quit_key:
                    runtime.shutdown_event.set()
                    logger.record({"event": "operator_quit", "phase": phase,
                                   "published_steps": published_steps})
                    return
                if key == args.success_key:
                    logger.record({"event": "operator_success", "phase": phase,
                                   "published_steps": published_steps})
                    print("Operator confirmed success.")
                    return
                raw_action = np.asarray(action_value, dtype=np.float32)
                command = raw_action.copy()
                offset, command[6], floor_applied = v52.gripper_command_with_safety_floor(
                    float(command[6]), gripper_offset=args.gripper_offset,
                    gripper_min=args.gripper_min,
                )
                _, before_timestamp = v52._latest_feedback(operator)
                command_timestamp: list[float | None] = [None]

                def publish(value, timestamp_slot=command_timestamp):
                    if not args.no_publish:
                        timestamp_slot[0] = operator.puppet_arm_publish(value)

                if not gate.try_publish_action(command, generation=action_generation, publish=publish):
                    break
                control_rate.sleep()
                feedback, feedback_timestamp = v53._wait_feedback_after(
                    args, operator, before_timestamp,
                )
                hold_target[0] = feedback.copy()
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--instruction", default=BOOK_INSTRUCTION)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--norm-stats-file", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--phase-change-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--need-recovery-rate-hz", type=float, default=7.0)
    parser.add_argument("--adjustment-end-rate-hz", type=float, default=7.0)
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--publish_rate", type=int, default=20)
    parser.add_argument("--observation-poll-rate", type=int, default=200)
    parser.add_argument("--state-history-len", type=int, default=100)
    parser.add_argument("--state-history-fps", type=float, default=30.0)
    parser.add_argument("--state-history-max-gap-seconds", type=float, default=0.02)
    parser.add_argument("--img_front_topic", default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", default="/camera_l/color/image_raw")
    parser.add_argument("--puppet_arm_cmd_topic", default="/master/joint_right")
    parser.add_argument("--puppet_arm_topic", default="/puppet/joint_right")
    parser.add_argument("--tactile_left_force_topic", default="/xense/OG001251/force")
    parser.add_argument("--tactile_right_force_topic", default="/xense/OG000991/force")
    parser.add_argument("--tactile_left_mesh_3d_topic", default="/xense/OG001251/mesh_3d")
    parser.add_argument("--tactile_right_mesh_3d_topic", default="/xense/OG000991/mesh_3d")
    parser.add_argument("--tactile_left_mesh_3d_flow_topic", default="/xense/OG001251/mesh_3d_flow")
    parser.add_argument("--tactile_right_mesh_3d_flow_topic", default="/xense/OG000991/mesh_3d_flow")
    parser.add_argument("--tactile_window_size", type=int, default=30)
    parser.add_argument("--captioner_checkpoint", type=Path, default=DEFAULT_CAPTIONER)
    parser.add_argument("--captioner_device", default="auto")
    parser.add_argument("--no-captioner", action="store_true")
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--success-key", default="s")
    parser.add_argument("--quit-key", default="q")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--gripper_offset", type=float, default=0.001)
    parser.add_argument("--gripper-min", dest="gripper_min", type=float, required=True)
    parser.add_argument(
        "--classification-gripper-open-threshold",
        type=float,
        help="Phase-model qpos[6] values above this threshold are replaced",
    )
    parser.add_argument(
        "--classification-gripper-open-value",
        type=float,
        help="Replacement value for open-gripper qpos[6] in phase-model inputs",
    )
    args = parser.parse_args(argv)
    return args, parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.noise_seed < 0:
        parser.error("--noise-seed must be non-negative")
    if args.instruction != BOOK_INSTRUCTION:
        parser.error("Book V9.7 requires its original book instruction")
    if args.no_captioner:
        parser.error("Book V9.7 phase inference requires the tactile captioner")
    if not args.norm_stats_file.is_file():
        parser.error(f"--norm-stats-file does not exist: {args.norm_stats_file}")
    if not 1 <= args.max_attempts <= 10:
        parser.error("--max-attempts must be in [1,10]")
    if args.tactile_window_size != 30 or args.state_history_fps != 30.0:
        parser.error("Book V9.7 requires the W30 captioner and a 30 Hz H100 history grid")
    if not 1 <= args.chunk_size <= ACTION_HORIZON:
        parser.error(f"--chunk_size must be in [1,{ACTION_HORIZON}]")
    if args.max_publish_step <= 0 or args.publish_rate <= 0 or args.observation_poll_rate <= 0:
        parser.error("publish limits and rates must be positive")
    if args.state_history_len != 100 or args.state_history_fps <= 0 or args.state_history_max_gap_seconds <= 0:
        parser.error("runtime ROS history must use len=100 with positive fps/max-gap")
    if not 0.0 < args.need_recovery_rate_hz <= args.publish_rate:
        parser.error("--need-recovery-rate-hz must be in (0,publish_rate]")
    if not 0.0 < args.adjustment_end_rate_hz <= args.publish_rate:
        parser.error("--adjustment-end-rate-hz must be in (0,publish_rate]")
    if args.request_timeout_seconds <= 0 or args.phase_change_timeout_seconds <= 0:
        parser.error("inference timeouts must be positive")
    if not np.isfinite(args.gripper_offset) or args.gripper_offset < 0:
        parser.error("--gripper_offset must be finite and non-negative")
    if not np.isfinite(args.gripper_min) or not 0.0 <= args.gripper_min <= 0.08:
        parser.error("--gripper-min must be finite and in [0,0.08]")
    if (args.classification_gripper_open_threshold is not None
            or args.classification_gripper_open_value is not None):
        parser.error("Book V9.7 uses raw qpos; classification gripper remapping is forbidden")
    if len(args.success_key) != 1 or len(args.quit_key) != 1 or args.success_key == args.quit_key:
        parser.error("success/quit keys must be distinct single characters")


def main() -> None:
    args, parser = get_arguments()
    runtime.apply_yaml_defaults(args, parser)
    validate_args(args, parser)
    if not sys.stdin.isatty():
        parser.error("Book V9.7 real-robot inference requires an interactive TTY")
    runtime.shutdown_event.clear()
    signal.signal(signal.SIGINT, runtime._on_sigint)
    captioner = runtime.load_captioner(args)
    validate_captioner(captioner)
    print(
        f"[CAPTIONER] {args.captioner_checkpoint} sha256={sha256_file(args.captioner_checkpoint)} "
        f"schema={captioner.schema_version}",
        flush=True,
    )
    action_policy = v53.TimeoutWebsocketPolicy(args.host, args.port)
    phase_client = StreamingPhaseClient(args.host, args.port, timeout=args.request_timeout_seconds)
    operator = runtime.RosOperator(args)
    v53._install_tactile_arrival_timestamps(operator)
    class NoRecording:
        def record(self, event):
            pass

    logger = NoRecording()
    try:
        if not args.start_immediately:
            input("Press enter to start Book V9.7 EXECUTION")
        with runtime.KeyboardPoller() as keyboard:
            run_book_v9_7_async(
                args, operator, action_policy, phase_client, captioner, keyboard, logger,
            )
    except Exception as exc:
        logger.record({"event": "fail_closed_safety_stop", "error": repr(exc)})
        feedback, _ = v52._latest_feedback(operator)
        if feedback is not None and not args.no_publish:
            hold = np.asarray(feedback, dtype=np.float32).copy()
            hold[6] = max(float(args.gripper_min), float(hold[6]))
            operator.puppet_arm_publish(hold)
        print(f"FAIL-CLOSED safety stop: {exc}. No more model actions will be published.")
        raise
    finally:
        phase_client.close()
        close = getattr(getattr(action_policy, "_ws", None), "close", None)
        if callable(close):
            close()


__all__ = [
    "BookV97AsyncControlGate",
    "PhaseAssessmentResult",
    "StreamingPhaseClient",
    "run_book_v9_7_async",
    "validate_server_metadata",
]


if __name__ == "__main__":
    main()
