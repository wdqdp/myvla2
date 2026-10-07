#!/usr/bin/env python3
"""Executable Book V9.4 asynchronous single-arm ROS inference.

Version-specific client adapted from V7.7; the original source is unchanged.
Recovery attempts are unlimited; memory retains the initial pair and latest three.

Action generation and phase assessment use separate WebSocket connections.
The phase worker applies a true decision as soon as its first streamed event
arrives, so the current H30 chunk is stopped without waiting for failure or
plan text generation.
"""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from inspect import signature
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
from tactile_vla.common.labels_v4 import LABEL_FIELDS
from tactile_vla.common.labels_v4 import LABEL_SCHEMA_VERSION
from tactile_vla.common.labels_v4 import neutral_caption
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.book_v9_4_memory import append_runtime_memory
from tactile_vla.vla.book_v9_4_multitask_data import DATA_PROFILE
from tactile_vla.vla.book_v9_4_runtime import BOOK_INSTRUCTION
from tactile_vla.vla.book_v9_4_runtime import DEFAULT_CAPTIONER
from tactile_vla.vla.book_v9_4_runtime import DEFAULT_NORM_DIR
from tactile_vla.vla.book_v9_4_runtime import HISTORY_OFFSETS
from tactile_vla.vla.book_v9_4_runtime import validate_server_metadata as validate_book_server_metadata
from tactile_vla.vla.book_v9_4_runtime import validate_tactile_caption
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE
from tactile_vla.vla.prompts import build_recovery_prompt
from tactile_vla.vla.structured_text import legal_failure_reasons
from tactile_vla.vla.structured_text import legal_recovery_plans
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v5_3_phase_change import discretize_state_qpos
from tactile_vla.vla.v7_7_async_state import AsyncPhaseState
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt

DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/book_v9_4_async"
DEFAULT_NORM_STATS = DEFAULT_NORM_DIR / "norm_stats.json"
ACTION_PROMPT_PROFILE = "phase_v2"
ACTION_HORIZON = 30
ACTION_DIM = 32
DIRECT_CONNECT_OPTIONS = {"proxy": None} if "proxy" in signature(websockets.sync.client.connect).parameters else {}


class DirectActionClient(v53.TimeoutWebsocketPolicy):
    """Direct LAN connection with bounded handshake, without an environment proxy."""

    def __init__(self, host: str, port: int, *, timeout: float):
        self._connection_timeout = timeout
        super().__init__(host, port)

    def _wait_for_server(self):
        connection = websockets.sync.client.connect(
            self._uri, compression=None, max_size=None,
            open_timeout=self._connection_timeout, close_timeout=min(1.0, self._connection_timeout),
            **DIRECT_CONNECT_OPTIONS,
        )
        try:
            metadata = msgpack_numpy.unpackb(connection.recv(timeout=self._connection_timeout))
        except Exception:
            connection.close()
            raise
        return connection, metadata


class StreamingPhaseClient:
    """One-request/two-event WebSocket client used only by phase assessment."""

    def __init__(self, host: str, port: int, *, timeout: float = 30.0):
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None,
            open_timeout=timeout, close_timeout=min(1.0, timeout),
            **DIRECT_CONNECT_OPTIONS,
        )
        self._timeout = timeout
        self._packer = msgpack_numpy.Packer()
        self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=timeout))
        if self.metadata.get("data_profile") != DATA_PROFILE:
            raise ValueError("phase server is not a Book V9.4 checkpoint")
        if not self.metadata.get("supports_streamed_phase_events"):
            raise ValueError("phase server does not support streamed phase events")
        self._lock = threading.Lock()

    def events(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield decision first and conditional failure second under one lock."""
        with self._lock:
            self._ws.send(self._packer.pack(payload))
            decision = msgpack_numpy.unpackb(self._ws.recv(timeout=self._timeout))
            if decision.get("event") != "phase_decision":
                raise ValueError(f"expected phase_decision, got {decision.get('event')!r}")
            if decision.get("request_id") != payload.get("request_id"):
                raise ValueError("phase decision request_id mismatch")
            phase = payload.get("phase")
            if decision.get("phase") != phase or phase not in {"execution", "adjustment"}:
                raise ValueError("phase decision phase mismatch")
            task = "need_recovery" if phase == "execution" else "adjustment_end"
            probabilities = np.asarray(decision.get(f"{task}_probs"), dtype=np.float64)
            if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0) or np.any(probabilities > 1)
                    or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)):
                raise ValueError(f"Invalid streamed {task} probabilities")
            if (not isinstance(decision.get(task), bool)
                    or decision[task] != bool(probabilities[1] >= float(self.metadata[f"{task}_threshold"]))):
                raise ValueError(f"Streamed {task} decision differs from advertised threshold")
            yield decision
            if phase == "execution" and decision[task]:
                failure = msgpack_numpy.unpackb(self._ws.recv(timeout=self._timeout))
                if failure.get("event") != "failure_reason" or failure.get("request_id") != decision.get("request_id"):
                    raise ValueError("conditional failure event does not match its need decision")
                yield failure

    def close(self):
        self._ws.close()


class BookV94AsyncControlGate:
    """Serialize action commands, immediate holds and permanent shutdown."""

    def __init__(self, *, hold: Callable[[], None], reset_hold: Callable[[], None] | None = None):
        self.state = AsyncPhaseState()
        self._hold = hold
        self._lock = threading.Lock()
        self._reset_hold = reset_hold or (lambda: None)
        self._terminal_stop = False

    def stop(self) -> None:
        with self._lock:
            if not self._terminal_stop:
                self._terminal_stop = True
                self.state.stop_latched = True
                self.state.action_generation += 1
                self.state.pending_actions.clear()
                self.state.accepted_phase_request_id = None
                self._reset_hold()
            self._hold()

    def handle_phase_event(self, event: dict[str, Any]) -> bool:
        with self._lock:
            if self._terminal_stop:
                return False
            if event.get("event") == "phase_decision":
                triggered = self.state.apply_phase_decision(event)
                if triggered:
                    self._reset_hold()
                    # Called in the assessment worker as soon as the first WS
                    # event arrives; action publication also observes the latch.
                    self._hold()
                return triggered
            if event.get("event") == "failure_reason":
                return self.state.accept_failure_event(event)
            raise ValueError(f"unknown Book V9.4 event {event.get('event')!r}")

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
            self._reset_hold()
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
            if self._terminal_stop:
                return False
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


class ContinuousH100:
    """Episode-continuous runtime qpos buffer; phase/attempt changes do not reset it."""

    def __init__(self):
        self._values: deque[np.ndarray] = deque(maxlen=100)

    def reset_episode(self):
        self._values.clear()

    def append(self, qpos):
        value = np.asarray(qpos, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("runtime qpos must be finite [7]")
        self._values.append(value.copy())

    def snapshot(self) -> list[np.ndarray]:
        return [value.copy() for value in self._values]

    @classmethod
    def from_values(cls, values) -> ContinuousH100:
        result = cls()
        for value in values:
            result.append(value)
        return result

    def sampled_discrete(self, stats: StateQuantileStats) -> np.ndarray:
        if not self._values:
            raise ValueError("H100 is empty")
        values = list(self._values)
        padded = [values[0]] * (100 - len(values)) + values
        offsets = np.linspace(0, 99, 11, dtype=np.int64)
        return discretize_state_qpos(np.stack([padded[int(i)] for i in offsets]), stats)

    def prompt(self, *, instruction, tactile_caption, recovery_plan, stats):
        return build_phase_prompt(
            instruction=instruction, tactile_caption=tactile_caption,
            recovery_plan=recovery_plan,
            qpos_h100_11_discrete=self.sampled_discrete(stats),
        )


def phase_payload(*, phase: str, request_id: str, attempt_id: int, generation: int,
                  action_step: int, prompt: str, image, wrist_image, qpos) -> dict[str, Any]:
    if phase not in {"execution", "adjustment"}:
        raise ValueError(phase)
    return {
        "mode": "phase", "phase": phase, "request_id": request_id,
        "attempt_id": int(attempt_id), "generation": int(generation),
        "action_step": int(action_step), "prompt": prompt,
        "observation/image": image, "observation/wrist_image": wrist_image,
        "observation/state": np.asarray(qpos, dtype=np.float32),
    }


def plan_payload_after_failure(*, instruction: str, tactile_caption: str,
                               executed_recovery_plan: str, failure_reason: str,
                               memory: list[dict[str, str]], image, wrist_image, qpos
                               ) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Append the current real failure, retaining initial plus latest three pairs."""
    if failure_reason not in legal_failure_reasons():
        raise ValueError("Invalid failure_reason in runtime memory")
    updated = append_runtime_memory(memory, {
        "recovery_plan": str(executed_recovery_plan).strip() or "initial plan",
        "failure_reason": failure_reason,
    })
    prompt = build_recovery_prompt(
        instruction=instruction, failed_tactile_caption=tactile_caption,
        failure_recovery_memory=updated, prompt_profile=MINIMAL_PROMPT_PROFILE,
    )
    return ({
        "mode": "reasoning", "prompt": prompt,
        "observation/image": image, "observation/wrist_image": wrist_image,
        "observation/state": np.asarray(qpos, dtype=np.float32),
    }, updated)


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
    qpos_h100_raw: list[list[float]] | None = None
    decision_received_monotonic: float | None = None

    @property
    def decision(self) -> dict[str, Any]:
        return self.events[0]

    @property
    def triggered(self) -> bool:
        key = "need_recovery" if self.phase == "execution" else "adjustment_end"
        return bool(self.decision.get(key))


def validate_server_metadata(metadata: dict[str, Any]) -> None:
    validate_book_server_metadata(metadata)


def _assess_phase(
    *, args: argparse.Namespace, client: StreamingPhaseClient, operator: Any,
    captioner: Any, stats: StateQuantileStats, gate: BookV94AsyncControlGate,
    phase: str, generation: int, attempt_id: int, request_id: str,
    captured_step: int, after_timestamp: float,
    recovery_plan: str, submitted_monotonic: float,
) -> PhaseAssessmentResult:
    observation, timestamps = v53._capture_classification_observation(
        args, operator, captioner, after_timestamp=after_timestamp,
    )
    dense_h100, valid_mask, diagnostics = operator.state_history.snapshot(
        current_timestamp=float(observation.timestamp), current_state=observation.qpos,
        return_diagnostics=True,
    )
    if dense_h100.shape != (100, 7):
        raise ValueError(f"runtime H100 must be [100,7], got {dense_h100.shape}")
    targets = float(observation.timestamp) + (np.arange(100) - 99) / args.state_history_fps
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if valid_mask.shape != (100,) or not np.isfinite(dense_h100).all():
        raise ValueError("Invalid runtime H100 mask or qpos")
    # Only timestamps preceding the actual trial start may be left-padded.
    sampled_offsets = np.asarray(HISTORY_OFFSETS, dtype=np.int64)
    sampled_mask = np.zeros(100, dtype=np.bool_)
    sampled_mask[sampled_offsets] = True
    missing = ~valid_mask & sampled_mask & (targets >= args.episode_start_timestamp - 1e-6)
    if missing.any():
        details = [{
            "offset": int(index),
            "target_timestamp": float(diagnostics["target_timestamps"][index]),
            "nearest_timestamp": float(diagnostics["nearest_timestamps"][index]),
            "gap_ms": float(diagnostics["nearest_gap_seconds"][index]) * 1000.0,
        } for index in np.flatnonzero(missing)]
        message = f"Runtime H100 has invalid sampled ROS points: {details}; limit_ms={args.state_history_max_gap_seconds * 1000:g}"
        print(f"[H100] {message}", flush=True)
        raise ValueError(message)
    dense_h100[targets < args.episode_start_timestamp - 1e-6] = args.episode_start_qpos
    validate_tactile_caption(observation.tactile_caption)
    discrete = discretize_state_qpos(dense_h100[sampled_offsets], stats)
    prompt = build_phase_prompt(
        instruction=args.instruction,
        tactile_caption=observation.tactile_caption,
        recovery_plan=recovery_plan,
        qpos_h100_11_discrete=discrete,
    )
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
    decision_received = None
    for event in client.events(payload):
        if event.get("event") == "phase_decision":
            decision_received = time.monotonic()
        gate.handle_phase_event(event)
        if phase == "execution" and event.get("event") == "phase_decision":
            print(
                f"[NEED_RECOVERY] request_id={request_id} "
                f"caption={observation.tactile_caption} "
                f"result={bool(event.get('need_recovery'))} "
                f"probs(false,true)={event.get('need_recovery_probs')}",
                flush=True,
            )
        events.append(event)
    if not events:
        raise ValueError("Book V9.4 phase request returned no events")
    return PhaseAssessmentResult(
        request_id=request_id, phase=phase, generation=generation,
        attempt_id=attempt_id, captured_step=captured_step,
        observation=observation, synchronized_timestamps=timestamps,
        prompt=prompt, qpos_h100_11_discrete=discrete.tolist(), events=events,
        submitted_monotonic=submitted_monotonic,
        finished_monotonic=time.monotonic(),
        qpos_h100_raw=dense_h100.tolist(), decision_received_monotonic=decision_received,
    )


def _run_phase_assessment(**kwargs) -> PhaseAssessmentResult:
    try:
        return _assess_phase(**kwargs)
    except Exception:
        # A failed assessment must stop commands even if action inference is still running.
        kwargs["gate"].stop()
        raise


def _capture_action_observation(args, operator) -> v52.FrozenObservation:
    """Bound camera/qpos acquisition and keep it interruptible from the control loop."""
    started = time.monotonic()
    rate = operator.rate(args.observation_poll_rate)
    while not operator.is_shutdown() and not runtime.shutdown_event.is_set():
        if time.monotonic() - started > args.phase_change_timeout_seconds:
            raise TimeoutError("Timed out waiting for action camera/qpos observation")
        frame = operator.get_frame()
        if frame:
            front, left, joint = frame
            qpos = np.asarray(joint.position, dtype=np.float32)
            timestamp = v52._joint_timestamp(joint)
            if qpos.shape != (7,) or not np.isfinite(qpos).all() or timestamp is None or not np.isfinite(timestamp):
                raise ValueError("Action observation requires finite [7] qpos and timestamp")
            return v52.FrozenObservation(
                img_front=np.asarray(front).copy(), img_left=np.asarray(left).copy(),
                qpos=qpos.copy(), timestamp=float(timestamp),
                state_history=np.empty((0, 7), dtype=np.float32),
                state_history_mask=np.empty((0,), dtype=np.bool_),
                tactile_caption=neutral_caption(),
            )
        rate.sleep()
    raise RuntimeError("Stopped while waiting for action observation")


def _request_action_chunk(
    *, args: argparse.Namespace, policy: v53.TimeoutWebsocketPolicy,
    observation: v52.FrozenObservation, phase: str, phase_index: int,
    recovery_plan: str, logger: v52.TrialLogger,
) -> np.ndarray:
    prompt = v52.build_phase_prompt(
        phase=phase, instruction=args.instruction,
        recovery_plan=recovery_plan if phase == "adjustment" else "",
        prompt_profile=ACTION_PROMPT_PROFILE,
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
        raise ValueError(f"Book V9.4 server returned invalid actions {actions.shape}")
    if raw.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(raw).all():
        raise ValueError(f"Book V9.4 server returned invalid raw_model_actions {raw.shape}")
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
    future: Future, *, gate: BookV94AsyncControlGate, operator: Any, publish_rate: int,
    poll_control: Callable[[], None],
):
    rate = operator.rate(publish_rate)
    while not future.done() and not operator.is_shutdown() and not runtime.shutdown_event.is_set():
        poll_control()
        gate.control_tick()
        rate.sleep()
    poll_control()
    if not future.done():
        raise RuntimeError("stopped while waiting for Book V9.4 inference")
    return future.result()


class OperatorStopError(Exception):
    """Normal operator completion, distinct from an inference failure."""


@contextmanager
def inference_executor(*, gate, action_policy, phase_client):
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="book-v9-4-inference")
    try:
        yield executor
    finally:
        # Latch and issue a hold before socket cleanup or executor joining.
        runtime.shutdown_event.set()
        try:
            gate.stop()
        finally:
            try:
                phase_client.close()
            finally:
                try:
                    action_policy._ws.close()
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)


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
        "qpos_h100_raw": result.qpos_h100_raw,
        "observation_qpos": result.observation.qpos,
        "observation_timestamp": result.observation.timestamp,
        "tactile_caption": result.observation.tactile_caption,
        "synchronized_timestamps": result.synchronized_timestamps,
        "need_recovery": decision.get("need_recovery"),
        "need_recovery_probs": decision.get("need_recovery_probs"),
        "adjustment_end": decision.get("adjustment_end"),
        "adjustment_end_probs": decision.get("adjustment_end_probs"),
        "total_async_ms": (result.finished_monotonic - result.submitted_monotonic) * 1000.0,
        "decision_async_ms": ((result.decision_received_monotonic - result.submitted_monotonic) * 1000.0
                              if result.decision_received_monotonic is not None else None),
    }


def run_book_v9_4_async(
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
    memory: list[dict[str, str]] = []
    phase_indices = {"execution": 0, "adjustment": 0}
    published_steps = 0
    phase_request_count = 0
    last_phase_submit = {"execution": None, "adjustment": None}
    last_feedback_timestamp: float | None = None
    phase_future: Future[PhaseAssessmentResult] | None = None
    hold_target: list[np.ndarray | None] = [None]

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

    gate = BookV94AsyncControlGate(hold=hold, reset_hold=reset_hold)
    logger.record({"event": "run_start", "server_metadata": action_metadata,
                   "args": vars(args), "continuous_h100": True})
    print(
        "Book V9.4 async started: need and adjustment checks run independently of action chunks; "
        f"need={args.need_recovery_rate_hz:g}Hz adjustment={args.adjustment_end_rate_hz:g}Hz. "
        f"{args.success_key}=success, {args.quit_key}=quit."
    )

    def poll_control() -> None:
        key = keyboard.get_key()
        if key in {args.quit_key, args.success_key}:
            gate.stop()
            runtime.shutdown_event.set()
            event = "operator_quit" if key == args.quit_key else "operator_success"
            logger.record({"event": event, "phase": gate.snapshot()[0],
                           "published_steps": published_steps})
            if key == args.success_key:
                print("Operator confirmed success.")
            raise OperatorStopError(event)
        if runtime.shutdown_event.is_set() or operator.is_shutdown():
            gate.stop()
            raise OperatorStopError("shutdown")
        if phase_future is not None and phase_future.done():
            error = phase_future.exception()
            if error is not None:
                raise error

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
        pending_observation: v52.FrozenObservation | None = initial

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
            nonlocal phase_future, recovery_plan, memory, pending_observation
            if phase_future is None:
                return False, False
            if not phase_future.done() and not wait:
                return False, False
            result = (_wait_future_while_holding(
                phase_future, gate=gate, operator=operator, publish_rate=args.publish_rate,
                poll_control=poll_control,
            ) if wait else phase_future.result())
            phase_future = None
            logger.record(_phase_result_log(result, handled_step=published_steps))
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

            failure_events = [event for event in result.events if event.get("event") == "failure_reason"]
            if len(failure_events) != 1:
                raise ValueError("need_recovery=true did not return exactly one failure_reason")
            failure_reason = str(failure_events[0].get("failure_reason", "")).strip()
            if failure_reason not in legal_failure_reasons():
                raise ValueError(f"server returned illegal failure_reason {failure_reason!r}")
            plan_payload, updated_memory = plan_payload_after_failure(
                instruction=args.instruction,
                tactile_caption=result.observation.tactile_caption,
                executed_recovery_plan=recovery_plan,
                failure_reason=failure_reason, memory=memory,
                image=runtime.prepare_rgb(result.observation.img_front),
                wrist_image=runtime.prepare_rgb(result.observation.img_left),
                qpos=result.observation.qpos,
            )
            plan_future = executor.submit(
                action_policy.infer_with_timeout, plan_payload,
                timeout=args.request_timeout_seconds,
            )
            plan_response = _wait_future_while_holding(
                plan_future, gate=gate, operator=operator, publish_rate=args.publish_rate,
                poll_control=poll_control,
            )
            new_plan = str(plan_response.get("recovery_plan", "")).strip()
            if new_plan not in legal_recovery_plans():
                raise ValueError(f"server returned illegal recovery_plan {new_plan!r}")
            memory = updated_memory
            recovery_plan = new_plan
            print(f"[RECOVERY] failure={failure_reason} plan={recovery_plan}")
            logger.record({
                "event": "failure_and_plan", "attempt_id": active_attempt,
                "failure_reason": failure_reason, "recovery_plan": recovery_plan,
                "memory": memory, "plan_prompt": plan_payload["prompt"],
                "server_plan_infer_ms": plan_response.get("policy_timing", {}).get("infer_ms"),
            })
            gate.switch_phase("adjustment", increment_attempt=True)
            last_phase_submit["adjustment"] = None
            pending_observation = None
            logger.record({"event": "phase_transition", "previous_phase": "execution",
                           "phase": "adjustment", "trigger": "need_recovery",
                           "attempt_id": active_attempt + 1, "published_steps": published_steps})
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


def build_argument_parser(*, direction_keys: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(direction_keys=direction_keys)
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--instruction", default=BOOK_INSTRUCTION)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--norm-stats-file", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--phase-change-timeout-seconds", type=float, default=10.0)
    if direction_keys:
        parser.set_defaults(need_recovery_rate_hz=7.0, success_key=None)
    else:
        parser.add_argument("--need-recovery-rate-hz", type=float, default=7.0)
    parser.add_argument("--adjustment-end-rate-hz", type=float, default=7.0)
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.set_defaults(max_attempts=None)  # No attempt-count cap; operator/timeout/action budget stops remain.
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--publish_rate", type=int, default=30)
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
    if not direction_keys:
        parser.add_argument("--success-key", default="s")
    parser.add_argument("--quit-key", default="q")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--allow-experimental-thresholds", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gripper_offset", type=float, default=0.001)
    parser.add_argument("--gripper-min", dest="gripper_min", type=float, required=True)
    parser.add_argument(
        "--classification-gripper-open-threshold",
        type=float,
        help="Legacy option; rejected because Book V9.4 requires raw qpos",
    )
    parser.add_argument(
        "--classification-gripper-open-value",
        type=float,
        help="Legacy option; rejected because Book V9.4 requires raw qpos",
    )
    return parser


def get_arguments(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = build_argument_parser()
    return parser.parse_args(argv), parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in ("need_recovery_rate_hz", "adjustment_end_rate_hz", "request_timeout_seconds",
                 "phase_change_timeout_seconds", "state_history_fps", "state_history_max_gap_seconds"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"{name} must be finite")
    if args.noise_seed < 0:
        parser.error("--noise-seed must be non-negative")
    if args.no_captioner:
        parser.error("Book V9.4 phase inference requires the tactile captioner")
    if not args.norm_stats_file.is_file():
        parser.error(f"--norm-stats-file does not exist: {args.norm_stats_file}")
    if args.instruction != BOOK_INSTRUCTION:
        parser.error("Book V9.4 requires the original book instruction")
    if not args.captioner_checkpoint.is_file():
        parser.error("--captioner_checkpoint does not exist")
    if args.tactile_window_size != 30:
        parser.error("Book V9.4 requires captioner window=30")
    if args.state_history_fps != 30.0:
        parser.error("Book V9.4 H100 requires the training 30 Hz time grid")
    if (args.classification_gripper_open_threshold is not None
            or args.classification_gripper_open_value is not None):
        parser.error("Book V9.4 uses raw qpos; classification gripper remapping is forbidden")
    if not 1 <= args.chunk_size <= ACTION_HORIZON:
        parser.error(f"--chunk_size must be in [1,{ACTION_HORIZON}]")
    if args.max_publish_step <= 0 or args.publish_rate <= 0 or args.observation_poll_rate <= 0:
        parser.error("publish limits and rates must be positive")
    if args.state_history_len != 100 or args.state_history_fps <= 0 or args.state_history_max_gap_seconds <= 0:
        parser.error("runtime ROS history must use len=100 with positive fps/max-gap")
    if not args.direction_keys and not 0.0 < args.need_recovery_rate_hz <= args.publish_rate:
        parser.error("--need-recovery-rate-hz must be in (0,publish_rate]")
    if not 0.0 < args.adjustment_end_rate_hz <= args.publish_rate:
        parser.error("--adjustment-end-rate-hz must be in (0,publish_rate]")
    if args.request_timeout_seconds <= 0 or args.phase_change_timeout_seconds <= 0:
        parser.error("inference timeouts must be positive")
    if not np.isfinite(args.gripper_offset) or args.gripper_offset < 0:
        parser.error("--gripper_offset must be finite and non-negative")
    if not np.isfinite(args.gripper_min) or not 0.0 <= args.gripper_min <= 0.08:
        parser.error("--gripper-min must be finite and in [0,0.08]")
    if len(args.quit_key) != 1 or (args.success_key is not None and (
        len(args.success_key) != 1 or args.success_key == args.quit_key
    )):
        parser.error("success/quit keys must be distinct single characters")


def main(
    *, argument_parser=get_arguments, argument_validator=validate_args,
    run=run_book_v9_4_async, deployment_label: str = "Book V9.4",
) -> None:
    args, parser = argument_parser()
    runtime.apply_yaml_defaults(args, parser)
    argument_validator(args, parser)
    if not sys.stdin.isatty():
        parser.error("Book V9.4 real-robot inference requires an interactive TTY")
    runtime.shutdown_event.clear()
    signal.signal(signal.SIGINT, runtime._on_sigint)
    captioner = runtime.load_captioner(args)
    action_policy = DirectActionClient(args.host, args.port, timeout=args.request_timeout_seconds)
    phase_client = StreamingPhaseClient(args.host, args.port, timeout=args.request_timeout_seconds)
    operator = runtime.RosOperator(args)
    v53._install_tactile_arrival_timestamps(operator)
    trial_id = args.trial_id
    if trial_id and (args.log_dir.resolve() / trial_id).exists():
        trial_id = f"{trial_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = v52.TrialLogger(args.log_dir, trial_id=trial_id)
    print(f"Trial log directory: {logger.directory}")
    try:
        with runtime.KeyboardPoller() as keyboard:
            if not args.start_immediately:
                print(f"Press enter to start {deployment_label} EXECUTION; q quits.", flush=True)
                while not operator.is_shutdown() and not runtime.shutdown_event.is_set():
                    key = keyboard.get_key()
                    if key in {"\n", "\r"}:
                        break
                    if key == args.quit_key:
                        runtime.shutdown_event.set()
                        break
                    time.sleep(0.01)
            if operator.is_shutdown() or runtime.shutdown_event.is_set():
                raise OperatorStopError("stopped before execution")
            run(
                args, operator, action_policy, phase_client, captioner, keyboard, logger,
            )
    except OperatorStopError as exc:
        print(f"{deployment_label} stopped: {exc}.")
    except Exception as exc:
        logger.record({"event": "fail_closed_safety_stop", "error": repr(exc)})
        print(f"FAIL-CLOSED safety stop: {exc}. No more model actions will be published.")
        raise
    finally:
        runtime.shutdown_event.set()
        try:
            feedback, _ = v52._latest_feedback(operator)
            if (feedback is not None and np.asarray(feedback).shape == (7,)
                    and np.isfinite(feedback).all() and not args.no_publish):
                command = np.asarray(feedback, dtype=np.float32).copy()
                command[6] = max(float(args.gripper_min), float(command[6]))
                operator.puppet_arm_publish(command)
        finally:
            try:
                phase_client.close()
            finally:
                close = getattr(getattr(action_policy, "_ws", None), "close", None)
                if callable(close):
                    close()


__all__ = [
    "BookV94AsyncControlGate",
    "ContinuousH100",
    "PhaseAssessmentResult",
    "StreamingPhaseClient",
    "phase_payload",
    "plan_payload_after_failure",
    "run_book_v9_4_async",
    "validate_server_metadata",
]


if __name__ == "__main__":
    main()
