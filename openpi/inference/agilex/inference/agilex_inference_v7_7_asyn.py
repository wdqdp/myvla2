#!/usr/bin/env python3
"""Executable V7.7 asynchronous single-arm ROS inference.

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
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE
from tactile_vla.vla.prompts import build_recovery_prompt
from tactile_vla.vla.prompts import update_failure_recovery_memory
from tactile_vla.vla.structured_text import legal_failure_reasons
from tactile_vla.vla.structured_text import legal_recovery_plans
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v5_3_phase_change import discretize_state_qpos
from tactile_vla.vla.v7_7_async_state import AsyncPhaseState
from tactile_vla.vla.v7_7_multitask_data import DATA_PROFILE
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt

DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/v7_7_async"
DEFAULT_NORM_STATS = Path(
    "/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/rotation_v4/norm_stats/norm_stats.json"
)
DEFAULT_CAPTIONER = Path("/data1/outputs/tactile_captioner/tcn_v3_w30_rotation_head/best.pt")
ACTION_PROMPT_PROFILE = "phase_v2"
ACTION_HORIZON = 30
ACTION_DIM = 32


def _remap_classification_qpos(
    qpos: np.ndarray,
    *,
    open_threshold: float | None,
    open_value: float | None,
) -> tuple[np.ndarray, int]:
    """Map open-gripper feedback for phase-model inputs only."""

    values = np.asarray(qpos, dtype=np.float32)
    if values.shape[-1] != 7:
        raise ValueError(f"classification qpos must end in dimension 7, got {values.shape}")
    if (open_threshold is None) != (open_value is None):
        raise ValueError("classification gripper threshold and value must be provided together")
    mapped = values.copy()
    if open_threshold is None:
        return mapped, 0
    mask = mapped[..., 6] > float(open_threshold)
    mapped[..., 6] = np.where(mask, float(open_value), mapped[..., 6])
    return mapped, int(np.count_nonzero(mask))


class StreamingPhaseClient:
    """One-request/two-event WebSocket client used only by phase assessment."""

    def __init__(self, host: str, port: int, *, timeout: float = 30.0):
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None,
            open_timeout=timeout, close_timeout=timeout,
        )
        self._timeout = timeout
        self._packer = msgpack_numpy.Packer()
        self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=timeout))
        if self.metadata.get("data_profile") != DATA_PROFILE:
            raise ValueError("phase server is not a V7.7 checkpoint")
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
            yield decision
            if bool(decision.get("need_recovery")):
                failure = msgpack_numpy.unpackb(self._ws.recv(timeout=self._timeout))
                if failure.get("event") != "failure_reason" or failure.get("request_id") != decision.get("request_id"):
                    raise ValueError("conditional failure event does not match its need decision")
                yield failure

    def close(self):
        self._ws.close()


class V77AsyncControlGate:
    """Thread-safe bridge between assessment events and a 25 Hz action loop."""

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
            if event.get("event") == "failure_reason":
                return self.state.accept_failure_event(event)
            raise ValueError(f"unknown V7.7 event {event.get('event')!r}")

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
    """Update failure-recovery memory, then create the mandatory new plan prefill."""
    entry = {
        "recovery_plan": executed_recovery_plan or "initial plan",
        "failure_reason": failure_reason,
    }
    updated = update_failure_recovery_memory(
        memory, entry, prompt_profile=MINIMAL_PROMPT_PROFILE
    )
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

    @property
    def decision(self) -> dict[str, Any]:
        return self.events[0]

    @property
    def triggered(self) -> bool:
        key = "need_recovery" if self.phase == "execution" else "adjustment_end"
        return bool(self.decision.get(key))


def validate_server_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "name": "tactile_vla_v7_7",
        "data_profile": DATA_PROFILE,
        "phase_prompt_profile": PROMPT_PROFILE,
        "supports_streamed_phase_events": True,
        "supports_action_noise": True,
        "requires_action_noise": True,
        "supports_failure_generation": True,
        "supports_recovery_generation": True,
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "output_action_dim": 7,
        "use_state_history": False,
        "state_history_len": 0,
    }
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items()
                if metadata.get(key) != value}
    if mismatch:
        raise ValueError(f"V7.7 client/server metadata mismatch: {mismatch}")
    for key in ("need_recovery_threshold", "adjustment_end_threshold"):
        value = float(metadata.get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid server {key}={value}")


def _run_phase_assessment(
    *, args: argparse.Namespace, client: StreamingPhaseClient, operator: Any,
    captioner: Any, stats: StateQuantileStats, gate: V77AsyncControlGate,
    phase: str, generation: int, attempt_id: int, request_id: str,
    captured_step: int, after_timestamp: float,
    recovery_plan: str, submitted_monotonic: float,
) -> PhaseAssessmentResult:
    observation, timestamps = v53._capture_classification_observation(
        args, operator, captioner, after_timestamp=after_timestamp,
    )
    dense_h100, _ = operator.state_history.snapshot(
        current_timestamp=float(observation.timestamp), current_state=observation.qpos,
    )
    if dense_h100.shape != (100, 7):
        raise ValueError(f"runtime H100 must be [100,7], got {dense_h100.shape}")
    open_threshold = args.classification_gripper_open_threshold
    open_value = args.classification_gripper_open_value
    mapped_h100, _ = _remap_classification_qpos(
        dense_h100, open_threshold=open_threshold, open_value=open_value,
    )
    mapped_current_qpos, _ = _remap_classification_qpos(
        observation.qpos, open_threshold=open_threshold, open_value=open_value,
    )
    discrete = discretize_state_qpos(mapped_h100[np.linspace(0, 99, 11, dtype=np.int64)], stats)
    prompt = build_phase_prompt(
        instruction=args.instruction,
        tactile_caption=observation.tactile_caption,
        recovery_plan=recovery_plan,
        qpos_h100_11_discrete=discrete,
    )
    payload = runtime.build_payload(
        mode="phase", img_front_bgr=observation.img_front,
        img_left_bgr=observation.img_left, qpos=mapped_current_qpos,
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
        raise ValueError("V7.7 phase request returned no events")
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
        raise ValueError(f"V7.7 server returned invalid actions {actions.shape}")
    if raw.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(raw).all():
        raise ValueError(f"V7.7 server returned invalid raw_model_actions {raw.shape}")
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
    future: Future, *, gate: V77AsyncControlGate, operator: Any, publish_rate: int,
):
    rate = operator.rate(publish_rate)
    while not future.done() and not operator.is_shutdown() and not runtime.shutdown_event.is_set():
        gate.control_tick()
        rate.sleep()
    if not future.done():
        raise RuntimeError("stopped while waiting for V7.7 inference")
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


def run_v7_7_async(
    args: argparse.Namespace, operator: Any, action_policy: v53.TimeoutWebsocketPolicy,
    phase_client: StreamingPhaseClient, captioner: Any, keyboard: Any,
    logger: v52.TrialLogger,
) -> None:
    action_metadata = action_policy.get_server_metadata()
    validate_server_metadata(action_metadata)
    if action_metadata != phase_client.metadata:
        raise ValueError("V7.7 action and phase connections expose different metadata")
    args.use_state_history = False
    stats = load_state_quantiles(args.norm_stats_file)
    recovery_plan = ""
    memory: list[dict[str, str]] = []
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

    gate = V77AsyncControlGate(hold=hold)
    logger.record({"event": "run_start", "server_metadata": action_metadata,
                   "args": vars(args), "continuous_h100": True})
    print(
        "V7.7 async started: need and adjustment checks run independently of action chunks; "
        f"need={args.need_recovery_rate_hz:g}Hz adjustment={args.adjustment_end_rate_hz:g}Hz. "
        f"{args.success_key}=success, {args.quit_key}=quit."
    )

    # This is the only H100 reset: phase and attempt transitions preserve it.
    operator.reset_state_history()
    # Action prompts contain no touch text; reserve the captioner for the
    # asynchronous phase worker to avoid serializing action generation on it.
    initial = v52._capture_observation(args, operator, None, reset_history=False)
    if initial.timestamp is None:
        raise ValueError("initial ROS qpos does not have a timestamp")
    operator.state_history.push(float(initial.timestamp), initial.qpos)
    hold_target[0] = initial.qpos.copy()
    last_feedback_timestamp = initial.timestamp
    pending_observation: v52.FrozenObservation | None = initial

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="v7-7-inference") as executor:
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
            nonlocal phase_future, recovery_plan, memory, pending_observation
            if phase_future is None:
                return False, False
            if not phase_future.done() and not wait:
                return False, False
            result = (_wait_future_while_holding(
                phase_future, gate=gate, operator=operator, publish_rate=args.publish_rate,
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
            if active_attempt >= args.max_attempts:
                print("need_recovery=true at max_attempts; remaining stopped.")
                logger.record({"event": "max_attempts_reached", "attempt_id": active_attempt,
                               "failure_reason": failure_reason})
                return False, True
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
                    if args.no_publish:
                        runtime.published_actions_history.append(np.asarray(value).copy())
                    else:
                        timestamp_slot[0] = operator.puppet_arm_publish(value)
                        runtime.published_actions_history.append(np.asarray(value).copy())

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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--instruction", default=v52.DEFAULT_INSTRUCTION)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--norm-stats-file", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--phase-change-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--need-recovery-rate-hz", type=float, default=7.0)
    parser.add_argument("--adjustment-end-rate-hz", type=float, default=7.0)
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--publish_rate", type=int, default=25)
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
    if args.no_captioner:
        parser.error("V7.7 phase inference requires the tactile captioner")
    if not args.norm_stats_file.is_file():
        parser.error(f"--norm-stats-file does not exist: {args.norm_stats_file}")
    if not 1 <= args.max_attempts <= 5:
        parser.error("--max-attempts must be in [1,5]")
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
    threshold = args.classification_gripper_open_threshold
    value = args.classification_gripper_open_value
    if (threshold is None) != (value is None):
        parser.error(
            "--classification-gripper-open-threshold and "
            "--classification-gripper-open-value must be provided together"
        )
    if threshold is not None and (
        not np.isfinite(threshold)
        or not np.isfinite(value)
        or not 0.0 <= threshold <= value <= 0.2
    ):
        parser.error("classification gripper mapping requires 0 <= threshold <= value <= 0.2")
    if len(args.success_key) != 1 or len(args.quit_key) != 1 or args.success_key == args.quit_key:
        parser.error("success/quit keys must be distinct single characters")


def main() -> None:
    args, parser = get_arguments()
    runtime.apply_yaml_defaults(args, parser)
    validate_args(args, parser)
    if not sys.stdin.isatty():
        parser.error("V7.7 real-robot inference requires an interactive TTY")
    runtime.shutdown_event.clear()
    signal.signal(signal.SIGINT, runtime._on_sigint)
    captioner = runtime.load_captioner(args)
    action_policy = v53.TimeoutWebsocketPolicy(args.host, args.port)
    phase_client = StreamingPhaseClient(args.host, args.port, timeout=args.request_timeout_seconds)
    operator = runtime.RosOperator(args)
    v53._install_tactile_arrival_timestamps(operator)
    trial_id = args.trial_id
    if trial_id and (args.log_dir.resolve() / trial_id).exists():
        trial_id = f"{trial_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = v52.TrialLogger(args.log_dir, trial_id=trial_id)
    print(f"Trial log directory: {logger.directory}")
    if args.classification_gripper_open_threshold is not None:
        print(
            "Classification gripper remapping enabled for current qpos and H100: "
            f"qpos[6]>{args.classification_gripper_open_threshold:.6f} -> "
            f"{args.classification_gripper_open_value:.6f}."
        )
    try:
        if not args.start_immediately:
            input("Press enter to start V7.7 EXECUTION")
        with runtime.KeyboardPoller() as keyboard:
            run_v7_7_async(
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
    "ContinuousH100",
    "PhaseAssessmentResult",
    "StreamingPhaseClient",
    "V77AsyncControlGate",
    "phase_payload",
    "plan_payload_after_failure",
    "run_v7_7_async",
    "validate_server_metadata",
]


if __name__ == "__main__":
    main()
