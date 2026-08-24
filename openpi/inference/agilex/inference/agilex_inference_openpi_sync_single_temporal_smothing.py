#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""V4 single-arm tactile VLA inference with short-horizon temporal smoothing.

The filename intentionally keeps ``smothing`` for compatibility with the requested
entrypoint.  This client reuses the observation, tactile, prompt, monitor, failure,
reasoning, and ROS interfaces from ``agilex_inference_tactile_vla_sync_single``.

Unlike the synchronous base client, action inference runs in a background worker
while cached actions publish at 30 Hz.  A returned H30 chunk is aligned to the
actual action step at which its request started, its elapsed prefix is discarded,
and only a short overlap with the previous chunk is blended.  Attempt/recovery
boundaries hard-reset the buffer and invalidate late results from the old phase.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import importlib
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

base = importlib.import_module("agilex_inference_tactile_vla_sync_single")


shutdown_event = base.shutdown_event
ACTION_HORIZON = 30


def _on_sigint(signum, frame) -> None:
    del signum, frame
    shutdown_event.set()
    try:
        import rospy

        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


@dataclass(frozen=True)
class TemporalIntegrationStats:
    accepted: bool
    request_start_step: int
    receive_step: int
    latency_steps: int
    first_action_index: int
    retained_actions: int
    blended_actions: int
    anchored_to_qpos: bool
    reason: str | None = None


class ShortHorizonTemporalSmoother:
    """Align and blend the newest action chunk on an absolute action-step grid."""

    def __init__(self, *, smooth_steps: int = 5, smooth_joints: int = 6) -> None:
        if smooth_steps <= 0:
            raise ValueError("smooth_steps must be positive")
        if smooth_joints not in (6, 7):
            raise ValueError("smooth_joints must be 6 or 7")
        self.smooth_steps = int(smooth_steps)
        self.smooth_joints = int(smooth_joints)
        self._actions: dict[int, np.ndarray] = {}
        self._generation = 0
        self._initialized = False

    @property
    def generation(self) -> int:
        return self._generation

    def reset(self, *, generation: int) -> None:
        self._actions.clear()
        self._generation = int(generation)
        self._initialized = False

    def has_action(self, step: int) -> bool:
        return int(step) in self._actions

    def pop(self, step: int) -> np.ndarray | None:
        step = int(step)
        for old_step in [value for value in self._actions if value < step]:
            del self._actions[old_step]
        action = self._actions.pop(step, None)
        return None if action is None else action.copy()

    def integrate(
        self,
        actions: np.ndarray,
        *,
        generation: int,
        request_start_step: int,
        receive_step: int,
        max_latency_steps: int,
        anchor_qpos: np.ndarray | None = None,
    ) -> TemporalIntegrationStats:
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected action chunk [T,7], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Refusing temporal integration of NaN or Inf actions")
        request_start_step = int(request_start_step)
        receive_step = int(receive_step)
        latency_steps = max(0, receive_step - request_start_step)
        if int(generation) != self._generation:
            return TemporalIntegrationStats(
                accepted=False,
                request_start_step=request_start_step,
                receive_step=receive_step,
                latency_steps=latency_steps,
                first_action_index=latency_steps,
                retained_actions=0,
                blended_actions=0,
                anchored_to_qpos=False,
                reason="generation_mismatch",
            )
        if latency_steps > int(max_latency_steps):
            return TemporalIntegrationStats(
                accepted=False,
                request_start_step=request_start_step,
                receive_step=receive_step,
                latency_steps=latency_steps,
                first_action_index=latency_steps,
                retained_actions=0,
                blended_actions=0,
                anchored_to_qpos=False,
                reason="latency_limit",
            )
        first_index = max(0, receive_step - request_start_step)
        if first_index >= len(actions):
            return TemporalIntegrationStats(
                accepted=False,
                request_start_step=request_start_step,
                receive_step=receive_step,
                latency_steps=latency_steps,
                first_action_index=first_index,
                retained_actions=0,
                blended_actions=0,
                anchored_to_qpos=False,
                reason="chunk_expired",
            )

        old_future = {
            step: action.copy()
            for step, action in self._actions.items()
            if step >= receive_step
        }
        anchored = False
        if not self._initialized and anchor_qpos is not None:
            anchor = np.asarray(anchor_qpos, dtype=float)
            if anchor.shape != (7,) or not np.isfinite(anchor).all():
                raise ValueError(f"Expected finite anchor_qpos [7], got {anchor.shape}")
            for offset in range(min(self.smooth_steps, len(actions) - first_index)):
                old_future.setdefault(receive_step + offset, anchor.copy())
            anchored = True

        new_future = {
            request_start_step + action_index: actions[action_index].copy()
            for action_index in range(first_index, len(actions))
        }
        overlap_steps: list[int] = []
        for step in range(receive_step, receive_step + self.smooth_steps):
            if step not in old_future or step not in new_future:
                break
            overlap_steps.append(step)

        new_weights = _short_transition_weights(len(overlap_steps))
        for step, new_weight in zip(overlap_steps, new_weights, strict=True):
            blended = new_future[step].copy()
            blended[: self.smooth_joints] = (
                (1.0 - new_weight) * old_future[step][: self.smooth_joints]
                + new_weight * new_future[step][: self.smooth_joints]
            )
            new_future[step] = blended

        # The latest observation owns all future commands.  Do not retain an old
        # tail beyond the new chunk, because that would silently replay stale phase
        # decisions after the newest H30 horizon expires.
        self._actions = new_future
        self._initialized = True
        return TemporalIntegrationStats(
            accepted=True,
            request_start_step=request_start_step,
            receive_step=receive_step,
            latency_steps=latency_steps,
            first_action_index=first_index,
            retained_actions=len(new_future),
            blended_actions=len(overlap_steps),
            anchored_to_qpos=anchored,
        )


def _short_transition_weights(length: int) -> np.ndarray:
    """Return newest-chunk weights from old to new over a short overlap."""

    length = int(length)
    if length <= 0:
        return np.zeros((0,), dtype=float)
    if length == 1:
        return np.ones((1,), dtype=float)
    return np.linspace(0.0, 1.0, length, dtype=float)


class AttemptStepClock:
    """Thread-safe attempt/generation/action-step snapshot for async requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempt_id = 0
        self._generation = 0
        self._step = 0

    def reset(self, *, attempt_id: int, generation: int) -> None:
        with self._lock:
            self._attempt_id = int(attempt_id)
            self._generation = int(generation)
            self._step = 0

    def update_step(self, step: int) -> None:
        with self._lock:
            self._step = int(step)

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return self._attempt_id, self._generation, self._step


class LockedCaptioner:
    """Serialize captioner GPU calls from action and assessment workers."""

    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor
        self._lock = threading.Lock()
        self.window_size = predictor.window_size

    def predict(self, *args, **kwargs):
        with self._lock:
            return self._predictor.predict(*args, **kwargs)


@dataclass(frozen=True)
class AsyncActionResult:
    attempt_id: int
    generation: int
    request_start_step: int
    qpos: np.ndarray
    state_history_valid_frames: int
    tactile_caption: str
    prompt: str
    actions: np.ndarray
    client_infer_ms: float
    server_infer_ms: float | None


def _run_async_action_once(
    *,
    args: argparse.Namespace,
    operator: base.RosOperator | base.ReplayOperator,
    policy,
    captioner,
    clock: AttemptStepClock,
    expected_attempt_id: int,
    expected_generation: int,
    input_recovery_plan: str,
) -> AsyncActionResult | None:
    """Capture the latest input and issue one action request on its own socket."""

    img_front, img_left, puppet_arm = base.get_latest_observation(args, operator)
    qpos = np.asarray(puppet_arm.position, dtype=float)
    state_history, state_history_mask = operator.get_state_history(puppet_arm)
    # Actions are conditioned on this observation, so latency starts at capture,
    # not after tactile-caption computation or network serialization.
    attempt_id, generation, request_start_step = clock.snapshot()
    if (attempt_id, generation) != (expected_attempt_id, expected_generation):
        return None
    tactile_caption = base.current_tactile_caption(operator, captioner)
    prompt = base.build_execution_prompt(
        instruction=args.instruction,
        tactile_caption=tactile_caption,
        input_recovery_plan=input_recovery_plan,
        case_id=args.case_id,
        attempt_id=expected_attempt_id,
        prompt_profile=args.prompt_profile,
    )
    payload = base.build_payload(
        mode="execution",
        img_front_bgr=img_front,
        img_left_bgr=img_left,
        qpos=qpos,
        state_history=state_history,
        state_history_mask=state_history_mask,
        prompt=prompt,
    )
    latest_attempt_id, latest_generation, _ = clock.snapshot()
    if (latest_attempt_id, latest_generation) != (expected_attempt_id, expected_generation):
        return None

    started = time.perf_counter()
    response = policy.infer(payload)
    client_infer_ms = (time.perf_counter() - started) * 1000.0
    actions = np.asarray(response["actions"], dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions with shape [T,7], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Refusing to publish actions containing NaN or Inf")
    server_ms = response.get("policy_timing", {}).get("infer_ms")
    return AsyncActionResult(
        attempt_id=attempt_id,
        generation=generation,
        request_start_step=request_start_step,
        qpos=qpos.copy(),
        state_history_valid_frames=int(np.asarray(state_history_mask, dtype=np.bool_).sum()),
        tactile_caption=tactile_caption,
        prompt=prompt,
        actions=actions,
        client_infer_ms=client_infer_ms,
        server_infer_ms=None if server_ms is None else float(server_ms),
    )


def _configure_from_server_metadata(args: argparse.Namespace, metadata: dict[str, Any]) -> int:
    print(f"Server metadata: {metadata}")
    server_data_profile = str(metadata.get("data_profile", "legacy"))
    if args.expected_data_profile is not None and server_data_profile != args.expected_data_profile:
        raise ValueError(
            "Client/server data profile mismatch: "
            f"expected={args.expected_data_profile!r}, server={server_data_profile!r}"
        )
    args.prompt_profile = base.resolve_prompt_profile(metadata.get("prompt_profile"))
    print(f"Using checkpoint prompt profile: {args.prompt_profile}")

    server_max_memory_pairs = int(metadata.get("max_memory_pairs", base.MAX_MEMORY_PAIRS))
    server_max_attempts = int(metadata.get("max_supported_attempts", base.MAX_SUPPORTED_ATTEMPTS))
    if server_max_memory_pairs != base.MAX_MEMORY_PAIRS:
        raise ValueError(
            "Client/server recovery memory mismatch: "
            f"client={base.MAX_MEMORY_PAIRS}, server={server_max_memory_pairs}"
        )
    if server_max_attempts != base.MAX_SUPPORTED_ATTEMPTS:
        raise ValueError(
            "Client/server attempt limit mismatch: "
            f"client={base.MAX_SUPPORTED_ATTEMPTS}, server={server_max_attempts}"
        )
    if not 1 <= args.max_attempts <= base.MAX_SUPPORTED_ATTEMPTS:
        raise ValueError(
            f"Requested max_attempts={args.max_attempts}, but server supports at most "
            f"{base.MAX_SUPPORTED_ATTEMPTS} attempts"
        )

    args.v3_autoregressive = str(metadata.get("stage_b_version", "")).startswith("v3_")
    args.v3_shared_assessment = args.v3_autoregressive and bool(
        metadata.get("supports_shared_assessment", False)
    )
    if args.v3_autoregressive and not bool(metadata.get("supports_failure_generation", False)):
        raise ValueError("V3 server does not advertise supports_failure_generation=true")
    if not bool(metadata.get("supports_step_monitor", False)):
        raise ValueError("Server does not advertise supports_step_monitor=true")

    server_action_horizon = int(metadata.get("action_horizon", 0))
    if server_action_horizon != ACTION_HORIZON:
        raise ValueError(
            f"V4 temporal smoothing requires action_horizon={ACTION_HORIZON}, "
            f"got {server_action_horizon}"
        )
    if int(metadata.get("output_action_dim", 0)) != 7:
        raise ValueError(
            f"V4 temporal smoothing requires output_action_dim=7, "
            f"got {metadata.get('output_action_dim')}"
        )
    if args.chunk_size > server_action_horizon:
        raise ValueError(
            f"Requested chunk_size={args.chunk_size}, server action_horizon={server_action_horizon}"
        )

    server_uses_history = bool(metadata.get("use_state_history", False))
    if args.v3_autoregressive:
        if server_uses_history:
            print("V3/V4 action expert uses dense proprioceptive state history.")
        else:
            print("Checkpoint ignores the state-history payload.")
        if args.v3_shared_assessment:
            print("Shared assessment enabled for need_recovery and failure_reason.")
    elif not server_uses_history:
        raise ValueError("This inference path requires a server with use_state_history=true")

    if server_uses_history:
        server_history = (
            int(metadata.get("state_history_len", 0)),
            int(metadata.get("state_history_dim", 0)),
        )
        if server_history != (args.state_history_len, 7):
            raise ValueError(
                "Client/server state-history shape mismatch: "
                f"client=[{args.state_history_len},7], server={list(server_history)}"
            )
        server_history_fps = float(metadata.get("state_history_fps", 0.0))
        if not np.isclose(server_history_fps, args.state_history_fps):
            raise ValueError(
                f"Client state_history_fps={args.state_history_fps} does not match "
                f"server={server_history_fps}"
            )

    if args.max_action_latency_steps is None:
        args.max_action_latency_steps = server_action_horizon - 1
    if not 0 <= args.max_action_latency_steps < server_action_horizon:
        raise ValueError(
            "--max-action-latency-steps must be in "
            f"[0,{server_action_horizon - 1}] for H{server_action_horizon}"
        )
    return server_action_horizon


def _metadata_identity(metadata: dict[str, Any]) -> tuple[Any, ...]:
    keys = (
        "data_profile",
        "prompt_profile",
        "stage_b_version",
        "action_horizon",
        "output_action_dim",
        "use_state_history",
        "state_history_len",
        "state_history_dim",
        "state_history_fps",
    )
    return tuple(metadata.get(key) for key in keys)


def _drain_obsolete_action_future(
    future: Future[AsyncActionResult | None] | None,
) -> None:
    if future is None:
        return
    if not future.cancel():
        # At recovery boundaries it is safer to let the one running request finish
        # while the robot is stopped than to queue a new-phase request behind it.
        future.result()


def run_closed_loop(
    args: argparse.Namespace,
    operator: base.RosOperator | base.ReplayOperator,
    action_policy,
    assessment_policy,
    captioner,
) -> None:
    action_metadata = action_policy.get_server_metadata()
    assessment_metadata = assessment_policy.get_server_metadata()
    if _metadata_identity(action_metadata) != _metadata_identity(assessment_metadata):
        raise ValueError("Action and assessment WebSocket connections expose different checkpoint metadata")
    server_action_horizon = _configure_from_server_metadata(args, action_metadata)

    request_interval_steps = max(1, round(args.publish_rate / args.action_inference_rate))
    print(
        "Temporal smoothing enabled: "
        f"H{server_action_horizon}, request every {request_interval_steps} published steps, "
        f"short overlap={args.temporal_smooth_steps}, smooth joints=0:{args.temporal_smooth_joints}, "
        f"max latency={args.max_action_latency_steps} steps."
    )
    print(
        "Action and assessment requests use separate WebSocket connections; "
        "one request per connection may be in flight."
    )
    if not args.start_immediately and args.replay_attempt_dir is None:
        input("Press enter to continue")

    pre_action: np.ndarray | None = None
    input_recovery_plan = ""
    memory: list[dict[str, Any]] = []
    previous_monitor_finished: float | None = None
    smoother = ShortHorizonTemporalSmoother(
        smooth_steps=args.temporal_smooth_steps,
        smooth_joints=args.temporal_smooth_joints,
    )
    clock = AttemptStepClock()
    generation = 0

    def record_monitor(result: base.AsyncMonitorResult, *, handled_step: int) -> bool:
        nonlocal previous_monitor_finished
        base.report_async_monitor_result(
            args=args,
            result=result,
            handled_step=handled_step,
            previous_finished_monotonic=previous_monitor_finished,
        )
        previous_monitor_finished = result.finished_monotonic
        return bool(result.response.get("need_recovery", False))

    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tactile-vla-action") as action_executor,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tactile-vla-monitor") as monitor_executor,
        base.KeyboardPoller() as keyboard,
    ):
        action_future: Future[AsyncActionResult | None] | None = None
        monitor_future: Future[base.AsyncMonitorResult] | None = None

        for attempt_id in range(1, args.max_attempts + 1):
            if monitor_future is not None:
                raise RuntimeError("Async monitor leaked across attempt boundary")
            _drain_obsolete_action_future(action_future)
            action_future = None
            generation += 1
            step = 0
            clock.reset(attempt_id=attempt_id, generation=generation)
            smoother.reset(generation=generation)
            operator.reset_state_history()
            pre_action = None
            next_action_request_step = 0
            recovery_result: base.AsyncMonitorResult | None = None
            print(f"Starting attempt {attempt_id} recovery_plan={input_recovery_plan or 'none'}")
            if attempt_id > 1 and args.recovery_tactile_ignore_seconds > 0:
                ignore_steps = round(args.recovery_tactile_ignore_seconds * args.publish_rate)
                print(
                    "Suppressing recovery assessment during homing for "
                    f"{ignore_steps} published steps ({args.recovery_tactile_ignore_seconds:.3f} s)"
                )

            control_rate = operator.rate(args.publish_rate)
            while step < args.max_publish_step and not operator.is_shutdown():
                # Recovery always wins over a newly completed action request.  This
                # prevents an old normal chunk from being integrated on the trigger step.
                if monitor_future is not None and monitor_future.done():
                    result = monitor_future.result()
                    monitor_future = None
                    if record_monitor(result, handled_step=step):
                        recovery_result = result
                        generation += 1
                        clock.reset(attempt_id=attempt_id, generation=generation)
                        smoother.reset(generation=generation)
                        break

                if action_future is not None and action_future.done():
                    action_result = action_future.result()
                    action_future = None
                    if action_result is not None:
                        current_attempt, current_generation, current_step = clock.snapshot()
                        actions = action_result.actions[: args.chunk_size]
                        stats = smoother.integrate(
                            actions,
                            generation=action_result.generation,
                            request_start_step=action_result.request_start_step,
                            receive_step=current_step,
                            max_latency_steps=args.max_action_latency_steps,
                            anchor_qpos=action_result.qpos,
                        )
                        base.append_runtime_log(
                            args,
                            {
                                "event": "temporal_action_chunk",
                                "attempt_id": action_result.attempt_id,
                                "generation": action_result.generation,
                                "request_start_step": action_result.request_start_step,
                                "receive_step": current_step,
                                "latency_steps": stats.latency_steps,
                                "accepted": stats.accepted,
                                "reason": stats.reason,
                                "first_action_index": stats.first_action_index,
                                "retained_actions": stats.retained_actions,
                                "blended_actions": stats.blended_actions,
                                "anchored_to_qpos": stats.anchored_to_qpos,
                                "smooth_joints": args.temporal_smooth_joints,
                                "client_infer_ms": action_result.client_infer_ms,
                                "server_infer_ms": action_result.server_infer_ms,
                                "tactile_caption": action_result.tactile_caption,
                                "state_history_valid_frames": action_result.state_history_valid_frames,
                            },
                        )
                        if (current_attempt, current_generation) != (
                            action_result.attempt_id,
                            action_result.generation,
                        ):
                            print("Discarded late action chunk from an obsolete attempt/generation")
                        elif stats.accepted:
                            next_action_request_step = (
                                action_result.request_start_step + request_interval_steps
                            )
                            print(
                                f"Temporal chunk accepted: request_step={action_result.request_start_step} "
                                f"receive_step={current_step} latency={stats.latency_steps} "
                                f"retained={stats.retained_actions} blended={stats.blended_actions}"
                            )
                        else:
                            next_action_request_step = current_step
                            print(
                                f"Temporal chunk discarded: reason={stats.reason} "
                                f"latency={stats.latency_steps}"
                            )

                current_attempt, current_generation, current_step = clock.snapshot()
                if (
                    action_future is None
                    and current_attempt == attempt_id
                    and current_generation == generation
                    and (
                        current_step >= next_action_request_step
                        or not smoother.has_action(current_step)
                    )
                ):
                    action_future = action_executor.submit(
                        _run_async_action_once,
                        args=args,
                        operator=operator,
                        policy=action_policy,
                        captioner=captioner,
                        clock=clock,
                        expected_attempt_id=attempt_id,
                        expected_generation=generation,
                        input_recovery_plan=input_recovery_plan,
                    )

                raw_action = smoother.pop(step)
                if raw_action is None:
                    # Hold the last physical target while waiting.  No synthetic
                    # model step is added to history or to latency accounting.
                    time.sleep(0.001)
                    continue
                if pre_action is None:
                    latest = base.get_latest_observation(args, operator)
                    pre_action = np.asarray(latest[2].position, dtype=float)
                pre_action, _ = base.publish_single_action(
                    args=args,
                    operator=operator,
                    raw_action=raw_action,
                    pre_action=pre_action,
                    rate=control_rate,
                )
                step += 1
                clock.update_step(step)
                print(f"Published Step {step}")

                key = keyboard.get_key()
                if key == args.quit_key:
                    print("Operator requested safety stop")
                    shutdown_event.set()
                    base.append_runtime_log(
                        args,
                        {"event": "operator_quit", "attempt_id": attempt_id, "step": step},
                    )
                    return
                if key == args.success_key:
                    print("Operator confirmed success")
                    base.append_runtime_log(
                        args,
                        {"event": "operator_success", "attempt_id": attempt_id, "step": step},
                    )
                    return

                if monitor_future is not None and monitor_future.done():
                    result = monitor_future.result()
                    monitor_future = None
                    if record_monitor(result, handled_step=step):
                        recovery_result = result
                        generation += 1
                        clock.reset(attempt_id=attempt_id, generation=generation)
                        smoother.reset(generation=generation)
                        break

                monitor_ignored, _ = base.recovery_monitor_ignore_status(
                    attempt_id=attempt_id,
                    published_step=step,
                    ignore_seconds=args.recovery_tactile_ignore_seconds,
                    publish_rate=args.publish_rate,
                )
                if monitor_future is None and not monitor_ignored:
                    monitor_future = monitor_executor.submit(
                        base.run_async_monitor_once,
                        args=args,
                        operator=operator,
                        policy=assessment_policy,
                        captioner=captioner,
                        attempt_id=attempt_id,
                        captured_step=step,
                        input_recovery_plan=input_recovery_plan,
                    )
                control_rate.sleep()

            if recovery_result is None:
                print(f"Attempt {attempt_id} ended without recovery trigger")
                return

            # Hard phase boundary: wait for at most the single old action request;
            # its result is intentionally ignored and cannot enter the next attempt.
            _drain_obsolete_action_future(action_future)
            action_future = None
            if args.v3_shared_assessment:
                failure_reason = str(recovery_result.response.get("failure_reason", "")).strip()
                if not failure_reason:
                    raise ValueError("Assessment returned need_recovery=true without failure_reason")
                print(f"Shared assessment failure_reason: {failure_reason}")
            elif args.v3_autoregressive:
                failure_reason = base.run_failure_diagnosis(
                    args=args,
                    policy=assessment_policy,
                    result=recovery_result,
                    input_recovery_plan=input_recovery_plan,
                )
            else:
                failure_reason = str(recovery_result.response.get("failure_reason", "unknown"))

            memory_plan = "initial plan" if attempt_id == 1 else input_recovery_plan
            entry = {"recovery_plan": memory_plan, "failure_reason": failure_reason}
            will_reason = attempt_id < args.max_attempts
            if will_reason:
                memory = base.update_failure_recovery_memory(
                    memory,
                    entry,
                    prompt_profile=args.prompt_profile,
                )
            base.append_runtime_log(
                args,
                {
                    "event": "need_recovery",
                    "attempt_id": attempt_id,
                    "step": step,
                    "captured_step": recovery_result.captured_step,
                    "lag_steps": max(0, step - recovery_result.captured_step),
                    "memory_entry": entry,
                    "memory_pairs": len(memory),
                    "reasoning_skipped": not will_reason,
                    "need_recovery_probs": recovery_result.response.get("need_recovery_probs"),
                    "failure_reason_probs": recovery_result.response.get("failure_reason_probs"),
                    "tactile_caption": recovery_result.tactile_caption,
                },
            )
            print(f"need_recovery=true failure_reason={failure_reason}; temporal buffer cleared")
            if not will_reason:
                print("Reached max attempts after recovery trigger")
                return
            input_recovery_plan = base.run_reasoning(
                args=args,
                policy=assessment_policy,
                img_front=recovery_result.img_front,
                img_left=recovery_result.img_left,
                qpos=recovery_result.qpos,
                state_history=recovery_result.state_history,
                state_history_mask=recovery_result.state_history_mask,
                tactile_caption=recovery_result.tactile_caption,
                memory=memory,
                failed_attempt_id=attempt_id,
            )


def get_arguments() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    # Reuse the complete base CLI without maintaining a second copy.  Build its
    # parser with an empty argv first, add temporal options, then parse real argv.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        _, parser = base.get_arguments()
    finally:
        sys.argv = original_argv
    parser.description = __doc__
    parser.add_argument(
        "--temporal-smooth-joints",
        type=int,
        choices=(6, 7),
        default=6,
        help="Smooth arm joints 0:6 only (6), or arm plus gripper 0:7 (7).",
    )
    parser.add_argument(
        "--temporal-smooth-steps",
        type=int,
        default=5,
        help="Short new/old overlap in 30 Hz action steps; recommended 3 to 5.",
    )
    parser.add_argument(
        "--action-inference-rate",
        type=float,
        default=3.0,
        help="Target action request rate in Hz; only one action request is in flight.",
    )
    parser.add_argument(
        "--max-action-latency-steps",
        type=int,
        default=None,
        help="Discard a result whose measured action-step delay exceeds this value; default H-1.",
    )
    return parser.parse_args(), parser


def main() -> None:
    args, parser = get_arguments()
    base.apply_yaml_defaults(args, parser)
    if args.publish_rate <= 0:
        parser.error("--publish_rate must be positive")
    if args.action_inference_rate <= 0:
        parser.error("--action-inference-rate must be positive")
    if args.action_inference_rate > args.publish_rate:
        parser.error("--action-inference-rate cannot exceed --publish_rate")
    if not 1 <= args.max_attempts <= base.MAX_SUPPORTED_ATTEMPTS:
        parser.error(f"--max_attempts must be in [1, {base.MAX_SUPPORTED_ATTEMPTS}]")
    if args.observation_poll_rate <= 0:
        parser.error("--observation-poll-rate must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")
    if not 1 <= args.temporal_smooth_steps <= args.chunk_size:
        parser.error("--temporal-smooth-steps must be in [1, chunk_size]")
    if args.state_history_len <= 0 or args.state_history_fps <= 0:
        parser.error("state-history length and fps must be positive")
    if args.state_history_max_gap_seconds <= 0:
        parser.error("--state-history-max-gap-seconds must be positive")
    if args.recovery_tactile_ignore_seconds < 0:
        parser.error("--recovery-need-ignore-seconds must be non-negative")
    if args.use_actions_interpolation:
        parser.error(
            "--use_actions_interpolation is incompatible with temporal action-step alignment; "
            "use the short temporal smoother instead"
        )
    if args.seed is not None:
        base.set_seed(args.seed)
    signal.signal(signal.SIGINT, _on_sigint)

    captioner = base.load_captioner(args)
    if captioner is not None:
        captioner = LockedCaptioner(captioner)
    if args.mock_policy:
        action_policy = base.MockPolicy()
        assessment_policy = base.MockPolicy()
    else:
        action_policy = base.websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        assessment_policy = base.websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    if args.replay_attempt_dir is not None:
        operator: base.RosOperator | base.ReplayOperator = base.ReplayOperator(args)
    else:
        operator = base.RosOperator(args)
    try:
        run_closed_loop(args, operator, action_policy, assessment_policy, captioner)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        if isinstance(operator, base.ReplayOperator):
            operator.close()


if __name__ == "__main__":
    main()
