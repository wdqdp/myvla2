from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    name = "forced_phase_ablation_client_test_module"
    spec = importlib.util.spec_from_file_location(
        name,
        PROJECT_ROOT
        / "openpi/inference/agilex/inference/agilex_inference_forced_phase_anlation.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


client = _load_module()
PLAN = "recovery_plan=move horizontally right slightly, move vertically none moderately."


class _Policy:
    def __init__(self):
        self.requests = []

    def infer(self, payload):
        self.requests.append(payload)
        return {
            "raw_model_actions": np.zeros((30, 32), dtype=np.float32),
            "actions": np.zeros((30, 7), dtype=np.float32),
            "policy_timing": {"infer_ms": 1.0},
        }


class _Keyboard:
    def __init__(self, keys):
        self.keys = list(keys)

    def get_key(self):
        return self.keys.pop(0) if self.keys else None


class _Operator:
    def is_shutdown(self):
        return False


class _ControlRate:
    def sleep(self):
        pass


class _PauseOperator:
    def __init__(self):
        self.history_paused = False
        self.resumed = []

    def is_shutdown(self):
        return False

    def rate(self, _hz):
        return _ControlRate()

    def pause_state_history(self):
        self.history_paused = True

    def resume_state_history(self, history, mask, *, current_timestamp):
        self.resumed.append(
            (
                np.asarray(history).copy(),
                np.asarray(mask).copy(),
                float(current_timestamp),
            )
        )
        self.history_paused = False


def _args(profile: str):
    return argparse.Namespace(
        instruction="Pick object",
        forced_recovery_plan=PLAN,
        forced_failure_reason="failure_reason=rotate right,grasp appropriate.",
        prompt_profile=profile,
        noise_seed=42,
        rotation_direction="right",
        quit_key="q",
        continue_key="s",
        history_freeze_delay_seconds=1.0,
    )


def _observation():
    return client.FrozenObservation(
        img_front=np.zeros((8, 8, 3), dtype=np.uint8),
        img_left=np.zeros((8, 8, 3), dtype=np.uint8),
        qpos=np.zeros((7,), dtype=np.float32),
        timestamp=1.0,
        state_history=np.zeros((60, 7), dtype=np.float32),
        state_history_mask=np.ones((60,), dtype=np.bool_),
        tactile_caption="Touch[area=none; Fz=near_zero]",
    )


def test_v5_uses_three_distinct_prompts_and_execution_mode(tmp_path: Path) -> None:
    args = _args("phase_v1")
    policy = _Policy()
    logger = client.TrialLogger(tmp_path, trial_id=None)
    for phase in ("execution", "reposition", "adjustment"):
        client._request_action_chunk(
            args=args,
            policy=policy,
            logger=logger,
            observation=_observation(),
            phase=phase,
            phase_index=0,
        )
    prompts = [request["prompt"] for request in policy.requests]
    assert prompts[0] == "Mode: execution. Task: Pick object."
    assert prompts[1].startswith("Mode: reposition.")
    assert prompts[2].startswith("Mode: adjustment.") and PLAN in prompts[2]
    assert all(request["mode"] == "execution" for request in policy.requests)
    assert all("Touch[" not in prompt for prompt in prompts)


def test_v52_uses_two_prompts_and_never_sends_reposition(tmp_path: Path) -> None:
    args = _args("phase_v2")
    policy = _Policy()
    logger = client.TrialLogger(tmp_path, trial_id=None)
    for phase in ("execution", "adjustment"):
        client._request_action_chunk(
            args=args,
            policy=policy,
            logger=logger,
            observation=_observation(),
            phase=phase,
            phase_index=0,
        )
    prompts = [request["prompt"] for request in policy.requests]
    assert prompts == [
        "Mode: execution. Task: Pick object.",
        (
            "Mode: adjustment. Task: Pick object.\n"
            "Put the object back, and follow this recovery plan: " + PLAN
        ),
    ]
    assert all(request["mode"] == "execution" for request in policy.requests)
    with pytest.raises(ValueError, match="reposition"):
        client._request_action_chunk(
            args=args,
            policy=policy,
            logger=logger,
            observation=_observation(),
            phase="reposition",
            phase_index=0,
        )


def test_v4_reposition_and_adjustment_use_same_old_recovery_prompt(tmp_path: Path) -> None:
    args = _args("minimal_v1")
    policy = _Policy()
    logger = client.TrialLogger(tmp_path, trial_id=None)
    for phase in ("reposition", "adjustment"):
        client._request_action_chunk(
            args=args,
            policy=policy,
            logger=logger,
            observation=_observation(),
            phase=phase,
            phase_index=0,
        )
    assert policy.requests[0]["prompt"] == policy.requests[1]["prompt"]
    assert policy.requests[0]["prompt"].startswith("Mode: execution.")


def test_phase_selection_is_pause_only_and_does_not_execute() -> None:
    args = _args("phase_v1")
    keyboard = _Keyboard(["2", "s"])
    selected = client._poll_control_key(args, keyboard, phase="execution", chunk_paused=True)
    assert selected.kind == "select"
    assert selected.selected_phase == "reposition"
    continued = client._poll_control_key(args, keyboard, phase="execution", chunk_paused=True)
    assert continued.kind == "continue"
    assert client._poll_control_key(
        args, _Keyboard(["3"]), phase="execution", chunk_paused=False
    ) is None


def test_v52_phase_controls_start_attempt2_in_adjustment() -> None:
    args = _args("phase_v2")
    selected = client._poll_control_key(
        args,
        _Keyboard(["2"]),
        phase="execution",
        chunk_paused=True,
    )
    assert selected.kind == "select"
    assert selected.selected_phase == "adjustment"
    assert client._poll_control_key(
        args,
        _Keyboard(["3"]),
        phase="execution",
        chunk_paused=True,
    ) is None
    assert client.forced_attempt_start_phase("phase_v2") == "adjustment"
    assert client.forced_attempt_start_phase("phase_v1") == "reposition"


def test_v52_server_metadata_pair_is_strict() -> None:
    args = argparse.Namespace(
        expected_data_profile="rotation_phase_v5_adjustment_v2",
        state_history_len=60,
        state_history_fps=30.0,
        chunk_size=30,
    )
    metadata = {
        "action_only_ablation": True,
        "supports_action_noise": True,
        "action_noise_shape": [30, 32],
        "action_horizon": 30,
        "action_dim": 32,
        "output_action_dim": 7,
        "use_state_history": True,
        "state_history_len": 60,
        "state_history_dim": 7,
        "state_history_fps": 30.0,
        "checkpoint_kind": "stage-a",
        "data_profile": "rotation_phase_v5_adjustment_v2",
        "prompt_profile": "phase_v2",
        "experiment_kind": "phase_prompt_h30_terminal_hold",
    }
    client.validate_server_metadata(args, metadata)
    client.validate_server_metadata(
        args,
        metadata
        | {
            "stage_a_protocol": "v6_1_no_state_history",
            "use_state_history": False,
            "state_history_len": 0,
        },
    )
    with pytest.raises(ValueError, match="stage_a_protocol"):
        client.validate_server_metadata(
            args,
            metadata
            | {
                "use_state_history": False,
                "state_history_len": 0,
            },
        )
    with pytest.raises(ValueError, match="experiment/prompt profile mismatch"):
        client.validate_server_metadata(
            args,
            metadata | {"experiment_kind": "phase_prompt_only"},
        )
    with pytest.raises(ValueError, match="requires a Stage A checkpoint"):
        client.validate_server_metadata(
            args,
            metadata | {"checkpoint_kind": "stage-b"},
        )


def test_history_snapshot_waits_then_requires_a_post_chunk_frame(monkeypatch) -> None:
    args = _args("phase_v1")
    args.history_freeze_delay_seconds = 1.0
    sleep_calls = []
    capture_calls = []
    expected = _observation()

    monkeypatch.setattr(client.time, "sleep", sleep_calls.append)

    def capture(args, operator, captioner, *, reset_history, after_timestamp=None):
        capture_calls.append((args, operator, captioner, reset_history, after_timestamp))
        return expected

    monkeypatch.setattr(client, "_capture_observation", capture)
    operator = _Operator()
    result = client._capture_delayed_history_snapshot(
        args,
        operator,
        None,
        chunk_completed_timestamp=12.5,
    )

    assert result is expected
    assert sleep_calls == [1.0]
    assert capture_calls == [(args, operator, None, False, 12.5)]


def test_continue_uses_live_observation_with_frozen_history(monkeypatch, tmp_path: Path) -> None:
    args = _args("phase_v1")
    args.observation_poll_rate = 200
    locked_history = _observation()
    locked_history = client.replace(
        locked_history,
        qpos=np.full(7, 1.0, dtype=np.float32),
        state_history=np.full((60, 7), 3.0, dtype=np.float32),
    )
    realtime = client.replace(
        _observation(),
        qpos=np.full(7, 8.0, dtype=np.float32),
        timestamp=51.0,
        state_history=np.full((60, 7), 9.0, dtype=np.float32),
    )
    capture_after = []

    monkeypatch.setattr(client.time, "time", lambda: 50.0)

    def capture(_args, _operator, _captioner, *, reset_history, after_timestamp=None):
        assert not reset_history
        capture_after.append(after_timestamp)
        return realtime

    monkeypatch.setattr(client, "_capture_observation", capture)
    operator = _PauseOperator()
    logger = client.TrialLogger(tmp_path, trial_id=None)
    signal = client._wait_for_chunk_continue(
        args=args,
        operator=operator,
        captioner=None,
        keyboard=_Keyboard(["3", "s"]),
        logger=logger,
        phase="reposition",
        phase_index=4,
        published_steps=30,
        locked_history=locked_history,
    )

    assert signal.kind == "continue"
    assert signal.selected_phase == "adjustment"
    assert signal.observation is not None
    np.testing.assert_array_equal(signal.observation.qpos, realtime.qpos)
    np.testing.assert_array_equal(signal.observation.state_history, locked_history.state_history)
    np.testing.assert_array_equal(
        signal.observation.state_history_mask,
        locked_history.state_history_mask,
    )
    assert capture_after == [50.0]
    assert not operator.history_paused
    assert len(operator.resumed) == 1
    np.testing.assert_array_equal(operator.resumed[0][0], locked_history.state_history)
    assert operator.resumed[0][2] == realtime.timestamp


def test_noise_is_stable_for_same_phase_chunk_and_seed() -> None:
    for phase in ("execution", "reposition", "adjustment"):
        first = client.deterministic_action_noise(9, phase=phase, index=4)
        second = client.deterministic_action_noise(9, phase=phase, index=4)
        np.testing.assert_array_equal(first, second)
    assert not np.array_equal(
        client.deterministic_action_noise(9, phase="reposition", index=4),
        client.deterministic_action_noise(9, phase="adjustment", index=4),
    )


def test_gripper_safety_floor_applies_after_offset() -> None:
    after_offset, published, was_clamped = client.gripper_command_with_safety_floor(
        0.020,
        gripper_offset=0.001,
        gripper_min=0.024,
    )
    assert after_offset == 0.019
    assert published == 0.024
    assert was_clamped

    after_offset, published, was_clamped = client.gripper_command_with_safety_floor(
        0.030,
        gripper_offset=0.001,
        gripper_min=0.024,
    )
    assert after_offset == 0.028999999999999998
    assert published == after_offset
    assert not was_clamped


def test_published_action_cannot_go_below_gripper_safety_floor(tmp_path: Path) -> None:
    args = argparse.Namespace(
        use_actions_interpolation=False,
        quit_key="q",
        gripper_offset=0.001,
        gripper_min=0.024,
        no_publish=True,
    )
    raw_action = np.zeros(7, dtype=np.float32)
    raw_action[6] = 0.020
    client.runtime.shutdown_event.clear()
    client.runtime.published_actions_history.clear()
    logger = client.TrialLogger(tmp_path, trial_id=None)

    _, published_count, control = client._publish_raw_action(
        args=args,
        operator=_Operator(),
        keyboard=_Keyboard([]),
        logger=logger,
        phase="execution",
        phase_index=0,
        action_index=0,
        raw_action=raw_action,
        pre_action=np.zeros(7, dtype=np.float32),
        control_rate=_ControlRate(),
    )

    assert published_count == 1
    assert control is None
    assert np.isclose(client.runtime.published_actions_history[-1][6], 0.024)
    event = logger.events_path.read_text().strip()
    assert '"gripper_min_applied":true' in event
    client.runtime.published_actions_history.clear()


def test_rotation_targets_support_all_directions_and_both_magnitudes() -> None:
    for direction in ("right", "left", "front", "back"):
        for magnitude in ("moderately", "slightly"):
            failure, plan = client.rotation_targets(direction, magnitude)
            assert f"rotate {direction}" in failure
            assert f"{direction} {magnitude}" in plan
