from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inference = _load_module(
    "agilex_inference_tactile_vla_sync_single_test_module",
    PROJECT_ROOT / "openpi/inference/agilex/inference/agilex_inference_tactile_vla_sync_single.py",
)
server = _load_module(
    "serve_tactile_vla_policy_test_module",
    PROJECT_ROOT / "scripts/serve_tactile_vla_policy.py",
)
v3_server = _load_module(
    "serve_tactile_vla_policy_v3_test_module",
    PROJECT_ROOT / "scripts/serve_tactile_vla_policy_v3.py",
)


class _FakeJointState:
    def __init__(self, value: float) -> None:
        self.position = np.full((7,), value, dtype=np.float32)


class _FakeOperator:
    tactile = None

    def __init__(self) -> None:
        self.frame_index = 0
        self.published: list[np.ndarray] = []

    def reset_state_history(self) -> None:
        pass

    def rate(self, hz: int):
        assert hz in {30, 200}
        return inference.NoopRate()

    def is_shutdown(self) -> bool:
        return False

    def get_frame(self, *, after_timestamp=None):
        del after_timestamp
        value = float(self.frame_index)
        self.frame_index += 1
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return image, image, _FakeJointState(value)

    def get_latest_frame(self):
        return self.get_frame()

    def get_state_history(self, puppet_arm):
        history = np.repeat(np.asarray(puppet_arm.position)[None, :], 60, axis=0)
        return history.astype(np.float32), np.ones((60,), dtype=np.bool_)

    def puppet_arm_publish(self, action: np.ndarray) -> None:
        self.published.append(np.asarray(action))


class _FakePolicy:
    def __init__(self) -> None:
        self.modes: list[str] = []
        self.monitor_states: list[np.ndarray] = []

    def get_server_metadata(self):
        return {
            "action_horizon": 30,
            "use_state_history": True,
            "state_history_len": 60,
            "state_history_dim": 7,
            "state_history_fps": 30.0,
            "supports_step_monitor": True,
        }

    def infer(self, payload):
        mode = payload["mode"]
        self.modes.append(mode)
        if mode == "execution":
            state = np.asarray(payload["observation/state"])
            return {
                "actions": np.repeat(state[None, :], 2, axis=0),
                "policy_timing": {"infer_ms": 1.0},
            }
        if mode == "monitor":
            self.monitor_states.append(np.asarray(payload["observation/state"]))
            time.sleep(0.05)
            trigger = True
            return {
                "need_recovery": trigger,
                "need_recovery_probs": [0.0, 1.0] if trigger else [1.0, 0.0],
                "failure_reason": "tilted left",
                "failure_reason_probs": [1.0, 0.0, 0.0, 0.0],
                "policy_timing": {"infer_ms": 2.0},
            }
        raise AssertionError(f"Unexpected mode: {mode}")


class _FakeV3SharedPolicy(_FakePolicy):
    def get_server_metadata(self):
        return {
            "stage_b_version": "v3_autoregressive",
            "action_horizon": 30,
            "use_state_history": True,
            "state_history_len": 60,
            "state_history_dim": 7,
            "state_history_fps": 30.0,
            "supports_step_monitor": True,
            "supports_shared_assessment": True,
            "supports_failure_generation": True,
        }

    def infer(self, payload):
        mode = payload["mode"]
        self.modes.append(mode)
        if mode == "execution":
            state = np.asarray(payload["observation/state"])
            return {
                "actions": np.repeat(state[None, :], 2, axis=0),
                "policy_timing": {"infer_ms": 1.0},
            }
        if mode == "assessment":
            return {
                "need_recovery": True,
                "need_recovery_probs": [0.0, 1.0],
                "failure_reason": "failure_reason=rotate right,grasp appropriate.",
                "policy_timing": {"infer_ms": 2.0},
            }
        raise AssertionError(f"Shared V3 client must not send mode={mode!r}")


class _FakeTactile:
    caption_text = "Touch[area=medium; Fx=positive; Fy=negative; Fz=negative; rotation=clockwise]"

    def caption(self, captioner):
        del captioner
        return self.caption_text


class _FakeV3RecoveryPolicy(_FakeV3SharedPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.execution_prompts: list[str] = []

    def infer(self, payload):
        mode = payload["mode"]
        self.modes.append(mode)
        if mode == "execution":
            self.execution_prompts.append(str(payload["prompt"]))
            state = np.asarray(payload["observation/state"])
            return {
                "actions": np.repeat(state[None, :], 2, axis=0),
                "policy_timing": {"infer_ms": 1.0},
            }
        if mode == "assessment":
            return {
                "need_recovery": True,
                "need_recovery_probs": [0.0, 1.0],
                "failure_reason": "failure_reason=rotate right,grasp appropriate.",
                "policy_timing": {"infer_ms": 2.0},
            }
        if mode == "reasoning":
            return {"recovery_plan": "move horizontally left moderately, move vertically none slightly"}
        raise AssertionError(f"Unexpected mode: {mode!r}")


def test_closed_loop_runs_monitor_in_background_with_latest_observation() -> None:
    inference.shutdown_event.clear()
    inference.published_actions_history.clear()
    args = argparse.Namespace(
        chunk_size=2,
        state_history_len=60,
        state_history_fps=30.0,
        state_history_max_gap_seconds=0.02,
        start_immediately=True,
        replay_attempt_dir=None,
        max_attempts=1,
        recovery_tactile_ignore_seconds=0.0,
        publish_rate=30,
        observation_poll_rate=200,
        max_publish_step=2,
        instruction="test",
        case_id="test",
        use_actions_interpolation=False,
        arm_steps_length=[0.01] * 7,
        gripper_offset=0.0,
        no_publish=False,
        memory_log=None,
        quit_key="q",
        success_key="s",
    )
    operator = _FakeOperator()
    policy = _FakePolicy()

    inference.run_closed_loop(args, operator, policy, captioner=None)

    assert policy.modes == ["execution", "monitor"]
    assert len(operator.published) == 2
    np.testing.assert_array_equal(policy.monitor_states[0], np.ones((7,), dtype=np.float32))


def test_v3_closed_loop_gets_need_and_failure_from_one_assessment_request() -> None:
    inference.shutdown_event.clear()
    inference.published_actions_history.clear()
    args = argparse.Namespace(
        chunk_size=2,
        state_history_len=60,
        state_history_fps=30.0,
        state_history_max_gap_seconds=0.02,
        start_immediately=True,
        replay_attempt_dir=None,
        max_attempts=1,
        recovery_tactile_ignore_seconds=0.0,
        publish_rate=30,
        observation_poll_rate=200,
        max_publish_step=2,
        instruction="test",
        case_id="test",
        use_actions_interpolation=False,
        arm_steps_length=[0.01] * 7,
        gripper_offset=0.0,
        no_publish=False,
        memory_log=None,
        quit_key="q",
        success_key="s",
    )
    operator = _FakeOperator()
    policy = _FakeV3SharedPolicy()

    inference.run_closed_loop(args, operator, policy, captioner=None)

    assert policy.modes == ["execution", "assessment"]


def test_recovery_ignore_suppresses_assessment_but_keeps_live_tactile_for_action() -> None:
    inference.shutdown_event.clear()
    inference.published_actions_history.clear()
    args = argparse.Namespace(
        chunk_size=2,
        state_history_len=60,
        state_history_fps=30.0,
        state_history_max_gap_seconds=0.02,
        start_immediately=True,
        replay_attempt_dir=None,
        max_attempts=2,
        recovery_tactile_ignore_seconds=5.0,
        publish_rate=30,
        observation_poll_rate=200,
        max_publish_step=2,
        instruction="test",
        case_id="test",
        use_actions_interpolation=False,
        arm_steps_length=[0.01] * 7,
        gripper_offset=0.0,
        no_publish=False,
        memory_log=None,
        quit_key="q",
        success_key="s",
    )
    operator = _FakeOperator()
    operator.tactile = _FakeTactile()
    policy = _FakeV3RecoveryPolicy()

    inference.run_closed_loop(args, operator, policy, captioner=None)

    assert policy.modes == ["execution", "assessment", "reasoning", "execution"]
    assert _FakeTactile.caption_text in policy.execution_prompts[-1]


def test_server_monitor_mode_never_samples_actions() -> None:
    policy = object.__new__(server.TactileVLAPolicy)
    policy._prepare_observation = lambda request: ("transformed", "observation")
    policy._predict_heads = lambda observation: {
        "need_recovery": np.asarray([0.0, 1.0], dtype=np.float32),
        "failure_reason": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    policy._need_recovery_threshold = 0.5
    policy._sample_absolute_actions = lambda transformed, observation: (_ for _ in ()).throw(
        AssertionError("monitor mode must not sample actions")
    )

    response = policy.infer({"mode": "monitor"})

    assert response["need_recovery"] is True
    assert response["failure_reason"] == "tilted left"
    assert "actions" not in response


def test_v3_execution_does_not_run_the_need_head() -> None:
    policy = object.__new__(v3_server.TactileVLAPolicyV3)
    policy._prepare = lambda request, mode: ("transformed", "observation")
    policy._actions = lambda transformed, observation: np.zeros((30, 7), dtype=np.float32)
    policy._need_response = lambda observation: (_ for _ in ()).throw(
        AssertionError("V3 execution must leave recovery decisions to shared assessment")
    )

    response = policy.infer({"mode": "execution"})

    assert np.asarray(response["actions"]).shape == (30, 7)
    assert "need_recovery" not in response


def test_v3_server_model_config_restores_state_history_from_checkpoint_config() -> None:
    args = argparse.Namespace(precision="float32")
    config = {
        "action_dim": 32,
        "action_horizon": 30,
        "max_token_len": 200,
        "use_state_history": True,
        "state_history_len": 60,
        "state_history_dim": 7,
        "history_hidden_dim": 256,
    }

    model_config = v3_server._model_config(args, config)

    assert model_config.use_state_history is True
    assert model_config.state_history_len == 60
    assert model_config.state_history_dim == 7
    assert model_config.history_hidden_dim == 256
