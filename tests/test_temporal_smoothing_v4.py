from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "openpi/inference/agilex/inference/agilex_inference_openpi_sync_single_temporal_smothing.py"
)


def _load_module():
    name = "agilex_inference_openpi_sync_single_temporal_smothing_test_module"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


client = _load_module()


def _chunk(value: float, *, gripper: float | None = None) -> np.ndarray:
    actions = np.full((30, 7), value, dtype=np.float64)
    if gripper is not None:
        actions[:, 6] = gripper
    return actions


def test_initial_chunk_is_anchored_to_current_qpos_over_short_window() -> None:
    smoother = client.ShortHorizonTemporalSmoother(smooth_steps=5, smooth_joints=6)
    smoother.reset(generation=1)

    stats = smoother.integrate(
        _chunk(1.0, gripper=0.8),
        generation=1,
        request_start_step=0,
        receive_step=0,
        max_latency_steps=29,
        anchor_qpos=np.zeros((7,), dtype=float),
    )

    assert stats.accepted
    assert stats.anchored_to_qpos
    assert stats.blended_actions == 5
    expected_arm = [0.0, 0.25, 0.5, 0.75, 1.0]
    for step, expected in enumerate(expected_arm):
        action = smoother.pop(step)
        assert action is not None
        np.testing.assert_allclose(action[:6], expected)
        # With smooth_joints=6 the gripper always follows the newest chunk.
        assert action[6] == 0.8


def test_new_chunk_drops_measured_latency_prefix_and_blends_only_five_steps() -> None:
    smoother = client.ShortHorizonTemporalSmoother(smooth_steps=5, smooth_joints=6)
    smoother.reset(generation=7)
    smoother.integrate(
        _chunk(0.0),
        generation=7,
        request_start_step=0,
        receive_step=0,
        max_latency_steps=29,
        anchor_qpos=np.zeros((7,), dtype=float),
    )
    for step in range(4):
        assert smoother.pop(step) is not None

    new = _chunk(10.0, gripper=0.9)
    stats = smoother.integrate(
        new,
        generation=7,
        request_start_step=2,
        receive_step=4,
        max_latency_steps=29,
        anchor_qpos=np.full((7,), -1.0),
    )

    assert stats.accepted
    assert stats.latency_steps == 2
    assert stats.first_action_index == 2
    assert not stats.anchored_to_qpos
    expected_arm = [0.0, 2.5, 5.0, 7.5, 10.0, 10.0]
    for step, expected in zip(range(4, 10), expected_arm, strict=True):
        action = smoother.pop(step)
        assert action is not None
        np.testing.assert_allclose(action[:6], expected)
        assert action[6] == 0.9


def test_seven_joint_mode_smooths_gripper_too() -> None:
    smoother = client.ShortHorizonTemporalSmoother(smooth_steps=3, smooth_joints=7)
    smoother.reset(generation=1)
    stats = smoother.integrate(
        _chunk(1.0, gripper=0.6),
        generation=1,
        request_start_step=0,
        receive_step=0,
        max_latency_steps=29,
        anchor_qpos=np.zeros((7,), dtype=float),
    )

    assert stats.accepted
    assert [smoother.pop(step)[6] for step in range(3)] == [0.0, 0.3, 0.6]


def test_hard_reset_rejects_late_old_generation_and_clears_actions() -> None:
    smoother = client.ShortHorizonTemporalSmoother(smooth_steps=5, smooth_joints=6)
    smoother.reset(generation=1)
    smoother.integrate(
        _chunk(1.0),
        generation=1,
        request_start_step=0,
        receive_step=0,
        max_latency_steps=29,
        anchor_qpos=np.zeros((7,), dtype=float),
    )
    smoother.reset(generation=2)

    stats = smoother.integrate(
        _chunk(2.0),
        generation=1,
        request_start_step=5,
        receive_step=6,
        max_latency_steps=29,
        anchor_qpos=np.zeros((7,), dtype=float),
    )

    assert not stats.accepted
    assert stats.reason == "generation_mismatch"
    assert smoother.pop(6) is None


def test_expired_or_excessively_delayed_chunks_do_not_replace_old_plan() -> None:
    smoother = client.ShortHorizonTemporalSmoother(smooth_steps=5, smooth_joints=6)
    smoother.reset(generation=1)
    smoother.integrate(
        _chunk(1.0),
        generation=1,
        request_start_step=0,
        receive_step=0,
        max_latency_steps=29,
        anchor_qpos=np.zeros((7,), dtype=float),
    )

    stats = smoother.integrate(
        _chunk(2.0),
        generation=1,
        request_start_step=0,
        receive_step=9,
        max_latency_steps=8,
        anchor_qpos=None,
    )

    assert not stats.accepted
    assert stats.reason == "latency_limit"
    action = smoother.pop(9)
    assert action is not None
    np.testing.assert_allclose(action, 1.0)


def test_cli_defaults_to_five_steps_and_first_six_joints(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(MODULE_PATH)])
    args, _ = client.get_arguments()

    assert args.temporal_smooth_steps == 5
    assert args.temporal_smooth_joints == 6
    assert args.action_inference_rate == 3.0
    assert args.max_action_latency_steps is None


def test_cli_can_select_all_seven_joints(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--temporal-smooth-joints",
            "7",
            "--temporal-smooth-steps",
            "3",
            "--action-inference-rate",
            "2",
        ],
    )
    args, _ = client.get_arguments()

    assert args.temporal_smooth_joints == 7
    assert args.temporal_smooth_steps == 3
    assert args.action_inference_rate == 2.0


class _FakeOperator:
    tactile = None

    def __init__(self) -> None:
        self.qpos = np.zeros((7,), dtype=np.float32)
        self.published: list[np.ndarray] = []

    def is_shutdown(self) -> bool:
        return False

    def reset_state_history(self) -> None:
        pass

    def rate(self, hz: int):
        assert hz in (30, 200)
        return client.base.NoopRate()

    def get_latest_frame(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return image, image, SimpleNamespace(position=self.qpos.copy())

    def get_state_history(self, puppet_arm):
        history = np.repeat(np.asarray(puppet_arm.position)[None, :], 60, axis=0)
        return history.astype(np.float32), np.ones((60,), dtype=np.bool_)

    def puppet_arm_publish(self, action: np.ndarray) -> float:
        self.qpos = np.asarray(action, dtype=np.float32).copy()
        self.published.append(self.qpos.copy())
        return float(len(self.published)) / 30.0


class _FakePolicy:
    def __init__(self, *, action: bool) -> None:
        self.action = action
        self.modes: list[str] = []

    def get_server_metadata(self):
        return {
            "data_profile": "rotation_v4",
            "prompt_profile": "minimal_v1",
            "action_horizon": 30,
            "output_action_dim": 7,
            "use_state_history": True,
            "state_history_len": 60,
            "state_history_dim": 7,
            "state_history_fps": 30.0,
            "supports_step_monitor": True,
            "max_memory_pairs": 4,
            "max_supported_attempts": 5,
        }

    def infer(self, payload):
        mode = str(payload["mode"])
        self.modes.append(mode)
        if self.action:
            assert mode == "execution"
            state = np.asarray(payload["observation/state"], dtype=np.float32)
            return {
                "actions": np.repeat((state + 0.1)[None, :], 30, axis=0),
                "policy_timing": {"infer_ms": 1.0},
            }
        assert mode == "monitor"
        return {
            "need_recovery": False,
            "need_recovery_probs": [1.0, 0.0],
            "failure_reason": "unused",
            "policy_timing": {"infer_ms": 1.0},
        }


def test_closed_loop_uses_async_action_socket_and_publishes_smoothed_h30_prefix() -> None:
    client.shutdown_event.clear()
    operator = _FakeOperator()
    action_policy = _FakePolicy(action=True)
    assessment_policy = _FakePolicy(action=False)
    args = SimpleNamespace(
        expected_data_profile="rotation_v4",
        max_attempts=1,
        prompt_profile=None,
        chunk_size=30,
        state_history_len=60,
        state_history_fps=30.0,
        state_history_max_gap_seconds=0.02,
        max_action_latency_steps=None,
        publish_rate=30,
        action_inference_rate=3.0,
        temporal_smooth_steps=5,
        temporal_smooth_joints=6,
        start_immediately=True,
        replay_attempt_dir=None,
        instruction="test task",
        case_id="test",
        recovery_tactile_ignore_seconds=0.0,
        max_publish_step=5,
        use_actions_interpolation=False,
        arm_steps_length=[0.01] * 6 + [0.2],
        gripper_offset=0.0,
        no_publish=False,
        quit_key="q",
        success_key="s",
        memory_log=None,
        v3_autoregressive=False,
        v3_shared_assessment=False,
        observation_poll_rate=200,
    )

    client.run_closed_loop(args, operator, action_policy, assessment_policy, None)

    assert len(operator.published) == 5
    assert action_policy.modes == ["execution"]
    assert assessment_policy.modes
    np.testing.assert_allclose([action[0] for action in operator.published], [0.0, 0.025, 0.05, 0.075, 0.1])
