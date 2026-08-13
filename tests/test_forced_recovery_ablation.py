from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


client = _load_module(
    "forced_recovery_ablation_client_test_module",
    PROJECT_ROOT
    / "openpi/inference/agilex/inference/agilex_inference_forced_recovery_ablation.py",
)
server = _load_module(
    "forced_recovery_ablation_server_test_module",
    PROJECT_ROOT / "scripts/serve_tactile_vla_action_ablation.py",
)


FAILURE_REASON = "failure_reason=rotate right,grasp appropriate."
RECOVERY_PLAN = (
    "recovery_plan=move horizontally right moderately, "
    "move vertically none moderately."
)


class _FakeJointState:
    def __init__(self, value: float, timestamp: float) -> None:
        self.position = np.full((7,), value, dtype=np.float32)
        self.timestamp = timestamp


class _FakeOperator:
    tactile = None

    def __init__(self) -> None:
        self.frame_index = 0
        self.reset_count = 0
        self.published: list[np.ndarray] = []
        self.feedback: _FakeJointState | None = None

    def is_shutdown(self) -> bool:
        return False

    def rate(self, hz: int):
        assert hz in {30, 200}
        return client.runtime.NoopRate()

    def get_frame(self, *, after_timestamp=None):
        del after_timestamp
        value = float(self.frame_index)
        self.frame_index += 1
        image = np.full((8, 8, 3), int(value), dtype=np.uint8)
        return image, image, _FakeJointState(value, 100.0 + value)

    def reset_state_history(self) -> None:
        self.reset_count += 1

    def get_state_history(self, puppet_arm):
        history = np.repeat(np.asarray(puppet_arm.position)[None, :], 60, axis=0).astype(np.float32)
        mask = np.ones((60,), dtype=np.bool_)
        if self.reset_count:
            mask[:-1] = False
        return history, mask

    def puppet_arm_publish(self, action: np.ndarray) -> float:
        value = np.asarray(action, dtype=np.float32).copy()
        self.published.append(value)
        self.feedback = _FakeJointState(float(value[0]), 200.0 + len(self.published))
        return 300.0 + len(self.published)

    def get_latest_joint_state(self):
        return self.feedback


class _FakeKeyboard:
    def __init__(self, keys: list[str | None]) -> None:
        self.keys = list(keys)

    def get_key(self) -> str | None:
        return self.keys.pop(0) if self.keys else None


class _FakePolicy:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def get_server_metadata(self):
        return {
            "name": "test_action_ablation",
            "action_only_ablation": True,
            "checkpoint_kind": "stage-a",
            "checkpoint": "/tmp/stage-a/15000",
            "supports_action_noise": True,
            "requires_action_noise": True,
            "action_noise_shape": [30, 32],
            "action_horizon": 30,
            "action_dim": 32,
            "output_action_dim": 7,
            "use_state_history": True,
            "state_history_len": 60,
            "state_history_dim": 7,
            "state_history_fps": 30.0,
        }

    def infer(self, payload):
        self.requests.append(payload)
        state = np.asarray(payload["observation/state"], dtype=np.float32)
        increment = 0.005 if "Recovery plan: none." in payload["prompt"] else 0.006
        actions = np.repeat((state + increment)[None, :], 30, axis=0)
        return {
            "raw_model_actions": np.zeros((30, 32), dtype=np.float32),
            "actions": actions,
            "policy_timing": {"infer_ms": 1.0},
        }


def _args(tmp_path: Path, *, max_publish_step: int) -> argparse.Namespace:
    return argparse.Namespace(
        rotation_direction="right",
        state_history_len=60,
        state_history_fps=30.0,
        state_history_max_gap_seconds=0.02,
        chunk_size=30,
        publish_rate=30,
        observation_poll_rate=200,
        max_publish_step=max_publish_step,
        instruction="test task",
        forced_failure_reason=FAILURE_REASON,
        forced_recovery_plan=RECOVERY_PLAN,
        noise_seed=42,
        quit_key="q",
        success_key="s",
        use_actions_interpolation=True,
        arm_steps_length=[0.01] * 7,
        gripper_offset=0.0,
        no_publish=False,
        log_dir=tmp_path,
        trial_id=None,
    )


def _events(logger) -> list[dict]:
    return [json.loads(line) for line in logger.events_path.read_text().splitlines()]


def test_deterministic_noise_is_phase_indexed() -> None:
    normal_0 = client.deterministic_action_noise(42, phase="normal", index=0)
    normal_0_again = client.deterministic_action_noise(42, phase="normal", index=0)
    normal_1 = client.deterministic_action_noise(42, phase="normal", index=1)
    recovery_0 = client.deterministic_action_noise(42, phase="recovery", index=0)

    assert normal_0.shape == (30, 32)
    assert normal_0.dtype == np.float32
    np.testing.assert_array_equal(normal_0, normal_0_again)
    assert not np.array_equal(normal_0, normal_1)
    assert not np.array_equal(normal_0, recovery_0)
    assert client.noise_sha256(normal_0) == client.noise_sha256(normal_0_again)


@pytest.mark.parametrize("direction", ["right", "left", "front", "back"])
def test_rotation_direction_builds_moderately_targets(direction: str, tmp_path: Path) -> None:
    args = _args(tmp_path, max_publish_step=1)
    args.rotation_direction = direction
    args.forced_failure_reason = None
    args.forced_recovery_plan = None
    parser = argparse.ArgumentParser()

    client.validate_args(args, parser)

    assert args.forced_failure_reason == f"failure_reason=rotate {direction},grasp appropriate."
    assert args.forced_recovery_plan == (
        f"recovery_plan=move horizontally {direction} moderately, "
        "move vertically none moderately."
    )


def test_rotation_direction_rejects_conflicting_explicit_plan(tmp_path: Path) -> None:
    args = _args(tmp_path, max_publish_step=1)
    args.rotation_direction = "left"
    parser = argparse.ArgumentParser()

    with pytest.raises(SystemExit):
        client.validate_args(args, parser)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "must include action_noise"),
        (np.zeros((30, 31), dtype=np.float32), "shape"),
        (np.full((30, 32), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_server_rejects_invalid_action_noise(value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        server.validate_action_noise(value, action_horizon=30, action_dim=32)


def test_server_passes_fixed_noise_to_backbone_and_returns_both_action_spaces() -> None:
    policy = object.__new__(server.ActionOnlyAblationPolicy)
    policy._config = SimpleNamespace(action_horizon=30, action_dim=32)
    policy._sample_rng = "rng"
    policy._num_inference_steps = 10
    captured = {}

    def sample_actions(rng, observation, *, num_steps, noise):
        captured.update(rng=rng, observation=observation, num_steps=num_steps, noise=np.asarray(noise))
        return noise

    policy._sample_actions = sample_actions
    policy._output_transform = lambda data: data
    noise = np.arange(30 * 32, dtype=np.float32).reshape(30, 32)

    raw, actions = policy._actions({"state": np.zeros((32,), dtype=np.float32)}, "observation", noise)

    assert captured["rng"] == "rng"
    assert captured["observation"] == "observation"
    assert captured["num_steps"] == 10
    np.testing.assert_array_equal(captured["noise"], noise[None, ...])
    np.testing.assert_array_equal(raw, noise)
    np.testing.assert_array_equal(actions, noise[:, :7])


def test_space_discards_normal_chunk_and_only_requests_forced_recovery(tmp_path: Path) -> None:
    client.runtime.shutdown_event.clear()
    client.runtime.published_actions_history.clear()
    args = _args(tmp_path, max_publish_step=2)
    operator = _FakeOperator()
    policy = _FakePolicy()
    # Boundary poll, first NORMAL action, second NORMAL action -> trigger,
    # then RECOVERY boundary and first RECOVERY action.
    keyboard = _FakeKeyboard([None, None, " ", None, None])
    logger = client.TrialLogger(tmp_path, trial_id=None)

    client.run_ablation(args, operator, policy, captioner=None, keyboard=keyboard, logger=logger)

    assert operator.reset_count == 1
    assert len(policy.requests) == 2
    assert len(operator.published) == 2
    normal_request, recovery_request = policy.requests
    assert "Recovery plan: none." in normal_request["prompt"]
    assert RECOVERY_PLAN in recovery_request["prompt"]
    assert FAILURE_REASON not in normal_request["prompt"]
    assert FAILURE_REASON not in recovery_request["prompt"]
    np.testing.assert_array_equal(
        recovery_request["action_noise"],
        client.deterministic_action_noise(42, phase="recovery", index=0),
    )
    # The trigger frame has qpos=1.0. One interpolation command proves
    # pre_action was reset to that qpos instead of retaining the NORMAL target.
    np.testing.assert_allclose(operator.published[1], np.full((7,), 1.006, dtype=np.float32))
    np.testing.assert_array_equal(
        recovery_request["observation/state_history_mask"],
        np.r_[np.zeros(59, dtype=np.bool_), np.ones(1, dtype=np.bool_)],
    )

    events = _events(logger)
    assert sum(event["event"] == "forced_recovery_trigger" for event in events) == 1
    assert sum(event["event"] == "action_chunk_inference" for event in events) == 2
    assert any(event["event"] == "action_publish" and event["feedback_qpos"] is not None for event in events)
    trigger = next(event for event in events if event["event"] == "forced_recovery_trigger")
    # max_publish_step=2 limits this synthetic chunk to two raw actions.
    assert trigger["discarded_raw_actions"] == 1
    assert Path(trigger["front_image"]).is_file()
    assert Path(trigger["left_image"]).is_file()


def test_space_at_chunk_boundary_switches_before_another_normal_request(tmp_path: Path) -> None:
    client.runtime.shutdown_event.clear()
    client.runtime.published_actions_history.clear()
    args = _args(tmp_path, max_publish_step=1)
    operator = _FakeOperator()
    policy = _FakePolicy()
    keyboard = _FakeKeyboard([" ", None])
    logger = client.TrialLogger(tmp_path, trial_id=None)

    client.run_ablation(args, operator, policy, captioner=None, keyboard=keyboard, logger=logger)

    assert len(policy.requests) == 1
    assert RECOVERY_PLAN in policy.requests[0]["prompt"]
    assert "Recovery plan: none." not in policy.requests[0]["prompt"]
    assert operator.reset_count == 1


def test_server_metadata_must_identify_action_ablation() -> None:
    args = argparse.Namespace(state_history_len=60, state_history_fps=30.0, chunk_size=30)
    with pytest.raises(ValueError, match="not the action-only"):
        client.validate_server_metadata(args, {"action_only_ablation": False})
