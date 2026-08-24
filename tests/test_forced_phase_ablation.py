from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


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


def test_noise_is_stable_for_same_phase_chunk_and_seed() -> None:
    for phase in ("execution", "reposition", "adjustment"):
        first = client.deterministic_action_noise(9, phase=phase, index=4)
        second = client.deterministic_action_noise(9, phase=phase, index=4)
        np.testing.assert_array_equal(first, second)
    assert not np.array_equal(
        client.deterministic_action_noise(9, phase="reposition", index=4),
        client.deterministic_action_noise(9, phase="adjustment", index=4),
    )


def test_rotation_targets_support_all_directions_and_both_magnitudes() -> None:
    for direction in ("right", "left", "front", "back"):
        for magnitude in ("moderately", "slightly"):
            failure, plan = client.rotation_targets(direction, magnitude)
            assert f"rotate {direction}" in failure
            assert f"{direction} {magnitude}" in plan
