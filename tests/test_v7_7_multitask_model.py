from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_v77_trainable_filter_names_both_heads():
    source = Path("src/tactile_vla/vla/v7_7_multitask_model.py").read_text()
    assert 'PathRegex(".*need_head.*")' in source
    assert 'PathRegex(".*adjustment_head.*")' in source
    assert 'PathRegex(".*backbone.*llm.*lora.*")' in source


def test_v77_training_cycle_and_v74_initialization_are_pinned():
    source = Path("scripts/train_vla_multitask_v7_7.py").read_text()
    assert "pi05_delta_tac_rotation_phase_v7_4_no_history/15000" in source
    assert "base.TASK_CYCLE = TASK_CYCLE" in source
    assert '"delta_params"' in source and '"full_params"' in source


def test_v77_server_uses_client_supplied_action_noise():
    from scripts.serve_tactile_vla_v7_7 import V77Policy

    policy = object.__new__(V77Policy)
    policy._model = SimpleNamespace(backbone=SimpleNamespace(action_horizon=30, action_dim=32))
    policy._rng = None
    policy._num_inference_steps = 10
    policy._output_action_dim = 7
    received = {}

    def sample(_rng, _observation, *, num_steps, noise):
        received["num_steps"] = num_steps
        received["noise"] = np.asarray(noise)
        return noise

    policy._sample_actions = sample
    policy._output_transform = lambda values: values
    noise = np.arange(30 * 32, dtype=np.float32).reshape(30, 32)
    raw, actions = policy._actions_with_noise(
        {"state": np.zeros(7, dtype=np.float32)}, object(), noise,
    )
    assert received["num_steps"] == 10
    np.testing.assert_array_equal(received["noise"], noise[None])
    np.testing.assert_array_equal(raw, noise)
    np.testing.assert_array_equal(actions, noise[:, :7])
