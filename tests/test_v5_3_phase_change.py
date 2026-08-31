from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.v5_3_phase_change import QPOS_SAMPLE_OFFSETS
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v5_3_phase_change import build_adjustment_end_prompt
from tactile_vla.vla.v5_3_phase_change import build_phase_change_prompt
from tactile_vla.vla.v5_3_phase_change import discretize_normalized_state
from tactile_vla.vla.v5_3_phase_change import h30_endpoint_indices
from tactile_vla.vla.v5_3_phase_change import pi05_phase_change_prefix
from tactile_vla.vla.v5_3_phase_change import pi05_phase_change_token_length
from tactile_vla.vla.v5_3_phase_change import runtime_reachable_endpoint
from tactile_vla.vla.v5_3_phase_change import sample_qpos_h30


def test_sample_offsets_are_exact() -> None:
    history = np.arange(30 * 7, dtype=np.float64).reshape(30, 7)
    sampled = sample_qpos_h30(history)
    np.testing.assert_array_equal(sampled, history[list(QPOS_SAMPLE_OFFSETS)])


def test_pi05_discretization_has_no_extra_clip() -> None:
    normalized = np.asarray([[-1.1, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]])
    result = discretize_normalized_state(normalized)
    bins = np.linspace(-1.0, 1.0, 257)[:-1]
    np.testing.assert_array_equal(result, np.digitize(normalized, bins=bins) - 1)
    assert result[0, 0] == -1
    assert result[0, -1] == 255


def test_adjustment_prompt_is_byte_exact() -> None:
    discrete = np.arange(70, dtype=np.int32).reshape(10, 7)
    prompt = build_phase_change_prompt(
        mode="adjustment",
        instruction="Pick object",
        tactile_caption="Touch[area=none]",
        recovery_plan="move left",
        qpos_h10_discrete=discrete,
    )
    assert prompt == (
        "Mode: adjustment.\n"
        "Task: Pick object.\n"
        "Tac: Touch[area=none]\n"
        "Recovery plan: move left.\n"
        "qpos_h30:[[0,1,2,3,4,5,6],[7,8,9,10,11,12,13],"
        "[14,15,16,17,18,19,20],[21,22,23,24,25,26,27],"
        "[28,29,30,31,32,33,34],[35,36,37,38,39,40,41],"
        "[42,43,44,45,46,47,48],[49,50,51,52,53,54,55],"
        "[56,57,58,59,60,61,62],[63,64,65,66,67,68,69]]"
    )


def test_adjustment_prompt_builds_from_raw_h30() -> None:
    stats = StateQuantileStats(q01=np.zeros(7), q99=np.ones(7))
    history = np.linspace(0.0, 1.0, 30 * 7).reshape(30, 7)
    prompt, discrete = build_adjustment_end_prompt(
        instruction="Pick object.",
        tactile_caption="Touch[area=none]",
        recovery_plan="move right.",
        qpos_h30=history,
        stats=stats,
    )
    assert discrete.shape == (10, 7)
    assert prompt.startswith("Mode: adjustment.\nTask: Pick object.\n")
    assert "qpos_h30: [[" not in prompt


def test_endpoint_rule_is_start_plus_30_not_29() -> None:
    assert h30_endpoint_indices(start_offset=0, stop_frame=95) == [30, 60, 90]
    assert h30_endpoint_indices(start_offset=1, stop_frame=95) == [31, 61, 91]
    assert not runtime_reachable_endpoint(29)
    assert runtime_reachable_endpoint(30)
    assert not runtime_reachable_endpoint(31)


def test_rejects_bad_history_shape_and_empty_fields() -> None:
    with pytest.raises(ValueError, match="shape"):
        sample_qpos_h30(np.zeros((29, 7)))
    with pytest.raises(ValueError, match="instruction"):
        build_phase_change_prompt(
            mode="adjustment",
            instruction="",
            tactile_caption="Touch[area=none]",
            recovery_plan="move left",
            qpos_h10_discrete=np.zeros((10, 7), dtype=np.int32),
        )


def test_pi05_token_audit_uses_exact_cleaned_prefix() -> None:
    class FakeTokenizer:
        def __init__(self) -> None:
            self.seen: tuple[str, bool] | None = None

        def encode_text(self, text: str, *, add_bos: bool = False) -> list[int]:
            self.seen = (text, add_bos)
            return [1, 2, 3]

    prompt = "Mode: adjustment.\nTask: pick_object."
    state = np.asarray([-1.1, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    expected = (
        "Task: Mode: adjustment. Task: pick object., "
        "State: -1 0 64 128 192 255 255;\nAction: "
    )
    assert pi05_phase_change_prefix(prompt=prompt, normalized_current_qpos=state) == expected
    tokenizer = FakeTokenizer()
    assert pi05_phase_change_token_length(
        tokenizer=tokenizer,
        prompt=prompt,
        normalized_current_qpos=state,
    ) == 3
    assert tokenizer.seen == (expected, True)
