from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v7_5_runtime_history import RUNTIME_SAMPLE_OFFSETS
from tactile_vla.vla.v7_5_runtime_history import build_runtime_adjustment_end_prompt


def test_runtime_h100_includes_current_qpos_and_uniformly_samples_11_rows() -> None:
    past = np.arange(99 * 7, dtype=np.float32).reshape(99, 7)
    current = np.full(7, 9_999, dtype=np.float32)
    prompt, discrete, h100 = build_runtime_adjustment_end_prompt(
        instruction="Pick object",
        tactile_caption="Touch[area=none; Fx=near_zero; Fy=near_zero; Fz=near_zero; rotation=none]",
        recovery_plan="move horizontally left slightly",
        past_qpos_h99=past,
        current_qpos=current,
        stats=StateQuantileStats(q01=np.zeros(7), q99=np.full(7, 10_000)),
    )
    assert RUNTIME_SAMPLE_OFFSETS == (0, 10, 20, 30, 40, 50, 59, 69, 79, 89, 99)
    np.testing.assert_array_equal(h100[-1], current)
    assert discrete.shape == (11, 7)
    assert prompt.startswith("Mode: adjustment.\n")
    assert "qpos_h100_11:[[" in prompt
