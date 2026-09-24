from __future__ import annotations

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vla.vla.v7_4_2_adjustment_data import phase_and_trainable


def test_stop_is_last_adjustment_frame_and_h30_cannot_cross_it() -> None:
    def phase(frame: int) -> tuple[str, bool]:
        return phase_and_trainable(attempt_id=2, frame_index=frame, stop=100, horizon=30)
    assert phase(71) == ("adjustment", True)
    assert phase(72) == ("adjustment", False)
    assert phase(100) == ("adjustment", False)
    assert phase(101) == ("execution", True)
    assert phase_and_trainable(attempt_id=1, frame_index=100, stop=None, horizon=30) == (
        "execution", True
    )
    with pytest.raises(ValueError, match="requires arm_adjustment_stop"):
        phase_and_trainable(attempt_id=2, frame_index=0, stop=None, horizon=30)
