from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vla.vla.book_stage_a_data import phase_and_trainable


def test_rexecution_is_first_execution_frame_and_h30_must_not_cross() -> None:
    def phase(frame: int) -> tuple[str, bool]:
        return phase_and_trainable(
            attempt_id=2, frame_index=frame, rexecution_frame=100, horizon=30,
        )

    assert phase(70) == ("adjustment", True)
    assert phase(71) == ("adjustment", False)
    assert phase(99) == ("adjustment", False)
    assert phase(100) == ("execution", True)
    assert phase_and_trainable(
        attempt_id=1, frame_index=100, rexecution_frame=None, horizon=30,
    ) == ("execution", True)
    with pytest.raises(ValueError, match="requires native rexecution"):
        phase_and_trainable(attempt_id=2, frame_index=0, rexecution_frame=None, horizon=30)
