from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from scripts.build_book_v9_3_multitask_data import need_negative_source
from tactile_vla.vla.book_v9_3_multitask_data import (
    EXPECTED_COUNTS, NEED_RECOVERY_NEGATIVE_START,
)
from tactile_vla.vla.v7_7_multitask_data import TASK_CYCLE
from tactile_vla.vla.v7_7_phase_prompt import uniform_positions


def test_book_recovery_need_negatives_begin_at_native_r_inclusive():
    meta = {"task": "moderate_lift", "result": "success", "rexecution_frame_index": 150}
    frame = SimpleNamespace(attempt_id=2, frame_index=149)
    assert need_negative_source(frame, meta) is None
    frame.frame_index = 150
    assert need_negative_source(frame, meta) == "successful_recovery_easy_negative"
    frame.frame_index = 151
    assert need_negative_source(frame, meta) == "successful_recovery_easy_negative"
    assert NEED_RECOVERY_NEGATIVE_START == "native_reexecution_R_inclusive"
    with pytest.raises(ValueError, match="lacks native R"):
        need_negative_source(frame, meta | {"rexecution_frame_index": None})


def test_book_failure_and_one_success_negative_domains():
    frame = SimpleNamespace(attempt_id=1, frame_index=99)
    failure = {"task": "moderate_lift", "result": "failure", "shift_frame_index": 100}
    assert need_negative_source(frame, failure) == "pre_failure_hard_negative"
    frame.frame_index = 100
    assert need_negative_source(frame, failure) is None
    assert need_negative_source(frame, {"task": "one_success", "result": "success"}) == "one_success_easy_negative"


def test_book_v9_3_five_task_and_h100_protocol():
    assert TASK_CYCLE == ("action", "adjustment", "need", "failure", "plan")
    assert uniform_positions(100).tolist() == [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    assert EXPECTED_COUNTS["train"] == {"adjustment": 792, "need": 9480, "failure": 360, "plan": 360}
