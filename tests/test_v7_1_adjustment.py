from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.v7_1_adjustment_data import contiguous_true_runs  # noqa: E402
from tactile_vla.vla.v7_1_adjustment_data import select_boundary_exclusions  # noqa: E402


def _row(frame: int, *, static: bool, opened: bool = True) -> dict:
    return {
        "frame_index": frame,
        "h30_static": static,
        "gripper_stably_open": opened,
    }


def test_contiguous_true_runs_breaks_on_missing_frame() -> None:
    assert contiguous_true_runs([1, 2, 4, 5, 6], [True, True, True, False, True]) == [
        (1, 2),
        (4, 4),
        (6, 6),
    ]


def test_boundary_filter_selects_only_last_open_pre_move_run_and_reexecution_run() -> None:
    rows = [_row(frame, static=False) for frame in range(100)]
    for frame in range(5, 15):
        rows[frame] = _row(frame, static=True, opened=False)
    for frame in range(30, 42):
        rows[frame] = _row(frame, static=True, opened=True)
    for frame in range(45, 60):
        rows[frame] = _row(frame, static=True, opened=True)
    for frame in range(70, 86):
        rows[frame] = _row(frame, static=True, opened=True)

    result = select_boundary_exclusions(
        rows,
        move_start_frame=65,
        rexecution_frame=82,
        horizon=30,
        minimum_run=10,
    )

    assert set(frame for frame, reason in result.items() if reason == "pre_move_static_wait") == set(range(45, 60))
    assert set(frame for frame, reason in result.items() if reason == "pre_rexecution_static_wait") == set(range(70, 82))
    assert set(frame for frame, reason in result.items() if reason == "post_reexecution_static_wait") == set(range(82, 86))
    assert not (set(range(5, 15)) & result.keys())
    assert not (set(range(30, 42)) & result.keys())


def test_boundary_filter_keeps_short_and_motion_runs() -> None:
    rows = [_row(frame, static=False) for frame in range(70)]
    for frame in range(20, 29):
        rows[frame] = _row(frame, static=True)
    result = select_boundary_exclusions(
        rows,
        move_start_frame=35,
        rexecution_frame=50,
        horizon=30,
        minimum_run=10,
    )
    assert result == {}
