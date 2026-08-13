from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vla.vla.index import extend_splits  # noqa: E402
from tactile_vla.vla.index import execution_indices  # noqa: E402
from tactile_vla.vla.index import failure_reason_indices  # noqa: E402
from tactile_vla.vla.index import FrameRecord  # noqa: E402
from tactile_vla.vla.index import index_payload  # noqa: E402
from tactile_vla.vla.index import load_or_create_splits  # noqa: E402
from tactile_vla.vla.index import SplitConfig  # noqa: E402
from tactile_vla.vla.index import stratified_status_indices  # noqa: E402
from tactile_vla.vla.index import validate_index_action_horizon  # noqa: E402


def _record(
    *,
    global_index: int,
    episode_index: int,
    frame_index: int,
    stored_valid: bool,
    execution_eligible: bool = True,
    stage_a_eligible: bool = True,
    need_recovery: bool = False,
    failure_reason: str = "",
) -> FrameRecord:
    return FrameRecord(
        global_index=global_index,
        lerobot_episode_index=episode_index,
        original_episode_id=episode_index,
        attempt_id=1,
        frame_index=frame_index,
        case_id="case_001",
        instruction="test",
        rotation_state_name="none",
        tactile_caption="Tactile: no rotation.",
        input_recovery_plan="",
        failure_recovery_memory="[]",
        failure_reason=failure_reason,
        recovery_plan="",
        need_recovery=need_recovery,
        action_chunk_valid=stored_valid,
        reasoning_has_sample=False,
        reasoning_failed_attempt_id=-1,
        reasoning_failed_tactile_caption="",
        reasoning_failure_reason="",
        reasoning_failure_recovery_memory="",
        reasoning_recovery_plan="",
        execution_eligible=execution_eligible,
        stage_a_eligible=stage_a_eligible,
    )


def test_execution_indices_use_requested_horizon_per_episode() -> None:
    records = [
        *[_record(global_index=i, episode_index=0, frame_index=i, stored_valid=True) for i in range(5)],
        *[
            _record(global_index=10 + i, episode_index=1, frame_index=i, stored_valid=False)
            for i in range(3)
        ],
    ]

    assert execution_indices(records, action_horizon=3) == [0, 1, 2, 10]
    assert execution_indices(records, action_horizon=5) == [0]
    assert execution_indices(records, action_chunk_valid_only=True) == [0, 1, 2, 3, 4]


def test_execution_indices_exclude_stage_a_ineligible_frames() -> None:
    records = [
        _record(global_index=0, episode_index=0, frame_index=0, stored_valid=True),
        _record(
            global_index=1,
            episode_index=0,
            frame_index=1,
            stored_valid=True,
            execution_eligible=False,
        ),
        _record(global_index=2, episode_index=0, frame_index=2, stored_valid=True),
    ]
    assert execution_indices(records, action_horizon=1) == [0, 2]
    assert execution_indices(records, action_chunk_valid_only=False) == [0, 2]

    records[2] = _record(
        global_index=2,
        episode_index=0,
        frame_index=2,
        stored_valid=True,
        stage_a_eligible=False,
    )
    assert execution_indices(records, action_horizon=1) == [0]


def test_stage_a_ineligible_flag_is_excluded_even_if_execution_flag_is_true() -> None:
    records = [
        _record(
            global_index=0,
            episode_index=0,
            frame_index=0,
            stored_valid=True,
            stage_a_eligible=False,
            execution_eligible=True,
        )
    ]
    assert execution_indices(records, action_horizon=1) == []


def test_index_payload_records_horizon_and_dynamic_count() -> None:
    records = [_record(global_index=i, episode_index=0, frame_index=i, stored_valid=True) for i in range(5)]
    payload = index_payload(
        records,
        {"train": [0], "val": [], "test": []},
        seed=42,
        negative_ratio=3.0,
        action_horizon=3,
    )

    assert payload["action_horizon"] == 3
    assert payload["splits"]["train"]["execution_indices"] == [0, 1, 2]
    assert payload["splits"]["train"]["summary"]["execution_indices"] == 3


def test_v3_failure_reason_window_uses_15_train_frames_and_one_eval_frame() -> None:
    records = [
        _record(
            global_index=index,
            episode_index=0,
            frame_index=index,
            stored_valid=True,
            need_recovery=index >= 5,
            failure_reason=(
                "failure_reason=rotate left,grasp appropriate." if index >= 5 else ""
            ),
        )
        for index in range(20)
    ]

    assert failure_reason_indices(records, window_frames=15, training=True) == list(range(5, 20))
    assert failure_reason_indices(records, window_frames=15, training=False) == [19]


def test_v3_failure_reason_window_rejects_incomplete_attempt() -> None:
    records = [
        _record(
            global_index=index,
            episode_index=0,
            frame_index=index,
            stored_valid=True,
            need_recovery=index >= 5,
            failure_reason=(
                "failure_reason=rotate left,grasp appropriate." if index >= 5 else ""
            ),
        )
        for index in range(19)
    ]

    with pytest.raises(ValueError, match="failure window is incomplete"):
        failure_reason_indices(records, window_frames=15, training=True)


def test_need_evaluation_indices_are_deterministic_interleaved_and_have_both_classes() -> None:
    records = [
        _record(
            global_index=index,
            episode_index=0,
            frame_index=index,
            stored_valid=True,
            need_recovery=index >= 100,
            failure_reason=(
                "failure_reason=rotate left,grasp appropriate." if index >= 100 else ""
            ),
        )
        for index in range(110)
    ]
    indices = stratified_status_indices(records, negative_ratio=3.0, seed=42)
    assert indices == stratified_status_indices(records, negative_ratio=3.0, seed=42)
    lookup = {record.global_index: record.need_recovery for record in records}
    first_eight = [lookup[index] for index in indices[:8]]
    assert any(first_eight)
    assert not all(first_eight)
    assert sum(not lookup[index] for index in indices) == 30
    assert sum(lookup[index] for index in indices) == 10


def test_index_horizon_validation_is_backward_compatible_only_with_h30() -> None:
    validate_index_action_horizon({}, 30)
    validate_index_action_horizon({"action_horizon": 50}, 50)

    with pytest.raises(ValueError, match="Index action_horizon=30"):
        validate_index_action_horizon({}, 50, index_path="old.json")
    with pytest.raises(ValueError, match="Index action_horizon=50"):
        validate_index_action_horizon({"action_horizon": 50}, 30)


def test_extend_splits_preserves_base_and_allocates_new_episodes() -> None:
    records = [_record(global_index=i, episode_index=i + 1, frame_index=0, stored_valid=True) for i in range(20)]
    base = {
        "train": list(range(1, 9)),
        "val": [9],
        "test": [10],
    }

    splits, additions = extend_splits(records, base, SplitConfig(seed=42))

    assert {name: len(values) for name, values in splits.items()} == {"train": 16, "val": 2, "test": 2}
    assert {name: len(values) for name, values in additions.items()} == {"train": 8, "val": 1, "test": 1}
    for name in ("train", "val", "test"):
        assert set(base[name]).issubset(splits[name])
    assert set().union(*map(set, splits.values())) == set(range(1, 21))


def test_existing_split_must_cover_all_dataset_episodes(tmp_path: Path) -> None:
    records = [_record(global_index=i, episode_index=i + 1, frame_index=0, stored_valid=True) for i in range(3)]
    split_file = tmp_path / "splits.json"
    split_file.write_text(
        json.dumps(
            {
                "original_episode_ids": {
                    "train": [1],
                    "val": [2],
                    "test": [],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="missing episode IDs=\\[3\\]"):
        load_or_create_splits(records, split_file, SplitConfig())
