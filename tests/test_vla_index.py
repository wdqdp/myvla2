from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vla.vla.index import extend_splits  # noqa: E402
from tactile_vla.vla.index import execution_indices  # noqa: E402
from tactile_vla.vla.index import FrameRecord  # noqa: E402
from tactile_vla.vla.index import index_payload  # noqa: E402
from tactile_vla.vla.index import load_or_create_splits  # noqa: E402
from tactile_vla.vla.index import SplitConfig  # noqa: E402
from tactile_vla.vla.index import validate_index_action_horizon  # noqa: E402


def _record(*, global_index: int, episode_index: int, frame_index: int, stored_valid: bool) -> FrameRecord:
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
        failure_reason="",
        recovery_plan="",
        need_recovery=False,
        action_chunk_valid=stored_valid,
        reasoning_has_sample=False,
        reasoning_failed_attempt_id=-1,
        reasoning_failed_tactile_caption="",
        reasoning_failure_reason="",
        reasoning_failure_recovery_memory="",
        reasoning_recovery_plan="",
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
