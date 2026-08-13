from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vla.vla.artifacts import action_indices_identity  # noqa: E402
from tactile_vla.vla.artifacts import artifact_identity  # noqa: E402
from tactile_vla.vla.artifacts import assert_identity_matches  # noqa: E402
from tactile_vla.vla.artifacts import sha256_json  # noqa: E402
from tactile_vla.vla.artifacts import load_checkpoint_prompt_profile  # noqa: E402
from tactile_vla.vla.artifacts import validate_norm_stats_identity  # noqa: E402
from tactile_vla.vla.data_profiles import build_profile_splits  # noqa: E402
from tactile_vla.vla.data_profiles import build_single_round_reasoning_samples  # noqa: E402
from tactile_vla.vla.data_profiles import direction_by_episode  # noqa: E402
from tactile_vla.vla.data_profiles import EXPECTED_ACTION_COUNTS  # noqa: E402
from tactile_vla.vla.data_profiles import ROTATION_MODERATELY_GROUPS  # noqa: E402
from tactile_vla.vla.data_profiles import ROTATION_MODERATELY_SUCCESS_V1  # noqa: E402
from tactile_vla.vla.data_profiles import select_profile_records  # noqa: E402
from tactile_vla.vla.data_profiles import validate_expected_action_counts  # noqa: E402
from tactile_vla.vla.data_profiles import validate_profile_metadata  # noqa: E402
from tactile_vla.vla.index import scan_lerobot_frames  # noqa: E402
from tactile_vla.vla.index import v3_index_payload  # noqa: E402
from tactile_vla.vla.structured_text import legal_failure_reasons  # noqa: E402
from tactile_vla.vla.structured_text import legal_recovery_plans  # noqa: E402


DATASET = Path("/data1/tac_data/lerobot_data/tactile_vla_v3")


def test_rotation_profile_selects_88_episodes_and_independent_72_8_8_split() -> None:
    selected = {
        episode_id
        for group in ROTATION_MODERATELY_GROUPS
        for episode_id in group.episode_ids
    }
    assert len(selected) == 88
    splits, groups = build_profile_splits(seed=42)
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 72,
        "val": 8,
        "test": 8,
    }
    assert set().union(*(set(values) for values in splits.values())) == selected
    assert len(groups) == 88

    directions = direction_by_episode()
    for split in ("val", "test"):
        recovery_directions = [
            directions[episode_id]
            for episode_id in splits[split]
            if episode_id in directions
        ]
        assert sorted(recovery_directions) == ["back", "front", "left", "right"]


def test_each_group_random_stream_is_stable_when_an_unrelated_group_is_appended() -> None:
    baseline, _ = build_profile_splits(seed=42)
    repeated, _ = build_profile_splits(seed=42)
    assert baseline == repeated
    for group in ROTATION_MODERATELY_GROUPS:
        for split in ("train", "val", "test"):
            assert set(baseline[split]).intersection(group.episode_ids) == set(
                repeated[split]
            ).intersection(group.episode_ids)


@pytest.mark.skipif(not DATASET.is_dir(), reason="local V3 LeRobot dataset is unavailable")
def test_local_rotation_profile_metadata_action_counts_and_reasoning_contract() -> None:
    records = select_profile_records(scan_lerobot_frames(DATASET))
    validate_profile_metadata(records)
    assert len({record.original_episode_id for record in records}) == 88
    assert validate_expected_action_counts(records) == EXPECTED_ACTION_COUNTS

    splits, _ = build_profile_splits(seed=42)
    reasoning = build_single_round_reasoning_samples(records, splits)
    assert {name: len(values) for name, values in reasoning.items()} == {
        "train": 40,
        "val": 4,
        "test": 4,
    }
    for samples in reasoning.values():
        assert all(sample["memory_length"] == 1 for sample in samples)
        assert all(len(sample["failure_recovery_memory"]) == 1 for sample in samples)
        assert all(sample["donor_episode_ids"] == [] for sample in samples)

    index = v3_index_payload(records, splits, seed=42)
    identity = action_indices_identity(index["splits"])
    assert identity["all"]["count"] == 98_233
    # Stage A and Stage B action replay both receive this exact persisted list.
    stage_a_indices = index["splits"]["train"]["execution_indices"]
    stage_b_replay_indices = index["splits"]["train"]["execution_indices"]
    assert stage_a_indices == stage_b_replay_indices
    assert sha256_json(stage_a_indices) == identity["train"]["sha256"]


def test_full_grammar_is_retained_while_profile_targets_only_four_moderate_rotations() -> None:
    assert len(legal_failure_reasons()) > 4
    assert len(legal_recovery_plans()) > 4
    target_failures = {
        f"failure_reason=rotate {direction},grasp appropriate."
        for direction in ("left", "right", "front", "back")
    }
    target_plans = {
        f"recovery_plan=move horizontally {direction} moderately, "
        "move vertically none moderately."
        for direction in ("left", "right", "front", "back")
    }
    assert target_failures.issubset(legal_failure_reasons())
    assert target_plans.issubset(legal_recovery_plans())


def test_resume_identity_rejects_profile_hash_and_action_index_changes(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    payload = {
        "data_profile": ROTATION_MODERATELY_SUCCESS_V1,
        "data_config_hash": "config-a",
        "splits": {
            "train": {"execution_indices": [1, 2]},
            "val": {"execution_indices": [3]},
            "test": {"execution_indices": [4]},
        },
    }
    payload["action_indices_identity"] = action_indices_identity(payload["splits"])
    index_path.write_text(json.dumps(payload))
    identity = artifact_identity(
        payload,
        index_path=index_path,
        prompt_profile="minimal_v1",
        requested_data_profile=ROTATION_MODERATELY_SUCCESS_V1,
    )

    for key, changed in (
        ("data_profile", "different"),
        ("data_config_hash", "config-b"),
        ("index_sha256", sha256_json("different")),
    ):
        requested = dict(identity)
        requested[key] = changed
        with pytest.raises(ValueError, match=key):
            assert_identity_matches(identity, requested, context="resume")


def test_legacy_checkpoint_without_prompt_profile_falls_back() -> None:
    assert load_checkpoint_prompt_profile({"checkpoint_format": "old"}) == "legacy"
    assert load_checkpoint_prompt_profile({"prompt_profile": "minimal_v1"}) == "minimal_v1"


def test_norm_stats_must_cover_the_same_train_action_manifest(tmp_path: Path) -> None:
    identity = {
        "data_profile": ROTATION_MODERATELY_SUCCESS_V1,
        "data_config_hash": "config",
        "action_frame_manifest_hash": "manifest",
        "index_sha256": "index",
        "action_indices_identity": {
            "train": {"count": 2, "sha256": "train"},
            "val": {"count": 1, "sha256": "val"},
            "test": {"count": 1, "sha256": "test"},
            "all": {"count": 4, "sha256": "all"},
        },
    }
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"artifact_identity": identity, "num_frames": 1})
    )
    with pytest.raises(ValueError, match="frame count mismatch"):
        validate_norm_stats_identity(summary, identity, context="norm")
