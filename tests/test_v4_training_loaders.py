from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import action_indices_identity  # noqa: E402
from tactile_vla.vla.artifacts import artifact_identity  # noqa: E402
from tactile_vla.vla.artifacts import assert_identity_matches  # noqa: E402
from tactile_vla.vla.artifacts import sha256_file  # noqa: E402
from tactile_vla.vla.artifacts import sha256_json  # noqa: E402
from tactile_vla.vla.artifacts import validate_norm_stats_identity  # noqa: E402
from tactile_vla.vla.openpi_bridge import V4DirectManifestDataset  # noqa: E402
from tactile_vla.vla.v4_data import V4_NEED_SCHEMA  # noqa: E402
from tactile_vla.vla.v4_data import V4_REASONING_SCHEMA  # noqa: E402


FAILURE = "failure_reason=rotate right,grasp appropriate."
PLAN = (
    "recovery_plan=move horizontally right moderately, "
    "move vertically none moderately."
)


class FakeLeRobotDataset:
    def __init__(self, rows: dict[int, dict]) -> None:
        self.rows = rows
        self.calls: list[int] = []

    def __getitem__(self, index: int) -> dict:
        self.calls.append(index)
        return self.rows[index]


def _item(
    index: int,
    *,
    need: bool,
    failure: str = "",
    plan: str = "",
) -> dict:
    return {
        "index": index,
        "episode_id": 10,
        "attempt_id": 1,
        "frame_index": index,
        "observation.images.front": np.zeros((3, 4, 4), dtype=np.uint8),
        "observation.images.left": np.zeros((3, 4, 4), dtype=np.uint8),
        "observation.state": np.zeros((7,), dtype=np.float32),
        "instruction": "pick object",
        "tactile_caption": f"Touch frame {index}",
        "input_recovery_plan": "initial plan",
        "need_recovery": need,
        "failure_reason": failure,
        "failure_reason_mask": bool(failure),
        "recovery_plan": plan,
        "recovery_plan_mask": bool(plan),
    }


def _observation(frame_index: int) -> dict:
    return {
        "source_type": "real",
        "episode_id": 10,
        "attempt_id": 1,
        "frame_index": frame_index,
        "frame_offset": frame_index - 5,
        "tactile_caption": f"Touch frame {frame_index}",
        "failure_reason": FAILURE,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_v4_need_manifest_target_is_authoritative_and_cross_checked(tmp_path: Path) -> None:
    path = tmp_path / "need.jsonl"
    rows = [
        {
            "schema_version": V4_NEED_SCHEMA,
            "split": "train",
            "global_index": 5,
            "episode_id": 10,
            "attempt_id": 1,
            "frame_index": 5,
            "need_recovery": True,
            "source": "failure_active",
        },
        {
            "schema_version": V4_NEED_SCHEMA,
            "split": "train",
            "global_index": 4,
            "episode_id": 10,
            "attempt_id": 1,
            "frame_index": 4,
            "need_recovery": False,
            "source": "pre_failure_hard_negative",
        },
    ]
    _write_jsonl(path, rows)
    fake = FakeLeRobotDataset({4: _item(4, need=False), 5: _item(5, need=True, failure=FAILURE)})
    dataset = V4DirectManifestDataset(
        dataset_dir=tmp_path,
        manifest_file=path,
        manifest_row_indices=[0, 1],
        global_indices=[5, 4],
        task="need",
        expected_manifest_sha256=sha256_file(path),
        expected_split="train",
        prompt_profile="minimal_v1",
        lerobot_dataset=fake,
    )
    assert len(dataset) == 2
    assert dataset[0]["need_recovery_label"] == 1
    assert dataset[1]["need_recovery_label"] == 0
    assert fake.calls == [5, 4]

    mismatched = FakeLeRobotDataset({5: _item(5, need=False)})
    dataset = V4DirectManifestDataset(
        dataset_dir=tmp_path,
        manifest_file=path,
        manifest_row_indices=[0],
        global_indices=[5],
        task="need",
        lerobot_dataset=mismatched,
    )
    with pytest.raises(ValueError, match="differs from LeRobot label"):
        dataset[0]


def test_v4_failure_and_plan_are_direct_rows_not_window_expansion(tmp_path: Path) -> None:
    failure_path = tmp_path / "failure.jsonl"
    failure_rows = [
        {
            "schema_version": V4_REASONING_SCHEMA,
            "split": "val",
            "sample_type": "failure_reason",
            "window_start": 5,
            "frame_offset": offset,
            "frame_index": 5 + offset,
            "current_observation": _observation(5 + offset),
            "target_failure_reason": FAILURE,
        }
        for offset in range(15)
    ]
    _write_jsonl(failure_path, failure_rows)
    fake = FakeLeRobotDataset({19: _item(19, need=True, failure=FAILURE, plan=PLAN)})
    failure_dataset = V4DirectManifestDataset(
        dataset_dir=tmp_path,
        manifest_file=failure_path,
        manifest_row_indices=[14],
        global_indices=[19],
        task="failure",
        expected_split="val",
        prompt_profile="minimal_v1",
        lerobot_dataset=fake,
    )
    assert len(failure_dataset) == 1
    failure_sample = failure_dataset[0]
    assert failure_sample["global_index"] == 19
    assert failure_sample["target_text"] == FAILURE

    plan_path = tmp_path / "plan.jsonl"
    variant_id = "fixture-l1"
    plan_row = {
        "schema_version": V4_REASONING_SCHEMA,
        "split": "val",
        "sample_type": "recovery_plan",
        "window_start": 5,
        "frame_offset": 14,
        "frame_index": 19,
        "current_observation": _observation(19),
        "memory_length": 1,
        "failure_recovery_memory": [
            {
                "recovery_plan": "initial plan",
                "failure_reason": FAILURE,
                "source_type": "synthetic",
                "rule_version": "rotation_distance_v1",
                "seed": 42,
                "variant_id": variant_id,
                "pair_index": 0,
            }
        ],
        "target_recovery_plan": PLAN,
        "target_source": {
            "source_type": "real",
            "episode_id": 10,
            "failed_attempt_id": 1,
            "plan_attempt_id": 2,
        },
        "variant_id": variant_id,
        "rule_version": "rotation_distance_v1",
        "seed": 42,
    }
    _write_jsonl(plan_path, [plan_row])
    plan_dataset = V4DirectManifestDataset(
        dataset_dir=tmp_path,
        manifest_file=plan_path,
        manifest_row_indices=[0],
        global_indices=[19],
        task="plan",
        expected_split="val",
        prompt_profile="minimal_v1",
        lerobot_dataset=fake,
    )
    assert len(plan_dataset) == 1
    plan_sample = plan_dataset[0]
    assert plan_sample["target_text"] == PLAN
    assert "Failure-recovery memory:" in plan_sample["prompt"]
    assert fake.calls == [19, 19]

    plan_row["failure_recovery_memory"][0]["variant_id"] = "forged"
    _write_jsonl(plan_path, [plan_row])
    with pytest.raises(ValueError, match="pair provenance mismatch"):
        V4DirectManifestDataset(
            dataset_dir=tmp_path,
            manifest_file=plan_path,
            manifest_row_indices=[0],
            global_indices=[19],
            task="plan",
            expected_split="val",
            lerobot_dataset=fake,
        )


def _v4_index_payload() -> dict:
    splits = {
        "train": {"execution_indices": [0, 1]},
        "val": {"execution_indices": [2]},
        "test": {"execution_indices": [3]},
    }
    payload = {
        "schema_version": "tactile_vla_v4_training_index_v1",
        "data_profile": "rotation_v4",
        "data_config_hash": "profile-hash",
        "profile_config_hash": "profile-hash",
        "selection_hash": "selection-hash",
        "splits": splits,
        "action_indices_identity": action_indices_identity(splits),
        "need_identity": {split: {"count": 1, "sha256": f"need-{split}"} for split in ("train", "val", "test")},
        "failure_manifest_identity": {split: {"count": 1, "sha256": f"failure-{split}"} for split in ("train", "val", "test")},
        "reasoning_manifest_identity": {split: {"count": 1, "sha256": f"plan-{split}"} for split in ("train", "val", "test")},
        "lerobot_identity": {"frame_count": 4, "attempt_count": 1, "frame_key_sha256": "frames"},
        "source_files": {
            "selection": {"path": "/selection.json", "sha256": "a" * 64},
            "profile": {"path": "/profile.json", "sha256": "b" * 64},
            "splits": {"path": "/splits.json", "sha256": "c" * 64},
            "lerobot_parquet": {"data/chunk-000/episode.parquet": "d" * 64},
        },
    }
    payload["training_data_hash"] = sha256_json(payload)
    return payload


def test_v4_artifact_identity_covers_all_data_side_hashes_and_v3_stays_compatible(tmp_path: Path) -> None:
    payload = _v4_index_payload()
    index_path = tmp_path / "v4_index.json"
    index_path.write_text(json.dumps(payload))
    identity = artifact_identity(
        payload,
        index_path=index_path,
        prompt_profile="minimal_v1",
        requested_data_profile="rotation_v4",
    )
    assert identity["selection_hash"] == "selection-hash"
    assert identity["profile_config_hash"] == "profile-hash"
    assert identity["training_data_hash"] == payload["training_data_hash"]
    assert identity["source_file_hashes"]["selection"] == "a" * 64
    assert identity["source_file_hashes"]["lerobot_parquet"]["data/chunk-000/episode.parquet"] == "d" * 64
    changed = dict(identity)
    changed["reasoning_manifest_identity"] = {"changed": True}
    with pytest.raises(ValueError, match="reasoning_manifest_identity"):
        assert_identity_matches(identity, changed, context="V4 resume")

    v3_payload = {
        "data_profile": "rotation_moderately_success_v1",
        "data_config_hash": "v3",
        "splits": {
            "train": {"execution_indices": [1]},
            "val": {"execution_indices": [2]},
            "test": {"execution_indices": [3]},
        },
    }
    v3_path = tmp_path / "v3_index.json"
    v3_path.write_text(json.dumps(v3_payload))
    v3_identity = artifact_identity(
        v3_payload,
        index_path=v3_path,
        prompt_profile="minimal_v1",
        requested_data_profile="rotation_moderately_success_v1",
    )
    assert "training_data_hash" not in v3_identity
    assert_identity_matches(v3_identity, v3_identity, context="V3 resume")


def test_v4_artifact_identity_rejects_mutated_index_payload(tmp_path: Path) -> None:
    payload = _v4_index_payload()
    payload["selection_hash"] = "mutated"
    path = tmp_path / "index.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training_data_hash"):
        artifact_identity(
            payload,
            index_path=path,
            prompt_profile="minimal_v1",
            requested_data_profile="rotation_v4",
        )


def test_v4_norm_hash_is_validated_without_self_reference(tmp_path: Path) -> None:
    payload = _v4_index_payload()
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(payload))
    identity = artifact_identity(
        payload,
        index_path=index_path,
        prompt_profile="minimal_v1",
        requested_data_profile="rotation_v4",
    )
    norm_dir = tmp_path / "norm"
    norm_dir.mkdir()
    norm_file = norm_dir / "norm_stats.json"
    norm_file.write_text('{"state": {}, "actions": {}}\n')
    summary = {
        "artifact_identity": {**identity, "prompt_profile": "not_applicable"},
        "num_frames": identity["action_indices_identity"]["train"]["count"],
        "norm_stats_sha256": sha256_file(norm_file),
    }
    (norm_dir / "summary.json").write_text(json.dumps(summary))
    validated = validate_norm_stats_identity(
        norm_dir / "summary.json",
        identity,
        context="V4 norm",
    )
    assert validated["norm_stats_sha256"] == sha256_file(norm_file)
    identity["norm_stats_sha256"] = validated["norm_stats_sha256"]

    norm_file.write_text("tampered\n")
    with pytest.raises(ValueError, match="norm_stats.json hash mismatch"):
        validate_norm_stats_identity(
            norm_dir / "summary.json",
            identity,
            context="V4 norm",
        )
