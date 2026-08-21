from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.artifacts import sha256_json  # noqa: E402
from tactile_vla.vla.artifacts import sha256_file  # noqa: E402
from tactile_vla.vla.v4_data import V4_ATTEMPT_SCHEMA  # noqa: E402
from tactile_vla.vla.v4_data import V4_NEED_SCHEMA  # noqa: E402
from tactile_vla.vla.v4_data import V4_PROFILE_SCHEMA  # noqa: E402
from tactile_vla.vla.v4_data import V4_REASONING_SCHEMA  # noqa: E402
from tactile_vla.vla.v4_data import V4_SELECTION_SCHEMA  # noqa: E402
from tactile_vla.vla.v4_data import V4_SPLIT_SCHEMA  # noqa: E402
from tactile_vla.vla.v4_data import scan_v4_lerobot_frames  # noqa: E402
from tactile_vla.vla.v4_data import validate_v4_lerobot_frames  # noqa: E402
from tactile_vla.vla.v4_data import validate_v4_index_dataset  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_v4_training_index_test_module",
        PROJECT_ROOT / "scripts" / "prepare_v4_training_index.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _manifest_row(
    *,
    split: str,
    episode_id: int,
    frame_offset: int,
    task: str,
    direction: str,
) -> dict:
    frame_index = 5 + frame_offset
    failure = f"failure_reason=rotate {direction},grasp appropriate."
    observation = {
        "source_type": "real",
        "episode_id": episode_id,
        "attempt_id": 1,
        "hdf5_path": f"episode{episode_id}/attempt1/data.hdf5",
        "window_start": 5,
        "window_end": 19,
        "frame_offset": frame_offset,
        "frame_index": frame_index,
        "ros_timestamp": float(frame_index),
        "tactile_caption": f"Touch frame {episode_id}/1/{frame_index}",
        "failure_reason": failure,
    }
    common = {
        "schema_version": V4_REASONING_SCHEMA,
        "split": split,
        "window_start": 5,
        "frame_offset": frame_offset,
        "frame_index": frame_index,
        "current_observation": observation,
    }
    if task == "failure":
        return {
            **common,
            "sample_type": "failure_reason",
            "target_failure_reason": failure,
            "target_failure_reason_mask": True,
        }
    variant_id = f"fixture-{split}-{episode_id}-{frame_index}"
    return {
        **common,
        "sample_type": "recovery_plan",
        "memory_length": 1,
        "failure_recovery_memory": [
            {
                "recovery_plan": "initial plan",
                "failure_reason": failure,
                "source_type": "synthetic",
                "rule_version": "rotation_distance_v1",
                "seed": 42,
                "variant_id": variant_id,
                "pair_index": 0,
            }
        ],
        "target_recovery_plan": (
            f"recovery_plan=move horizontally {direction} moderately, "
            "move vertically none moderately."
        ),
        "target_recovery_plan_mask": True,
        "target_source": {
            "source_type": "real",
            "episode_id": episode_id,
            "failed_attempt_id": 1,
            "plan_attempt_id": 2,
            "plan_meta_sha256": f"meta-{episode_id}-2",
        },
        "variant_id": variant_id,
        "rule_version": "rotation_distance_v1",
        "seed": 42,
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    dataset_dir = tmp_path / "lerobot"
    profile_dir = tmp_path / "profile"
    reasoning_dir = tmp_path / "reasoning_manifests"
    output_dir = tmp_path / "training"
    split_pairs = {
        "train": (1, 2, "right"),
        "val": (3, 4, "left"),
        "test": (5, 6, "front"),
    }
    selection_attempts: list[dict] = []
    profile_attempts: list[dict] = []
    frame_rows: list[dict] = []
    action_rows: list[dict] = []
    global_index = 0
    lerobot_episode_index = 0
    for split, (one_episode, failed_episode, direction) in split_pairs.items():
        attempts = [
            (one_episode, 1, "one_success", "success", "none", "none", "moderately", True),
            (failed_episode, 1, "moderate_lift", "failure", direction, "none", "moderately", True),
            (failed_episode, 2, "moderate_lift", "success", "none", direction, "moderately", True),
        ]
        for episode_id, attempt_id, task, result, rotation, horizontal, magnitude, stage_gate in attempts:
            selection_attempts.append(
                {
                    "episode_id": episode_id,
                    "attempt_id": attempt_id,
                    "task": task,
                    "subgroup": [direction if task != "one_success" else "horizontal", "fixture"],
                    "meta_path": f"episode{episode_id}/attempt{attempt_id}/meta.json",
                    "meta_sha256": f"meta-{episode_id}-{attempt_id}",
                }
            )
            failure_start = 5 if result == "failure" else None
            profile_attempts.append(
                {
                    "episode_id": episode_id,
                    "attempt_id": attempt_id,
                    "split": split,
                    "task": task,
                    "subgroup": [direction if task != "one_success" else "horizontal", "fixture"],
                    "hdf5_path": f"episode{episode_id}/attempt{attempt_id}/data.hdf5",
                    "meta_sha256": f"meta-{episode_id}-{attempt_id}",
                    "result": result,
                    "rotation_direction": rotation,
                    "grasp_position": "appropriate",
                    "horizontal_direction": horizontal,
                    "horizontal_magnitude": magnitude,
                    "vertical_direction": "none",
                    "vertical_magnitude": "moderately",
                    "stage_a_eligible": stage_gate,
                    "valid": True,
                    "frame_count": 20,
                    "failure_window_start": failure_start,
                }
            )
            for frame_index in range(20):
                full_horizon = frame_index + 3 <= 20
                input_plan = (
                    f"recovery_plan=move horizontally {direction} moderately, "
                    "move vertically none moderately."
                    if attempt_id == 2 and task == "moderate_lift"
                    else "initial plan"
                )
                frame_rows.append(
                    {
                        "index": global_index,
                        "episode_index": lerobot_episode_index,
                        "episode_id": episode_id,
                        "attempt_id": attempt_id,
                        "frame_index": frame_index,
                        "ros_timestamp": float(frame_index),
                        "schema_version": V4_ATTEMPT_SCHEMA,
                        "result": result,
                        "rotation_direction": rotation,
                        "grasp_position": "appropriate",
                        "horizontal_direction": horizontal,
                        "horizontal_magnitude": magnitude,
                        "valid": True,
                        "stage_a_eligible": stage_gate,
                        "execution_eligible": full_horizon,
                        "action_chunk_valid": full_horizon,
                        "tactile_caption": f"Touch frame {episode_id}/{attempt_id}/{frame_index}",
                        "instruction": "pick object",
                        "input_recovery_plan": input_plan,
                    }
                )
                if stage_gate and full_horizon:
                    action_rows.append(
                        {
                            "split": split,
                            "episode_id": episode_id,
                            "attempt_id": attempt_id,
                            "frame_index": frame_index,
                            "hdf5_path": f"episode{episode_id}/attempt{attempt_id}/data.hdf5",
                            "stage_a_eligible": True,
                            "execution_eligible": True,
                            "action_horizon": 3,
                        }
                    )
                global_index += 1
            lerobot_episode_index += 1

        failure_rows = [
            _manifest_row(
                split=split,
                episode_id=failed_episode,
                frame_offset=offset,
                task="failure",
                direction=direction,
            )
            for offset in range(15)
        ]
        plan_rows = [
            _manifest_row(
                split=split,
                episode_id=failed_episode,
                frame_offset=offset,
                task="plan",
                direction=direction,
            )
            for offset in range(15)
        ]
        _write_jsonl(reasoning_dir / "failure_reason" / f"{split}.jsonl", failure_rows)
        _write_jsonl(reasoning_dir / "reasoning" / f"{split}.jsonl", plan_rows)

    selection = {
        "schema_version": V4_SELECTION_SCHEMA,
        "selected_episode_ids": [1, 2, 3, 4, 5, 6],
        "selected_attempt_count": len(selection_attempts),
        "attempts": selection_attempts,
    }
    selection["selection_hash"] = sha256_json(selection)
    profile = {
        "schema_version": V4_PROFILE_SCHEMA,
        "data_profile": "rotation_v4",
        "profile_config_hash": "profile-hash",
        "selection_hash": selection["selection_hash"],
        "config": {"action_horizon": 3, "failure_window_length": 15},
        "selected_episode_ids": [1, 2, 3, 4, 5, 6],
        "attempts": profile_attempts,
    }
    splits = {
        "schema_version": V4_SPLIT_SCHEMA,
        "profile_config_hash": "profile-hash",
        "selection_hash": selection["selection_hash"],
        "original_episode_ids": {
            "train": [1, 2],
            "val": [3, 4],
            "test": [5, 6],
        },
    }
    _write_json(profile_dir / "selection.json", selection)
    _write_json(profile_dir / "profile.json", profile)
    _write_json(profile_dir / "splits.json", splits)
    _write_jsonl(profile_dir / "action_frame_manifest.jsonl", action_rows)
    _write_json(
        reasoning_dir / "summary.json",
        {
            "schema_version": V4_REASONING_SCHEMA,
            "profile_config_hash": "profile-hash",
            "failure_window_length": 15,
        },
    )
    parquet_path = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(frame_rows), parquet_path)
    return {
        "dataset_dir": dataset_dir,
        "selection_file": profile_dir / "selection.json",
        "profile_file": profile_dir / "profile.json",
        "split_file": profile_dir / "splits.json",
        "action_manifest_file": profile_dir / "action_frame_manifest.jsonl",
        "reasoning_manifest_dir": reasoning_dir,
        "output_dir": output_dir,
    }


def test_v4_index_maps_local_frames_and_builds_deterministic_need(tmp_path: Path) -> None:
    module = _load_script()
    paths = _fixture(tmp_path)
    first, first_need, summary = module.build_index(**paths, seed=42, negative_ratio=3.0, failure_window_length=15)
    second, second_need, _ = module.build_index(**paths, seed=42, negative_ratio=3.0, failure_window_length=15)

    assert first == second
    assert first_need == second_need
    assert first["data_profile"] == "rotation_v4"
    assert first["action_indices_identity"]["all"]["count"] == 9 * 18
    assert first["splits"]["train"]["execution_indices"][:3] == [0, 1, 2]
    assert first["splits"]["val"]["failure_reason_indices"] == [99]
    assert first["splits"]["test"]["reasoning_indices"] == [159]
    assert summary["splits"]["train"]["failure"] == 15
    assert summary["splits"]["val"]["failure"] == 1
    assert summary["splits"]["val"]["plan"] == 1

    train_need = first_need["train"]
    assert all(row["schema_version"] == V4_NEED_SCHEMA for row in train_need)
    assert sum(bool(row["need_recovery"]) for row in train_need) == 15
    assert sum(not bool(row["need_recovery"]) for row in train_need) == 45
    assert sum(row["source"] == "pre_failure_hard_negative" for row in train_need) == 5
    assert {bool(row["need_recovery"]) for row in first_need["val"][:4]} == {False, True}
    for name in (
        "selection",
        "profile",
        "splits",
        "action_frame_manifest",
        "failure_reason_train",
        "reasoning_train",
        "reasoning_summary",
    ):
        assert len(first["source_files"][name]["sha256"]) == 64


def test_v4_action_manifest_rejects_wrong_hdf_local_identity(tmp_path: Path) -> None:
    module = _load_script()
    paths = _fixture(tmp_path)
    rows = [json.loads(line) for line in paths["action_manifest_file"].read_text().splitlines()]
    rows[0]["hdf5_path"] = "episode999/attempt1/data.hdf5"
    _write_jsonl(paths["action_manifest_file"], rows)
    with pytest.raises(ValueError, match="wrong HDF5 attempt identity"):
        module.build_index(**paths, seed=42, negative_ratio=3.0, failure_window_length=15)


def test_v4_lerobot_validation_rejects_duplicate_or_noncontiguous_global_index(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    profile = json.loads(paths["profile_file"].read_text())
    frames = scan_v4_lerobot_frames(paths["dataset_dir"])
    changed = list(frames)
    changed[0] = type(changed[0])(**{**changed[0].__dict__, "global_index": 999})
    with pytest.raises(ValueError, match="exact contiguous range"):
        validate_v4_lerobot_frames(changed, profile)


def test_v4_plan_manifest_rejects_synthetic_chain_or_terminal_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    paths = _fixture(tmp_path)
    path = paths["reasoning_manifest_dir"] / "reasoning" / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["failure_recovery_memory"][-1]["failure_reason"] = (
        "failure_reason=rotate left,grasp appropriate."
    )
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="terminal failure differs"):
        module.build_index(**paths, seed=42, negative_ratio=3.0, failure_window_length=15)


def test_v4_cli_persists_need_file_hashes_and_dry_run_is_write_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    paths = _fixture(tmp_path)
    argv = [
        "prepare_v4_training_index.py",
        "--dataset-dir", str(paths["dataset_dir"]),
        "--selection-file", str(paths["selection_file"]),
        "--profile-file", str(paths["profile_file"]),
        "--split-file", str(paths["split_file"]),
        "--action-manifest-file", str(paths["action_manifest_file"]),
        "--reasoning-manifest-dir", str(paths["reasoning_manifest_dir"]),
        "--output-dir", str(paths["output_dir"]),
        "--failure-window-length", "15",
    ]
    monkeypatch.setattr(sys, "argv", [*argv, "--dry-run"])
    assert module.main() == 0
    assert not paths["output_dir"].exists()

    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    index = json.loads((paths["output_dir"] / "v4_training_index.json").read_text())
    frames, global_lookup = validate_v4_index_dataset(index, paths["dataset_dir"])
    assert len(frames) == 180
    assert set(global_lookup) == set(range(180))
    for split in ("train", "val", "test"):
        need_file = paths["output_dir"] / "need" / f"{split}.jsonl"
        assert index["splits"][split]["status_manifest_sha256"] == sha256_file(need_file)
        assert index["source_files"][f"need_{split}"]["sha256"] == sha256_file(need_file)
    assert len(index["training_data_hash"]) == 64


def test_v4_runtime_rejects_changed_profile_or_parquet_source(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    paths = _fixture(tmp_path)
    argv = [
        "prepare_v4_training_index.py",
        "--dataset-dir", str(paths["dataset_dir"]),
        "--selection-file", str(paths["selection_file"]),
        "--profile-file", str(paths["profile_file"]),
        "--split-file", str(paths["split_file"]),
        "--action-manifest-file", str(paths["action_manifest_file"]),
        "--reasoning-manifest-dir", str(paths["reasoning_manifest_dir"]),
        "--output-dir", str(paths["output_dir"]),
        "--failure-window-length", "15",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    index = json.loads((paths["output_dir"] / "v4_training_index.json").read_text())

    original_profile = paths["profile_file"].read_text()
    paths["profile_file"].write_text(original_profile + " \n")
    with pytest.raises(ValueError, match="source file hash mismatch for profile"):
        validate_v4_index_dataset(index, paths["dataset_dir"])
    paths["profile_file"].write_text(original_profile)

    parquet = next((paths["dataset_dir"] / "data").glob("chunk-*/episode_*.parquet"))
    parquet.write_bytes(parquet.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="parquet file hashes"):
        validate_v4_index_dataset(index, paths["dataset_dir"])
