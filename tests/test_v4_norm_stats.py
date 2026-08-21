from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi" / "src"))

from tactile_vla.vla.artifacts import action_indices_identity  # noqa: E402
from tactile_vla.vla.artifacts import sha256_file  # noqa: E402
from tactile_vla.vla.artifacts import sha256_json  # noqa: E402
from tactile_vla.vla.v4_data import V4_ATTEMPT_SCHEMA  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "compute_v4_norm_stats_test_module",
        PROJECT_ROOT / "scripts" / "compute_vla_norm_stats.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset_dir = tmp_path / "lerobot"
    rows = []
    for attempt_index, episode_id in enumerate((10, 20, 30)):
        for frame_index in range(3):
            global_index = attempt_index * 3 + frame_index
            full_h2 = frame_index + 2 <= 3
            rows.append(
                {
                    "index": global_index,
                    "episode_index": attempt_index,
                    "episode_id": episode_id,
                    "attempt_id": 1,
                    "frame_index": frame_index,
                    "ros_timestamp": float(frame_index),
                    "schema_version": V4_ATTEMPT_SCHEMA,
                    "result": "success",
                    "rotation_direction": "none",
                    "grasp_position": "appropriate",
                    "horizontal_direction": "none",
                    "horizontal_magnitude": "moderately",
                    "valid": True,
                    "stage_a_eligible": True,
                    "execution_eligible": full_h2,
                    "action_chunk_valid": full_h2,
                    "tactile_caption": f"Touch {global_index}",
                    "instruction": "pick object",
                    "input_recovery_plan": "initial plan",
                    "observation.state": np.full((7,), global_index, dtype=np.float32).tolist(),
                    "action": np.full((7,), global_index + 0.5, dtype=np.float32).tolist(),
                }
            )
    parquet_path = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    splits = {
        "train": {"execution_indices": [0, 1]},
        "val": {"execution_indices": [3, 4]},
        "test": {"execution_indices": [6, 7]},
    }
    required_names = [
        "selection",
        "profile",
        "splits",
        "action_frame_manifest",
        "reasoning_summary",
        *(f"need_{split}" for split in ("train", "val", "test")),
        *(f"failure_reason_{split}" for split in ("train", "val", "test")),
        *(f"reasoning_{split}" for split in ("train", "val", "test")),
    ]
    source_files = {}
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for name in required_names:
        path = source_dir / f"{name}.json"
        path.write_text(f"{name}\n")
        source_files[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    source_files["lerobot_parquet"] = {
        "data/chunk-000/episode_000000.parquet": sha256_file(parquet_path)
    }
    payload = {
        "schema_version": "tactile_vla_v4_training_index_v1",
        "data_profile": "rotation_v4",
        "data_config_hash": "profile-hash",
        "profile_config_hash": "profile-hash",
        "selection_hash": "selection-hash",
        "action_horizon": 2,
        "splits": splits,
        "action_indices_identity": action_indices_identity(splits),
        "need_identity": {name: {"count": 1, "sha256": name} for name in splits},
        "failure_manifest_identity": {name: {"count": 1, "sha256": name} for name in splits},
        "reasoning_manifest_identity": {name: {"count": 1, "sha256": name} for name in splits},
        "lerobot_identity": {
            "frame_count": len(rows),
            "attempt_count": 3,
            "frame_key_sha256": sha256_json(
                [
                    [row["episode_id"], row["attempt_id"], row["frame_index"], row["index"]]
                    for row in rows
                ]
            ),
        },
        "source_files": source_files,
    }
    payload["training_data_hash"] = sha256_json(payload)
    index_file = tmp_path / "v4_training_index.json"
    index_file.write_text(json.dumps(payload))
    return dataset_dir, index_file


def test_v4_norm_consumes_only_train_index_and_records_actual_file_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script()
    dataset_dir, index_file = _fixture(tmp_path)
    output_dir = tmp_path / "norm"
    args = argparse.Namespace(
        dataset_dir=dataset_dir,
        split_file=tmp_path / "splits.json",
        index_file=index_file,
        data_profile="rotation_v4",
        output_dir=output_dir,
        seed=42,
        action_horizon=2,
        delta_action_dims=7,
        max_frames=None,
        overwrite_splits=False,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    module.main()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["num_frames"] == 2
    assert summary["artifact_identity"]["data_profile"] == "rotation_v4"
    assert "norm_stats_sha256" not in summary["artifact_identity"]
    assert summary["norm_stats_sha256"] == sha256_file(output_dir / "norm_stats.json")
