"""V7.7.2 data policy for the V7.4.2 new-environment fine-tune."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_7_multitask_data import TASK_CYCLE
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE


DATA_PROFILE = "rotation_phase_v7_7_2_five_task_h100"
INDEX_SCHEMA = "tactile_vla_v7_7_2_multitask_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_v7_7_2_multitask_manifest_v1"
FACTUAL_TRUE_BEFORE = 10
FACTUAL_TRUE_AFTER = 10


def factual_adjustment_end(frame_index: int, stop: int) -> bool:
    return stop - FACTUAL_TRUE_BEFORE <= frame_index <= stop + FACTUAL_TRUE_AFTER


def validate_index(index: Mapping[str, Any]) -> None:
    if index.get("schema_version") != INDEX_SCHEMA or index.get("data_profile") != DATA_PROFILE:
        raise ValueError("V7.7.2 index header mismatch")
    if index.get("prompt_profile") != PROMPT_PROFILE or tuple(index.get("task_cycle", ())) != TASK_CYCLE:
        raise ValueError("V7.7.2 prompt/task protocol mismatch")
    if (
        index.get("adjustment_sampling_policy") != "ratio_1_to_2"
        or index.get("adjustment_label_policy") != "factual_true_inclusive_[S-10,S+10]"
        or index.get("history_policy", {}).get("idle_perturbation") != "none_raw_contiguous"
        or index.get("need_successful_recovery_start") != "arm_adjustment_stop_plus_1"
    ):
        raise ValueError("V7.7.2 boundary/sampling policy mismatch")
    if index.get("training_data_hash") != sha256_json({
        key: value for key, value in index.items() if key != "training_data_hash"
    }):
        raise ValueError("V7.7.2 training data hash mismatch")
    for name, expected_hash in index["source_hashes"].items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"V7.7.2 source hash changed: {path}")
    for split in ("train", "val", "test"):
        streams = index["splits"][split]
        for task in ("adjustment", "need", "failure", "plan"):
            if len(streams[task]["manifest_row_indices"]) != len(streams[task]["global_indices"]):
                raise ValueError(f"{split}/{task} manifest identity lengths differ")


__all__ = [
    "DATA_PROFILE", "INDEX_SCHEMA", "MANIFEST_SCHEMA", "FACTUAL_TRUE_BEFORE",
    "FACTUAL_TRUE_AFTER", "factual_adjustment_end", "validate_index",
]
