"""Book V9.3 five-task index identity and training protocol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_7_multitask_data import TASK_CYCLE
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE, helper_identity


DATA_PROFILE = "book_v9_3_five_task_h100"
INDEX_SCHEMA = "tactile_vla_book_v9_3_multitask_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_book_v9_3_multitask_manifest_v1"
EXPECTED_COUNTS = {
    "train": {"adjustment": 792, "need": 9480, "failure": 360, "plan": 360},
    "val": {"adjustment": 507, "need": 1236, "failure": 3, "plan": 3},
    "test": {"adjustment": 471, "need": 1252, "failure": 3, "plan": 3},
}
ADJUSTMENT_LABEL_POLICY = "book_v9_2_factual_true_inclusive_[R-10,R]"
NEED_RECOVERY_NEGATIVE_START = "native_reexecution_R_inclusive"


def validate_index(index: Mapping[str, Any]) -> None:
    if index.get("schema_version") != INDEX_SCHEMA or index.get("data_profile") != DATA_PROFILE:
        raise ValueError("Book V9.3 index header mismatch")
    if index.get("prompt_profile") != PROMPT_PROFILE or tuple(index.get("task_cycle", ())) != TASK_CYCLE:
        raise ValueError("Book V9.3 prompt/task cycle mismatch")
    if index.get("history_policy") != helper_identity() | {"idle_perturbation": "none_raw_contiguous"}:
        raise ValueError("Book V9.3 H100 policy mismatch")
    if index.get("adjustment_label_policy") != ADJUSTMENT_LABEL_POLICY:
        raise ValueError("Book V9.3 adjustment label policy mismatch")
    if index.get("need_successful_recovery_start") != NEED_RECOVERY_NEGATIVE_START:
        raise ValueError("Book V9.3 need negative start mismatch")
    if index.get("training_data_hash") != sha256_json({
        key: value for key, value in index.items() if key != "training_data_hash"
    }):
        raise ValueError("Book V9.3 training data hash mismatch")
    for name, expected_hash in index["source_hashes"].items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Book V9.3 source changed: {path}")
    for split in ("train", "val", "test"):
        streams = index["splits"][split]
        for task in ("adjustment", "need", "failure", "plan"):
            sample = streams[task]
            if sample["sample_count"] != EXPECTED_COUNTS[split][task]:
                raise ValueError(f"Book V9.3 {split}/{task} count mismatch")
            if (
                len(sample["manifest_row_indices"]) != sample["sample_count"]
                or len(sample["global_indices"]) != sample["sample_count"]
            ):
                raise ValueError(f"Book V9.3 {split}/{task} identity length mismatch")


__all__ = [
    "ADJUSTMENT_LABEL_POLICY", "DATA_PROFILE", "EXPECTED_COUNTS", "INDEX_SCHEMA",
    "MANIFEST_SCHEMA", "NEED_RECOVERY_NEGATIVE_START", "validate_index",
]
