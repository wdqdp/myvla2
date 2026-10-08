"""Versioned Book V9.4.3 data contract; only need supervision differs from V9.4.2."""

from __future__ import annotations

import json
from pathlib import Path

from tactile_vla.vla import book_v9_4_multitask_data as previous
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.book_v9_4_3_need_data import LABEL_POLICY, SAMPLING_POLICY, positive_counts, validate_need_rows
from tactile_vla.vla.v4_data import SPLITS, load_jsonl

ROOT = previous.ROOT
VERSION_TAG = "book_v9_4_3"
DATA_PROFILE = "book_v9_4_3_five_task_h100"
INDEX_SCHEMA = "tactile_vla_book_v9_4_3_multitask_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_book_v9_4_3_multitask_manifest_v1"
DEFAULT_ACTION_INDEX = ROOT / "outputs/book_stage_a_v9_4_2/book_stage_a_training_index.json"
DEFAULT_ADJUSTMENT_DIR = ROOT / "outputs/book_adjustment_end_v9_4_2"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_book_stage_a_v9_4_2_no_history/15000"
DEFAULT_MULTITASK_DIR = ROOT / "outputs/book_v9_4_3_multitask"
DEFAULT_INDEX = DEFAULT_MULTITASK_DIR / "book_v9_4_3_multitask_training_index.json"
DEFAULT_OUTPUT = ROOT / "outputs/multitask_v9_4_3"
RUN_NAME = "pi05_book_v9_4_3_five_task_h100_no_history"


def expected_counts(v4_dir: Path, *, expand_plan=True):
    counts = previous.expected_counts(v4_dir, expand_plan=expand_plan)
    profile = json.loads((v4_dir / "profile.json").read_text())
    for split, count in positive_counts(profile).items():
        counts[split]["need"] = 4 * count
    return counts


def validate_manifest_rows(index, manifests):
    previous.validate_manifest_rows(index, manifests, data_profile=DATA_PROFILE, manifest_schema=MANIFEST_SCHEMA)
    profile = json.loads((Path(index["v4_dir"]) / "profile.json").read_text())
    audit = validate_need_rows(profile, manifests["need"])
    for split in SPLITS:
        selected = index["splits"][split]["need"]["manifest_row_indices"]
        expected = [i for i, row in enumerate(manifests["need"]) if row["split"] == split]
        if selected != expected:
            raise ValueError("V9.4.3 need stream must select its entire sampled manifest")
    return audit


def validate_index(index):
    if (
        index.get("experiment_version") != VERSION_TAG
        or index.get("need_label_policy") != LABEL_POLICY
        or index.get("need_sampling_policy") != SAMPLING_POLICY
    ):
        raise ValueError("V9.4.3 requires delayed/ignored need labels and boundary-first sampling; rebuild old data")
    previous.validate_index(
        index,
        data_profile=DATA_PROFILE,
        index_schema=INDEX_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
        counts_fn=expected_counts,
    )
    manifests = {
        task: load_jsonl(Path(index[f"{task}_manifest_file"])) for task in ("adjustment", "need", "failure", "plan")
    }
    audit = validate_manifest_rows(index, manifests)
    path = Path(index["need_boundary_audit_file"])
    if sha256_file(path) != index["need_boundary_audit_sha256"] or json.loads(path.read_text()) != audit:
        raise ValueError("V9.4.3 need boundary audit differs from the actual sampled frames")


validate_stage_a_for_training = previous.validate_stage_a_for_training
