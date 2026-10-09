"""Exact real-only, no-touch conversion of the already built V9.5 dataset."""

from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

from tactile_vla.vla import book_v9_5_multitask_data as source
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_4_memory import validate_dataset_plan_tokens
from tactile_vla.vla.book_v9_5_stage_a_data import reject_archive
from tactile_vla.vla.book_v9_6_prompts import INPUT_POLICY, PROMPT_PROFILE, remove_touch, validate_no_touch
from tactile_vla.vla.v4_data import SPLITS, load_jsonl

ROOT = source.ROOT
VERSION_TAG = "book_v9_6"
DATA_PROFILE = "book_v9_6_five_task_h100_visual"
INDEX_SCHEMA = "tactile_vla_book_v9_6_multitask_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_book_v9_6_multitask_manifest_v1"
DEFAULT_SOURCE_INDEX = source.DEFAULT_INDEX
DEFAULT_STAGE_A = source.DEFAULT_STAGE_A
DEFAULT_MULTITASK_DIR = ROOT / "outputs/book_v9_6_multitask"
DEFAULT_INDEX = DEFAULT_MULTITASK_DIR / "book_v9_6_multitask_training_index.json"
REASONING_WINDOW_POLICY = source.REASONING_WINDOW_POLICY
LABEL_POLICY = {
    "schema_version": "book_v9_6_need_labels_v1",
    "real_labels": source.LABEL_POLICY,
    "counterfactuals": "removed_no_reconstruction",
}
SAMPLING_POLICY = {
    "schema_version": "book_v9_6_need_sampling_v1",
    "source": "all_selected_v9_5_real_rows_in_original_order",
    "new_sampling": False,
    "negative_ratio": "natural_after_removing_rotation_none_not_forced_1_to_3",
    "preserve_pre_F_hard_negatives": True,
}
TASKS = ("adjustment", "need", "failure", "plan")
MANIFEST_NAMES = {
    "adjustment": "adjustment_end_manifest.jsonl",
    "need": "need_recovery_manifest.jsonl",
    "failure": "failure_reason_manifest.jsonl",
    "plan": "recovery_plan_manifest.jsonl",
}


def convert_manifests(source_index, source_rows):
    manifests, remaps, removed = {}, {}, []
    for task in TASKS:
        rows, remap = [], {}
        for position, original in enumerate(source_rows[task]):
            if task == "need" and original["need_variant"] != "real":
                if (
                    original["need_variant"] != "rotation_none"
                    or original["split"] != "train"
                    or original["need_recovery"]
                ):
                    raise ValueError("Unexpected V9.5 need counterfactual")
                removed.append(
                    {key: original[key] for key in ("split", "global_index", "episode_id", "attempt_id", "frame_index")}
                )
                continue
            row = copy.deepcopy(original)
            row.update(schema_version=MANIFEST_SCHEMA, data_profile=DATA_PROFILE)
            row["prompt"] = remove_touch(row["prompt"], task=task)
            for key in ("need_pair_id", "need_counterfactual", "plan_token_lengths"):
                row.pop(key, None)
            remap[position] = len(rows)
            rows.append(row)
        manifests[task], remaps[task] = rows, remap
    splits = copy.deepcopy(source_index["splits"])
    for split in SPLITS:
        for task in TASKS:
            positions = [
                remaps[task][p]
                for p in source_index["splits"][split][task]["manifest_row_indices"]
                if p in remaps[task]
            ]
            splits[split][task] = {
                "manifest_row_indices": positions,
                "global_indices": [manifests[task][p]["global_index"] for p in positions],
                "sample_count": len(positions),
            }
    return manifests, splits, removed


def conversion_audit(source_index, manifests, splits, removed):
    return {
        "schema_version": "book_v9_6_need_conversion_audit_v1",
        "source_training_data_hash": source_index["training_data_hash"],
        "input_policy": INPUT_POLICY,
        "sampling_policy": SAMPLING_POLICY,
        "removed_counterfactual_count": len(removed),
        "removed_counterfactual_frames": removed,
        "splits": {
            split: {
                "positive_count": sum(
                    bool(manifests["need"][p]["need_recovery"]) for p in splits[split]["need"]["manifest_row_indices"]
                ),
                "negative_count": sum(
                    not manifests["need"][p]["need_recovery"] for p in splits[split]["need"]["manifest_row_indices"]
                ),
                "sample_count": splits[split]["need"]["sample_count"],
            }
            for split in SPLITS
        },
    }


def derive(source_index, source_rows, source_path):
    manifests, splits, removed = convert_manifests(source_index, source_rows)
    token_summary = validate_dataset_plan_tokens(
        manifests["plan"],
        dataset_dir=Path(source_index["dataset_dir"]),
        norm_stats_dir=Path(source_index["v4_dir"]) / "norm_stats",
    )
    index = copy.deepcopy(source_index)
    index.pop("training_data_hash", None)
    for task in TASKS:
        index.pop(f"{task}_manifest_file", None)
        index.pop(f"{task}_manifest_sha256", None)
    for name in ("need_boundary_audit", "reasoning_boundary_audit"):
        index.pop(f"{name}_file", None)
        index.pop(f"{name}_sha256", None)
    index.update(
        schema_version=INDEX_SCHEMA,
        data_profile=DATA_PROFILE,
        experiment_version=VERSION_TAG,
        prompt_profile=PROMPT_PROFILE,
        history_policy=source_index["history_policy"] | {"prompt_profile": PROMPT_PROFILE},
        input_policy=INPUT_POLICY,
        captioner_identity_scope="offline_label_provenance_only",
        need_label_policy=LABEL_POLICY,
        need_sampling_policy=SAMPLING_POLICY,
        reasoning_window_policy=REASONING_WINDOW_POLICY,
        splits=splits,
        source_multitask_index_file=str(source_path.resolve()),
        source_multitask_index_sha256=sha256_file(source_path),
        source_multitask_training_data_hash=source_index["training_data_hash"],
        manifest_content_hashes={task: sha256_json(rows) for task, rows in manifests.items()},
        plan_token_validation=token_summary,
    )
    need_audit = conversion_audit(source_index, manifests, splits, removed)
    reasoning_audit = json.loads(Path(source_index["reasoning_boundary_audit_file"]).read_text())
    reasoning_audit.update(schema_version="book_v9_6_reasoning_boundary_audit_v1", input_policy=INPUT_POLICY)
    return index, manifests, need_audit, reasoning_audit


def load_source(path):
    path = reject_archive(path)
    index = json.loads(path.read_text())
    source.validate_index(index)  # Existing captions are only audited; no captioner is instantiated.
    rows = {task: load_jsonl(Path(index[f"{task}_manifest_file"])) for task in TASKS}
    return index, rows


def validate_index(index):
    if index.get("experiment_version") != VERSION_TAG or index.get("schema_version") != INDEX_SCHEMA:
        raise ValueError("V9.6 requires its independent real-only no-touch index")
    source_path = reject_archive(Path(index["source_multitask_index_file"]))
    if sha256_file(source_path) != index["source_multitask_index_sha256"]:
        raise ValueError("V9.6 source V9.5 index changed")
    source_index, source_rows = load_source(source_path)
    expected, manifests, need_audit, reasoning_audit = derive(source_index, source_rows, source_path)
    excluded = {"training_data_hash"}
    for task in TASKS:
        excluded.update((f"{task}_manifest_file", f"{task}_manifest_sha256"))
        path = reject_archive(Path(index[f"{task}_manifest_file"]))
        if sha256_file(path) != index[f"{task}_manifest_sha256"] or load_jsonl(path) != manifests[task]:
            raise ValueError(f"V9.6 {task} is not the exact no-touch real V9.5 conversion")
        for row in manifests[task]:
            validate_no_touch(row["prompt"])
    for name, audit in (("need_boundary_audit", need_audit), ("reasoning_boundary_audit", reasoning_audit)):
        excluded.update((f"{name}_file", f"{name}_sha256"))
        path = reject_archive(Path(index[f"{name}_file"]))
        if sha256_file(path) != index[f"{name}_sha256"] or json.loads(path.read_text()) != audit:
            raise ValueError(f"V9.6 {name} mismatch")
    if {k: v for k, v in index.items() if k not in excluded} != expected:
        raise ValueError("V9.6 protocol/selection differs from the exact real-only V9.5 conversion")
    if index["training_data_hash"] != sha256_json({k: v for k, v in index.items() if k != "training_data_hash"}):
        raise ValueError("V9.6 training data hash mismatch")


def validate_stage_a_for_training(checkpoint, index):
    # Deliberately validate book_v9_5, not book_v9_6: there is no new Stage A run.
    return source.validate_stage_a_for_training(checkpoint, index) | {"experiment_version": "book_v9_5"}


def summary(index, manifests, need_audit):
    return {
        "experiment_version": VERSION_TAG,
        "data_profile": DATA_PROFILE,
        "input_policy": INPUT_POLICY,
        "model_dependency": "none_data_only",
        "source_multitask_training_data_hash": index["source_multitask_training_data_hash"],
        "need": need_audit,
        "selected_counts": {
            split: {task: index["splits"][split][task]["sample_count"] for task in TASKS} for split in SPLITS
        },
        "manifest_counts": {task: dict(Counter(row["split"] for row in rows)) for task, rows in manifests.items()},
        "plan_token_validation": index["plan_token_validation"],
    }
