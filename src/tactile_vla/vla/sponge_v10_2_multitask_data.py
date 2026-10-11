"""Exact no-touch ablation of Sponge V10.1; no relabeling or resampling."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from tactile_vla.vla import sponge_v10_1_multitask_data as source
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.sponge_v10_2_prompts import INPUT_POLICY, PROMPT_PROFILE, remove_touch, validate_no_touch
from tactile_vla.vla.v4_data import SPLITS, load_jsonl
from tactile_vla.vla.v5_3_adjustment_end_data import scan_selected_qpos

ROOT = source.ROOT
DEFAULT_STAGE_A = source.DEFAULT_STAGE_A
DEFAULT_SOURCE_INDEX = source.DEFAULT_INDEX
VERSION_TAG = "sponge_v10_2"
DATA_PROFILE = "sponge_v10_2_five_task_h100_visual"
INDEX_SCHEMA = "tactile_vla_sponge_v10_2_multitask_index_v1"
MANIFEST_SCHEMA = "tactile_vla_sponge_v10_2_multitask_manifest_v1"
DEFAULT_OUTPUT = ROOT / "outputs/sponge_v10_2_multitask"
DEFAULT_INDEX = DEFAULT_OUTPUT / "sponge_v10_2_multitask_training_index.json"
RUN_NAME = "pi05_sponge_v10_2_five_task_h100_no_history"
FAILURE_TARGET = source.FAILURE_TARGET
PLAN_TARGET = source.PLAN_TARGET
NEED_POLICY = source.NEED_POLICY | {"caption": "absent_from_model_input_offline_labels_unchanged"}
ADJUSTMENT_POLICY = source.ADJUSTMENT_POLICY
REASONING_POLICY = source.REASONING_POLICY
MEMORY_POLICY = source.MEMORY_POLICY
HISTORY_POLICY = source.HISTORY_POLICY | {"prompt_profile": PROMPT_PROFILE}
TASKS = ("adjustment", "need", "failure", "plan")
MANIFEST_NAMES = {
    "adjustment": "adjustment_end_manifest.jsonl",
    "need": "need_recovery_manifest.jsonl",
    "failure": "failure_reason_manifest.jsonl",
    "plan": "recovery_plan_manifest.jsonl",
}


def convert_manifests(source_rows):
    """Keep row identity/order, labels, H100 and memory exactly; strip Touch only."""
    rows = copy.deepcopy(source_rows)
    for task in TASKS:
        for row in rows[task]:
            row.update(schema_version=MANIFEST_SCHEMA, data_profile=DATA_PROFILE)
            row["prompt"] = remove_touch(row["prompt"], task=task)
            row.pop("token_lengths", None)  # Recomputed with the actual no-touch input.
    return rows


def validate_tokens(index, manifests):
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.transforms import Normalize

    qpos = scan_selected_qpos(
        dataset_dir=Path(index["dataset_dir"]),
        selected_episode_ids={r["episode_id"] for rows in manifests.values() for r in rows},
    )
    tokenizer = PaligemmaTokenizer(512)
    normalizer = Normalize(normalize.load(Path(index["v4_dir"]) / "norm_stats"), use_quantiles=True)
    summary = {}
    for task, rows in manifests.items():
        lengths = {}
        for row in rows:
            validate_no_touch(row["prompt"])
            state = normalizer({"state": np.asarray(qpos[row["global_index"]], dtype=np.float32)})["state"]
            target_text = row.get("target_failure_reason", row.get("target_recovery_plan"))
            target = None if target_text is None else tokenizer.encode_text(target_text, add_eos=True)
            _, mask, _, _, prefix = tokenizer.tokenize_structured_response(
                row["prompt"], state, target, max_len=320 if task == "plan" else 512
            )
            row["token_lengths"] = {
                "prefix": int(prefix),
                "total": int(mask.sum()),
                "target": 0 if target is None else len(target),
            }
            lengths.setdefault(str(row.get("memory_length", "all")), []).append(int(mask.sum()))
        summary[task] = {k: {"count": len(v), "min": min(v), "max": max(v)} for k, v in lengths.items()}
    return summary


def load_source(path):
    index = json.loads(Path(path).read_text())
    source.validate_index(index)
    return index, {task: load_jsonl(Path(index[f"{task}_manifest_file"])) for task in TASKS}


def derive(source_index, source_rows, source_path):
    """Construction is checkpoint-free; captions remain offline provenance only."""
    rows = convert_manifests(source_rows)
    token_summary = validate_tokens(source_index, rows)
    index = copy.deepcopy(source_index)
    index.pop("training_data_hash", None)
    for task in TASKS:
        index.pop(f"{task}_manifest_file", None)
        index.pop(f"{task}_manifest_sha256", None)
    index.update(
        schema_version=INDEX_SCHEMA,
        experiment_version=VERSION_TAG,
        data_profile=DATA_PROFILE,
        prompt_profile=PROMPT_PROFILE,
        input_policy=INPUT_POLICY,
        captioner_identity_scope="offline_label_provenance_only",
        need_policy=NEED_POLICY,
        history_policy=HISTORY_POLICY,
        source_multitask_index_file=str(Path(source_path).resolve()),
        source_multitask_index_sha256=sha256_file(source_path),
        source_multitask_training_data_hash=source_index["training_data_hash"],
        token_validation=token_summary,
        manifest_content_hashes={task: sha256_json(items) for task, items in rows.items()},
    )
    index["source_hashes"][str(Path(source_path).resolve())] = sha256_file(source_path)
    # No remapping: every task stream and physical-frame selection is identical.
    assert index["splits"] == source_index["splits"]
    return index, rows


def summary(index, manifests):
    return {
        "experiment_version": VERSION_TAG,
        "input_policy": INPUT_POLICY,
        "source_multitask_training_data_hash": index["source_multitask_training_data_hash"],
        "model_dependency": "none_data_only",
        "selected_counts": {
            s: {"action": len(index["splits"][s]["action"]["indices"])}
            | {t: index["splits"][s][t]["sample_count"] for t in TASKS}
            for s in SPLITS
        },
        "token_validation": index["token_validation"],
        "need_labels": {
            s: {
                "positive": sum(
                    manifests["need"][i]["need_recovery"] for i in index["splits"][s]["need"]["manifest_row_indices"]
                ),
                "negative": sum(
                    not manifests["need"][i]["need_recovery"]
                    for i in index["splits"][s]["need"]["manifest_row_indices"]
                ),
            }
            for s in SPLITS
        },
    }


def validate_index(index):
    if (
        index.get("schema_version") != INDEX_SCHEMA
        or index.get("experiment_version") != VERSION_TAG
        or index.get("data_profile") != DATA_PROFILE
    ):
        raise ValueError("V10.2 requires its independent visual-only index")
    if index.get("training_data_hash") != sha256_json({k: v for k, v in index.items() if k != "training_data_hash"}):
        raise ValueError("V10.2 index hash mismatch")
    path = Path(index["source_multitask_index_file"])
    if sha256_file(path) != index["source_multitask_index_sha256"]:
        raise ValueError("V10.2 source V10.1 index changed")
    original, original_rows = load_source(path)
    expected, rows = derive(original, original_rows, path)
    excluded = {"training_data_hash"}
    for task in TASKS:
        excluded.update((f"{task}_manifest_file", f"{task}_manifest_sha256"))
        manifest = Path(index[f"{task}_manifest_file"])
        if sha256_file(manifest) != index[f"{task}_manifest_sha256"] or load_jsonl(manifest) != rows[task]:
            raise ValueError(f"V10.2 {task} differs from the exact V10.1 no-touch conversion")
    if {k: v for k, v in index.items() if k not in excluded} != expected:
        raise ValueError("V10.2 labels/selection/history/input policy differ from V10.1 ablation")


def validate_stage_a_for_training(checkpoint, index):
    # Deliberately reuse V10.1 Stage A, never a tactile multitask checkpoint.
    return source.validate_stage_a_for_training(checkpoint, index)
