"""Exact no-touch conversion of Bottle V11.2, including its compressed action/history."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from tactile_vla.vla import bottle_v11_2_multitask_data as source
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.bottle_v11_3_prompts import INPUT_POLICY, PROMPT_PROFILE, remove_touch, validate_no_touch
from tactile_vla.vla.sponge_v10_2_multitask_data import validate_tokens as validate_tokens
from tactile_vla.vla.structured_text import ConstrainedTokenGrammar
from tactile_vla.vla.v4_data import SPLITS, load_jsonl

ROOT = source.ROOT
VERSION_TAG = "bottle_v11_3"
DATA_PROFILE = "bottle_v11_3_five_task_h100_visual"
INDEX_SCHEMA = "tactile_vla_bottle_v11_3_multitask_index_v1"
MANIFEST_SCHEMA = "tactile_vla_bottle_v11_3_multitask_manifest_v1"
DEFAULT_SOURCE_INDEX = source.DEFAULT_INDEX
DEFAULT_OUTPUT = ROOT / "outputs/bottle_v11_3_multitask"
DEFAULT_INDEX = DEFAULT_OUTPUT / "bottle_v11_3_multitask_training_index.json"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_bottle_stage_a_v11_2_no_history/7500"
RUN_NAME = "pi05_bottle_v11_3_five_task_h100_no_history"
NEED_POLICY = source.previous.NEED_POLICY | {"caption": "absent_from_model_input_offline_labels_unchanged"}
ADJUSTMENT_POLICY = source.previous.ADJUSTMENT_POLICY
REASONING_POLICY = source.previous.REASONING_POLICY
MEMORY_POLICY = source.previous.MEMORY_POLICY
HISTORY_POLICY = source.HISTORY_POLICY | {"prompt_profile": PROMPT_PROFILE}
TEXT_SCHEMA_POLICY = source.previous.TEXT_SCHEMA_POLICY
TASKS = ("adjustment", "need", "failure", "plan")
MANIFEST_NAMES = {
    "adjustment": "adjustment_end_manifest.jsonl",
    "need": "need_recovery_manifest.jsonl",
    "failure": "failure_reason_manifest.jsonl",
    "plan": "recovery_plan_manifest.jsonl",
}


def convert_manifests(original):
    rows = copy.deepcopy(original)
    for task in TASKS:
        for row in rows[task]:
            row.update(schema_version=MANIFEST_SCHEMA, data_profile=DATA_PROFILE)
            row["prompt"] = remove_touch(row["prompt"], task=task)
            row.pop("token_lengths", None)
    return rows


def load_source(path):
    index = json.loads(Path(path).read_text())
    source.validate_index(index)
    return index, {task: load_jsonl(Path(index[f"{task}_manifest_file"])) for task in TASKS}


def derive(original, original_rows, source_path):
    rows = convert_manifests(original_rows)
    tokens = validate_tokens(original, rows)
    index = copy.deepcopy(original)
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
        source_multitask_training_data_hash=original["training_data_hash"],
        token_validation=tokens,
        manifest_content_hashes={task: sha256_json(items) for task, items in rows.items()},
    )
    index["source_hashes"][str(Path(source_path).resolve())] = sha256_file(source_path)
    for key in ("splits", "action_index_file", "action_training_data_hash", "action_target_policy", "idle_gap_audit"):
        if index[key] != original[key]:
            raise AssertionError(f"V11.3 must preserve source V11.2 {key}")
    return index, rows


def summary(index, rows):
    return {
        "experiment_version": VERSION_TAG,
        "input_policy": INPUT_POLICY,
        "model_dependency": "none_data_only",
        "source_multitask_training_data_hash": index["source_multitask_training_data_hash"],
        "selected_counts": {
            s: {"action": len(index["splits"][s]["action"]["indices"])}
            | {task: index["splits"][s][task]["sample_count"] for task in TASKS}
            for s in SPLITS
        },
        "removed_idle_frames": sum(g["removed_frame_count"] for g in index["idle_gap_audit"]["attempts"]),
        "history_note": "V11.2_sources_and_discrete_qpos_unchanged_no_second_compression",
        "token_validation": index["token_validation"],
        "need_labels": {
            s: dict(
                positive=sum(
                    bool(rows["need"][i]["need_recovery"]) for i in index["splits"][s]["need"]["manifest_row_indices"]
                ),
                negative=sum(
                    not rows["need"][i]["need_recovery"] for i in index["splits"][s]["need"]["manifest_row_indices"]
                ),
            )
            for s in SPLITS
        },
    }


def validate_index(index):
    if (index.get("schema_version"), index.get("experiment_version"), index.get("data_profile")) != (
        INDEX_SCHEMA,
        VERSION_TAG,
        DATA_PROFILE,
    ):
        raise ValueError("V11.3 requires its own visual-only index")
    if index.get("training_data_hash") != sha256_json({k: v for k, v in index.items() if k != "training_data_hash"}):
        raise ValueError("V11.3 index hash mismatch")
    path = Path(index["source_multitask_index_file"])
    if sha256_file(path) != index["source_multitask_index_sha256"]:
        raise ValueError("V11.3 source V11.2 index changed")
    original, original_rows = load_source(path)
    expected, rows = derive(original, original_rows, path)
    excluded = {"training_data_hash"}
    for task in TASKS:
        excluded.update((f"{task}_manifest_file", f"{task}_manifest_sha256"))
        manifest = Path(index[f"{task}_manifest_file"])
        if sha256_file(manifest) != index[f"{task}_manifest_sha256"] or load_jsonl(manifest) != rows[task]:
            raise ValueError(f"V11.3 {task} differs from the exact V11.2 no-touch conversion")
    if {k: v for k, v in index.items() if k not in excluded} != expected:
        raise ValueError("V11.3 labels/history/compression/input policy differ from the source ablation")


def validate_stage_a_for_training(checkpoint, index):
    return source.validate_stage_a_for_training(checkpoint, index)


def build_training_grammars(tokenizer, *, args, index):
    if (
        args.data_profile != DATA_PROFILE
        or index.get("experiment_version") != VERSION_TAG
        or index.get("text_schema_policy") != TEXT_SCHEMA_POLICY
    ):
        raise ValueError("Bottle V11.3 requires its own versioned grammar contract")

    def encode(text):
        return tokenizer.encode_text(text, add_eos=True)

    return (
        ConstrainedTokenGrammar(source.previous.bottle_failure_reasons(), encode=encode),
        ConstrainedTokenGrammar(source.previous.bottle_recovery_plans(), encode=encode),
    )


def validate_prompts(index):
    for task in TASKS:
        for row in load_jsonl(Path(index[f"{task}_manifest_file"])):
            validate_no_touch(row["prompt"])
