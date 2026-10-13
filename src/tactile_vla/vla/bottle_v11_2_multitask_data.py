"""Bottle V11.2: compressed action targets and adjustment-only retained-frame H100."""

from __future__ import annotations

import json
from pathlib import Path

from tactile_vla.vla import bottle_v11_1_multitask_data as previous
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.bottle_v11_2_stage_a_data import (
    ROOT,
    VERSION_TAG,
    DEFAULT_STAGE_A as DEFAULT_STAGE_A,
    validate_training_index as validate_action_index,
)
from tactile_vla.vla.v4_data import load_jsonl
from tactile_vla.vla.v7_7_phase_prompt import sample_episode_history
from tactile_vla.vla.structured_text import ConstrainedTokenGrammar

DATA_PROFILE = "bottle_v11_2_five_task_h100"
INDEX_SCHEMA = "tactile_vla_bottle_v11_2_multitask_index_v1"
MANIFEST_SCHEMA = "tactile_vla_bottle_v11_2_multitask_manifest_v1"
DEFAULT_INDEX = ROOT / "outputs/bottle_v11_2_multitask/bottle_v11_2_multitask_training_index.json"
RUN_NAME = "pi05_bottle_v11_2_five_task_h100_no_history"
HISTORY_POLICY = previous.HISTORY_POLICY | {
    "idle_perturbation": "adjustment_end_only_open_rotation_gap_compressed",
    "adjustment_end": {
        "clock": "last_100_retained_frames_inclusive_then_uniform_11",
        "removed_interval_source": "action_index.idle_gap_audit",
        "raw_current_frame": "always_keep_as_last_point_even_inside_removed_interval",
        "early_attempt2_prefix": "same_episode_attempt1_tail_unchanged",
        "labels_and_selection": "unchanged_raw_frame_R_and_R_minus_10",
        "future_access": "none_only_prefix_through_current",
    },
    "need_failure_plan": "unchanged_V11.1_raw_history",
}


def build_training_grammars(tokenizer, *, args, index):
    if (
        args.data_profile != DATA_PROFILE
        or index.get("experiment_version") != VERSION_TAG
        or index.get("text_schema_policy") != previous.TEXT_SCHEMA_POLICY
    ):
        raise ValueError("Bottle V11.2 text grammar requires its versioned data/schema contract")

    def encode(text):
        return tokenizer.encode_text(text, add_eos=True)

    return (
        ConstrainedTokenGrammar(previous.bottle_failure_reasons(), encode=encode),
        ConstrainedTokenGrammar(previous.bottle_recovery_plans(), encode=encode),
    )


def gap_lookup(action):
    return {(r["episode_id"], r["attempt_id"]): r for r in action["idle_gap_audit"]["attempts"]}


def sample_compressed_adjustment_history(timeline, current_position, *, gaps):
    if not 0 <= current_position < len(timeline):
        raise IndexError(current_position)
    current = timeline[current_position]
    retained = []
    for point in timeline[: current_position + 1]:
        gap = gaps.get((point.episode_id, point.attempt_id))
        removed = gap is not None and gap["removed_start"] <= point.frame_index < gap["removed_stop"]
        if not removed or point.global_index == current.global_index:
            retained.append(point)
    return sample_episode_history(retained, len(retained) - 1)


def build_artifacts(*, dataset_dir, v4_dir, action_index_file, incremental_state_dir):
    action = json.loads(Path(action_index_file).read_text())
    gaps = gap_lookup(action)
    index, summary, rows = previous.build_artifacts(
        dataset_dir=dataset_dir,
        v4_dir=v4_dir,
        action_index_file=action_index_file,
        incremental_state_dir=incremental_state_dir,
        action_validator=validate_action_index,
        version_tag=VERSION_TAG,
        data_profile=DATA_PROFILE,
        index_schema=INDEX_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
        history_policy=HISTORY_POLICY,
        adjustment_history_sampler=lambda timeline, p: sample_compressed_adjustment_history(timeline, p, gaps=gaps),
    )
    index["action_target_policy"] = action["target_policy"]
    index["idle_gap_audit"] = action["idle_gap_audit"]
    summary["adjustment_history_note"] = HISTORY_POLICY["adjustment_end"]
    summary["removed_idle_frames"] = sum(g["removed_frame_count"] for g in gaps.values())
    return index, summary, rows


def validate_index(index):
    if (index.get("schema_version"), index.get("experiment_version"), index.get("data_profile")) != (
        INDEX_SCHEMA,
        VERSION_TAG,
        DATA_PROFILE,
    ):
        raise ValueError("V11.2 multitask index header mismatch")
    if index.get("training_data_hash") != sha256_json({k: v for k, v in index.items() if k != "training_data_hash"}):
        raise ValueError("V11.2 multitask index hash mismatch")
    for path, digest in index["source_hashes"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"V11.2 source changed: {path}")
    expected, _, rows = build_artifacts(
        dataset_dir=Path(index["dataset_dir"]),
        v4_dir=Path(index["v4_dir"]),
        action_index_file=Path(index["action_index_file"]),
        incremental_state_dir=Path(index["incremental_state_dir"]),
    )
    for key, value in expected.items():
        if index.get(key) != value:
            raise ValueError(f"V11.2 source-derived index {key} mismatch")
    for task, expected_rows in rows.items():
        path = Path(index[f"{task}_manifest_file"])
        if sha256_file(path) != index[f"{task}_manifest_sha256"] or load_jsonl(path) != expected_rows:
            raise ValueError(f"V11.2 {task} manifest differs from source-derived prompts/history")


def validate_stage_a_for_training(checkpoint, index):
    # Keep all identity checks, but no required step or planned-total restriction.
    checkpoint = Path(checkpoint).resolve()
    config_path, metadata = checkpoint.parent / "config.json", checkpoint / "params/_METADATA"
    if not metadata.is_file():
        raise ValueError("V11.2 Stage A params metadata is missing")
    config = json.loads(config_path.read_text())
    action = json.loads(Path(index["action_index_file"]).read_text())
    identity = config.get("artifact_identity", {})
    if (
        config.get("experiment_version") != VERSION_TAG
        or config.get("use_state_history") is not False
        or config.get("data_profile") != "book_stage_a_v1"
        or config.get("source_scope") != action["source_scope"]
        or config.get("failure_action_policy") != action["target_policy"]["failure_action_filter"]
        or config.get("action_target_policy") != action["target_policy"]
        or config.get("idle_gap_audit_sha256") != sha256_json(action["idle_gap_audit"])
        or identity.get("training_data_hash") != index["action_training_data_hash"]
        or identity.get("index_sha256") != sha256_file(index["action_index_file"])
        or identity.get("v4_norm_stats_sha256") != action["v4_norm_stats_sha256"]
    ):
        raise ValueError("V11.2 Stage A version/data/idle compression/norm identity mismatch")
    return {
        "path": str(checkpoint),
        "step": int(checkpoint.name) if checkpoint.name.isdecimal() else None,
        "experiment_version": VERSION_TAG,
        "config_sha256": sha256_file(config_path),
        "params_metadata_sha256": sha256_file(metadata),
        "action_training_data_hash": index["action_training_data_hash"],
    }
