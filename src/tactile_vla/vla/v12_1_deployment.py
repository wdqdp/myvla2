"""Deployment contract for the joint Blackbox/Book V12.1 model."""

from __future__ import annotations

import math
from pathlib import Path
import re

from tactile_vla.common import labels as legacy_labels
from tactile_vla.common import labels_v4
from tactile_vla.vla.book_v9_4_memory import append_runtime_memory as append_book_memory
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, update_failure_recovery_memory
from tactile_vla.vla.structured_text import legal_failure_reasons, legal_recovery_plans
from tactile_vla.vla.v12_1_data import DATA_PROFILE, OUTPUT, POLICY, ROOT, canonical_touch
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE, uniform_positions

VERSION = "v12_1"
SERVER_NAME = "tactile_vla_v12_1"
DEFAULT_RUN = ROOT / "joint_blackbox_book/outputs/multitask_v12_1/pi05_blackbox_book_v12_1_five_task_h100_no_history"
DEFAULT_CHECKPOINT = DEFAULT_RUN / "1000"
DEFAULT_NORM_DIR = OUTPUT / "norm_stats"
HISTORY_OFFSETS = uniform_positions(100).tolist()
TASKS = {
    "blackbox": {
        "instruction": "Pick up and transfer the object stably.",
        "captioner": Path("/data1/outputs/tactile_captioner/tcn_v3_w30_rotation_head/best.pt"),
        "captioner_sha256": "10818320f415bfc27b2965496fa1c9be8498f13c17300e528f245398266a4118",
        "schema_version": legacy_labels.LABEL_SCHEMA_VERSION,
        "label_fields": list(legacy_labels.LABEL_FIELDS),
        "memory_policy": "retain_all_up_to_four_then_stop",
    },
    "book": {
        "instruction": "Pick up the book and place it horizontally on the bookshelf.",
        "captioner": ROOT.parent / "tac_cap_data/tactile_captioner/tcn_v4_w30_masked_six_head_mixed/best.pt",
        "captioner_sha256": "b13517e9ff35732ca9e178b740a42edece74afce4a8710f4d773d3559d6a5e53",
        "schema_version": labels_v4.LABEL_SCHEMA_VERSION,
        "label_fields": list(labels_v4.LABEL_FIELDS),
        "memory_policy": "initial_pair_plus_latest_three",
    },
}


def threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Classification thresholds must be finite and in [0,1]")
    return value


def validate_training_config(config: dict, norm_sha: str | None = None) -> None:
    expected = {
        "experiment_version": VERSION,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "grammar_profile": "v3_full_v1",
        "phase_prefill_protocol": "need_failure_shared_kv_v1",
        "action_horizon": 30,
        "action_dim": 32,
        "use_state_history": False,
        "state_history_len": 0,
        "history_hidden_dim": 0,
        "state_history_dim": 7,
        "state_history_fps": 30.0,
        "max_token_len": 512,
        "reasoning_max_token_len": 320,
        "joint_policy": POLICY,
        "failure_grammar": list(legal_failure_reasons()),
        "recovery_grammar": list(legal_recovery_plans()),
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    identity = config.get("artifact_identity", {})
    for key, value in (("data_profile", DATA_PROFILE), ("training_data_hash", config.get("training_data_hash"))):
        if not value or identity.get(key) != value:
            mismatches[f"artifact_identity.{key}"] = (identity.get(key), value)
    for key in ("training_data_hash", "norm_stats_sha256", "index_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(identity.get(key, ""))) is None:
            mismatches[key] = (identity.get(key), "SHA256")
    if norm_sha is not None and identity.get("norm_stats_sha256") != norm_sha:
        mismatches["norm_stats_sha256"] = (norm_sha, identity.get("norm_stats_sha256"))
    if mismatches:
        raise ValueError(f"V12.1 deployment config mismatch: {mismatches}")


def validate_tactile_caption(caption: str, task: str) -> None:
    match = re.fullmatch(r"Touch\[(.*)\]", caption)
    names = ("area", "Fx", "Fy", "Fz", "Fz_bias", "rotation")
    parts = match.group(1).split("; ") if match else []
    if len(parts) != len(names):
        raise ValueError("V12.1 requires six canonical Touch fields")
    for part, name, field in zip(parts, names, labels_v4.LABEL_FIELDS, strict=True):
        key, separator, value = part.partition("=")
        allowed = {"unknown"} if task == "blackbox" and field == "fz_bias" else labels_v4.LABEL_MAPS[field]
        if key != name or not separator or value not in allowed:
            raise ValueError(f"Invalid V12.1 {task} tactile field: {part!r}")


def runtime_caption(caption: str, task: str) -> str:
    # The blackbox source has five measured fields; do not invent an Fz_bias label.
    result = canonical_touch(caption) if task == "blackbox" else caption
    validate_tactile_caption(result, task)
    return result


def append_memory(memory: list[dict], entry: dict, task: str) -> list[dict]:
    if task == "book":
        return append_book_memory(memory, entry)
    return update_failure_recovery_memory(memory, entry, prompt_profile=MINIMAL_PROMPT_PROFILE)


def deployment_fields(task: str, *, captioner_sha: str | None = None) -> dict:
    spec = TASKS[task]
    return {
        "name": SERVER_NAME,
        "experiment_version": VERSION,
        "data_profile": DATA_PROFILE,
        "task": task,
        "instruction": spec["instruction"],
        "phase_prompt_profile": PROMPT_PROFILE,
        "action_prompt_profile": "phase_v2",
        "action_horizon": 30,
        "action_dim": 32,
        "output_action_dim": 7,
        "use_state_history": False,
        "state_history_len": 0,
        "state_history_fps": 30.0,
        "qpos_h100_sample_offsets": HISTORY_OFFSETS,
        "classification_qpos_policy": "raw_no_gripper_remap",
        "episode_start_padding": "left_pad_episode_frame_0",
        "max_memory_pairs": 4,
        "memory_policy": spec["memory_policy"],
        "max_supported_attempts": 5 if task == "blackbox" else None,
        "captioner_window_size": 30,
        "captioner_schema_version": spec["schema_version"],
        "captioner_label_fields": spec["label_fields"],
        "captioner_checkpoint_sha256": captioner_sha or spec["captioner_sha256"],
        "captioner_identity_scope": "runtime_client_server_source_defaults_not_recorded_in_joint_config",
        "direction_mapping": POLICY["direction_mapping"][task],
        "supports_streamed_phase_events": True,
        "supports_action_noise": True,
        "requires_action_noise": True,
        "supports_failure_generation": True,
        "supports_recovery_generation": True,
        "worker_policy": "single_model_worker_stream_events_before_continuation",
    }


def validate_server_metadata(metadata: dict, task: str) -> None:
    expected = deployment_fields(task)
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"V12.1 client/server metadata mismatch: {mismatches}")
    config = metadata.get("config", {})
    validate_training_config(config, metadata.get("norm_stats_sha256"))
    for key in ("need_recovery_threshold", "adjustment_end_threshold"):
        threshold(metadata[key])
    if metadata.get("training_data_hash") != config["training_data_hash"]:
        raise ValueError("V12.1 client/server training data hash mismatch")
    if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("full_params_sha256", ""))) is None:
        raise ValueError("V12.1 server full_params SHA256 is missing")
