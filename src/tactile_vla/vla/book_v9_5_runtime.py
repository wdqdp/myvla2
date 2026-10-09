"""Book V9.5 deployment identities, six-field captions and threshold contracts."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from tactile_vla.common.labels_v4 import LABEL_FIELDS, LABEL_MAPS, LABEL_SCHEMA_VERSION, labels_to_caption
from tactile_vla.vla.book_v9_4_5_action_replay import ACTION_SAMPLING_POLICY
from tactile_vla.vla.book_v9_4_memory import MEMORY_POLICY as TRAINING_MEMORY_POLICY
from tactile_vla.vla.book_v9_5_multitask_data import DATA_PROFILE, LABEL_POLICY, REASONING_WINDOW_POLICY, SAMPLING_POLICY
from tactile_vla.vla.book_v9_5_stage_a_data import SCOPE_POLICY
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE, uniform_positions

BOOK_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book")
BOOK_INSTRUCTION = "Pick up the book and place it horizontally on the bookshelf."
DEFAULT_RUN = BOOK_ROOT / "outputs/multitask_v9_5/pi05_book_v9_5_five_task_h100_no_history"
DEFAULT_NORM_DIR = BOOK_ROOT / "outputs/rotation_v4/norm_stats"
DEFAULT_INDEX = BOOK_ROOT / "outputs/book_v9_5_multitask/book_v9_5_multitask_training_index.json"
SERVER_NAME = "tactile_vla_book_v9_5"
THRESHOLD_SCHEMA = "book_v9_5_val_thresholds_v1"
DEFAULT_THRESHOLD = 0.5
DEFAULT_CHECKPOINT = DEFAULT_RUN / "2000"
DEFAULT_CAPTIONER = Path(
    "/data1/qxh/tac_vla_new/tac_data/tac_cap_data/tactile_captioner/"
    "tcn_v4_w30_masked_six_head_mixed/best.pt"
)
DEFAULT_CAPTIONER_SHA256 = "b13517e9ff35732ca9e178b740a42edece74afce4a8710f4d773d3559d6a5e53"
HISTORY_OFFSETS = uniform_positions(100).tolist()
MEMORY_POLICY = TRAINING_MEMORY_POLICY["runtime_retention"]
VERSION_TAG = "book_v9_5"


def deployment_version(data_profile: str | None) -> str:
    if data_profile != DATA_PROFILE:
        raise ValueError(f"Unsupported Book V9.5 deployment data_profile: {data_profile!r}")
    return VERSION_TAG


def deployment_policy_fields(version: str) -> dict[str, Any]:
    """Compact training contracts checked in config, export and client metadata."""
    if version != VERSION_TAG:
        raise ValueError(f"Unsupported Book V9.5 deployment version: {version!r}")
    return {
        "training_profile": "book_v9_5_phase_balanced_action_replay",
        "data_experiment_version": version,
        "need_label_policy": LABEL_POLICY,
        "need_sampling_policy": SAMPLING_POLICY,
        "reasoning_window_policy": REASONING_WINDOW_POLICY,
        "action_sampling_policy": ACTION_SAMPLING_POLICY,
    }


def validate_source_identity(payload: dict[str, Any], training_data_hash: str) -> None:
    scope = payload.get("source_scope")
    if not isinstance(scope, dict) or scope.get("policy") != SCOPE_POLICY:
        raise ValueError("Book V9.5 requires its current-data source_scope policy")
    if not re.fullmatch(r"[0-9a-f]{64}", str(training_data_hash)) or payload.get("training_data_hash") != training_data_hash:
        raise ValueError("Book V9.5 training_data_hash differs from its artifact identity")
    stage_a = payload.get("stage_a_initialization_identity", {})
    if not isinstance(stage_a, dict) or stage_a.get("step") != 15000:
        raise ValueError("Book V9.5 requires its Stage A 15000 initialization identity")
    for key in ("action_training_data_hash", "config_sha256", "params_metadata_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(stage_a.get(key, ""))):
            raise ValueError(f"Book V9.5 Stage A initialization identity is missing {key}")
    for key in ("need_boundary_audit_sha256", "reasoning_boundary_audit_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get(key, ""))):
            raise ValueError(f"Book V9.5 identity is missing {key}")


def validate_captioner_identity(identity: dict[str, Any]) -> None:
    expected = {"label_schema_version": LABEL_SCHEMA_VERSION,
                "label_fields": list(LABEL_FIELDS), "window_size": 30}
    mismatch = {key: (identity.get(key), value) for key, value in expected.items()
                if identity.get(key) != value}
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("checkpoint_sha256", ""))):
        mismatch["checkpoint_sha256"] = (identity.get("checkpoint_sha256"), "SHA256")
    if mismatch:
        raise ValueError(f"Book V9.5 captioner identity mismatch: {mismatch}")


def validate_tactile_caption(caption: str) -> None:
    names = ("area", "Fx", "Fy", "Fz", "Fz_bias", "rotation")
    match = re.fullmatch(r"Touch\[(.*)\]", caption)
    if match is None:
        raise ValueError("Book V9.5 requires a six-field Touch caption")
    parts = match.group(1).split("; ")
    if len(parts) != len(names):
        raise ValueError("Book V9.5 Touch caption is missing six-field labels")
    labels = {}
    for part, name, field in zip(parts, names, LABEL_FIELDS, strict=True):
        key, separator, value = part.partition("=")
        if key != name or not separator or value not in LABEL_MAPS[field]:
            raise ValueError(f"Invalid Book V9.5 tactile field: {part!r}")
        labels[field] = value
    if labels_to_caption(labels) != caption:
        raise ValueError("Book V9.5 Touch caption must retain the training format")


def validate_training_config(config: dict[str, Any], norm_sha: str) -> None:
    version = deployment_version(config.get("data_profile"))
    expected = {
        "prompt_profile": PROMPT_PROFILE,
        "action_horizon": 30, "action_dim": 32, "max_token_len": 512,
        "reasoning_max_token_len": 320, "use_state_history": False, "state_history_len": 0,
        "history_hidden_dim": 0, "state_history_dim": 7, "state_history_fps": 30.0,
        "grammar_profile": "v3_full_v1", "phase_prefill_protocol": "need_failure_shared_kv_v1",
        "plan_memory_policy": TRAINING_MEMORY_POLICY,
    }
    expected["experiment_version"] = version
    expected.update(deployment_policy_fields(version))
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    identity = config.get("artifact_identity", {})
    if identity.get("data_profile") != "book_stage_a_v1":
        mismatch["artifact_identity.data_profile"] = (identity.get("data_profile"), "book_stage_a_v1")
    if identity.get("v4_norm_stats_sha256") != norm_sha:
        mismatch["norm_stats_sha256"] = (norm_sha, identity.get("v4_norm_stats_sha256"))
    training_hash_key = f"{version}_training_data_hash"
    if not identity.get(training_hash_key):
        mismatch[training_hash_key] = (None, "nonempty")
    if mismatch:
        raise ValueError(f"Book V9.5 deployment config mismatch: {mismatch}")
    validate_captioner_identity(config.get("captioner_identity", {}))
    validate_source_identity(config, identity[training_hash_key])
    if config["stage_a_initialization_identity"]["action_training_data_hash"] != identity.get("training_data_hash"):
        raise ValueError("Book V9.5 Stage A action data identity differs from initialization")


def resolve_thresholds(
    *, calibration: dict[str, Any] | None, step: int, full_params_sha: str,
    norm_sha: str, training_data_hash: str, need_override: float | None,
    adjustment_override: float | None,
    data_profile: str = DATA_PROFILE,
) -> tuple[dict[str, float], dict[str, bool]]:
    version = deployment_version(data_profile)
    if calibration is not None:
        expected = {
            "schema_version": f"{version}_val_thresholds_v1", "data_profile": data_profile,
            "prompt_profile": PROMPT_PROFILE, "checkpoint_step": step,
            "full_params_sha256": full_params_sha, "norm_stats_sha256": norm_sha,
            "training_data_hash": training_data_hash, "selection_split": "val",
            "thresholds_status": "calibrated_on_book_val",
        }
        mismatch = {key: (calibration.get(key), value) for key, value in expected.items()
                    if calibration.get(key) != value}
        if mismatch:
            raise ValueError(f"Book V9.5 threshold identity mismatch: {mismatch}")
    thresholds, overrides = {}, {}
    for task, override in (("need_recovery", need_override), ("adjustment_end", adjustment_override)):
        value = override if override is not None else (
            DEFAULT_THRESHOLD if calibration is None else calibration.get("thresholds", {}).get(task)
        )
        if value is None:
            raise ValueError(f"{task}: supplied thresholds file is missing this threshold")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{task} threshold must be finite and in [0,1]")
        thresholds[task] = value
        overrides[task] = override is not None
    return thresholds, overrides


def validate_server_metadata(metadata: dict[str, Any]) -> None:
    version = deployment_version(metadata.get("data_profile"))
    expected = {
        "name": SERVER_NAME, "phase_prompt_profile": PROMPT_PROFILE,
        "action_prompt_profile": "phase_v2", "supports_streamed_phase_events": True,
        "supports_action_noise": True, "requires_action_noise": True,
        "supports_failure_generation": True, "supports_recovery_generation": True,
        "action_horizon": 30, "action_dim": 32, "output_action_dim": 7,
        "use_state_history": False, "state_history_len": 0, "state_history_fps": 30.0,
        "qpos_h100_sample_offsets": HISTORY_OFFSETS, "max_memory_pairs": 4,
        "memory_policy": MEMORY_POLICY, "max_supported_attempts": None,
        "classification_qpos_policy": "raw_no_gripper_remap",
        "episode_start_padding": "left_pad_episode_frame_0", "captioner_window_size": 30,
    }
    expected["experiment_version"] = version
    expected.update(deployment_policy_fields(version))
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items()
                if metadata.get(key) != value}
    if mismatch:
        raise ValueError(f"Book V9.5 client/server metadata mismatch: {mismatch}")
    validate_captioner_identity(metadata.get("captioner_identity", {}))
    validate_source_identity(metadata, metadata.get("training_data_hash", ""))
    if metadata.get("captioner_checkpoint_sha256") != metadata["captioner_identity"]["checkpoint_sha256"]:
        raise ValueError("Book V9.5 runtime captioner differs from training provenance")
    for task in ("need_recovery", "adjustment_end"):
        value = float(metadata.get(f"{task}_threshold", -1))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid {task} threshold")
    if metadata.get("thresholds_status") not in {
        "default_0_5", "calibrated_on_book_val", "explicit_manual_override",
    }:
        raise ValueError("Book V9.5 requires default, calibrated or explicitly overridden thresholds")
