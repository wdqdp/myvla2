"""Sponge V10.2 no-touch deployment identities, memory and threshold contracts."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from tactile_vla.vla.book_v9_4_5_action_replay import ACTION_SAMPLING_POLICY
from tactile_vla.vla.sponge_v10_2_multitask_data import (
    DATA_PROFILE,
    FAILURE_TARGET,
    PLAN_TARGET,
    ADJUSTMENT_POLICY,
    HISTORY_POLICY,
    MEMORY_POLICY as TRAINING_MEMORY_POLICY,
    NEED_POLICY,
    REASONING_POLICY,
)
from tactile_vla.vla.sponge_v10_1_stage_a_data import FAILURE_ACTION_POLICY
from tactile_vla.vla.structured_text import legal_failure_reasons, legal_recovery_plans
from tactile_vla.vla.v7_7_phase_prompt import uniform_positions
from tactile_vla.vla.sponge_v10_2_prompts import INPUT_POLICY, PROMPT_PROFILE

SPONGE_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/sponge")
SPONGE_INSTRUCTION = "Put the black sponge into the black frame."
DEFAULT_RUN = SPONGE_ROOT / "outputs/multitask_v10_2/pi05_sponge_v10_2_five_task_h100_no_history"
DEFAULT_NORM_DIR = SPONGE_ROOT / "outputs/rotation_v4/norm_stats"
DEFAULT_INDEX = SPONGE_ROOT / "outputs/sponge_v10_2_multitask/sponge_v10_2_multitask_training_index.json"
SERVER_NAME = "tactile_vla_sponge_v10_2"
THRESHOLD_SCHEMA = "sponge_v10_2_val_thresholds_v1"
DEFAULT_THRESHOLD = 0.5
DEFAULT_CHECKPOINT = DEFAULT_RUN / "1000"
HISTORY_OFFSETS = uniform_positions(100).tolist()
MEMORY_POLICY = TRAINING_MEMORY_POLICY
VERSION_TAG = "sponge_v10_2"


def deployment_version(data_profile: str | None) -> str:
    if data_profile != DATA_PROFILE:
        raise ValueError(f"Unsupported Sponge V10.2 deployment data_profile: {data_profile!r}")
    return VERSION_TAG


def deployment_policy_fields(version: str) -> dict[str, Any]:
    """Compact training contracts checked in config, export and client metadata."""
    if version != VERSION_TAG:
        raise ValueError(f"Unsupported Sponge V10.2 deployment version: {version!r}")
    return {
        "include_fz_bias_failure_grammar": True,
        "input_policy": INPUT_POLICY,
        "captioner_identity_scope": "offline_label_provenance_only",
        "failure_action_policy": FAILURE_ACTION_POLICY,
        "need_policy": NEED_POLICY,
        "adjustment_policy": ADJUSTMENT_POLICY,
        "reasoning_policy": REASONING_POLICY,
        "history_policy": HISTORY_POLICY,
        "memory_policy": TRAINING_MEMORY_POLICY,
        "action_sampling_policy": ACTION_SAMPLING_POLICY
        | {
            "action_candidates": "V10.1_Stage_A_phase_pure_and_failure_safe_H30",
        },
        "training_target_coverage": {"failure_reason": [FAILURE_TARGET], "recovery_plan": [PLAN_TARGET]},
    }


def validate_source_identity(payload: dict[str, Any], training_data_hash: str) -> None:
    scope = payload.get("source_scope")
    if (
        not isinstance(scope, dict)
        or scope.get("schema_version") != "sponge_v10_1_source_scope_v1"
        or scope.get("selection") != "explicit_selected_dataset_and_hashed_V4_sources_no_recursive_archive_scan"
        or not scope.get("dataset_dir")
        or not scope.get("v4_index_file")
    ):
        raise ValueError("Sponge V10.2 requires its selected-data source_scope")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(training_data_hash))
        or payload.get("training_data_hash") != training_data_hash
    ):
        raise ValueError("Sponge V10.2 training_data_hash differs from its artifact identity")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("source_multitask_training_data_hash", ""))):
        raise ValueError("V10.2 requires source V10.1 multitask data identity")
    stage_a = payload.get("stage_a_initialization_identity", {})
    if (
        not isinstance(stage_a, dict)
        or stage_a.get("step") != 15000
        or stage_a.get("experiment_version") != "sponge_v10_1"
    ):
        raise ValueError("Sponge V10.2 requires V10.1 Stage A 15000 initialization identity")
    for key in ("action_training_data_hash", "config_sha256", "params_metadata_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(stage_a.get(key, ""))):
            raise ValueError(f"Sponge V10.2 Stage A initialization identity is missing {key}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("boundary_audit_sha256", ""))):
        raise ValueError("Sponge V10.2 identity is missing boundary_audit_sha256")


def append_runtime_memory(memory: list[dict], entry: dict) -> list[dict]:
    """Keep the real initial pair and latest three pairs across unlimited recoveries."""
    if len(memory) > 4:
        raise ValueError("Sponge V10.2 runtime memory exceeds four pairs")
    updated = [dict(pair) for pair in memory] + [dict(entry)]
    for index, pair in enumerate(updated):
        if pair.get("failure_reason") not in legal_failure_reasons(include_fz_bias=True):
            raise ValueError("Invalid Sponge V10.2 runtime failure memory")
        plan = pair.get("recovery_plan")
        if (index == 0 and plan not in {"initial plan", "recovery_plan=initial plan"}) or (
            index > 0 and plan not in legal_recovery_plans()
        ):
            raise ValueError("Sponge V10.2 memory must preserve the initial pair and executed plans")
    return updated if len(updated) <= 4 else [updated[0], *updated[-3:]]


def validate_training_config(config: dict[str, Any], norm_sha: str) -> None:
    version = deployment_version(config.get("data_profile"))
    expected = {
        "prompt_profile": PROMPT_PROFILE,
        "action_horizon": 30,
        "action_dim": 32,
        "max_token_len": 512,
        "reasoning_max_token_len": 320,
        "use_state_history": False,
        "state_history_len": 0,
        "history_hidden_dim": 0,
        "state_history_dim": 7,
        "state_history_fps": 30.0,
        "grammar_profile": "v3_full_v1",
        "phase_prefill_protocol": "need_failure_shared_kv_v1",
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
        raise ValueError(f"Sponge V10.2 deployment config mismatch: {mismatch}")
    if config.get("failure_grammar") != list(legal_failure_reasons(include_fz_bias=True)):
        raise ValueError("Sponge V10.2 requires the complete extended failure grammar")
    if config.get("recovery_grammar") != list(legal_recovery_plans()):
        raise ValueError("Sponge V10.2 requires the complete recovery grammar")
    if config.get("dataset_dir") != config.get("source_scope", {}).get("dataset_dir"):
        raise ValueError("Sponge V10.2 source_scope differs from the training dataset")
    validate_source_identity(config, identity[training_hash_key])
    if config["stage_a_initialization_identity"]["action_training_data_hash"] != identity.get("training_data_hash"):
        raise ValueError("Sponge V10.2 Stage A action data identity differs from initialization")


def resolve_thresholds(
    *,
    calibration: dict[str, Any] | None,
    step: int,
    full_params_sha: str,
    norm_sha: str,
    training_data_hash: str,
    need_override: float | None,
    adjustment_override: float | None,
    data_profile: str = DATA_PROFILE,
) -> tuple[dict[str, float], dict[str, bool]]:
    version = deployment_version(data_profile)
    if calibration is not None:
        expected = {
            "schema_version": f"{version}_val_thresholds_v1",
            "data_profile": data_profile,
            "prompt_profile": PROMPT_PROFILE,
            "checkpoint_step": step,
            "full_params_sha256": full_params_sha,
            "norm_stats_sha256": norm_sha,
            "training_data_hash": training_data_hash,
            "selection_split": "val",
            "thresholds_status": "calibrated_on_sponge_val",
        }
        mismatch = {
            key: (calibration.get(key), value) for key, value in expected.items() if calibration.get(key) != value
        }
        if mismatch:
            raise ValueError(f"Sponge V10.2 threshold identity mismatch: {mismatch}")
    thresholds, overrides = {}, {}
    for task, override in (("need_recovery", need_override), ("adjustment_end", adjustment_override)):
        value = (
            override
            if override is not None
            else (DEFAULT_THRESHOLD if calibration is None else calibration.get("thresholds", {}).get(task))
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
        "name": SERVER_NAME,
        "phase_prompt_profile": PROMPT_PROFILE,
        "action_prompt_profile": "phase_v2",
        "supports_streamed_phase_events": True,
        "supports_action_noise": True,
        "requires_action_noise": True,
        "supports_failure_generation": True,
        "supports_recovery_generation": True,
        "action_horizon": 30,
        "action_dim": 32,
        "output_action_dim": 7,
        "use_state_history": False,
        "state_history_len": 0,
        "state_history_fps": 30.0,
        "qpos_h100_sample_offsets": HISTORY_OFFSETS,
        "max_memory_pairs": 4,
        "memory_policy": MEMORY_POLICY,
        "max_supported_attempts": None,
        "classification_qpos_policy": "raw_no_gripper_remap",
        "episode_start_padding": "left_pad_episode_frame_0",
        "requires_captioner": False,
        "requires_tactile_topics": False,
    }
    expected["experiment_version"] = version
    expected.update(deployment_policy_fields(version))
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise ValueError(f"Sponge V10.2 client/server metadata mismatch: {mismatch}")
    validate_source_identity(metadata, metadata.get("training_data_hash", ""))
    for task in ("need_recovery", "adjustment_end"):
        value = float(metadata.get(f"{task}_threshold", -1))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid {task} threshold")
    if metadata.get("thresholds_status") not in {
        "default_0_5",
        "calibrated_on_sponge_val",
        "explicit_manual_override",
    }:
        raise ValueError("Sponge V10.2 requires default, calibrated or explicitly overridden thresholds")
