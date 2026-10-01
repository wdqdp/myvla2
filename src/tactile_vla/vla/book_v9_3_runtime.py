"""Book V9.3 deployment identities, single-pair memory and threshold contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tactile_vla.vla.book_v9_3_multitask_data import DATA_PROFILE
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE, uniform_positions

BOOK_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book")
BOOK_INSTRUCTION = "Pick up the book and place it horizontally on the bookshelf."
DEFAULT_RUN = BOOK_ROOT / "outputs/multitask_v9_3/pi05_book_v9_3_five_task_h100_no_history"
DEFAULT_NORM_DIR = BOOK_ROOT / "outputs/rotation_v4/norm_stats"
DEFAULT_INDEX = BOOK_ROOT / "outputs/book_v9_3_multitask/book_v9_3_multitask_training_index.json"
SERVER_NAME = "tactile_vla_book_v9_3"
THRESHOLD_SCHEMA = "book_v9_3_val_thresholds_v1"
HISTORY_OFFSETS = uniform_positions(100).tolist()
MEMORY_POLICY = "latest_failed_plan_reason_pair_only"


def latest_failure_memory(*, executed_recovery_plan: str, failure_reason: str) -> list[dict[str, str]]:
    reason = str(failure_reason).strip()
    if not reason:
        raise ValueError("failure_reason must not be empty")
    return [{
        "recovery_plan": str(executed_recovery_plan).strip() or "initial plan",
        "failure_reason": reason,
    }]


def validate_training_config(config: dict[str, Any], norm_sha: str) -> None:
    expected = {
        "data_profile": DATA_PROFILE, "prompt_profile": PROMPT_PROFILE,
        "action_horizon": 30, "action_dim": 32, "max_token_len": 512,
        "reasoning_max_token_len": 320, "use_state_history": False, "state_history_len": 0,
        "grammar_profile": "v3_full_v1", "phase_prefill_protocol": "need_failure_shared_kv_v1",
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    identity = config.get("artifact_identity", {})
    if identity.get("data_profile") != "book_stage_a_v1":
        mismatch["artifact_identity.data_profile"] = (identity.get("data_profile"), "book_stage_a_v1")
    if identity.get("v4_norm_stats_sha256") != norm_sha:
        mismatch["norm_stats_sha256"] = (norm_sha, identity.get("v4_norm_stats_sha256"))
    if not identity.get("book_v9_3_training_data_hash"):
        mismatch["book_v9_3_training_data_hash"] = (None, "nonempty")
    if mismatch:
        raise ValueError(f"Book V9.3 deployment config mismatch: {mismatch}")


def resolve_thresholds(
    *, calibration: dict[str, Any] | None, step: int, full_params_sha: str,
    norm_sha: str, training_data_hash: str, need_override: float | None,
    adjustment_override: float | None,
) -> tuple[dict[str, float], dict[str, bool]]:
    if calibration is not None:
        expected = {
            "schema_version": THRESHOLD_SCHEMA, "data_profile": DATA_PROFILE,
            "prompt_profile": PROMPT_PROFILE, "checkpoint_step": step,
            "full_params_sha256": full_params_sha, "norm_stats_sha256": norm_sha,
            "training_data_hash": training_data_hash, "selection_split": "val",
            "thresholds_status": "calibrated_on_book_val",
        }
        mismatch = {key: (calibration.get(key), value) for key, value in expected.items()
                    if calibration.get(key) != value}
        if mismatch:
            raise ValueError(f"Book V9.3 threshold identity mismatch: {mismatch}")
    thresholds, overrides = {}, {}
    for task, override in (("need_recovery", need_override), ("adjustment_end", adjustment_override)):
        value = override if override is not None else (calibration or {}).get("thresholds", {}).get(task)
        if value is None:
            raise ValueError(f"{task}: supply --thresholds-file or an explicit threshold override; "
                             "training 0.5 placeholders are not deployment thresholds")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{task} threshold must be finite and in [0,1]")
        thresholds[task] = value
        overrides[task] = override is not None
    return thresholds, overrides


def validate_server_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "name": SERVER_NAME, "data_profile": DATA_PROFILE, "phase_prompt_profile": PROMPT_PROFILE,
        "action_prompt_profile": "phase_v2", "supports_streamed_phase_events": True,
        "supports_action_noise": True, "requires_action_noise": True,
        "supports_failure_generation": True, "supports_recovery_generation": True,
        "action_horizon": 30, "action_dim": 32, "output_action_dim": 7,
        "use_state_history": False, "state_history_len": 0, "state_history_fps": 30.0,
        "qpos_h100_sample_offsets": HISTORY_OFFSETS, "max_memory_pairs": 1,
        "memory_policy": MEMORY_POLICY, "max_supported_attempts": None,
        "classification_qpos_policy": "raw_no_gripper_remap",
        "episode_start_padding": "left_pad_episode_frame_0", "captioner_window_size": 30,
    }
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items()
                if metadata.get(key) != value}
    if mismatch:
        raise ValueError(f"Book V9.3 client/server metadata mismatch: {mismatch}")
    for task in ("need_recovery", "adjustment_end"):
        value = float(metadata.get(f"{task}_threshold", -1))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid {task} threshold")
    if metadata.get("thresholds_status") not in {"calibrated_on_book_val", "explicit_manual_override"}:
        raise ValueError("Book V9.3 requires calibrated or explicitly overridden thresholds")

