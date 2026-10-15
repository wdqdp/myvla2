"""V7.8 visual/state deployment identity; training data is not needed online."""

from __future__ import annotations

import math
from pathlib import Path
import re

from tactile_vla.vla.structured_text import legal_failure_reasons, legal_recovery_plans
from tactile_vla.vla.v7_7_phase_prompt import helper_identity, uniform_positions
from tactile_vla.vla.v7_8_phase_prompt import INPUT_POLICY, PROMPT_PROFILE

VERSION = "v7_8"
DATA_PROFILE = "rotation_phase_v7_8_five_task_h100_visual"
SERVER_NAME = "tactile_vla_v7_8"
INSTRUCTION = "Pick up and transfer the object stably."
ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
RUN_NAME = "pi05_rotation_v7_8_five_task_h100_no_history"
DEFAULT_RUN = ROOT / "outputs/multitask_v7_8" / RUN_NAME
DEFAULT_CHECKPOINT = DEFAULT_RUN / "2000"
DEFAULT_NORM_DIR = ROOT / "outputs/rotation_v4_1_new_env/norm_stats"
HISTORY_OFFSETS = uniform_positions(100).tolist()
HISTORY_POLICY = helper_identity() | {"prompt_profile": PROMPT_PROFILE, "idle_perturbation": "none_raw_contiguous"}
EXPORT_FIELDS = (
    "experiment_version",
    "data_profile",
    "prompt_profile",
    "input_policy",
    "captioner_identity_scope",
    "training_data_hash",
    "source_multitask_training_data_hash",
    "source_multitask_index_sha256",
    "history_policy",
    "adjustment_sampling_policy",
    "adjustment_label_policy",
    "need_successful_recovery_start",
    "stage_a_initialization_identity",
    "phase_prefill_protocol",
)


def require_sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"V7.8 requires a SHA256 for {name}")
    return value


def threshold(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return value


def validate_training_config(config, norm_sha):
    expected = {
        "experiment_version": VERSION,
        "run_name": RUN_NAME,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "input_policy": INPUT_POLICY,
        "captioner_identity_scope": "offline_label_provenance_only",
        "use_state_history": False,
        "state_history_len": 0,
        "history_hidden_dim": 0,
        "state_history_dim": 7,
        "state_history_fps": 30.0,
        "action_horizon": 30,
        "action_dim": 32,
        "max_token_len": 512,
        "reasoning_max_token_len": 320,
        "grammar_profile": "v3_full_v1",
        "no_norm": False,
        "history_policy": HISTORY_POLICY,
        "phase_prefill_protocol": "need_failure_shared_kv_v1",
        "adjustment_sampling_policy": "ratio_1_to_2",
        "adjustment_label_policy": "factual_true_inclusive_[S-10,S+10]",
        "need_successful_recovery_start": "arm_adjustment_stop_plus_1",
        "failure_grammar": list(legal_failure_reasons()),
        "recovery_grammar": list(legal_recovery_plans()),
    }
    mismatch = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if mismatch:
        raise ValueError(f"V7.8 training/deployment contract mismatch: {mismatch}")
    identity = config.get("artifact_identity", {})
    if (
        identity.get("data_profile") != "rotation_phase_v7_4_2_adjustment"
        or identity.get("prompt_profile") != "phase_v2"
        or identity.get("target_policy", {}).get("action_targets")
        != "raw_contiguous_no_idle_compression_no_terminal_hold"
        or identity.get("v4_norm_stats_sha256") != norm_sha
        or identity.get("v7_8_prompt_profile") != PROMPT_PROFILE
    ):
        raise ValueError("V7.8 requires its V7.4.2 raw action/norm artifact identity")
    require_sha(norm_sha, "norm_stats")
    training_hash = require_sha(config.get("training_data_hash"), "training_data_hash")
    if training_hash != identity.get("v7_8_training_data_hash"):
        raise ValueError("V7.8 training_data_hash differs from its artifact identity")
    for key in ("source_multitask_training_data_hash", "source_multitask_index_sha256"):
        require_sha(config.get(key), key)
    stage = config.get("stage_a_initialization_identity", {})
    if (
        type(stage.get("step")) is not int
        or stage["step"] <= 0
        or Path(str(stage.get("path", ""))).name != str(stage["step"])
        or config.get("stage_a_checkpoint") != stage.get("path")
        or config.get("stage_a_checkpoint_step") != stage.get("step")
        or stage.get("action_training_data_hash") != identity.get("training_data_hash")
    ):
        raise ValueError("V7.8 Stage A initialization identity mismatch")
    for key in ("action_training_data_hash", "config_sha256", "params_metadata_sha256"):
        require_sha(stage.get(key), f"stage_a.{key}")


def validate_server_metadata(metadata):
    expected = {
        "name": SERVER_NAME,
        "experiment_version": VERSION,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "phase_prompt_profile": PROMPT_PROFILE,
        "instruction": INSTRUCTION,
        "input_policy": INPUT_POLICY,
        "captioner_identity_scope": "offline_label_provenance_only",
        "requires_captioner": False,
        "requires_tactile_topics": False,
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
        "action_prompt_profile": "phase_v2",
        "history_policy": HISTORY_POLICY,
        "qpos_h100_sample_offsets": HISTORY_OFFSETS,
        "history_frame_filter": "none",
        "max_memory_pairs": 5,
        "memory_policy": "latest_five_real_failure_plan_pairs",
        "phase_prefill_protocol": "need_failure_shared_kv_v1",
    }
    mismatch = {k: (metadata.get(k), v) for k, v in expected.items() if metadata.get(k) != v}
    if mismatch:
        raise ValueError(f"V7.8 client/server metadata mismatch: {mismatch}")
    if type(metadata.get("checkpoint_step")) is not int or metadata["checkpoint_step"] <= 0:
        raise ValueError("V7.8 server is missing its numeric checkpoint step")
    for key in ("norm_stats_sha256", "full_params_sha256", "training_data_hash"):
        require_sha(metadata.get(key), key)
    for name in ("need_recovery_threshold", "adjustment_end_threshold"):
        threshold(metadata.get(name, -1), name)
