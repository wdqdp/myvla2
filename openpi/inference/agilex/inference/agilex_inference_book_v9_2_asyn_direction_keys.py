#!/usr/bin/env python3
"""Book V9.2 async action execution with a/s/d/f recovery selection."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import agilex_inference_forced_phase_anlation_v7_5_asyn_direction_keys as controls

implementation = controls.implementation
from tactile_vla.vla.book_stage_a_data import DATA_PROFILE as ACTION_DATA_PROFILE
from tactile_vla.vla.book_stage_a_data import EXPERIMENT_KIND as ACTION_EXPERIMENT_KIND
from tactile_vla.vla.book_v9_2_adjustment_end_data import DATA_PROFILE, HISTORY_POLICY
from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN, PHASE_CHANGE_PROMPT_PROFILE
from tactile_vla.vla.v7_5_runtime_history import RUNTIME_SAMPLE_OFFSETS

BOOK_INSTRUCTION = "Pick up the book and place it horizontally on the bookshelf."
BOOK_NORM_STATS = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book/outputs/rotation_v4/norm_stats/norm_stats.json")
DEFAULT_LOG_ROOT = implementation.PROJECT_ROOT / "outputs/runtime/book_v9_2_async_direction_keys"


def validate_server_metadata(args, metadata: dict[str, Any]) -> None:
    expected = {
        "supports_action_noise": True,
        "requires_action_noise": True,
        "supports_adjustment_end": True,
        "prompt_profile": "phase_v2",
        "data_profile": ACTION_DATA_PROFILE,
        "experiment_kind": ACTION_EXPERIMENT_KIND,
        "stage_a_protocol": "book_stage_a_v1_no_state_history",
        "phase_change_prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
        "qpos_history_frames": 100,
        "qpos_history_includes_current": True,
        "qpos_sampled_frames": 11,
        "qpos_h100_sample_offsets": list(RUNTIME_SAMPLE_OFFSETS),
        "runtime_history_policy": "uniform_h100_no_ga_idle_compression",
        "captioner_window_size": 30,
        "action_horizon": 30,
        "action_dim": 32,
        "output_action_dim": 7,
        "state_history_len": 0,
        "state_history_dim": 7,
        "use_state_history": False,
        "adjustment_end_data_profile": DATA_PROFILE,
        "adjustment_end_checkpoint_format": "book_v9_2_adjustment_end_paligemma_lora_h100_no_history_v1",
        "adjustment_end_history_policy": HISTORY_POLICY,
    }
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise ValueError(f"Book V9.2 client/server metadata mismatch: {mismatch}")
    if not 0.0 <= float(metadata.get("adjustment_end_threshold", -1.0)) <= 1.0:
        raise ValueError("Book V9.2 server adjustment_end_threshold is invalid")
    if metadata.get("adjustment_end_experimental_override", False) and not getattr(
        args, "allow_experimental_adjustment_end", False
    ):
        raise ValueError("Pass --allow-experimental-adjustment-end to acknowledge a threshold override")
    if float(metadata.get("phase_change_timeout_seconds", -1.0)) != args.phase_change_timeout_seconds:
        raise ValueError("Book V9.2 client/server phase-change timeout mismatch")
    if metadata.get("captioner_checkpoint_sha256") != implementation.v53.sha256_file(args.captioner_checkpoint):
        raise ValueError("Client captioner SHA differs from the server deployment SHA")

    if metadata.get("norm_stats_sha256") != implementation.v53.sha256_file(args.norm_stats_file):
        raise ValueError("Client and server book norm stats differ")


def validate_raw_classification_qpos(args, parser) -> None:
    if args.classification_gripper_open_threshold is not None or args.classification_gripper_open_value is not None:
        parser.error("Book V9.2 requires raw qpos; classification gripper remapping is disabled")


def _validate_keyboard_args(original_validate, args, parser) -> None:
    controls._keyboard_selection_validate_args(original_validate, args, parser)
    if args.expected_data_profile != ACTION_DATA_PROFILE:
        parser.error("Book V9.2 requires --expected-data-profile book_stage_a_v1")
    if args.instruction != BOOK_INSTRUCTION:
        parser.error(f"Book V9.2 requires --instruction={BOOK_INSTRUCTION!r}")
    if args.quit_key != "q":
        parser.error("Book V9.2 uses q to stop; --quit-key must be q")
    args.deployment_label = "Book V9.2"


def main() -> None:
    # These process-local hooks reuse the V7.6 state machine and H100 construction.
    # Direction keys interrupt execution; adjustment_end alone returns to execution.
    original_validate = implementation.base.validate_args
    implementation.ROTATION_PHASE_V7_4_ADJUSTMENT = ACTION_DATA_PROFILE
    implementation.V7_4_EXPERIMENT_KIND = ACTION_EXPERIMENT_KIND
    implementation.DATA_PROFILE = DATA_PROFILE
    implementation.DEFAULT_NORM_STATS = BOOK_NORM_STATS
    implementation.DEFAULT_LOG_ROOT = DEFAULT_LOG_ROOT
    implementation.v52.DEFAULT_INSTRUCTION = BOOK_INSTRUCTION
    implementation.validate_server_metadata = validate_server_metadata
    implementation.validate_classification_gripper_probe_arguments = validate_raw_classification_qpos
    implementation.base._poll_key = controls._poll_key
    implementation.v52._poll_control_key = controls._poll_control_key
    implementation.base.validate_args = lambda args, parser: _validate_keyboard_args(original_validate, args, parser)
    implementation.base.run_v5_3_async = controls.run_v7_5_async_direction_keys
    implementation.__doc__ = __doc__
    print(
        "Book V9.2: a=left moderately, s=left slightly, d=right slightly, f=right moderately; q=stop; SPACE disabled."
    )
    implementation.main()


if __name__ == "__main__":
    main()
