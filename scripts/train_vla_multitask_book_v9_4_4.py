#!/usr/bin/env python3
"""Train Book V9.4.4 five tasks with failure/plan [C,C+14] supervision."""

# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_multitask_book_v9_4 as previous
from tactile_vla.vla.book_v9_4_4_multitask_data import (
    DATA_PROFILE,
    DEFAULT_INDEX,
    DEFAULT_OUTPUT,
    DEFAULT_STAGE_A,
    LABEL_POLICY,
    REASONING_WINDOW_POLICY,
    RUN_NAME,
    SAMPLING_POLICY,
    VERSION_TAG,
    validate_index,
)


def ensure_index(args):
    index, frames = previous.ensure_index(args, index_validator=validate_index)
    previous.training_base.EXTRA_CONFIG.update(
        {
            "experiment_version": VERSION_TAG,
            "need_label_policy": LABEL_POLICY,
            "need_sampling_policy": SAMPLING_POLICY,
            "reasoning_window_policy": REASONING_WINDOW_POLICY,
            "need_boundary_audit_sha256": index["need_boundary_audit_sha256"],
            "reasoning_boundary_audit_sha256": index["reasoning_boundary_audit_sha256"],
        }
    )
    return index, frames


def export_checkpoint(run_dir, state, step, filter_):
    previous.trainer.export_checkpoint(run_dir, state, step, filter_)
    path = run_dir / str(step) / f"{VERSION_TAG}_export.json"
    metadata = json.loads(path.read_text())
    metadata.update(
        {
            "experiment_version": VERSION_TAG,
            "data_profile": DATA_PROFILE,
            "need_label_policy": LABEL_POLICY,
            "need_sampling_policy": SAMPLING_POLICY,
            "reasoning_window_policy": REASONING_WINDOW_POLICY,
        }
    )
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def configure_training():
    previous.configure_training()
    previous.training_base.CHECKPOINT_EXPORT_HOOK = export_checkpoint
    previous.training_base.EXTRA_CONFIG.update(
        {
            "experiment_version": VERSION_TAG,
            "need_label_policy": LABEL_POLICY,
            "need_sampling_policy": SAMPLING_POLICY,
            "reasoning_window_policy": REASONING_WINDOW_POLICY,
            "plan_eval_policy": "all_C_plus_14_variants_grouped_by_memory_length_and_left_right",
            "target_support_note": "Book V9.4.4 real C-window observations; memory_length=1,2,3,4; synthetic text history only; no action/class counterfactuals",
        }
    )


def configure_version():
    previous.configure_version()
    trainer = previous.trainer
    trainer.DATA_PROFILE = DATA_PROFILE
    trainer.DEFAULT_INDEX = DEFAULT_INDEX
    trainer.DEFAULT_STAGE_A = DEFAULT_STAGE_A
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.RUN_NAME = RUN_NAME
    trainer.VERSION_TAG = VERSION_TAG
    trainer.validate_index = validate_index
    trainer.ensure_index = ensure_index
    trainer.configure = configure_training


def main():
    configure_version()
    previous.trainer.main()


if __name__ == "__main__":
    main()
