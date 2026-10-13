#!/usr/bin/env python3
"""Bottle V11.3 five-task visual/state ablation, initialized from V11.2 Stage A."""

# ruff: noqa: E402
from pathlib import Path
import argparse
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src"), str(PROJECT / "openpi/src")]

from scripts import train_vla_multitask_bottle_v11_1 as engine
from tactile_vla.vla import bottle_v11_3_multitask_data as data
from tactile_vla.vla.bottle_v11_2_stage_a_data import validate_training_index
from tactile_vla.vla.bottle_v11_3_prompts import INPUT_POLICY, PROMPT_PROFILE


def parse_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--bottle-root", type=Path, default=data.ROOT)
    extra, _ = p.parse_known_args(arguments)
    root = extra.bottle_root
    for flag, value in {
        "--index-file": root / "outputs/bottle_v11_3_multitask/bottle_v11_3_multitask_training_index.json",
        "--stage-a-checkpoint": root / "outputs/stage_a_action/pi05_delta_tac_bottle_stage_a_v11_2_no_history/7500",
        "--output-dir": root / "outputs/multitask_v11_3",
        "--run-name": data.RUN_NAME,
        "--prompt-profile": PROMPT_PROFILE,
    }.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in arguments):
            arguments.extend([flag, str(value)])
    return engine.parse_args(arguments)


def ensure_index(args):
    index, frames = engine.ensure_index(args)
    engine.base.EXTRA_CONFIG.update(
        {
            "input_policy": INPUT_POLICY,
            "captioner_identity_scope": "offline_label_provenance_only",
            "source_multitask_training_data_hash": index["source_multitask_training_data_hash"],
            "source_multitask_index_sha256": index["source_multitask_index_sha256"],
            "action_target_policy": index["action_target_policy"],
            "idle_gap_audit_sha256": engine.sha256_json(index["idle_gap_audit"]),
        }
    )
    return index, frames


def identity(index, **kwargs):
    result = engine.identity(index, **kwargs)
    for key in ("bottle_v11_1_training_data_hash", "bottle_v11_1_multitask_index_sha256"):
        result[key.replace("v11_1", "v11_3")] = result.pop(key)
    return result


def build_loaders(args, model_config, index, records, tokenizer, failure_codec, plan_codec):
    data.validate_prompts(index)
    return engine.build_loaders(args, model_config, index, records, tokenizer, failure_codec, plan_codec)


def validate_resume(saved, args, *, actual_precision):
    for key in (
        "input_policy",
        "captioner_identity_scope",
        "source_multitask_training_data_hash",
        "source_multitask_index_sha256",
        "action_target_policy",
        "idle_gap_audit_sha256",
    ):
        if saved.get(key) != engine.base.EXTRA_CONFIG.get(key):
            raise ValueError(f"V11.3 resume identity/policy mismatch: {key}")
    engine.validate_resume(saved, args, actual_precision=actual_precision)


def configure_training():
    engine.configure_training()
    engine.base.TEXT_GRAMMAR_FACTORY = data.build_training_grammars
    engine.base.EXTRA_CONFIG.update(input_policy=INPUT_POLICY, captioner_identity_scope="offline_label_provenance_only")
    engine.base.validate_v4_resume_config = validate_resume


def configure_version():
    # Only alter the in-process engine; original V11.2 files/running jobs are untouched.
    engine.DATA_PROFILE, engine.VERSION_TAG = data.DATA_PROFILE, data.VERSION_TAG
    engine.DEFAULT_INDEX, engine.DEFAULT_STAGE_A = data.DEFAULT_INDEX, data.DEFAULT_STAGE_A
    engine.RUN_NAME, engine.HISTORY_POLICY = data.RUN_NAME, data.HISTORY_POLICY
    engine.NEED_POLICY = data.NEED_POLICY
    engine.validate_index = data.validate_index
    engine.validate_stage_a_for_training = data.validate_stage_a_for_training
    engine.validate_action_index = validate_training_index
    engine.ACTION_POLICY = engine.ACTION_POLICY | {"action_candidates": "V11.2_compressed_phase_pure_failure_safe_H30"}
    engine.trainer.PROMPT_PROFILE = PROMPT_PROFILE
    engine.configure_version()
    engine.trainer.parse_args, engine.trainer.configure = parse_args, configure_training
    engine.trainer.ensure_index, engine.trainer.identity = ensure_index, identity
    engine.trainer.build_loaders = build_loaders
    engine.trainer.DEFAULT_OUTPUT = data.ROOT / "outputs/multitask_v11_3"


def main():
    configure_version()
    engine.trainer.main()


if __name__ == "__main__":
    main()
