#!/usr/bin/env python3
"""Sponge V10.1: five tasks, fz_bias grammar, 4+4 action replay and H100->11."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from torch.utils.data import DataLoader, Subset
from scripts import train_vla_multitask_v7_7 as trainer
from scripts import train_vla_stage_b_v3 as base
from tactile_vla.vla.artifacts import artifact_identity, sha256_file, sha256_json
from tactile_vla.vla.book_v9_4_5_action_replay import (
    ACTION_EVAL_POLICY,
    ACTION_SAMPLING_POLICY,
    PHASES,
    PhaseBalancedActionBatchSampler,
    phase_eval_positions,
    phase_positions,
)
from tactile_vla.vla.book_v9_4_multitask_data import validate_schedule
from tactile_vla.vla.sponge_v10_1_multitask_data import (
    ADJUSTMENT_POLICY,
    DATA_PROFILE,
    DEFAULT_INDEX,
    DEFAULT_STAGE_A,
    HISTORY_POLICY,
    MEMORY_POLICY,
    NEED_POLICY,
    REASONING_POLICY,
    ROOT,
    RUN_NAME,
    VERSION_TAG,
    validate_index,
    validate_stage_a_for_training,
)
from tactile_vla.vla.sponge_v10_1_stage_a_data import (
    FAILURE_ACTION_POLICY,
    validate_training_index as validate_action_index,
)
from tactile_vla.vla.v4_data import scan_v4_lerobot_frames

_BASE_PARSE = trainer.parse_args
_BASE_CONFIGURE = trainer.configure
_BASE_LOADERS = trainer.build_loaders
_BASE_EXPORT = trainer.export_checkpoint
_BASE_TEXT = base.evaluate_text
_BASE_NEED = base.evaluate_need
_BASE_RESUME = base.validate_v4_resume_config
ACTION_POLICY = ACTION_SAMPLING_POLICY | {"action_candidates": "V10.1_Stage_A_phase_pure_and_failure_safe_H30"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sponge-root", type=Path, default=ROOT)
    p.add_argument("--action-eval-samples-per-phase", type=int, default=256)
    p.add_argument("--data-only-dry-run", action="store_true")
    extra, rest = p.parse_known_args(sys.argv[1:] if argv is None else argv)
    root = extra.sponge_root.resolve()
    normalized = []
    for argument in rest:
        normalized.extend(argument.split("=", 1) if argument.startswith("--") and "=" in argument else [argument])
    defaults = {
        "--dataset-dir": root / "lerobot_data/tactile_vla_rotation_v4",
        "--index-file": root / "outputs/sponge_v10_1_multitask/sponge_v10_1_multitask_training_index.json",
        "--norm-stats-dir": root / "outputs/rotation_v4/norm_stats",
        "--stage-a-checkpoint": root / "outputs/stage_a_action/pi05_delta_tac_sponge_stage_a_v10_1_no_history/15000",
        "--output-dir": root / "outputs/multitask_v10_1",
        "--run-name": RUN_NAME,
        "--data-profile": DATA_PROFILE,
        "--prompt-profile": trainer.PROMPT_PROFILE,
        "--num-steps": 2000,
        "--eval-interval": 1000,
        "--save-interval": 1000,
        "--keep-period": 1000,
        "--eval-max-need-samples": 2147483647,
        "--eval-max-text-samples": 2147483647,
        "--fsdp-devices": 2,
    }
    for flag, value in defaults.items():
        if flag not in normalized:
            normalized.extend([flag, str(value)])
    if "--help" in normalized or "-h" in normalized:
        print(
            "V10.1 options: --sponge-root PATH; --action-eval-samples-per-phase N (256); --data-only-dry-run (no checkpoint access)"
        )
    if extra.data_only_dry_run and "--dry-run" not in normalized:
        normalized.append("--dry-run")
    saved = sys.argv
    try:
        sys.argv = [saved[0], *normalized]
        args = _BASE_PARSE()
    finally:
        sys.argv = saved
    validate_schedule(
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
    )
    required = {
        "data_profile": DATA_PROFILE,
        "prompt_profile": trainer.PROMPT_PROFILE,
        "batch_size": 8,
        "seed": 42,
        "action_horizon": 30,
        "action_dim": 32,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "max_token_len": 512,
        "reasoning_max_token_len": 320,
        "no_norm": False,
        "grammar_profile": "v3_full_v1",
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
        "action_loss_weight": 1.0,
        "need_loss_weight": 1.0,
        "failure_loss_weight": 1.0,
        "plan_loss_weight": 1.0,
        "use_state_history": False,
        "state_history_len": 0,
        "history_hidden_dim": 0,
    }
    mismatches = {k: (getattr(args, k), v) for k, v in required.items() if getattr(args, k) != v}
    if mismatches or extra.action_eval_samples_per_phase <= 0:
        raise ValueError(f"V10.1 training protocol mismatch: {mismatches}")
    args.data_only_dry_run = extra.data_only_dry_run
    args.action_eval_samples_per_phase = extra.action_eval_samples_per_phase
    return args


def ensure_index(args):
    index = json.loads(args.index_file.read_text())
    validate_index(index)
    if Path(index["dataset_dir"]).resolve() != args.dataset_dir.resolve() or index["source_hashes"].get(
        str((args.norm_stats_dir / "norm_stats.json").resolve())
    ) != sha256_file(args.norm_stats_dir / "norm_stats.json"):
        raise ValueError("V10.1 training dataset/norm does not match the index")
    stage_a = (
        {"status": "unverified_data_only_dry_run"}
        if args.data_only_dry_run
        else validate_stage_a_for_training(args.stage_a_checkpoint, index)
    )
    base.EXTRA_CONFIG.update(
        {
            "training_data_hash": index["training_data_hash"],
            "training_target_coverage": index["training_target_coverage"],
            "captioner_identity": index["captioner_identity"],
            "token_validation": index["token_validation"],
            "source_scope": index["source_scope"],
            "stage_a_initialization_identity": stage_a,
            "boundary_audit_sha256": sha256_json(index["boundary_audit"]),
        }
    )
    return index, scan_v4_lerobot_frames(args.dataset_dir)


def identity(index, **_):
    action_file = Path(index["action_index_file"])
    action = json.loads(action_file.read_text())
    result = artifact_identity(
        action, index_path=action_file, prompt_profile="phase_v2", requested_data_profile="book_stage_a_v1"
    )
    return result | {
        "sponge_v10_1_training_data_hash": index["training_data_hash"],
        "sponge_v10_1_multitask_index_sha256": sha256_file(trainer._ARGS.index_file),
    }


def build_loaders(args, model_config, index, records, tokenizer, failure_codec, plan_codec):
    from tactile_vla.vla.structured_text import legal_failure_reasons

    if failure_codec.texts != legal_failure_reasons(include_fz_bias=True):
        raise ValueError("V10.1 failure grammar did not enable fz_bias outputs")
    loaders = _BASE_LOADERS(args, model_config, index, records, tokenizer, failure_codec, plan_codec)
    action_file = Path(index["action_index_file"])
    _, lookup = validate_action_index(
        json.loads(action_file.read_text()), index_path=action_file, dataset_dir=args.dataset_dir
    )
    pools = phase_positions(index["splits"]["train"]["action"]["indices"], lookup)
    options = {"multiprocessing_context": "spawn", "persistent_workers": True} if args.num_workers else {}
    loaders["train"]["action"] = DataLoader(
        loaders["train"]["action"].dataset,
        batch_sampler=PhaseBalancedActionBatchSampler(pools, batch_size=args.batch_size, seed=args.seed),
        num_workers=args.num_workers,
        collate_fn=base.collate_numpy,
        **options,
    )
    indices = index["splits"]["val"]["action"]["indices"]
    selected = phase_eval_positions(
        indices, lookup, samples_per_phase=args.action_eval_samples_per_phase, seed=args.seed
    )
    audit = {}
    for phase in PHASES:
        loaders["val"]["action_" + phase] = base._loader(
            Subset(loaders["val"]["action"].dataset, selected[phase]),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        globals_ = [indices[p] for p in selected[phase]]
        audit[phase] = {
            "global_indices": globals_,
            "global_indices_sha256": sha256_json(globals_),
            "selected_count": len(globals_),
            "episode_attempt_groups": [
                list(v) for v in sorted({(lookup[g]["episode_id"], lookup[g]["attempt_id"]) for g in globals_})
            ],
        }
    need_loader = loaders["val"]["need"]
    raw = need_loader.dataset.dataset
    groups = {}
    for position, row_index in enumerate(raw.row_indices):
        groups.setdefault(raw.rows[row_index]["source"], []).append(position)
    need_loader.v10_1_groups = {
        name: base._loader(
            Subset(need_loader.dataset, positions), batch_size=args.batch_size, num_workers=0, shuffle=False
        )
        for name, positions in groups.items()
    }
    base.EXTRA_CONFIG.update(
        {
            "action_train_phase_candidates": {s: len(p) for s, p in pools.items()},
            "action_eval_selection": audit,
            "action_eval_policy": ACTION_EVAL_POLICY | {"samples_per_phase": args.action_eval_samples_per_phase},
        }
    )
    if args.dry_run:
        batch = next(iter(loaders["train"]["action"]))
        counts = Counter(lookup[int(g)]["phase"] for g in batch["global_index"])
        if counts != {"adjustment": 4, "execution": 4}:
            raise ValueError("V10.1 action batch is not 4 adjustment + 4 execution")
        for phase in PHASES:
            next(iter(loaders["val"]["action_" + phase]))
        logging.info(
            "V10.1 action replay dry-run: phase_counts=%s, val_selection=%s",
            dict(counts),
            {p: audit[p]["selected_count"] for p in PHASES},
        )
    return loaders


def evaluate_action_by_phase(state, loaders, data_sharding, *, seed):
    losses = {
        p: base.evaluate_action_loss(
            state, loaders["action_" + p], data_sharding, seed=seed, max_batches=None, sample_weighted=True
        )
        for p in PHASES
    }
    return {
        "loss": sum(losses.values()) / len(PHASES),
        "by_phase": losses,
        "support_by_phase": {p: len(loaders["action_" + p].dataset) for p in PHASES},
    }


def evaluate_text(state, loader, task, grammar, data_sharding, *, max_samples):
    if task != "plan":
        return _BASE_TEXT(state, loader, task, grammar, data_sharding, max_samples=max_samples)
    raw = loader.dataset.dataset
    grouped = {}
    for pos, row_index in enumerate(raw.row_indices):
        grouped.setdefault(raw.rows[row_index]["memory_length"], []).append(pos)
    metrics = {}
    for length, positions in sorted(grouped.items()):
        group = base._loader(
            Subset(loader.dataset, positions), batch_size=loader.batch_size, num_workers=0, shuffle=False
        )
        metrics[str(length)] = _BASE_TEXT(state, group, task, grammar, data_sharding, max_samples=None)
    support = sum(m["num_samples"] for m in metrics.values())
    return {
        "exact_match": sum(m["exact_match"] * m["num_samples"] for m in metrics.values()) / support,
        "num_samples": support,
        "by_memory_length": metrics,
        "target_support_note": "single real fz_bias-left/up class; exact match is not multiclass generalization",
    }


def evaluate_need(state, loader, data_sharding, *, max_samples):
    result = _BASE_NEED(state, loader, data_sharding, max_samples=max_samples)
    groups = {}
    for name, group in getattr(loader, "v10_1_groups", {}).items():
        metric = _BASE_NEED(state, group, data_sharding, max_samples=len(group.dataset))
        matrix = metric["confusion_matrix"]
        negatives = sum(matrix[0])
        metric["false_positive_rate"] = matrix[0][1] / negatives if negatives else None
        groups[name] = metric
    result["by_source"] = groups
    return result


def export_checkpoint(run_dir, state, step, filter_):
    from tactile_vla.vla.structured_text import legal_failure_reasons, legal_recovery_plans

    _BASE_EXPORT(run_dir, state, step, filter_)
    path = run_dir / str(step) / f"{VERSION_TAG}_export.json"
    metadata = json.loads(path.read_text())
    metadata.update(
        base.EXTRA_CONFIG
        | {
            "data_profile": DATA_PROFILE,
            "prompt_profile": trainer.PROMPT_PROFILE,
            "failure_grammar": list(legal_failure_reasons(include_fz_bias=True)),
            "recovery_grammar": list(legal_recovery_plans()),
        }
    )
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def validate_resume(saved, args, *, actual_precision):
    for key in (
        "experiment_version",
        "training_data_hash",
        "include_fz_bias_failure_grammar",
        "failure_action_policy",
        "need_policy",
        "adjustment_policy",
        "reasoning_policy",
        "memory_policy",
        "stage_a_initialization_identity",
    ):
        if saved.get(key) != base.EXTRA_CONFIG.get(key):
            raise ValueError(f"V10.1 resume identity/policy mismatch: {key}")
    _BASE_RESUME(saved, args, actual_precision=actual_precision)


def configure_training():
    _BASE_CONFIGURE()
    base.CHECKPOINT_EXPORT_HOOK = export_checkpoint
    base.ACTION_EVALUATION_HOOK = evaluate_action_by_phase
    base.EXTRA_CONFIG.update(
        {
            "experiment_version": VERSION_TAG,
            "include_fz_bias_failure_grammar": True,
            "failure_action_policy": FAILURE_ACTION_POLICY,
            "need_policy": NEED_POLICY,
            "adjustment_policy": ADJUSTMENT_POLICY,
            "reasoning_policy": REASONING_POLICY,
            "history_policy": HISTORY_POLICY,
            "memory_policy": MEMORY_POLICY,
            "action_sampling_policy": ACTION_POLICY,
            "thresholds_status": "uncalibrated_0.5_placeholders_validate_before_robot",
            "target_support_note": "only fz_bias-left -> up moderately; memory1..4; no counterfactuals",
        }
    )
    base.evaluate_text = evaluate_text
    base.evaluate_need = evaluate_need
    base.validate_v4_resume_config = validate_resume


def configure_version():
    trainer.DATA_PROFILE, trainer.VERSION_TAG = DATA_PROFILE, VERSION_TAG
    trainer.DEFAULT_INDEX, trainer.DEFAULT_STAGE_A = DEFAULT_INDEX, DEFAULT_STAGE_A
    trainer.DEFAULT_OUTPUT, trainer.RUN_NAME = ROOT / "outputs/multitask_v10_1", RUN_NAME
    trainer.DEFAULT_DATASET = ROOT / "lerobot_data/tactile_vla_rotation_v4"
    trainer.DEFAULT_NORM_STATS = ROOT / "outputs/rotation_v4/norm_stats"
    trainer.validate_v7_4_adjustment_training_index = validate_action_index
    trainer.parse_args, trainer.configure, trainer.ensure_index = parse_args, configure_training, ensure_index
    trainer.identity, trainer.build_loaders = identity, build_loaders


def main():
    configure_version()
    trainer.main()


if __name__ == "__main__":
    main()
