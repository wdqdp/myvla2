#!/usr/bin/env python3
"""Book V9.6: real-only visual/state five-task ablation of V9.5."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from torch.utils.data import DataLoader, Subset
from scripts import train_vla_multitask_book_v9_4_4 as previous
from tactile_vla.vla.artifacts import sha256_json, sha256_file
from tactile_vla.vla.book_v9_6_multitask_data import (
    DATA_PROFILE,
    DEFAULT_INDEX,
    DEFAULT_STAGE_A,
    LABEL_POLICY,
    REASONING_WINDOW_POLICY,
    ROOT,
    SAMPLING_POLICY,
    VERSION_TAG as DATA_VERSION,
    validate_index,
    validate_stage_a_for_training,
)
from tactile_vla.vla.book_v9_5_stage_a_data import reject_archive
from tactile_vla.vla.book_v9_6_prompts import INPUT_POLICY, PROMPT_PROFILE
from tactile_vla.vla.book_v9_4_5_action_replay import (
    ACTION_EVAL_POLICY,
    ACTION_SAMPLING_POLICY,
    PHASES,
    PhaseBalancedActionBatchSampler,
    phase_eval_positions,
    phase_positions,
)

VERSION_TAG = "book_v9_6"
TRAINING_PROFILE = "book_v9_6_phase_balanced_action_replay"
DEFAULT_OUTPUT = ROOT / "outputs/multitask_v9_6"
RUN_NAME = "pi05_book_v9_6_five_task_h100_no_history"
trainer = previous.previous.trainer
training_base = previous.previous.training_base
_BASE_BUILD_LOADERS = trainer.build_loaders
_BASE_EVALUATE_NEED = training_base.evaluate_need
_BASE_IDENTITY = trainer.identity


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action-eval-samples-per-phase", type=int, default=256)
    parser.add_argument("--data-only-dry-run", action="store_true")
    extra, rest = parser.parse_known_args(sys.argv[1:])
    sys.argv[:] = [sys.argv[0], *rest]
    if any(flag in rest for flag in ("-h", "--help")):
        print("V9.6 option: --action-eval-samples-per-phase N (default 256; trajectory-stratified per phase)")
        print("V9.6 option: --data-only-dry-run (forces dry-run; validates data without Stage A weights)")
    if extra.action_eval_samples_per_phase <= 0:
        raise ValueError("--action-eval-samples-per-phase must be positive")
    for flag, value in {
        "--num-steps": "2000",
        "--eval-interval": "1000",
        "--save-interval": "1000",
        "--keep-period": "1000",
    }.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in rest):
            sys.argv.extend([flag, value])
    args = previous.previous.parse_args()
    args.action_eval_samples_per_phase = extra.action_eval_samples_per_phase
    args.data_only_dry_run = extra.data_only_dry_run
    if args.data_only_dry_run:
        args.dry_run = True
    for name in ("dataset_dir", "index_file", "norm_stats_dir", "stage_a_checkpoint", "output_dir"):
        reject_archive(getattr(args, name))
    reject_archive(args.output_dir / args.run_name)
    return args


def ensure_index(args):
    index = json.loads(args.index_file.read_text())
    validate_index(index)
    if Path(index["dataset_dir"]).resolve() != args.dataset_dir.resolve():
        raise ValueError("V9.6 multitask dataset mismatch")
    norm_file = (args.norm_stats_dir / "norm_stats.json").resolve()
    if index["source_hashes"].get(str(norm_file)) != sha256_file(norm_file):
        raise ValueError("V9.6 multitask norm stats mismatch")
    model_identity = (
        {"status": "unverified_data_only_dry_run", "path": str(args.stage_a_checkpoint)}
        if args.data_only_dry_run
        else validate_stage_a_for_training(args.stage_a_checkpoint, index)
    )
    from tactile_vla.vla.v4_data import scan_v4_lerobot_frames

    frames = scan_v4_lerobot_frames(args.dataset_dir)
    training_base.EXTRA_CONFIG.update(
        {
            "experiment_version": VERSION_TAG,
            "training_profile": TRAINING_PROFILE,
            "data_experiment_version": DATA_VERSION,
            "need_label_policy": LABEL_POLICY,
            "need_sampling_policy": SAMPLING_POLICY,
            "reasoning_window_policy": REASONING_WINDOW_POLICY,
            "need_boundary_audit_sha256": index["need_boundary_audit_sha256"],
            "reasoning_boundary_audit_sha256": index["reasoning_boundary_audit_sha256"],
            "source_scope": index["source_scope"],
            "stage_a_initialization_identity": model_identity,
            "training_target_coverage": index["training_target_coverage"],
            "captioner_identity": index["captioner_identity"],
            "captioner_identity_scope": "offline_label_provenance_only",
            "input_policy": INPUT_POLICY,
            "source_multitask_training_data_hash": index["source_multitask_training_data_hash"],
            "plan_token_validation": index["plan_token_validation"],
            "training_data_hash": index["training_data_hash"],
        }
    )
    return index, frames


def identity(index, **kwargs):
    if trainer._ARGS.data_only_dry_run:
        return {
            "book_v9_6_training_data_hash": index["training_data_hash"],
            "initialization_status": "unverified_data_only_dry_run",
        }
    return _BASE_IDENTITY(index, **kwargs)


def build_loaders(args, model_config, index, records, tokenizer, failure_codec, plan_codec):
    loaders = _BASE_BUILD_LOADERS(args, model_config, index, records, tokenizer, failure_codec, plan_codec)
    raw = loaders["val"]["need"].dataset.dataset
    groups = need_eval_groups(raw.rows, raw.row_indices)
    loaders["val"]["need"].book_v9_6_groups = {
        name: training_base._loader(
            Subset(loaders["val"]["need"].dataset, positions), batch_size=args.batch_size, num_workers=0, shuffle=False
        )
        for name, positions in groups.items()
    }
    training_base.EXTRA_CONFIG["need_eval_group_support"] = {name: len(p) for name, p in groups.items()}
    action_index = json.loads(Path(index["action_index_file"]).read_text())
    _, lookup = trainer.validate_v7_4_adjustment_training_index(
        action_index,
        index_path=Path(index["action_index_file"]),
        dataset_dir=args.dataset_dir,
    )
    train_indices = index["splits"]["train"]["action"]["indices"]
    pools = phase_positions(train_indices, lookup)
    sampler = PhaseBalancedActionBatchSampler(pools, batch_size=args.batch_size, seed=args.seed)
    worker_options = {"multiprocessing_context": "spawn", "persistent_workers": True} if args.num_workers > 0 else {}
    loaders["train"]["action"] = DataLoader(
        loaders["train"]["action"].dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=training_base.collate_numpy,
        **worker_options,
    )
    val_indices = index["splits"]["val"]["action"]["indices"]
    selected = phase_eval_positions(
        val_indices,
        lookup,
        samples_per_phase=args.action_eval_samples_per_phase,
        seed=args.seed,
    )
    audit = {}
    val_pools = phase_positions(val_indices, lookup)
    for phase in PHASES:
        dataset = Subset(loaders["val"]["action"].dataset, selected[phase])
        loaders["val"]["action_" + phase] = training_base._loader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        globals_ = [val_indices[position] for position in selected[phase]]
        groups = sorted({(lookup[g]["episode_id"], lookup[g]["attempt_id"]) for g in globals_})
        audit[phase] = {
            "candidate_count": len(val_pools[phase]),
            "selected_count": len(globals_),
            "episode_attempt_groups": [list(group) for group in groups],
            "global_indices": globals_,
            "global_indices_sha256": sha256_json(globals_),
        }
        if args.dry_run:
            batch = next(iter(loaders["val"]["action_" + phase]))
            if any(lookup[int(g)]["phase"] != phase for g in batch["global_index"]):
                raise ValueError("Phase action evaluation loader contains the wrong phase")
            logging.info(
                "V9.6 %s eval dry-run: samples=%d first_batch=%d groups=%d",
                phase,
                len(dataset),
                len(batch["global_index"]),
                len(groups),
            )
    training_base.EXTRA_CONFIG.update(
        {
            "action_sampling_policy": ACTION_SAMPLING_POLICY,
            "action_eval_policy": ACTION_EVAL_POLICY | {"samples_per_phase": args.action_eval_samples_per_phase},
            "action_train_phase_candidates": {phase: len(pool) for phase, pool in pools.items()},
            "action_eval_selection": audit,
        }
    )
    logging.info(
        "V9.6 action replay: candidates=%s, every batch 4 adjustment + 4 execution",
        {phase: len(pool) for phase, pool in pools.items()},
    )
    if args.dry_run:
        batch = next(iter(loaders["train"]["action"]))
        counts = {phase: sum(lookup[int(g)]["phase"] == phase for g in batch["global_index"]) for phase in PHASES}
        if any(count != args.batch_size // 2 for count in counts.values()):
            raise ValueError("V9.6 real action batch is not phase-balanced")
        logging.info("V9.6 real action batch dry-run phase counts=%s", counts)
    return loaders


def need_eval_groups(rows, positions):
    groups = {}
    for position, row_index in enumerate(positions):
        row = rows[row_index]
        if row["need_variant"] != "real":
            raise ValueError("V9.6 validation must use real need observations only")
        boundary = row.get("need_boundary")
        if boundary is None:
            continue
        direction = boundary["rotation_direction"]
        groups.setdefault(direction, []).append(position)
        if boundary["transition_interval"]:
            groups.setdefault("transition_" + direction, []).append(position)
    return groups


def evaluate_need(state, loader, data_sharding, *, max_samples):
    result = _BASE_EVALUATE_NEED(state, loader, data_sharding, max_samples=max_samples)
    grouped = {}
    for name, group in getattr(loader, "book_v9_6_groups", {}).items():
        metrics = _BASE_EVALUATE_NEED(state, group, data_sharding, max_samples=len(group.dataset))
        matrix = metrics["confusion_matrix"]
        negatives = sum(matrix[0])
        metrics["false_positive_rate"] = matrix[0][1] / negatives if negatives else None
        grouped[name] = metrics
    result["by_direction_and_transition"] = grouped
    return result


def evaluate_action_by_phase(state, loaders, data_sharding, *, seed):
    losses, support = {}, {}
    for phase in PHASES:
        loader = loaders["action_" + phase]
        support[phase] = len(loader.dataset)
        losses[phase] = training_base.evaluate_action_loss(
            state,
            loader,
            data_sharding,
            seed=seed,
            max_batches=None,
            sample_weighted=True,
        )
    return {"loss": sum(losses.values()) / len(PHASES), "by_phase": losses, "support_by_phase": support}


def export_checkpoint(run_dir, state, step, filter_):
    trainer.export_checkpoint(run_dir, state, step, filter_)
    path = run_dir / str(step) / f"{VERSION_TAG}_export.json"
    metadata = json.loads(path.read_text())
    metadata.update(
        {
            "experiment_version": VERSION_TAG,
            "training_profile": TRAINING_PROFILE,
            "data_profile": DATA_PROFILE,
            "data_experiment_version": DATA_VERSION,
            "need_label_policy": LABEL_POLICY,
            "need_sampling_policy": SAMPLING_POLICY,
            "reasoning_window_policy": REASONING_WINDOW_POLICY,
            "action_sampling_policy": ACTION_SAMPLING_POLICY,
            "action_eval_policy": training_base.EXTRA_CONFIG["action_eval_policy"],
            "action_eval_selection": training_base.EXTRA_CONFIG["action_eval_selection"],
            "training_data_hash": training_base.EXTRA_CONFIG["training_data_hash"],
            "source_scope": training_base.EXTRA_CONFIG["source_scope"],
            "stage_a_initialization_identity": training_base.EXTRA_CONFIG["stage_a_initialization_identity"],
            "need_boundary_audit_sha256": training_base.EXTRA_CONFIG["need_boundary_audit_sha256"],
            "reasoning_boundary_audit_sha256": training_base.EXTRA_CONFIG["reasoning_boundary_audit_sha256"],
            "input_policy": INPUT_POLICY,
            "captioner_identity_scope": "offline_label_provenance_only",
            "source_multitask_training_data_hash": training_base.EXTRA_CONFIG["source_multitask_training_data_hash"],
        }
    )
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def configure_training():
    previous.configure_training()
    training_base.CHECKPOINT_EXPORT_HOOK = export_checkpoint
    training_base.ACTION_EVALUATION_HOOK = evaluate_action_by_phase
    training_base.EXTRA_CONFIG.update(
        {
            "experiment_version": VERSION_TAG,
            "training_profile": TRAINING_PROFILE,
            "data_experiment_version": DATA_VERSION,
            "need_label_policy": LABEL_POLICY,
            "need_sampling_policy": SAMPLING_POLICY,
            "reasoning_window_policy": REASONING_WINDOW_POLICY,
            "input_policy": INPUT_POLICY,
            "target_support_note": "Book V9.6: unchanged V9.5 real labels/frames without Touch input; no counterfactuals or resampling; real [C,C+14] failure/plan; plan history 1-4",
            "plan_eval_policy": "all_C_plus_14_variants_grouped_by_memory_length_and_left_right",
            "need_eval_policy": "real_only_full_val_including_F_to_C_grouped_by_direction_and_transition",
            "action_sampling_policy": ACTION_SAMPLING_POLICY,
            "action_eval_policy": ACTION_EVAL_POLICY,
        }
    )


def configure_version():
    previous.configure_version()
    trainer.DATA_PROFILE = DATA_PROFILE
    trainer.PROMPT_PROFILE = PROMPT_PROFILE
    trainer.validate_index = validate_index
    trainer.DEFAULT_INDEX = DEFAULT_INDEX
    trainer.DEFAULT_STAGE_A = DEFAULT_STAGE_A
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.RUN_NAME = RUN_NAME
    trainer.VERSION_TAG = VERSION_TAG
    trainer.ensure_index = ensure_index
    trainer.identity = identity
    trainer.parse_args = parse_args
    trainer.build_loaders = build_loaders
    trainer.configure = configure_training
    training_base.evaluate_need = evaluate_need


def main():
    configure_version()
    trainer.main()


if __name__ == "__main__":
    main()
