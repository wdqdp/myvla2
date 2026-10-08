#!/usr/bin/env python3
"""Book V9.4.5: 4+4 action replay, delayed C and rotation-none transition negatives."""

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
from tactile_vla.vla.artifacts import sha256_json
from tactile_vla.vla.book_v9_4_5_multitask_data import (
    DATA_PROFILE,
    DEFAULT_INDEX,
    DEFAULT_STAGE_A,
    LABEL_POLICY,
    REASONING_WINDOW_POLICY,
    ROOT,
    SAMPLING_POLICY,
    VERSION_TAG as DATA_VERSION,
    validate_index,
)
from tactile_vla.vla.book_v9_4_5_action_replay import (
    ACTION_EVAL_POLICY,
    ACTION_SAMPLING_POLICY,
    PHASES,
    PhaseBalancedActionBatchSampler,
    phase_eval_positions,
    phase_positions,
)

VERSION_TAG = "book_v9_4_5"
TRAINING_PROFILE = "book_v9_4_5_phase_balanced_action_replay"
DEFAULT_OUTPUT = ROOT / "outputs/multitask_v9_4_5"
RUN_NAME = "pi05_book_v9_4_5_five_task_h100_no_history"
trainer = previous.previous.trainer
training_base = previous.previous.training_base
_BASE_BUILD_LOADERS = trainer.build_loaders


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action-eval-samples-per-phase", type=int, default=256)
    extra, rest = parser.parse_known_args(sys.argv[1:])
    sys.argv[:] = [sys.argv[0], *rest]
    if any(flag in rest for flag in ("-h", "--help")):
        print("V9.4.5 option: --action-eval-samples-per-phase N (default 256; trajectory-stratified per phase)")
    if extra.action_eval_samples_per_phase <= 0:
        raise ValueError("--action-eval-samples-per-phase must be positive")
    if not any(arg == "--num-steps" or arg.startswith("--num-steps=") for arg in rest):
        sys.argv.extend(["--num-steps", "2000"])
    args = previous.previous.parse_args()
    args.action_eval_samples_per_phase = extra.action_eval_samples_per_phase
    return args


def ensure_index(args):
    index, frames = previous.previous.ensure_index(args, index_validator=validate_index)
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
        }
    )
    return index, frames


def build_loaders(args, model_config, index, records, tokenizer, failure_codec, plan_codec):
    loaders = _BASE_BUILD_LOADERS(args, model_config, index, records, tokenizer, failure_codec, plan_codec)
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
                "V9.4.5 %s eval dry-run: samples=%d first_batch=%d groups=%d",
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
        "V9.4.5 action replay: candidates=%s, every batch 4 adjustment + 4 execution",
        {phase: len(pool) for phase, pool in pools.items()},
    )
    if args.dry_run:
        batch = next(iter(loaders["train"]["action"]))
        counts = {phase: sum(lookup[int(g)]["phase"] == phase for g in batch["global_index"]) for phase in PHASES}
        if any(count != args.batch_size // 2 for count in counts.values()):
            raise ValueError("V9.4.5 real action batch is not phase-balanced")
        logging.info("V9.4.5 real action batch dry-run phase counts=%s", counts)
    return loaders


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
            "target_support_note": "Book V9.4.5: train-only rotation-none [F,C) need negatives; real [C,C+14] failure/plan; plan history 1-4",
            "action_sampling_policy": ACTION_SAMPLING_POLICY,
            "action_eval_policy": ACTION_EVAL_POLICY,
        }
    )


def configure_version():
    previous.configure_version()
    trainer.DATA_PROFILE = DATA_PROFILE
    trainer.validate_index = validate_index
    trainer.DEFAULT_INDEX = DEFAULT_INDEX
    trainer.DEFAULT_STAGE_A = DEFAULT_STAGE_A
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.RUN_NAME = RUN_NAME
    trainer.VERSION_TAG = VERSION_TAG
    trainer.ensure_index = ensure_index
    trainer.parse_args = parse_args
    trainer.build_loaders = build_loaders
    trainer.configure = configure_training


def main():
    configure_version()
    trainer.main()


if __name__ == "__main__":
    main()
