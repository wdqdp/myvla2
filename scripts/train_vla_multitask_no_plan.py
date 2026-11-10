#!/usr/bin/env python3
"""Train Book, Sponge, or Bottle no-Plan recovery from task-specific sources."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import orbax.checkpoint as ocp
from openpi.shared import normalize
from scripts import train_vla_multitask_v7_7 as trainer
from scripts import train_vla_stage_b_v3 as base
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.openpi_bridge import (
    TransformedTactileVLADataset, build_structured_inference_transform, build_transform,
)
from tactile_vla.vla.v7_11_data import (
    DATA_PROFILE, PROMPT_PROFILE, TASK_CYCLE,
    selected_label_counts,
)
from tactile_vla.vla.v7_11_datasets import DirectActionDataset, DirectPhaseDataset
from tactile_vla.vla.no_plan_selection import select_factual
from scripts.train_vla_stage_a_no_plan import TASKS, prompt_identity
from torch.utils.data import DataLoader, Subset
from tactile_vla.vla.book_v9_4_5_action_replay import (
    PhaseBalancedActionBatchSampler, phase_positions, phase_eval_positions,
)
from tactile_vla.vla.v4_data import scan_v4_lerobot_frames
from tactile_vla.vla.v7_7_multitask_data import TransformedV77Dataset


TASK = None
VERSION = None
SOURCE_MODULE = None
ACTION_MODULE = None
SOURCE_ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data")
_BASE_PARSE = trainer._BASE_PARSE_ARGS
_BASE_CONFIGURE = trainer.configure


def parse_args():
    defaults = {
        "--dataset-dir": str(trainer.DEFAULT_DATASET), "--index-file": str(trainer.DEFAULT_INDEX),
        "--norm-stats-dir": str(trainer.DEFAULT_NORM_STATS),
        "--stage-a-checkpoint": str(trainer.DEFAULT_STAGE_A),
        "--output-dir": str(SOURCE_ROOT / TASK / "outputs" / f"multitask_{VERSION}"),
        "--run-name": f"pi05_{TASK}_{VERSION}_direct_tactile_h100", "--data-profile": DATA_PROFILE,
        "--prompt-profile": PROMPT_PROFILE, "--batch-size": "8", "--num-steps": "2000",
        "--eval-interval": "2000", "--save-interval": "2000", "--keep-period": "2000",
        "--max-token-len": "512", "--reasoning-max-token-len": "320",
        "--state-history-len": "0", "--history-hidden-dim": "0",
    }
    supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:]}
    for flag, value in defaults.items():
        if flag not in supplied:
            sys.argv.extend([flag, value])
    if "--use-state-history" not in supplied and "--no-use-state-history" not in supplied:
        sys.argv.append("--no-use-state-history")
    args = _BASE_PARSE()
    if (args.num_steps, args.eval_interval, args.save_interval, args.keep_period) != (2000, 2000, 2000, 2000):
        raise ValueError("no-Plan Stage B requires 2000 steps and eval/save/keep interval 2000")
    if args.resume:
        raise ValueError("no-Plan Stage B saves full_params for inference only; --resume is not supported")
    if args.use_state_history or args.state_history_len != 0 or args.history_hidden_dim != 0:
        raise ValueError("no-Plan action expert uses no dense state history")
    if args.data_profile != DATA_PROFILE or args.prompt_profile != PROMPT_PROFILE:
        raise ValueError("no-Plan requires its direct tactile data/prompt profiles")
    args.adjustment_loss_weight = 1.0
    trainer._ARGS = args
    return args


def build_loaders(args, model_config, index, records, tokenizer, failure_codec, plan_codec):
    del failure_codec, plan_codec
    norm_stats = normalize.load(args.norm_stats_dir)
    action_index = json.loads(Path(index["action_index_file"]).read_text())
    action_kwargs = {"index_path": Path(index["action_index_file"]), "dataset_dir": args.dataset_dir}
    if TASK == "book":
        action_kwargs["book_root"] = SOURCE_ROOT / TASK
    _, phase_lookup = ACTION_MODULE.validate_training_index(action_index, **action_kwargs)
    frames = {int(frame.global_index): frame for frame in records}
    max_offset = max([args.action_horizon - 1] + [
        int(row["action_target_offsets"][-1]) for row in phase_lookup.values() if row.get("action_target_offsets")
    ])
    shared = LeRobotDataset(
        args.dataset_dir.name, root=args.dataset_dir,
        delta_timestamps={"action": [step / 30.0 for step in range(max_offset + 1)]},
        download_videos=False, video_backend=args.video_backend,
    )
    plain = LeRobotDataset(args.dataset_dir.name, root=args.dataset_dir,
                           download_videos=False, video_backend=args.video_backend)
    action_transform = build_transform(model_config, norm_stats=norm_stats, use_quantile_norm=True,
                                       use_delta_actions=True)
    phase_transform = build_structured_inference_transform(
        model_config, tokenizer=tokenizer, max_len=model_config.max_token_len,
        norm_stats=norm_stats, use_quantile_norm=True,
    )
    manifests = {task: trainer._manifest(index, task) for task in ("adjustment", "need")}
    loaders = {"train": {}, "val": {}}
    selection = {}
    for split in loaders:
        source_split = index["splits"][split]
        action = DirectActionDataset(
            dataset_dir=args.dataset_dir, indices=source_split["action"]["indices"], stage="execution",
            action_horizon=args.action_horizon, state_history_len=0, video_backend=args.video_backend,
            prompt_profile="phase_v2", action_phase_by_global_index=phase_lookup,
            dataset_repo_id=args.dataset_dir.name, lerobot_dataset=shared,
        )
        if args.dry_run and split == "train":
            phase_indices = {}
            for position, global_index in enumerate(action.indices):
                phase = phase_lookup[global_index]["phase"]
                phase_indices.setdefault(phase, position)
                if {"execution", "adjustment"} <= phase_indices.keys():
                    break
            if not {"execution", "adjustment"} <= phase_indices.keys():
                raise ValueError("no-Plan action training requires both execution and adjustment samples")
            for phase in ("execution", "adjustment"):
                sample = action[phase_indices[phase]]
                transformed = action_transform(sample)
                print(f"{TASK} {VERSION} {phase} action sample global_index={sample['global_index']}")
                print(sample["prompt"])
                print(f"{TASK} {VERSION} {phase} action target shape={transformed['actions'].shape}")
        loaders[split]["action"] = base._loader(
            TransformedTactileVLADataset(action, action_transform), batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=split == "train",
        )
        phase_counts = Counter(phase_lookup[int(i)]["phase"] for i in action.indices)
        missing_touch = [i for i in action.indices if phase_lookup[i]["phase"] == "adjustment"
                         and (i not in frames or not frames[i].tactile_caption.strip()
                              or frames[i].tactile_caption.strip().lower() == "none")]
        if missing_touch:
            raise ValueError(f"Missing current tactile captions for recovery action frames: {missing_touch[:10]}")
        selection[split] = {"action": {"count": len(action.indices), "phases": dict(phase_counts)}}
        for task in ("adjustment", "need"):
            stream = source_split[task]
            row_indices, global_indices = select_factual(manifests[task], stream)
            raw = DirectPhaseDataset(rows=manifests[task], row_indices=row_indices,
                                     global_indices=global_indices, task=task, lerobot_dataset=plain)
            loaders[split][task] = base._loader(
                TransformedV77Dataset(raw, phase_transform), batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=split == "train",
            )
            label = "adjustment_end" if task == "adjustment" else "need_recovery"
            selection[split][task] = {"count": len(row_indices),
                                      "labels": selected_label_counts(manifests[task], row_indices, label)}
    pools = phase_positions(index["splits"]["train"]["action"]["indices"], phase_lookup)
    loaders["train"]["action"] = DataLoader(
        loaders["train"]["action"].dataset,
        batch_sampler=PhaseBalancedActionBatchSampler(pools, batch_size=args.batch_size, seed=args.seed),
        num_workers=args.num_workers, collate_fn=base.collate_numpy,
        **({"multiprocessing_context": "spawn", "persistent_workers": True} if args.num_workers else {}),
    )
    if args.dry_run:
        batch = next(iter(loaders["train"]["action"]))
        counts = Counter(phase_lookup[int(g)]["phase"] for g in batch["global_index"])
        expected = {"execution": args.batch_size // 2, "adjustment": args.batch_size // 2}
        if counts != expected:
            raise ValueError(f"Unbalanced action batch: {counts} != {expected}")
        print(f"{TASK} {VERSION} action batch phases: {dict(counts)}")
    val_indices = index["splits"]["val"]["action"]["indices"]
    chosen = phase_eval_positions(val_indices, phase_lookup, samples_per_phase=256, seed=args.seed)
    for phase in ("execution", "adjustment"):
        loaders["val"]["action_" + phase] = base._loader(
            Subset(loaders["val"]["action"].dataset, chosen[phase]),
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
        )
    base.EXTRA_CONFIG["no_plan_selected_samples"] = selection
    return loaders


def ensure_index(args):
    index = json.loads(args.index_file.read_text())
    SOURCE_MODULE.validate_index(index)
    if Path(index["dataset_dir"]).resolve() != args.dataset_dir.resolve():
        raise ValueError("no-Plan dataset differs from the task source index")
    norm_file = (args.norm_stats_dir / "norm_stats.json").resolve()
    if index["source_hashes"].get(str(norm_file)) != sha256_file(norm_file):
        raise ValueError("no-Plan norm stats differ from the task source index")
    checkpoint = args.stage_a_checkpoint.resolve()
    config_path = checkpoint.parent / "config.json"
    metadata = checkpoint / "params/_METADATA"
    if checkpoint.name != "10000" or not config_path.is_file() or not metadata.is_file():
        raise ValueError(f"Expected completed no-Plan Stage A checkpoint at step 10000: {checkpoint}")
    config = json.loads(config_path.read_text())
    if config.get("no_plan_stage_a") != prompt_identity(TASK):
        raise ValueError("Expected the matching task-specific no-Plan Stage A checkpoint")
    action_file = Path(index["action_index_file"])
    action = json.loads(action_file.read_text())
    identity = config.get("artifact_identity", {})
    required = {
        "experiment_version": (config.get("experiment_version"), {"book": "book_v9_5", "sponge": "sponge_v10_1", "bottle": "bottle_v11_2"}[TASK]),
        "num_steps": (config.get("num_steps"), 10000),
        "data_profile": (config.get("data_profile"), "book_stage_a_v1"),
        "use_state_history": (config.get("use_state_history"), False),
        "source_scope": (config.get("source_scope"), action["source_scope"]),
        "action_training_data_hash": (identity.get("training_data_hash"), index["action_training_data_hash"]),
        "index_sha256": (identity.get("index_sha256"), sha256_file(action_file)),
        "norm_stats": (identity.get("v4_norm_stats_sha256"), action["v4_norm_stats_sha256"]),
    }
    if TASK in {"sponge", "bottle"}:
        required["failure_action_policy"] = (config.get("failure_action_policy"), action["target_policy"]["failure_action_filter"])
    if TASK == "bottle":
        required["action_target_policy"] = (config.get("action_target_policy"), action["target_policy"])
        required["idle_gap_audit_sha256"] = (config.get("idle_gap_audit_sha256"), sha256_json(action["idle_gap_audit"]))
    mismatches = {key: {"saved": saved, "expected": expected} for key, (saved, expected) in required.items() if saved != expected}
    if mismatches:
        raise ValueError(f"no-Plan Stage A checkpoint/data mismatch: {mismatches}")
    stage = {"path": str(checkpoint), "step": 10000, "config_sha256": sha256_file(config_path),
             "params_metadata_sha256": sha256_file(metadata), "action_training_data_hash": index["action_training_data_hash"]}
    base.EXTRA_CONFIG["stage_a_initialization_identity"] = stage
    return index, scan_v4_lerobot_frames(args.dataset_dir)


def configure():
    _BASE_CONFIGURE()
    base.TASK_CYCLE = TASK_CYCLE
    base.REQUIRE_FULL_TASK_CYCLE = False
    base.build_loaders = build_loaders
    base.ACTION_EVALUATION_HOOK = evaluate_action_by_phase
    base.ensure_v3_index = ensure_index
    base.save_state = lambda manager, state, step, train_filter, rng: None
    base.CHECKPOINT_EXPORT_HOOK = export_full_params
    base.save_best_checkpoint = lambda run_dir, state, metrics, train_filter: None
    base.EXTRA_CONFIG.update({
        "experiment_version": f"{TASK}_{VERSION}", "phase_prefill_protocol": "classification_only",
        "recovery_action_condition": "current_tactile_no_plan_no_recovery_history",
        "phase_condition": "current_tactile_h100_no_plan",
        "adjustment_supervision": "v7_7_2_selected_factual_only_no_counterfactual",
        "checkpoint_format": "full_params_inference_only_v1",
        "checkpoint_contents": ["full_params"],
        "checkpoint_score": "validation_only_no_best_selection",
        "updates_per_task": None,
        "updates_per_task_by_task": {"action": 667, "adjustment": 667, "need": 666},
    })


def export_full_params(run_dir, state, step, train_filter):
    del train_filter
    if base.jax.process_index() != 0:
        return
    step_dir = run_dir / str(step)
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / "full_params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, state.params.to_pure_dict(), force=True)
    (step_dir / f"{TASK}_{VERSION}_export.json").write_text(json.dumps({
        "step": step, "default_deployment": "full_params", "full_params": str(path.resolve()),
        "resume_supported": False,
    }, indent=2) + "\n")


def evaluate_action_by_phase(state, loaders, data_sharding, *, seed):
    phases = ("execution", "adjustment")
    losses = {p: base.evaluate_action_loss(state, loaders["action_" + p], data_sharding,
              seed=seed, max_batches=None, sample_weighted=True) for p in phases}
    return {"loss": sum(losses.values()) / len(phases), "by_phase": losses,
            "support_by_phase": {p: len(loaders["action_" + p].dataset) for p in phases}}


def main():
    global TASK, VERSION, SOURCE_MODULE, ACTION_MODULE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", choices=TASKS, required=True)
    args, remaining = parser.parse_known_args()
    TASK, VERSION = args.task, TASKS[args.task][0]
    source_versions = {"book": "book_v9_5", "sponge": "sponge_v10_1", "bottle": "bottle_v11_2"}
    source_version = source_versions[TASK]
    SOURCE_MODULE = importlib.import_module(f"tactile_vla.vla.{source_version}_multitask_data")
    ACTION_MODULE = importlib.import_module(f"tactile_vla.vla.{source_version}_stage_a_data")
    trainer.DEFAULT_DATASET = SOURCE_ROOT / TASK / "lerobot_data/tactile_vla_rotation_v4"
    trainer.DEFAULT_INDEX = SOURCE_ROOT / TASK / "outputs" / f"{source_version}_multitask" / f"{source_version}_multitask_training_index.json"
    trainer.DEFAULT_NORM_STATS = SOURCE_ROOT / TASK / "outputs/rotation_v4/norm_stats"
    trainer.DEFAULT_STAGE_A = (SOURCE_ROOT / TASK / "outputs/stage_a_action"
                               / f"pi05_{TASK}_{VERSION}_stage_a_direct_tactile" / "10000")
    trainer.DATA_PROFILE = DATA_PROFILE
    trainer.PROMPT_PROFILE = PROMPT_PROFILE
    trainer.VERSION_TAG = f"{TASK}_{VERSION}"
    trainer.parse_args = parse_args
    trainer.configure = configure
    trainer.validate_index = SOURCE_MODULE.validate_index
    sys.argv = [sys.argv[0], *remaining]
    trainer.main()


if __name__ == "__main__":
    main()
