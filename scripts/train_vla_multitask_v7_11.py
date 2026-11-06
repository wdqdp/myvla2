#!/usr/bin/env python3
"""Train the V7.11 direct-tactile recovery ablation from V7.7.2 data."""

# ruff: noqa: E402
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.shared import normalize
from scripts import train_vla_multitask_v7_7 as trainer
from scripts import train_vla_multitask_v7_7_2 as source
from scripts import train_vla_stage_b_v3 as base
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.openpi_bridge import (
    TransformedTactileVLADataset, build_structured_inference_transform, build_transform,
)
from tactile_vla.vla.v7_11_data import (
    DATA_PROFILE, PROMPT_PROFILE, TASK_CYCLE,
    factual_selection, selected_label_counts,
)
from tactile_vla.vla.v7_11_datasets import DirectActionDataset, DirectPhaseDataset
from tactile_vla.vla.v7_11_stage_a import DEFAULT_STAGE_A_CHECKPOINT, validate_stage_a_checkpoint
from tactile_vla.vla.v4_data import scan_v4_lerobot_frames
from tactile_vla.vla.v7_7_multitask_data import TransformedV77Dataset


ROOT = PROJECT_ROOT / "outputs" / "multitask_v7_11"
_BASE_PARSE = trainer._BASE_PARSE_ARGS
_BASE_CONFIGURE = trainer.configure


def parse_args():
    defaults = {
        "--dataset-dir": str(trainer.DEFAULT_DATASET), "--index-file": str(trainer.DEFAULT_INDEX),
        "--norm-stats-dir": str(trainer.DEFAULT_NORM_STATS),
        "--stage-a-checkpoint": str(trainer.DEFAULT_STAGE_A), "--output-dir": str(ROOT),
        "--run-name": "pi05_rotation_v7_11_direct_tactile_h100", "--data-profile": DATA_PROFILE,
        "--prompt-profile": PROMPT_PROFILE, "--batch-size": "8", "--num-steps": "2001",
        "--eval-interval": "1000", "--save-interval": "1000", "--keep-period": "1000",
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
    if args.num_steps % len(TASK_CYCLE):
        raise ValueError("V7.11 num_steps must be divisible by the three-task cycle")
    if args.use_state_history or args.state_history_len != 0 or args.history_hidden_dim != 0:
        raise ValueError("V7.11 action expert uses no dense state history")
    if args.data_profile != DATA_PROFILE or args.prompt_profile != PROMPT_PROFILE:
        raise ValueError("V7.11 requires its direct tactile data/prompt profiles")
    args.adjustment_loss_weight = 1.0
    trainer._ARGS = args
    return args


def build_loaders(args, model_config, index, records, tokenizer, failure_codec, plan_codec):
    del failure_codec, plan_codec
    norm_stats = normalize.load(args.norm_stats_dir)
    action_index = json.loads(Path(index["action_index_file"]).read_text())
    _, phase_lookup = trainer.validate_v7_4_adjustment_training_index(
        action_index, index_path=Path(index["action_index_file"]), dataset_dir=args.dataset_dir,
    )
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
                raise ValueError("V7.11 action training requires both execution and adjustment samples")
            for phase in ("execution", "adjustment"):
                sample = action[phase_indices[phase]]
                transformed = action_transform(sample)
                print(f"V7.11 {phase} action sample global_index={sample['global_index']}")
                print(sample["prompt"])
                print(f"V7.11 {phase} action target shape={transformed['actions'].shape}")
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
            if task == "adjustment":
                row_indices, global_indices = factual_selection(manifests[task], stream)
            else:
                row_indices, global_indices = stream["manifest_row_indices"], stream["global_indices"]
            raw = DirectPhaseDataset(rows=manifests[task], row_indices=row_indices,
                                     global_indices=global_indices, task=task, lerobot_dataset=plain)
            loaders[split][task] = base._loader(
                TransformedV77Dataset(raw, phase_transform), batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=split == "train",
            )
            label = "adjustment_end" if task == "adjustment" else "need_recovery"
            selection[split][task] = {"count": len(row_indices),
                                      "labels": selected_label_counts(manifests[task], row_indices, label)}
    base.EXTRA_CONFIG["v7_11_selected_samples"] = selection
    return loaders


def ensure_index(args):
    index = json.loads(args.index_file.read_text())
    trainer.validate_index(index)
    if Path(index["dataset_dir"]).resolve() != args.dataset_dir.resolve():
        raise ValueError("V7.11 dataset differs from the V7.7.2 source index")
    norm_file = (args.norm_stats_dir / "norm_stats.json").resolve()
    if index["source_hashes"].get(str(norm_file)) != sha256_file(norm_file):
        raise ValueError("V7.11 norm stats differ from the V7.7.2 source index")
    base.EXTRA_CONFIG["stage_a_initialization_identity"] = validate_stage_a_checkpoint(
        args.stage_a_checkpoint, index
    )
    return index, scan_v4_lerobot_frames(args.dataset_dir)


def configure():
    _BASE_CONFIGURE()
    base.TASK_CYCLE = TASK_CYCLE
    base.build_loaders = build_loaders
    base.ensure_v3_index = ensure_index
    base.EXTRA_CONFIG.update({
        "experiment_version": "v7_11", "phase_prefill_protocol": "classification_only",
        "recovery_action_condition": "current_tactile_no_plan_no_recovery_history",
        "phase_condition": "current_tactile_h100_no_plan",
        "adjustment_supervision": "v7_7_2_selected_factual_only_no_counterfactual",
        "checkpoint_score": "mean_need_and_end_macro_f1_with_action_retention_gate",
    })


def main():
    source.configure_version()
    trainer.DEFAULT_STAGE_A = DEFAULT_STAGE_A_CHECKPOINT
    trainer.DATA_PROFILE = DATA_PROFILE
    trainer.PROMPT_PROFILE = PROMPT_PROFILE
    trainer.VERSION_TAG = "v7_11"
    trainer.parse_args = parse_args
    trainer.configure = configure
    trainer.main()


if __name__ == "__main__":
    main()
