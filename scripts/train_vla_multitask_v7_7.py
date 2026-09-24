#!/usr/bin/env python3
"""Train V7.7 action + two phase heads + failure/plan generation."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]
os.environ.setdefault("USE_TF", "0")

import orbax.checkpoint as ocp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.shared import normalize
from tactile_vla.vla.artifacts import checkpoint_artifact_identity, sha256_file
from tactile_vla.vla.openpi_bridge import (
    TactileVLAFrameDataset, TransformedTactileVLADataset, build_structured_inference_transform,
    build_structured_text_transform, build_transform,
)
from tactile_vla.vla.stage_b_v3_checkpoint import delta_params
from tactile_vla.vla.v4_data import scan_v4_lerobot_frames
from tactile_vla.vla.v7_4_adjustment_data import validate_v7_4_adjustment_training_index
from tactile_vla.vla.v7_7_multitask_data import (
    DATA_PROFILE, TASK_CYCLE, TransformedV77Dataset, V77ManifestDataset, load_jsonl, validate_index,
)
from tactile_vla.vla.v7_7_multitask_model import V77MultitaskModel, trainable_filter
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from scripts import train_vla_stage_b_v3 as base

_BASE_PARSE_ARGS = base.parse_args

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DEFAULT_INDEX = ROOT / "outputs/rotation_v7_7_multitask/v7_7_multitask_training_index.json"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_no_history/15000"
DEFAULT_OUTPUT = ROOT / "outputs/multitask_v7_7"
DEFAULT_DATASET = ROOT / "lerobot_data/tactile_vla_rotation_v4"
DEFAULT_NORM_STATS = ROOT / "outputs/rotation_v4/norm_stats"
RUN_NAME = "pi05_rotation_v7_7_five_task_h100_no_history"
VERSION_TAG = "v7_7"
_ARGS = None


def parse_args():
    # Reuse the mature CLI, then pin/add V7.7 defaults before it is parsed.
    defaults = {
        "--dataset-dir": str(DEFAULT_DATASET),
        "--index-file": str(DEFAULT_INDEX), "--norm-stats-dir": str(DEFAULT_NORM_STATS),
        "--stage-a-checkpoint": str(DEFAULT_STAGE_A), "--output-dir": str(DEFAULT_OUTPUT),
        "--run-name": RUN_NAME, "--data-profile": DATA_PROFILE, "--prompt-profile": PROMPT_PROFILE,
        "--batch-size": "8", "--num-steps": "20000", "--eval-interval": "1000",
        "--save-interval": "5000", "--keep-period": "5000", "--max-token-len": "512",
        "--reasoning-max-token-len": "320", "--state-history-len": "0", "--history-hidden-dim": "0",
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    if "--use-state-history" not in sys.argv and "--no-use-state-history" not in sys.argv:
        sys.argv.append("--no-use-state-history")
    args = _BASE_PARSE_ARGS()
    args.adjustment_loss_weight = 1.0
    if args.num_steps % 5 or args.save_interval % 5:
        raise ValueError("V7.7 num_steps and save_interval must end on a five-task cycle boundary")
    global _ARGS
    _ARGS = args
    return args


def ensure_index(args):
    index = json.loads(args.index_file.read_text())
    validate_index(index)
    if Path(index["stage_a_checkpoint"]).resolve() != args.stage_a_checkpoint.resolve():
        raise ValueError("multitask index was built for a different Stage-A checkpoint")
    if Path(index["dataset_dir"]).resolve() != args.dataset_dir.resolve():
        raise ValueError("multitask index was built for a different LeRobot dataset")
    norm_file = (args.norm_stats_dir / "norm_stats.json").resolve()
    expected_norm_hash = index.get("source_hashes", {}).get(str(norm_file))
    if expected_norm_hash is None or sha256_file(norm_file) != expected_norm_hash:
        raise ValueError("multitask index and norm stats differ")
    return index, scan_v4_lerobot_frames(args.dataset_dir)


def identity(index, **_):
    assert _ARGS is not None
    config_path, config = base._find_checkpoint_config(_ARGS.stage_a_checkpoint)
    result = checkpoint_artifact_identity(config)
    result.update({
        f"{VERSION_TAG}_training_data_hash": index["training_data_hash"],
        f"{VERSION_TAG}_index_sha256": sha256_file(_ARGS.index_file),
        f"{VERSION_TAG}_prompt_profile": PROMPT_PROFILE,
    })
    return result


def _manifest(index, task):
    path = Path(index[f"{task}_manifest_file"])
    if sha256_file(path) != index[f"{task}_manifest_sha256"]:
        raise ValueError(f"V7.7 {task} manifest hash changed")
    return load_jsonl(path)


def build_loaders(args, model_config, index, records, tokenizer, failure_codec, plan_codec):
    del records
    norm_stats = normalize.load(args.norm_stats_dir)
    action_index = json.loads(Path(index["action_index_file"]).read_text())
    _, phase_lookup = validate_v7_4_adjustment_training_index(
        action_index, index_path=Path(index["action_index_file"]), dataset_dir=args.dataset_dir
    )
    max_offset = max(
        [args.action_horizon - 1] + [
            int(row["action_target_offsets"][-1])
            for row in phase_lookup.values() if row.get("action_target_offsets")
        ]
    )
    dataset_repo_id = args.dataset_dir.name
    shared = LeRobotDataset(
        dataset_repo_id, root=args.dataset_dir,
        delta_timestamps={"action": [step / 30.0 for step in range(max_offset + 1)]},
        download_videos=False, video_backend=args.video_backend,
    )
    plain_shared = LeRobotDataset(
        dataset_repo_id, root=args.dataset_dir, download_videos=False,
        video_backend=args.video_backend,
    )
    action_transform = build_transform(
        model_config, norm_stats=norm_stats, use_quantile_norm=True, use_delta_actions=True
    )
    classify_transform = build_structured_inference_transform(
        model_config, tokenizer=tokenizer, max_len=model_config.max_token_len,
        norm_stats=norm_stats, use_quantile_norm=True,
    )
    failure_transform = build_structured_text_transform(
        model_config, tokenizer=tokenizer, grammar=failure_codec, max_len=model_config.max_token_len,
        norm_stats=norm_stats, use_quantile_norm=True,
    )
    plan_transform = build_structured_text_transform(
        model_config, tokenizer=tokenizer, grammar=plan_codec, max_len=args.reasoning_max_token_len,
        norm_stats=norm_stats, use_quantile_norm=True,
    )
    manifests = {task: _manifest(index, task) for task in ("adjustment", "need", "failure", "plan")}
    output = {"train": {}, "val": {}}
    for split in output:
        split_index = index["splits"][split]
        action_raw = TactileVLAFrameDataset(
            dataset_dir=args.dataset_dir, indices=split_index["action"]["indices"], stage="execution",
            action_horizon=args.action_horizon, state_history_len=0, video_backend=args.video_backend,
            prompt_profile="phase_v2", action_phase_by_global_index=phase_lookup,
            dataset_repo_id=dataset_repo_id, lerobot_dataset=shared,
        )
        output[split]["action"] = base._loader(
            TransformedTactileVLADataset(action_raw, action_transform), batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=split == "train",
        )
        for task in ("adjustment", "need", "failure", "plan"):
            task_index = split_index[task]
            raw = V77ManifestDataset(
                rows=manifests[task], row_indices=task_index["manifest_row_indices"],
                global_indices=task_index["global_indices"], task=task, lerobot_dataset=plain_shared,
            )
            transform = classify_transform if task in {"adjustment", "need"} else (
                failure_transform if task == "failure" else plan_transform
            )
            output[split][task] = base._loader(
                TransformedV77Dataset(raw, transform), batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=split == "train",
            )
    return output


def export_checkpoint(run_dir, state, step, filter_):
    step_dir = run_dir / str(step)
    exports = {
        "delta_params": delta_params(state.params, filter_).to_pure_dict(),
        "full_params": state.params.to_pure_dict(),
    }
    metadata = {"default_deployment": "full_params", "step": step, "exports": {}}
    for name, tree in exports.items():
        path = step_dir / name
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(path, tree, force=True)
        source_state = state.params.filter(filter_) if name == "delta_params" else state.params
        metadata["exports"][name] = {
            "path": str(path.resolve()), "parameter_tree_sha256": parameter_tree_sha256(source_state)
        }
    (step_dir / f"{VERSION_TAG}_export.json").write_text(json.dumps(metadata, indent=2) + "\n")


def configure():
    base.TASK_CYCLE = TASK_CYCLE
    base.CHECKPOINT_EXPORT_HOOK = export_checkpoint
    base.EXTRA_CONFIG = {
        "thresholds": {"adjustment_end": 0.5, "need_recovery": 0.5},
        "threshold_policy": "checkpoint_metadata_only",
        "phase_prefill_protocol": "need_failure_shared_kv_v1",
    }
    base.StageBV3Model = V77MultitaskModel
    base.trainable_filter = trainable_filter
    base.V4_STAGE_B_TRAINABLE_COMPONENTS = ("paligemma_lora", "adjustment_end_head", "need_recovery_head")
    base.V4_STAGE_B_FROZEN_COMPONENTS = ("action_expert", "paligemma_non_lora")
    base.parse_args = parse_args
    base.ensure_v3_index = ensure_index
    base.artifact_identity = identity
    base.validate_norm_stats_identity = lambda path, identity, context: {
        "norm_stats_sha256": sha256_file(Path(path).parent / "norm_stats.json")
    }
    base.validate_reasoning_manifests = lambda *args, **kwargs: {
        f"{VERSION_TAG}_training_data_hash": json.loads(_ARGS.index_file.read_text())["training_data_hash"]
    }
    base.validate_stage_a_checkpoint_step = lambda profile, checkpoint: int(Path(checkpoint).name)
    base.resolve_prompt_profile = lambda value: value
    base.validate_v4_args = lambda args: None
    base.validate_v4_training_protocol = lambda args: None
    base.build_loaders = build_loaders


def main():
    configure()
    base.main()


if __name__ == "__main__":
    main()
