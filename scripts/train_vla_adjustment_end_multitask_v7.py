#!/usr/bin/env python3
"""Train V7 action replay plus native-R adjustment_end without H60 state history."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/src"))
os.environ.setdefault("USE_TF", "0")

import orbax.checkpoint as ocp

from openpi.models.pi0_config import Pi0Config
from openpi.shared import normalize
from tactile_vla.vla.openpi_bridge import build_transform
from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset
from tactile_vla.vla.openpi_bridge import TransformedTactileVLADataset
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_PROMPT_PROFILE
from tactile_vla.vla.v7_adjustment_data import ROTATION_PHASE_V7_ADJUSTMENT
from tactile_vla.vla.v7_adjustment_data import V7_EXPERIMENT_KIND
from tactile_vla.vla.v7_adjustment_data import validate_v7_adjustment_training_index
from tactile_vla.vla.v7_adjustment_end_data import AdjustmentEndManifestDataset
from tactile_vla.vla.v7_adjustment_end_data import ADJUSTMENT_END_END_OFFSET
from tactile_vla.vla.v7_adjustment_end_data import ADJUSTMENT_END_START_OFFSET
from tactile_vla.vla.v7_adjustment_end_data import DATA_PROFILE
from tactile_vla.vla.v7_adjustment_end_data import DeterministicOneToThreeBatchSampler
from tactile_vla.vla.v7_adjustment_end_data import EXPERIMENT_KIND
from tactile_vla.vla.v7_adjustment_end_data import LABEL_POLICY
from tactile_vla.vla.v7_adjustment_end_data import TransformedAdjustmentEndDataset
from tactile_vla.vla.v7_adjustment_end_data import load_indexed_manifest_rows
from tactile_vla.vla.v7_adjustment_end_evaluation import relative_probability_profile

from scripts import train_vla_adjustment_end_multitask_v5_3 as base


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DATASET_DIR = ROOT / "lerobot_data/tactile_vla_rotation_v4"
ACTION_INDEX = ROOT / "outputs/rotation_v7_adjustment/v7_prompt_training_index.json"
CLASSIFICATION_INDEX = ROOT / "outputs/rotation_v7_adjustment_end_r10_r0/adjustment_end_training_index.json"
NORM_DIR = ROOT / "outputs/rotation_v4/norm_stats"
STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_no_history/15000"
OUTPUT_DIR = ROOT / "outputs/adjustment_end_v7"
RUN_NAME = "pi05_adjustment_end_rotation_v7_multitask_r10_r0_no_history"
MULTITASK_CHECKPOINT_FORMAT = "v7_adjustment_end_paligemma_lora_no_history_v1"


def _validate_protocol(args) -> None:
    required = {
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "batch_size": 8,
        "num_workers": 0,
        "num_steps": 8000,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "seed": 42,
        "eval_interval": 1000,
        "save_interval": 1000,
        "action_max_token_len": 200,
        "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
        "action_horizon": 30,
        "action_dim": 32,
        "use_state_history": False,
        "state_history_len": 0,
        "state_history_dim": 7,
        "history_hidden_dim": 0,
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
    }
    mismatches = {
        key: {"requested": getattr(args, key), "required": value}
        for key, value in required.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"V7 multitask fixed protocol mismatch: {mismatches}")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")


def _validate_stage_a(checkpoint: Path):
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.name != "15000" or not (checkpoint / "params" / "_METADATA").is_file():
        raise ValueError(f"V7 requires the exact Stage A step 15000 checkpoint: {checkpoint}")
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    required = {
        "data_profile": ROTATION_PHASE_V7_ADJUSTMENT,
        "prompt_profile": "phase_v2",
        "experiment_kind": V7_EXPERIMENT_KIND,
        "stage_a_protocol": "v7_no_state_history",
        "num_steps": 15000,
        "action_horizon": 30,
        "action_dim": 32,
        "use_state_history": False,
        "state_history_len": 0,
        "state_history_dim": 7,
        "history_hidden_dim": 0,
        "seed": 42,
    }
    mismatches = {key: (config.get(key), value) for key, value in required.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"V7 Stage A checkpoint config mismatch: {mismatches}")
    return config_path, config


def _model_config(args, *, precision: str, max_token_len: int) -> Pi0Config:
    return Pi0Config(
        dtype=precision,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=max_token_len,
        pi05=True,
        use_state_history=False,
        state_history_len=0,
        state_history_dim=7,
        history_hidden_dim=0,
        pytorch_compile_mode=None,
    )


def _build_datasets(
    args,
    *,
    action_config,
    phase_config,
    action_index,
    classification_index,
    manifest,
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    _, action_phase_lookup = validate_v7_adjustment_training_index(
        action_index,
        index_path=args.action_index_file,
        dataset_dir=args.dataset_dir,
    )
    if action_index["selection_hash"] != classification_index["selection_hash"]:
        raise ValueError("V7 action and adjustment_end indices use different selections")
    if classification_index["action_training_data_hash"] != action_index["training_data_hash"]:
        raise ValueError("V7 adjustment_end data references a different action index")
    norm_stats = normalize.load(args.norm_stats_dir)
    shared = LeRobotDataset(
        "tactile_vla_rotation_v4",
        root=args.dataset_dir,
        delta_timestamps={"action": [step / 30.0 for step in range(args.action_horizon)]},
        download_videos=False,
        video_backend=args.video_backend,
    )
    action_raw = TactileVLAFrameDataset(
        dataset_dir=args.dataset_dir,
        indices=action_index["splits"]["train"]["execution_indices"],
        stage="execution",
        action_horizon=args.action_horizon,
        state_history_len=0,
        video_backend=args.video_backend,
        prompt_profile="phase_v2",
        action_phase_by_global_index=action_phase_lookup,
        dataset_repo_id="tactile_vla_rotation_v4",
        lerobot_dataset=shared,
    )
    action_dataset = TransformedTactileVLADataset(
        action_raw,
        build_transform(action_config, norm_stats=norm_stats, use_quantile_norm=True, use_delta_actions=True),
    )
    phase_transform = build_transform(
        phase_config,
        norm_stats=norm_stats,
        use_quantile_norm=True,
        use_delta_actions=False,
    )
    classification = {}
    for split in ("train", "val", "test"):
        split_index = classification_index["splits"][split]
        raw = AdjustmentEndManifestDataset(
            manifest=manifest,
            manifest_row_indices=split_index["manifest_row_indices"],
            global_indices=split_index["global_indices"],
            lerobot_dataset=shared,
            state_history_len=0,
        )
        classification[split] = TransformedAdjustmentEndDataset(raw, phase_transform)
    return action_dataset, classification


_base_evaluate_final = base._evaluate_final


def _evaluate_final(**kwargs):
    final = _base_evaluate_final(**kwargs)
    state = kwargs["state"]
    filter_ = kwargs["filter_"]
    step_dir = kwargs["run_dir"] / str(kwargs["step"])
    full_path = step_dir / "full_params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(full_path, state.params.to_pure_dict(), force=True)
    final.update({
        "parameter_exports": {
            "default_deployment": "full_params",
            "full_params": {
                "path": str(full_path.resolve()),
                "parameter_tree_sha256": parameter_tree_sha256(state.params),
                "lora_structure_preserved": True,
            },
            "delta_params": {
                "path": str((step_dir / "delta_params").resolve()),
                "parameter_tree_sha256": parameter_tree_sha256(state.params.filter(filter_)),
            },
        },
        "use_state_history": False,
        "state_history_len": 0,
        "stage_a_protocol": "v7_no_state_history",
    })
    text = json.dumps(final, indent=2, ensure_ascii=False) + "\n"
    (step_dir / "adjustment_end_metadata.json").write_text(text)
    (kwargs["run_dir"] / "final_metrics.json").write_text(text)
    return final


def _configure_base() -> None:
    base.__doc__ = __doc__
    base.DEFAULT_DATASET_DIR = DATASET_DIR
    base.DEFAULT_ACTION_INDEX = ACTION_INDEX
    base.DEFAULT_CLASSIFICATION_INDEX = CLASSIFICATION_INDEX
    base.DEFAULT_NORM_DIR = NORM_DIR
    base.DEFAULT_BACKBONE = STAGE_A
    base.DEFAULT_OUTPUT = OUTPUT_DIR
    base.DATA_PROFILE = DATA_PROFILE
    base.EXPERIMENT_KIND = EXPERIMENT_KIND
    base.LABEL_POLICY = LABEL_POLICY
    base.ADJUSTMENT_END_START_OFFSET = ADJUSTMENT_END_START_OFFSET
    base.ADJUSTMENT_END_END_OFFSET = ADJUSTMENT_END_END_OFFSET
    base.ACTION_DATA_PROFILE = ROTATION_PHASE_V7_ADJUSTMENT
    base.ACTION_EXPERIMENT_KIND = V7_EXPERIMENT_KIND
    base.MULTITASK_CHECKPOINT_FORMAT = MULTITASK_CHECKPOINT_FORMAT
    base.FROZEN_COMPONENTS = [
        "action_expert_all_parameters",
        "paligemma_non_lora",
        "action_projection_layers",
    ]
    base.CHECKPOINT_EXPORTS = ["delta_params", "full_params_final"]
    base.AdjustmentEndManifestDataset = AdjustmentEndManifestDataset
    base.DeterministicOneToThreeBatchSampler = DeterministicOneToThreeBatchSampler
    base.TransformedAdjustmentEndDataset = TransformedAdjustmentEndDataset
    base.load_indexed_manifest_rows = load_indexed_manifest_rows
    base.relative_probability_profile = relative_probability_profile
    base._validate_protocol = _validate_protocol
    base._validate_stage_a = _validate_stage_a
    base._model_config = _model_config
    base._build_datasets = _build_datasets
    base._evaluate_final = _evaluate_final


def main() -> None:
    _configure_base()
    defaults = {
        "--run-name": RUN_NAME,
        "--experiment-kind": EXPERIMENT_KIND,
        "--eval-interval": "1000",
        "--state-history-len": "0",
        "--state-history-dim": "7",
        "--history-hidden-dim": "0",
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    if "--use-state-history" not in sys.argv and "--no-use-state-history" not in sys.argv:
        sys.argv.append("--no-use-state-history")
    base.main()


if __name__ == "__main__":
    main()
