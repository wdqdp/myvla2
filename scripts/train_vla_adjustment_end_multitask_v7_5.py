#!/usr/bin/env python3
"""Train V7.4 action replay plus the V7.5 H100 adjustment-end classifier."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/src"))
os.environ.setdefault("USE_TF", "0")

from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp

from openpi.models.pi0_config import Pi0Config
from openpi.shared import normalize
from openpi.shared import nnx_utils
from tactile_vla.vla.openpi_bridge import build_transform, TactileVLAFrameDataset, TransformedTactileVLADataset
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v7_4_adjustment_data import ROTATION_PHASE_V7_4_ADJUSTMENT
from tactile_vla.vla.v7_4_adjustment_data import V7_4_EXPERIMENT_KIND
from tactile_vla.vla.v7_4_adjustment_data import validate_v7_4_adjustment_training_index
from tactile_vla.vla.v7_5_adjustment_end_data import AdjustmentEndManifestDataset
from tactile_vla.vla.v7_5_adjustment_end_data import DATA_PROFILE, EXPERIMENT_KIND, HISTORY_POLICY, LABEL_POLICY
from tactile_vla.vla.v7_5_adjustment_end_data import DeterministicOneToThreeBatchSampler
from tactile_vla.vla.v7_5_adjustment_end_data import TransformedAdjustmentEndDataset
from tactile_vla.vla.v7_5_adjustment_end_data import load_indexed_manifest_rows
from tactile_vla.vla.v7_5_adjustment_end_evaluation import relative_probability_profile
from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN, PHASE_CHANGE_PROMPT_PROFILE

from scripts import train_vla_adjustment_end_multitask_v5_3 as base


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DATASET_DIR = ROOT / "lerobot_data/tactile_vla_rotation_v4"
ACTION_INDEX = ROOT / "outputs/rotation_v7_4_adjustment/v7_4_prompt_training_index.json"
CLASSIFICATION_INDEX = ROOT / "outputs/rotation_v7_5_adjustment_end_h100/adjustment_end_training_index.json"
NORM_DIR = ROOT / "outputs/rotation_v4/norm_stats"
STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_no_history/15000"
OUTPUT_DIR = ROOT / "outputs/adjustment_end_v7_5"
RUN_NAME = "pi05_adjustment_end_rotation_v7_5_multitask_h100_no_history"
MULTITASK_CHECKPOINT_FORMAT = "v7_5_adjustment_end_paligemma_lora_h100_no_history_v1"


def _validate_protocol(args) -> None:
    smoke_test = args.num_steps == 4 and args.save_interval == 4
    standard_run = args.num_steps == 8000 and args.save_interval == 1000
    if not (standard_run or smoke_test):
        raise ValueError(
            "V7.5 requires either the standard num_steps/save_interval=8000/1000 "
            "or the checkpoint smoke-test values 4/4"
        )
    required = {
        "data_profile": DATA_PROFILE, "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND, "batch_size": 8, "num_workers": 0,
        "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
        "seed": 42, "eval_interval": 1000,
        "action_max_token_len": 200, "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
        "action_horizon": 30, "action_dim": 32, "use_state_history": False,
        "state_history_len": 0, "state_history_dim": 7, "history_hidden_dim": 0,
        "paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora",
    }
    mismatches = {key: {"requested": getattr(args, key), "required": value} for key, value in required.items() if getattr(args, key) != value}
    if mismatches:
        raise ValueError(f"V7.5 multitask fixed protocol mismatch: {mismatches}")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")


def _validate_stage_a(checkpoint: Path):
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.name != "15000" or not (checkpoint / "params" / "_METADATA").is_file():
        raise ValueError(f"V7.5 requires the exact V7.4 Stage A step 15000: {checkpoint}")
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    required = {
        "data_profile": ROTATION_PHASE_V7_4_ADJUSTMENT,
        "prompt_profile": "phase_v2",
        "experiment_kind": V7_4_EXPERIMENT_KIND,
        "num_steps": 15000, "action_horizon": 30, "action_dim": 32,
        "use_state_history": False, "state_history_len": 0,
        "state_history_dim": 7, "history_hidden_dim": 0, "seed": 42,
    }
    mismatches = {key: (config.get(key), value) for key, value in required.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"V7.4 Stage A checkpoint config mismatch: {mismatches}")
    return config_path, config


def _model_config(args, *, precision: str, max_token_len: int) -> Pi0Config:
    return Pi0Config(
        dtype=precision, paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant, action_dim=args.action_dim,
        action_horizon=args.action_horizon, max_token_len=max_token_len, pi05=True,
        use_state_history=False, state_history_len=0, state_history_dim=7,
        history_hidden_dim=0, pytorch_compile_mode=None,
    )


def _build_datasets(args, *, action_config, phase_config, action_index, classification_index, manifest):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    _, action_phase_lookup = validate_v7_4_adjustment_training_index(
        action_index, index_path=args.action_index_file, dataset_dir=args.dataset_dir
    )
    if action_index["selection_hash"] != classification_index["selection_hash"]:
        raise ValueError("V7.4 action and V7.5 classifier indices use different selections")
    if classification_index["action_training_data_hash"] != action_index["training_data_hash"]:
        raise ValueError("V7.5 classifier references a different V7.4 action index")
    max_action_offset = max(
        [args.action_horizon - 1]
        + [
            int(offsets[-1])
            for row in action_phase_lookup.values()
            if (offsets := row.get("action_target_offsets")) is not None
        ]
    )
    norm_stats = normalize.load(args.norm_stats_dir)
    shared = LeRobotDataset(
        "tactile_vla_rotation_v4", root=args.dataset_dir,
        delta_timestamps={"action": [step / 30.0 for step in range(max_action_offset + 1)]},
        download_videos=False, video_backend=args.video_backend,
    )
    action_raw = TactileVLAFrameDataset(
        dataset_dir=args.dataset_dir,
        indices=action_index["splits"]["train"]["execution_indices"],
        stage="execution", action_horizon=args.action_horizon, state_history_len=0,
        video_backend=args.video_backend, prompt_profile="phase_v2",
        action_phase_by_global_index=action_phase_lookup,
        dataset_repo_id="tactile_vla_rotation_v4", lerobot_dataset=shared,
    )
    action_dataset = TransformedTactileVLADataset(
        action_raw,
        build_transform(action_config, norm_stats=norm_stats, use_quantile_norm=True, use_delta_actions=True),
    )
    phase_transform = build_transform(phase_config, norm_stats=norm_stats, use_quantile_norm=True, use_delta_actions=False)
    classification = {}
    for split in ("train", "val", "test"):
        split_index = classification_index["splits"][split]
        raw = AdjustmentEndManifestDataset(
            manifest=manifest, manifest_row_indices=split_index["manifest_row_indices"],
            global_indices=split_index["global_indices"], lerobot_dataset=shared, state_history_len=0,
        )
        classification[split] = TransformedAdjustmentEndDataset(raw, phase_transform)
    return action_dataset, classification


def _pad_eval_batch(raw, *, multiple: int):
    """Repeat the last row so a short eval tail can be sharded, then trim outputs."""

    if multiple <= 0:
        raise ValueError("Evaluation sharding multiple must be positive")
    batch_size = int(np.asarray(raw["adjustment_end_label"]).shape[0])
    if batch_size <= 0:
        raise ValueError("Cannot pad an empty evaluation batch")
    padding = (-batch_size) % multiple
    if padding == 0:
        return raw, batch_size

    def pad(value):
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != batch_size:
            raise ValueError(
                "Every evaluation batch leaf must have the same leading dimension: "
                f"expected {batch_size}, got {array.shape}"
            )
        return np.concatenate((array, np.repeat(array[-1:], padding, axis=0)), axis=0)

    return jax.tree.map(pad, raw), batch_size


def _predict(state, loader, data_sharding) -> list[dict[str, Any]]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    infer = nnx_utils.module_jit(model.adjustment_end_logits)
    output = []
    for raw in loader:
        raw, valid_count = _pad_eval_batch(
            raw, multiple=len(data_sharding.device_set)
        )
        observation, labels = base._batch_to_jax(raw, "adjustment_end", data_sharding)
        probabilities = np.asarray(jax.device_get(jax.nn.softmax(infer(observation), axis=-1)))[:, 1]
        labels_np = np.asarray(jax.device_get(labels))
        for offset in range(valid_count):
            output.append({
                "label": int(labels_np[offset]), "probability": float(probabilities[offset]),
                "episode_id": int(np.asarray(raw["episode_id"])[offset]),
                "frame_index": int(np.asarray(raw["frame_index"])[offset]),
                "rexecution_frame": int(np.asarray(raw["rexecution_frame"])[offset]),
                "arm_adjustment_stop_frame": int(np.asarray(raw["arm_adjustment_stop_frame"])[offset]),
            })
    return output


def _h30_simulation(rows, threshold):
    by_episode: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], {})[row["frame_index"]] = row
    offsets = {}
    for start in range(30):
        counts = {"transition": 0, "early": 0, "miss": 0}
        for episode_rows in by_episode.values():
            stops = {int(row["arm_adjustment_stop_frame"]) for row in episode_rows.values()}
            if len(stops) != 1:
                raise ValueError("H30 simulation requires one arm_adjustment_stop per episode")
            stop = stops.pop()
            endpoints = list(range(start + 30, stop + 11, 30))
            triggers = [frame for frame in endpoints if frame in episode_rows and episode_rows[frame]["probability"] >= threshold]
            if not triggers:
                counts["miss"] += 1
            elif triggers[0] < stop - 10:
                counts["early"] += 1
            else:
                counts["transition"] += 1
        total = len(by_episode)
        offsets[str(start)] = {**counts, "episode_count": total, "transition_recall": counts["transition"] / total, "early_transition_episode_rate": counts["early"] / total, "miss_rate": counts["miss"] / total}
    return {"diagnostic_only": True, "acceptance_gate": None, "start_offsets": offsets, "actual_s0": offsets["0"]}


_base_evaluate_final = base._evaluate_final


def _evaluate_final(**kwargs):
    final = _base_evaluate_final(**kwargs)
    state, filter_ = kwargs["state"], kwargs["filter_"]
    step_dir = kwargs["run_dir"] / str(kwargs["step"])
    full_path = step_dir / "full_params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(full_path, state.params.to_pure_dict(), force=True)
    final.update({
        "parameter_exports": {
            "default_deployment": "full_params",
            "full_params": {"path": str(full_path.resolve()), "parameter_tree_sha256": parameter_tree_sha256(state.params), "lora_structure_preserved": True},
            "delta_params": {"path": str((step_dir / "delta_params").resolve()), "parameter_tree_sha256": parameter_tree_sha256(state.params.filter(filter_))},
        },
        "history_policy": HISTORY_POLICY,
        "use_state_history": False, "state_history_len": 0, "stage_a_protocol": "v7_4_no_state_history",
    })
    text = json.dumps(final, indent=2, ensure_ascii=False) + "\n"
    (step_dir / "adjustment_end_metadata.json").write_text(text)
    (kwargs["run_dir"] / "final_metrics.json").write_text(text)
    return final


def _configure_base() -> None:
    base.__doc__ = __doc__
    base.DEFAULT_DATASET_DIR, base.DEFAULT_ACTION_INDEX = DATASET_DIR, ACTION_INDEX
    base.DEFAULT_CLASSIFICATION_INDEX, base.DEFAULT_NORM_DIR = CLASSIFICATION_INDEX, NORM_DIR
    base.DEFAULT_BACKBONE, base.DEFAULT_OUTPUT = STAGE_A, OUTPUT_DIR
    base.DATA_PROFILE, base.EXPERIMENT_KIND = DATA_PROFILE, EXPERIMENT_KIND
    base.PHASE_CHANGE_PROMPT_PROFILE = PHASE_CHANGE_PROMPT_PROFILE
    base.LABEL_POLICY = LABEL_POLICY
    base.ADJUSTMENT_END_START_OFFSET, base.ADJUSTMENT_END_END_OFFSET = -10, 10
    base.ACTION_DATA_PROFILE, base.ACTION_EXPERIMENT_KIND = ROTATION_PHASE_V7_4_ADJUSTMENT, V7_4_EXPERIMENT_KIND
    base.MULTITASK_CHECKPOINT_FORMAT = MULTITASK_CHECKPOINT_FORMAT
    base.FROZEN_COMPONENTS = ["action_expert_all_parameters", "paligemma_non_lora", "action_projection_layers"]
    base.CHECKPOINT_EXPORTS = ["delta_params", "full_params_final"]
    base.AdjustmentEndManifestDataset = AdjustmentEndManifestDataset
    base.DeterministicOneToThreeBatchSampler = DeterministicOneToThreeBatchSampler
    base.TransformedAdjustmentEndDataset = TransformedAdjustmentEndDataset
    base.load_indexed_manifest_rows = load_indexed_manifest_rows
    base.relative_probability_profile = relative_probability_profile
    base._validate_protocol, base._validate_stage_a = _validate_protocol, _validate_stage_a
    base._model_config, base._build_datasets = _model_config, _build_datasets
    base._predict, base._h30_simulation, base._evaluate_final = _predict, _h30_simulation, _evaluate_final


def main() -> None:
    _configure_base()
    defaults = {
        "--run-name": RUN_NAME, "--experiment-kind": EXPERIMENT_KIND,
        "--eval-interval": "1000", "--state-history-len": "0",
        "--state-history-dim": "7", "--history-hidden-dim": "0",
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    if "--use-state-history" not in sys.argv and "--no-use-state-history" not in sys.argv:
        sys.argv.append("--no-use-state-history")
    base.main()


if __name__ == "__main__":
    main()
