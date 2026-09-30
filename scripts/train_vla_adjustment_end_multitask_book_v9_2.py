#!/usr/bin/env python3
"""Alternate book Stage A action replay and V9.2 adjustment-end classification."""

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

from flax import nnx
import jax
import numpy as np

from openpi.shared import nnx_utils, normalize
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.book_stage_a_data import DATA_PROFILE as ACTION_DATA_PROFILE
from tactile_vla.vla.book_stage_a_data import EXPERIMENT_KIND as ACTION_EXPERIMENT_KIND
from tactile_vla.vla.book_stage_a_data import validate_training_index
from tactile_vla.vla.book_v9_2_adjustment_end_data import (
    AdjustmentEndManifestDataset, DATA_PROFILE,
    EXPERIMENT_KIND, HISTORY_POLICY, LABEL_POLICY, TRAIN_SAMPLING_POLICY,
    TransformedAdjustmentEndDataset, load_indexed_manifest_rows,
)
from tactile_vla.vla.v7_6_adjustment_end_data import DeterministicNaturalBatchSampler
from tactile_vla.vla.openpi_bridge import (
    TactileVLAFrameDataset, TransformedTactileVLADataset, build_transform,
)
from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN, PHASE_CHANGE_PROMPT_PROFILE

from scripts import train_vla_adjustment_end_multitask_v5_3 as base
from scripts import train_vla_adjustment_end_multitask_v7_5 as v7_5
from scripts import train_vla_adjustment_end_multitask_v7_6 as v7_6

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book")
DATASET_DIR = ROOT / "lerobot_data/tactile_vla_rotation_v4"
ACTION_INDEX = ROOT / "outputs/book_stage_a_v1/book_stage_a_training_index.json"
CLASSIFICATION_INDEX = ROOT / "outputs/book_adjustment_end_v9_2/adjustment_end_training_index.json"
NORM_DIR = ROOT / "outputs/rotation_v4/norm_stats"
STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_book_stage_a_v1_no_history/15000"
OUTPUT_DIR = ROOT / "outputs/adjustment_end_v9_2"
RUN_NAME = "pi05_adjustment_end_book_v9_2_multitask_h100_no_history"
MULTITASK_CHECKPOINT_FORMAT = "book_v9_2_adjustment_end_paligemma_lora_h100_no_history_v1"


def _validate_protocol(args) -> None:
    smoke = (args.num_steps, args.eval_interval, args.save_interval, args.keep_period) == (4, 4, 4, 4)
    standard = (args.num_steps, args.eval_interval, args.save_interval, args.keep_period) == (4000, 800, 800, 800)
    if not (smoke or standard):
        raise ValueError("Book V9.2 requires 4000 steps, eval/save/keep every 800 (or smoke 4/4/4/4)")
    required = {
        "data_profile": DATA_PROFILE, "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND, "batch_size": 8, "num_workers": 0,
        "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0, "seed": 42,
        "action_max_token_len": 200, "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
        "action_horizon": 30, "action_dim": 32, "use_state_history": False,
        "state_history_len": 0, "state_history_dim": 7, "history_hidden_dim": 0,
        "paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora",
    }
    mismatches = {key: (getattr(args, key), value) for key, value in required.items() if getattr(args, key) != value}
    if mismatches:
        raise ValueError(f"Book V9.2 fixed protocol mismatch: {mismatches}")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")


def _validate_stage_a(checkpoint: Path):
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.name != "15000" or not (checkpoint / "params/_METADATA").is_file():
        raise ValueError("Book V9.2 requires the Stage A 15000 checkpoint")
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    required = {
        "data_profile": ACTION_DATA_PROFILE, "prompt_profile": "phase_v2",
        "experiment_kind": ACTION_EXPERIMENT_KIND, "num_steps": 15000,
        "action_horizon": 30, "action_dim": 32, "use_state_history": False,
        "state_history_len": 0, "state_history_dim": 7, "history_hidden_dim": 0, "seed": 42,
    }
    mismatches = {key: (config.get(key), value) for key, value in required.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Book Stage A config mismatch: {mismatches}")
    return config_path, config


def _build_datasets(args, *, action_config, phase_config, action_index, classification_index, manifest):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    _, action_phase_lookup = validate_training_index(
        action_index, index_path=args.action_index_file, dataset_dir=args.dataset_dir,
    )
    if action_index["selection_hash"] != classification_index["selection_hash"]:
        raise ValueError("Book action and classifier selections differ")
    if action_index["training_data_hash"] != classification_index["action_training_data_hash"]:
        raise ValueError("Book classifier references a different Stage A index")
    if classification_index["state_norm"]["norm_stats_sha256"] != sha256_file(args.norm_stats_dir / "norm_stats.json"):
        raise ValueError("Book classifier references different V4 norm stats")
    if Path(classification_index["stage_a_checkpoint"]["path"]).resolve() != args.stage_a_checkpoint.resolve():
        raise ValueError("Book classifier references a different Stage A checkpoint")
    if classification_index["source_files"]["stage_a_params_metadata"]["sha256"] != sha256_file(
        args.stage_a_checkpoint / "params/_METADATA"
    ):
        raise ValueError("Book Stage A checkpoint metadata changed since data preparation")
    shared = LeRobotDataset(
        "tactile_vla_rotation_v4", root=args.dataset_dir,
        delta_timestamps={"action": [step / 30.0 for step in range(args.action_horizon)]},
        download_videos=False, video_backend=args.video_backend,
    )
    norm_stats = normalize.load(args.norm_stats_dir)
    action_raw = TactileVLAFrameDataset(
        dataset_dir=args.dataset_dir,
        indices=action_index["splits"]["train"]["execution_indices"],
        stage="execution", action_horizon=args.action_horizon, state_history_len=0,
        video_backend=args.video_backend, prompt_profile="phase_v2",
        action_phase_by_global_index=action_phase_lookup,
        dataset_repo_id="tactile_vla_rotation_v4", lerobot_dataset=shared,
    )
    action_dataset = TransformedTactileVLADataset(
        action_raw, build_transform(
            action_config, norm_stats=norm_stats, use_quantile_norm=True, use_delta_actions=True,
        ),
    )
    phase_transform = build_transform(
        phase_config, norm_stats=norm_stats, use_quantile_norm=True, use_delta_actions=False,
    )
    classification = {}
    for split in ("train", "val", "test"):
        split_index = classification_index["splits"][split]
        raw = AdjustmentEndManifestDataset(
            manifest=manifest, manifest_row_indices=split_index["manifest_row_indices"],
            global_indices=split_index["global_indices"], lerobot_dataset=shared,
        )
        classification[split] = TransformedAdjustmentEndDataset(raw, phase_transform)
    return action_dataset, classification


def _predict(state, loader, data_sharding):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    infer = nnx_utils.module_jit(model.adjustment_end_logits)
    output = []
    for raw in loader:
        raw, count = v7_5._pad_eval_batch(raw, multiple=len(data_sharding.device_set))
        observation, labels = base._batch_to_jax(raw, "adjustment_end", data_sharding)
        probabilities = np.asarray(jax.device_get(jax.nn.softmax(infer(observation), axis=-1)))[:, 1]
        labels_np = np.asarray(jax.device_get(labels))
        for offset in range(count):
            output.append({
                "label": int(labels_np[offset]), "probability": float(probabilities[offset]),
                "episode_id": int(np.asarray(raw["episode_id"])[offset]),
                "frame_index": int(np.asarray(raw["frame_index"])[offset]),
                "rexecution_frame": int(np.asarray(raw["rexecution_frame"])[offset]),
                "manifest_row_index": int(np.asarray(raw["manifest_row_index"])[offset]),
            })
    return output


def _relative_probability_profile(rows):
    bins = []
    for lower, upper in ((-30, -26), (-25, -21), (-20, -16), (-15, -11), (-10, -6), (-5, -1), (0, 0)):
        values = [float(row["probability"]) for row in rows if lower <= int(row["frame_index"]) - int(row["rexecution_frame"]) <= upper]
        if not values:
            raise ValueError(f"No book V9.2 predictions in R-relative bin [{lower},{upper}]")
        bins.append({"relative_start": lower, "relative_end": upper,
                     "sample_count": len(values), "mean_probability": float(np.mean(values))})
    return {"boundary": "native_reexecution_R", "positive_window": "inclusive_[R-10,R]", "bins": bins}


def _async_diagnostic(rows, threshold):
    """Per-frame oracle only; online latency and chunk overlap require a separate async replay."""
    by_episode = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_id"]), []).append(row)
    outcomes = {"timely": 0, "early": 0, "miss": 0}
    for episode_rows in by_episode.values():
        triggers = [row for row in episode_rows if float(row["probability"]) >= threshold]
        if not triggers:
            outcomes["miss"] += 1
        elif int(triggers[0]["label"]) == 0:
            outcomes["early"] += 1
        else:
            outcomes["timely"] += 1
    return {"diagnostic_only": True, "mode": "framewise_async_oracle_no_latency",
            "acceptance_gate": None, "episode_count": len(by_episode), "outcomes": outcomes}


def _evaluate_final(**kwargs):
    final = v7_5._evaluate_final(**kwargs)
    step_dir = kwargs["run_dir"] / str(kwargs["step"])
    final.update({
        "history_policy": HISTORY_POLICY, "train_sampling_policy": TRAIN_SAMPLING_POLICY,
        "stage_a_protocol": "book_stage_a_v1_no_state_history",
        "inference_protocol": "asynchronous_adjustment_end_per_observation",
        "h30_simulation_applicable": False,
        "accepted_for_robot": False,
        "reason_not_auto_accepted": "Offline framewise validation does not verify asynchronous runtime timing or action regression",
    })
    final.pop("validation_h30_simulation", None)
    text = json.dumps(final, indent=2, ensure_ascii=False) + "\n"
    (step_dir / "adjustment_end_metadata.json").write_text(text)
    (kwargs["run_dir"] / "final_metrics.json").write_text(text)
    return final


def _configure_base():
    v7_6._configure_base()
    base.__doc__ = __doc__
    base.DEFAULT_DATASET_DIR, base.DEFAULT_ACTION_INDEX = DATASET_DIR, ACTION_INDEX
    base.DEFAULT_CLASSIFICATION_INDEX, base.DEFAULT_NORM_DIR = CLASSIFICATION_INDEX, NORM_DIR
    base.DEFAULT_BACKBONE, base.DEFAULT_OUTPUT = STAGE_A, OUTPUT_DIR
    base.DATA_PROFILE, base.EXPERIMENT_KIND = DATA_PROFILE, EXPERIMENT_KIND
    base.ACTION_DATA_PROFILE, base.ACTION_EXPERIMENT_KIND = ACTION_DATA_PROFILE, ACTION_EXPERIMENT_KIND
    base.PHASE_CHANGE_PROMPT_PROFILE = PHASE_CHANGE_PROMPT_PROFILE
    base.LABEL_POLICY, base.MULTITASK_CHECKPOINT_FORMAT = LABEL_POLICY, MULTITASK_CHECKPOINT_FORMAT
    base.CHECKPOINT_EXPORTS = ["delta_params", "full_params_final"]
    base.AdjustmentEndManifestDataset = AdjustmentEndManifestDataset
    base.DeterministicOneToThreeBatchSampler = DeterministicNaturalBatchSampler
    base.TransformedAdjustmentEndDataset = TransformedAdjustmentEndDataset
    base.load_indexed_manifest_rows = load_indexed_manifest_rows
    base.relative_probability_profile = _relative_probability_profile
    base.CLASSIFICATION_SAMPLING_RATIO = {"positive": 1, "negative": 2}
    base.CLASSIFICATION_SAMPLING_POLICY = {"strategy": "deterministic_natural_manifest_stream", **TRAIN_SAMPLING_POLICY}
    base._validate_protocol, base._validate_stage_a = _validate_protocol, _validate_stage_a
    base._model_config, base._build_datasets = v7_6._model_config, _build_datasets
    base._predict, base._h30_simulation, base._evaluate_final = _predict, _async_diagnostic, _evaluate_final


def main():
    _configure_base()
    defaults = {
        "--run-name": RUN_NAME, "--experiment-kind": EXPERIMENT_KIND,
        "--num-steps": "4000", "--eval-interval": "800", "--save-interval": "800", "--keep-period": "800",
        "--state-history-len": "0", "--state-history-dim": "7", "--history-hidden-dim": "0",
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    if "--use-state-history" not in sys.argv and "--no-use-state-history" not in sys.argv:
        sys.argv.append("--no-use-state-history")
    base.main()


if __name__ == "__main__":
    main()
