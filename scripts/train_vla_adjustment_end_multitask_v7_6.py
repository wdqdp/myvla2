#!/usr/bin/env python3
"""Train V7.4 action replay plus the V7.6 counterfactual adjustment-end classifier."""

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

from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils, normalize
from tactile_vla.vla.openpi_bridge import (
    TactileVLAFrameDataset,
    TransformedTactileVLADataset,
    build_transform,
)
from tactile_vla.vla.v7_4_adjustment_data import (
    validate_v7_4_adjustment_training_index,
)
from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN, PHASE_CHANGE_PROMPT_PROFILE
from tactile_vla.vla.v7_6_adjustment_end_data import (
    AdjustmentEndManifestDataset,
    COUNTERFACTUAL_SELECTION_POLICY,
    DATA_PROFILE,
    DeterministicNaturalBatchSampler,
    EXPERIMENT_KIND,
    HISTORY_POLICY,
    LABEL_POLICY,
    MAGNITUDE_IDS,
    SAMPLE_VARIANT_IDS,
    TransformedAdjustmentEndDataset,
    load_indexed_manifest_rows,
)
from tactile_vla.vla.v7_6_adjustment_end_evaluation import (
    counterfactual_pair_metrics,
    factual_relative_probability_profile,
)

from scripts import train_vla_adjustment_end_multitask_v5_3 as base
from scripts import train_vla_adjustment_end_multitask_v7_5 as v7_5


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
DATASET_DIR = ROOT / "lerobot_data/tactile_vla_rotation_v4"
ACTION_INDEX = ROOT / "outputs/rotation_v7_4_adjustment/v7_4_prompt_training_index.json"
CLASSIFICATION_INDEX = ROOT / "outputs/rotation_v7_6_adjustment_end_counterfactual_h100/adjustment_end_training_index.json"
NORM_DIR = ROOT / "outputs/rotation_v4/norm_stats"
STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_no_history/15000"
OUTPUT_DIR = ROOT / "outputs/adjustment_end_v7_6"
RUN_NAME = "pi05_adjustment_end_rotation_v7_6_counterfactual_h100_no_history"
MULTITASK_CHECKPOINT_FORMAT = "v7_6_adjustment_end_counterfactual_paligemma_lora_h100_no_history_v1"


def _validate_protocol(args) -> None:
    smoke_test = args.num_steps == 4 and args.save_interval == 4
    standard_run = args.num_steps == 8000 and args.save_interval == 4000
    if not (standard_run or smoke_test):
        raise ValueError(
            "V7.6 requires either num_steps/save_interval=8000/1000 or smoke-test values 4/4"
        )
    required = {
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "batch_size": 8,
        "num_workers": 0,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "seed": 42,
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
        raise ValueError(f"V7.6 multitask fixed protocol mismatch: {mismatches}")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")


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


def _build_datasets(args, *, action_config, phase_config, action_index, classification_index, manifest):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    _, action_phase_lookup = validate_v7_4_adjustment_training_index(
        action_index, index_path=args.action_index_file, dataset_dir=args.dataset_dir
    )
    if action_index["selection_hash"] != classification_index["selection_hash"]:
        raise ValueError("V7.4 action and V7.6 classifier indices use different selections")
    if classification_index["action_training_data_hash"] != action_index["training_data_hash"]:
        raise ValueError("V7.6 classifier references a different V7.4 action index")
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
        "tactile_vla_rotation_v4",
        root=args.dataset_dir,
        delta_timestamps={"action": [step / 30.0 for step in range(max_action_offset + 1)]},
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
        build_transform(
            action_config,
            norm_stats=norm_stats,
            use_quantile_norm=True,
            use_delta_actions=True,
        ),
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


def _predict(state, loader, data_sharding) -> list[dict[str, Any]]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    infer = nnx_utils.module_jit(model.adjustment_end_logits)
    output = []
    variant_names = {value: key for key, value in SAMPLE_VARIANT_IDS.items()}
    magnitude_names = {value: key for key, value in MAGNITUDE_IDS.items()}
    for raw in loader:
        raw, valid_count = v7_5._pad_eval_batch(raw, multiple=len(data_sharding.device_set))
        observation, labels = base._batch_to_jax(raw, "adjustment_end", data_sharding)
        probabilities = np.asarray(jax.device_get(jax.nn.softmax(infer(observation), axis=-1)))[:, 1]
        labels_np = np.asarray(jax.device_get(labels))
        for offset in range(valid_count):
            variant_id = int(np.asarray(raw["sample_variant_id"])[offset])
            source_id = int(np.asarray(raw["source_magnitude_id"])[offset])
            prompt_id = int(np.asarray(raw["prompt_magnitude_id"])[offset])
            output.append({
                "label": int(labels_np[offset]),
                "probability": float(probabilities[offset]),
                "manifest_row_index": int(np.asarray(raw["manifest_row_index"])[offset]),
                "episode_id": int(np.asarray(raw["episode_id"])[offset]),
                "frame_index": int(np.asarray(raw["frame_index"])[offset]),
                "rexecution_frame": int(np.asarray(raw["rexecution_frame"])[offset]),
                "arm_adjustment_stop_frame": int(np.asarray(raw["arm_adjustment_stop_frame"])[offset]),
                "pair_id": int(np.asarray(raw["pair_id"])[offset]),
                "sample_variant_id": variant_id,
                "sample_variant": variant_names[variant_id],
                "source_magnitude_id": source_id,
                "source_magnitude": magnitude_names[source_id],
                "prompt_magnitude_id": prompt_id,
                "prompt_magnitude": magnitude_names[prompt_id],
                "target_stop_frame": int(np.asarray(raw["target_stop_frame"])[offset]),
            })
    return output


def _h30_simulation(rows, threshold):
    factual = [row for row in rows if int(row["sample_variant_id"]) == SAMPLE_VARIANT_IDS["factual"]]
    result = v7_5._h30_simulation(factual, threshold)
    result["sample_scope"] = "factual_only"
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _evaluate_final(**kwargs):
    final = v7_5._evaluate_final(**kwargs)
    step_dir = kwargs["run_dir"] / str(kwargs["step"])
    val_rows = _read_jsonl(step_dir / "val_predictions.jsonl")
    test_rows = _read_jsonl(step_dir / "test_predictions.jsonl")
    final.update({
        "history_policy": HISTORY_POLICY,
        "counterfactual_selection_policy": COUNTERFACTUAL_SELECTION_POLICY,
        "validation_counterfactual_pair_metrics": counterfactual_pair_metrics(val_rows),
        "test_counterfactual_pair_metrics": counterfactual_pair_metrics(test_rows),
        "classification_sampling_policy": {
            "strategy": "deterministic_natural_manifest_stream",
            "positive_to_negative_ratio": "1:2",
            "tail_policy": "carry_into_next_epoch",
            "seed": 42,
        },
    })
    text = json.dumps(final, indent=2, ensure_ascii=False) + "\n"
    (step_dir / "adjustment_end_metadata.json").write_text(text)
    (kwargs["run_dir"] / "final_metrics.json").write_text(text)
    return final


def _configure_base() -> None:
    v7_5._configure_base()
    base.__doc__ = __doc__
    base.DEFAULT_DATASET_DIR, base.DEFAULT_ACTION_INDEX = DATASET_DIR, ACTION_INDEX
    base.DEFAULT_CLASSIFICATION_INDEX, base.DEFAULT_NORM_DIR = CLASSIFICATION_INDEX, NORM_DIR
    base.DEFAULT_BACKBONE, base.DEFAULT_OUTPUT = STAGE_A, OUTPUT_DIR
    base.DATA_PROFILE, base.EXPERIMENT_KIND = DATA_PROFILE, EXPERIMENT_KIND
    base.PHASE_CHANGE_PROMPT_PROFILE = PHASE_CHANGE_PROMPT_PROFILE
    base.LABEL_POLICY = LABEL_POLICY
    base.MULTITASK_CHECKPOINT_FORMAT = MULTITASK_CHECKPOINT_FORMAT
    base.AdjustmentEndManifestDataset = AdjustmentEndManifestDataset
    base.DeterministicOneToThreeBatchSampler = DeterministicNaturalBatchSampler
    base.TransformedAdjustmentEndDataset = TransformedAdjustmentEndDataset
    base.load_indexed_manifest_rows = load_indexed_manifest_rows
    base.relative_probability_profile = factual_relative_probability_profile
    base.CLASSIFICATION_SAMPLING_RATIO = {"positive": 1, "negative": 2}
    base.CLASSIFICATION_SAMPLING_POLICY = {
        "strategy": "deterministic_natural_manifest_stream",
        "positive": 1,
        "negative": 2,
        "tail_policy": "carry_into_next_epoch",
        "seed": 42,
    }
    base._validate_protocol = _validate_protocol
    base._validate_stage_a = v7_5._validate_stage_a
    base._model_config, base._build_datasets = _model_config, _build_datasets
    base._predict, base._h30_simulation, base._evaluate_final = _predict, _h30_simulation, _evaluate_final


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
