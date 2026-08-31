#!/usr/bin/env python3
"""Train the V5.3 adjustment-end head on a frozen V5.2 Stage A backbone."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache/huggingface/datasets"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache/torch"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("OPENPI_DATA_HOME", "/data1/outputs/openpi_cache")

# LeRobot currently imports torchvision's legacy video I/O module even though
# this trainer does not use torchvision video decoding.  Suppress only that
# upstream deprecation notice; all other warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
    module=r"torchvision\.io\._video_deprecation_warning",
)

from flax import nnx
from flax import traverse_util
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from torch.utils.data import DataLoader
import tqdm

from openpi.models import gemma as openpi_gemma
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.shared import normalize
from openpi.training import checkpoints as openpi_checkpoints
from openpi.training import sharding
from openpi.training import utils as training_utils
from openpi.training import weight_loaders
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.openpi_bridge import build_transform
from tactile_vla.vla.openpi_bridge import collate_numpy
from tactile_vla.vla.stage_b_v3_checkpoint import cast_frozen_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import CHECKPOINT_FORMAT
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import delta_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import merge_delta_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import resume_state
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import trainable_filter
from tactile_vla.vla.v5_3_adjustment_end_data import AdjustmentEndManifestDataset
from tactile_vla.vla.v5_3_adjustment_end_data import DATA_PROFILE
from tactile_vla.vla.v5_3_adjustment_end_data import DeterministicOneToThreeBatchSampler
from tactile_vla.vla.v5_3_adjustment_end_data import EXPERIMENT_KIND
from tactile_vla.vla.v5_3_adjustment_end_data import TransformedAdjustmentEndDataset
from tactile_vla.vla.v5_3_adjustment_end_data import load_indexed_manifest_rows
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_PROMPT_PROFILE
from tactile_vla.vla.v5_3_phase_change import h30_endpoint_indices
from tactile_vla.vla.v5_3_adjustment_end_model import AdjustmentEndModel


DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_rotation_v4")
DEFAULT_INDEX = Path(
    "/data1/outputs/vla/rotation_v5_adjustment_end_v1/adjustment_end_training_index.json"
)
DEFAULT_NORM_DIR = Path("/data1/outputs/vla/assets/tactile_vla_rotation_v4")
DEFAULT_BACKBONE = Path(
    "/data1/outputs/vla/stage_a_action/pi05_delta_tac_rotation_phase_v5_3/15000"
)
DEFAULT_OUTPUT = Path("/data1/outputs/vla/adjustment_end_v5_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument("--stage-a-checkpoint", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="pi05_adjustment_end_rotation_v5_3")
    parser.add_argument("--data-profile", default=DATA_PROFILE)
    parser.add_argument("--prompt-profile", default=PHASE_CHANGE_PROMPT_PROFILE)
    parser.add_argument("--experiment-kind", default=EXPERIMENT_KIND)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-period", type=int, default=1000)
    parser.add_argument("--phase-change-max-token-len", type=int, default=512)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--state-history-len", type=int, default=60)
    parser.add_argument("--state-history-dim", type=int, default=7)
    parser.add_argument("--history-hidden-dim", type=int, default=256)
    parser.add_argument("--paligemma-variant", default="gemma_2b_lora")
    parser.add_argument("--action-expert-variant", default="gemma_300m_lora")
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_protocol(args: argparse.Namespace) -> None:
    required = {
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "batch_size": 8,
        "num_steps": 4000,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "seed": 42,
        "eval_interval": 500,
        "save_interval": 1000,
        "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
        "action_horizon": 30,
        "action_dim": 32,
        "state_history_len": 60,
        "state_history_dim": 7,
        "history_hidden_dim": 256,
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
    }
    mismatches = {
        key: {"requested": getattr(args, key), "required": value}
        for key, value in required.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"V5.3 fixed training protocol mismatch: {mismatches}")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _validate_stage_a(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint = checkpoint.resolve()
    if checkpoint.name != "15000" or not (checkpoint / "params" / "_METADATA").is_file():
        raise ValueError(f"V5.3 requires the exact Stage A step 15000 checkpoint: {checkpoint}")
    config_path = checkpoint.parent / "config.json"
    config = _load_json(config_path)
    required = {
        "data_profile": "rotation_phase_v5_adjustment_v2",
        "prompt_profile": "phase_v2",
        "experiment_kind": "phase_prompt_h30_terminal_hold",
        "num_steps": 15000,
        "action_horizon": 30,
        "action_dim": 32,
        "state_history_len": 60,
        "state_history_dim": 7,
        "seed": 42,
    }
    mismatches = {key: (config.get(key), value) for key, value in required.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"V5.3 Stage A checkpoint config mismatch: {mismatches}")
    return config_path, config


def _build_datasets(args: argparse.Namespace, model_config: Pi0Config, index: dict[str, Any], manifest):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    norm_stats = normalize.load(args.norm_stats_dir)
    transform = build_transform(
        model_config,
        norm_stats=norm_stats,
        use_quantile_norm=True,
        use_delta_actions=False,
    )
    shared = LeRobotDataset(
        "tactile_vla_rotation_v4",
        root=args.dataset_dir,
        delta_timestamps={
            "observation.state": [step / 30.0 for step in range(-59, 1)]
        },
        download_videos=False,
        video_backend=args.video_backend,
    )
    datasets = {}
    for split in ("train", "val", "test"):
        split_index = index["splits"][split]
        raw = AdjustmentEndManifestDataset(
            manifest=manifest,
            manifest_row_indices=split_index["manifest_row_indices"],
            global_indices=split_index["global_indices"],
            lerobot_dataset=shared,
            state_history_len=args.state_history_len,
        )
        datasets[split] = TransformedAdjustmentEndDataset(raw, transform)
    return datasets


def _eval_loader(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_numpy,
        drop_last=False,
    )


def _train_loader(dataset, args: argparse.Namespace, *, start_step: int) -> DataLoader:
    labels = [bool(dataset.dataset.manifest[row]["adjustment_end"]) for row in dataset.dataset.row_indices]
    sampler = DeterministicOneToThreeBatchSampler(
        labels=labels,
        num_batches=args.num_steps,
        seed=args.seed,
        start_batch=start_step,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_numpy,
    )


def _resolve_params(path: Path) -> Path:
    return path / "params" if (path / "params").is_dir() else path


def _load_weights(loader, shape):
    loaded = loader.load(shape)
    at.check_pytree_equality(expected=shape, got=loaded, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(loaded).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )


def _optimizer(args: argparse.Namespace):
    return optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(args.lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=args.weight_decay),
    )


def _init_state(model_config, args, tx, filter_, init_rng, mesh):
    width = openpi_gemma.get_config(model_config.paligemma_variant).width

    def init(rng, backbone_params=None):
        backbone_rng, head_rng = jax.random.split(rng)
        backbone = model_config.create(backbone_rng)
        if backbone_params is not None:
            graphdef, state = nnx.split(backbone)
            state.replace_by_pure_dict(backbone_params)
            backbone = nnx.merge(graphdef, state)
        model = AdjustmentEndModel(backbone, paligemma_width=width, rngs=nnx.Rngs(head_rng))
        params = nnx.state(model)
        frozen = nnx.All(nnx.Param, nnx.Not(filter_))
        params = cast_frozen_params(params, frozen)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            opt_state=tx.init(params.filter(filter_)),
            tx=tx,
            ema_decay=None,
            ema_params=None,
        )

    shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(shape, mesh, log=True)
    backbone_params = _load_weights(
        weight_loaders.CheckpointWeightLoader(str(_resolve_params(args.stage_a_checkpoint))),
        shape.params["backbone"].to_pure_dict(),
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=(replicated, replicated),
        out_shardings=state_sharding,
    )(init_rng, backbone_params)
    return state, state_sharding


def _apply_gradients(state, model, grads, filter_):
    parameters = state.params.filter(filter_)
    updates, optimizer_state = state.tx.update(grads, state.opt_state, parameters)
    nnx.update(model, optax.apply_updates(parameters, updates))
    return dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=optimizer_state,
    )


def _train_step(filter_, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, labels = batch

    def loss_fn(module):
        logits = module.adjustment_end_logits(observation, rng=rng, train=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filter_))(model)
    state = _apply_gradients(state, model, grads, filter_)
    return state, {"loss": loss, "grad_norm": optax.global_norm(grads)}


def _batch_to_jax(batch, data_sharding):
    local = jax.tree.map(np.asarray, batch)
    local["tokenized_prompt"] = np.asarray(local["tokenized_prompt"], dtype=np.int32)
    local["tokenized_prompt_mask"] = np.asarray(local["tokenized_prompt_mask"], dtype=np.bool_)
    observation = Observation.from_dict(local)
    labels = np.asarray(local["adjustment_end_label"], dtype=np.int32)
    return jax.tree.map(
        lambda value: jax.make_array_from_process_local_data(data_sharding, value),
        (observation, labels),
    )


def _predict(state, loader, data_sharding) -> list[dict[str, Any]]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    infer = nnx_utils.module_jit(model.adjustment_end_logits)
    output = []
    for raw in loader:
        observation, labels = _batch_to_jax(raw, data_sharding)
        probabilities = np.asarray(jax.device_get(jax.nn.softmax(infer(observation), axis=-1)))[:, 1]
        labels_np = np.asarray(jax.device_get(labels))
        for offset in range(len(labels_np)):
            output.append(
                {
                    "label": int(labels_np[offset]),
                    "probability": float(probabilities[offset]),
                    "episode_id": int(np.asarray(raw["episode_id"])[offset]),
                    "frame_index": int(np.asarray(raw["frame_index"])[offset]),
                }
            )
    return output


def _ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int32)
    probs = np.asarray([row["probability"] for row in rows], dtype=np.float64)
    order = np.argsort(-probs, kind="stable")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    positives = max(int(np.count_nonzero(labels == 1)), 1)
    negatives = max(int(np.count_nonzero(labels == 0)), 1)
    tpr = np.concatenate(([0.0], tp / positives, [1.0]))
    fpr = np.concatenate(([0.0], fp / negatives, [1.0]))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    auroc = float(np.trapz(tpr, fpr))
    auprc = float(np.sum((recall - np.concatenate(([0.0], recall[:-1]))) * precision))
    return {"auroc": auroc, "auprc": auprc}


def _threshold_metrics(rows, threshold):
    labels = np.asarray([row["label"] for row in rows], dtype=np.bool_)
    pred = np.asarray([row["probability"] >= threshold for row in rows], dtype=np.bool_)
    tp = int(np.count_nonzero(labels & pred))
    fp = int(np.count_nonzero(~labels & pred))
    fn = int(np.count_nonzero(labels & ~pred))
    tn = int(np.count_nonzero(~labels & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "early_false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        **_ranking_metrics(rows),
    }


def _select_threshold(rows):
    candidates = sorted({0.0, 1.0, *(float(row["probability"]) for row in rows)})
    eligible = []
    for threshold in candidates:
        metrics = _threshold_metrics(rows, threshold)
        if metrics["recall"] >= 0.80 and metrics["early_false_positive_rate"] <= 0.01:
            eligible.append((threshold, metrics))
    if not eligible:
        raise ValueError("No val threshold satisfies recall>=80% and early FPR<=1%")
    return eligible[-1]


def _h30_simulation(rows, threshold):
    by_episode: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], {})[row["frame_index"]] = row
    offsets = {}
    for start in range(30):
        counts = {"transition": 0, "early": 0, "miss": 0}
        delays = []
        hittable = 0
        for episode_rows in by_episode.values():
            rexecution = max(frame for frame, row in episode_rows.items() if row["label"] == 1)
            endpoints = h30_endpoint_indices(start_offset=start, stop_frame=rexecution)
            if any(rexecution - 5 <= frame <= rexecution for frame in endpoints):
                hittable += 1
            triggers = [
                frame
                for frame in endpoints
                if frame in episode_rows and episode_rows[frame]["probability"] >= threshold
            ]
            if not triggers:
                counts["miss"] += 1
            elif triggers[0] < rexecution - 5:
                counts["early"] += 1
                delays.append(triggers[0] - rexecution)
            elif triggers[0] <= rexecution:
                counts["transition"] += 1
                delays.append(triggers[0] - rexecution)
            else:
                counts["miss"] += 1
        total = len(by_episode)
        offsets[str(start)] = {
            **counts,
            "episode_count": total,
            "structurally_hittable": hittable,
            "transition_recall": counts["transition"] / total,
            "early_transition_episode_rate": counts["early"] / total,
            "miss_rate": counts["miss"] / total,
            "first_trigger_relative_to_r": delays,
        }
    actual = offsets["0"]
    actual["gate_passed"] = (
        actual["transition_recall"] >= 0.80
        and actual["early_transition_episode_rate"] <= 0.05
        and actual["miss_rate"] <= 0.20
    )
    return {"start_offsets": offsets, "actual_s0": actual}


def _checkpoint_parts(state, filter_, rng):
    train_state = dataclasses.replace(state, params=nnx.State({}))
    return {
        "state": train_state,
        "rng_key_data": jax.random.key_data(rng),
    }, {"params": resume_state(state.params, filter_)}


def _save(manager, state, step, filter_, rng):
    training_payload, params = _checkpoint_parts(state, filter_, rng)
    manager.save(step, {"train_state": training_payload, "params": params})


def _restore(manager, state, filter_, rng):
    training_payload, params = _checkpoint_parts(state, filter_, rng)
    restored = manager.restore(None, items={"train_state": training_payload, "params": params})
    restored_state = dataclasses.replace(
        restored["train_state"]["state"],
        params=merge_delta_params(state.params, restored["params"]["params"]),
    )
    return restored_state, jax.random.wrap_key_data(restored["train_state"]["rng_key_data"])


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    _validate_protocol(args)
    if args.batch_size % jax.device_count():
        raise ValueError("batch size must be divisible by JAX device count")
    index = _load_json(args.index_file)
    manifest_path = Path(str(index["manifest_file"]))
    manifest = load_indexed_manifest_rows(index=index, manifest_path=manifest_path)
    stage_a_config_path, _ = _validate_stage_a(args.stage_a_checkpoint)
    if index["source_files"]["backbone_config"]["sha256"] != sha256_file(stage_a_config_path):
        raise ValueError("V5.3 data index references a different Stage A config")

    precision = args.precision
    if precision == "auto":
        precision = "bfloat16" if jax.default_backend() in {"gpu", "tpu"} else "float32"
    model_config = Pi0Config(
        dtype=precision,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.phase_change_max_token_len,
        pi05=True,
        use_state_history=True,
        state_history_len=args.state_history_len,
        state_history_dim=args.state_history_dim,
        history_hidden_dim=args.history_hidden_dim,
        pytorch_compile_mode=None,
    )
    datasets = _build_datasets(args, model_config, index, manifest)
    if args.dry_run:
        first = next(iter(_train_loader(datasets["train"], args, start_step=0)))
        labels = np.asarray(first["adjustment_end_label"])
        print(json.dumps({
            "batch_shapes": {key: list(np.asarray(value).shape) for key, value in first.items() if not isinstance(value, dict)},
            "positive_in_batch": int(labels.sum()),
            "negative_in_batch": int((labels == 0).sum()),
            "prompt": datasets["train"].dataset[0]["prompt"],
        }, indent=2, ensure_ascii=False))
        return

    cache_dir = PROJECT_ROOT / ".cache" / "jax"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    logging.info("host=%s devices=%s", platform.node(), jax.devices())
    run_dir = args.output_dir / args.run_name
    manager, resuming = openpi_checkpoints.initialize_checkpoint_dir(
        run_dir,
        keep_period=args.keep_period,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    filter_ = trainable_filter()
    state, state_sharding = _init_state(
        model_config, args, _optimizer(args), filter_, jax.random.key(args.seed), mesh
    )
    rng = jax.random.key(args.seed + 1)
    jax.block_until_ready(state)
    backbone_sha = parameter_tree_sha256(state.params["backbone"])
    if resuming:
        state, rng = _restore(manager, state, filter_, rng)
    start_step = int(jax.device_get(state.step))
    config_payload = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "precision": precision,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "trainable_components": ["adjustment_end_head"],
        "frozen_components": ["stage_a_backbone_all_parameters"],
        "stage_a_config_sha256": sha256_file(stage_a_config_path),
        "stage_a_backbone_parameter_tree_sha256": backbone_sha,
        "training_index_sha256": sha256_file(args.index_file),
        "manifest_sha256": sha256_file(manifest_path),
        "caption_source": index["caption_source"],
        "prompt_helper": index["prompt_helper"],
        "optimizer": {"name": "AdamW", "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "schedule": "constant"},
    }
    if jax.process_index() == 0 and not resuming:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n")

    train_loader = _train_loader(datasets["train"], args, start_step=start_step)
    val_loader = _eval_loader(datasets["val"], args)
    test_loader = _eval_loader(datasets["test"], args)
    ptrain = jax.jit(
        lambda step_rng, train_state, batch: _train_step(filter_, step_rng, train_state, batch),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    metrics_file = run_dir / "metrics.jsonl"
    recent = []
    started = time.time()
    progress = tqdm.tqdm(train_loader, initial=start_step, total=args.num_steps, desc="V5.3 adjustment_end")
    for numpy_batch in progress:
        batch = _batch_to_jax(numpy_batch, data_sharding)
        rng, step_rng = jax.random.split(rng)
        with sharding.set_mesh(mesh):
            state, info = ptrain(step_rng, state, batch)
        info = jax.device_get(info)
        recent.append(float(info["loss"]))
        step = int(jax.device_get(state.step))
        if step % args.log_interval == 0:
            payload = {"step": step, "loss": float(np.mean(recent)), "elapsed_sec": time.time() - started}
            progress.write(str(payload))
            if jax.process_index() == 0:
                with metrics_file.open("a") as stream:
                    stream.write(json.dumps(payload) + "\n")
            recent.clear()
        if step % args.eval_interval == 0:
            val_rows = _predict(state, val_loader, data_sharding)
            monitor = {"step": step, **_ranking_metrics(val_rows)}
            progress.write(json.dumps(monitor))
            if jax.process_index() == 0:
                with metrics_file.open("a") as stream:
                    stream.write(json.dumps(monitor) + "\n")
        if step % args.save_interval == 0 or step == args.num_steps:
            _save(manager, state, step, filter_, rng)

    manager.wait_until_finished()
    if int(jax.device_get(state.step)) != 4000:
        raise ValueError("Only the completed step=4000 model may be evaluated as official")
    val_rows = _predict(state, val_loader, data_sharding)
    threshold, val_metrics = _select_threshold(val_rows)
    simulation = _h30_simulation(val_rows, threshold)
    test_rows = _predict(state, test_loader, data_sharding)
    test_metrics = _threshold_metrics(test_rows, threshold)
    final = {
        "official_step": 4000,
        "adjustment_end_threshold": threshold,
        "val": val_metrics,
        "val_h30_simulation": simulation,
        "test": test_metrics,
        "h30_gate_passed": bool(simulation["actual_s0"]["gate_passed"]),
        "accepted_for_robot": bool(simulation["actual_s0"]["gate_passed"]),
        "stage_a_backbone_parameter_tree_sha256": backbone_sha,
        "caption_source": index["caption_source"],
        "prompt_helper": index["prompt_helper"],
    }
    if jax.process_index() == 0:
        head_dir = run_dir / "4000" / "head_params"
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(
                head_dir,
                delta_params(state.params, filter_).to_pure_dict(),
                force=True,
            )
        (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
        step_metadata = run_dir / "4000" / "adjustment_end_metadata.json"
        step_metadata.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    if not final["accepted_for_robot"]:
        raise ValueError("V5.3 final checkpoint failed the s=0 H30 endpoint gate")


if __name__ == "__main__":
    main()
