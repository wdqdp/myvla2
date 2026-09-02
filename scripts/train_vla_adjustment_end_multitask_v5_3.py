#!/usr/bin/env python3
"""Jointly train V5.2 action replay and the V5.3 adjustment-end classifier."""

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

warnings.filterwarnings(
    "ignore",
    message=r"The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
    module=r"torchvision\.io\._video_deprecation_warning",
)

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
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
from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset
from tactile_vla.vla.openpi_bridge import TransformedTactileVLADataset
from tactile_vla.vla.stage_b_v3_checkpoint import cast_frozen_params
from tactile_vla.vla.v5_adjustment_data import validate_v5_adjustment_training_index
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import delta_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import merge_delta_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import MULTITASK_CHECKPOINT_FORMAT
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import multitask_trainable_filter
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import resume_state
from tactile_vla.vla.v5_3_adjustment_end_data import AdjustmentEndManifestDataset
from tactile_vla.vla.v5_3_adjustment_end_data import ADJUSTMENT_END_END_OFFSET
from tactile_vla.vla.v5_3_adjustment_end_data import ADJUSTMENT_END_START_OFFSET
from tactile_vla.vla.v5_3_adjustment_end_data import DATA_PROFILE
from tactile_vla.vla.v5_3_adjustment_end_data import DeterministicOneToThreeBatchSampler
from tactile_vla.vla.v5_3_adjustment_end_data import EXPERIMENT_KIND
from tactile_vla.vla.v5_3_adjustment_end_data import LABEL_POLICY
from tactile_vla.vla.v5_3_adjustment_end_data import TransformedAdjustmentEndDataset
from tactile_vla.vla.v5_3_adjustment_end_data import load_indexed_manifest_rows
from tactile_vla.vla.v5_3_adjustment_end_evaluation import ranking_metrics
from tactile_vla.vla.v5_3_adjustment_end_evaluation import relative_probability_profile
from tactile_vla.vla.v5_3_adjustment_end_evaluation import select_max_recall_under_early_fpr
from tactile_vla.vla.v5_3_adjustment_end_evaluation import threshold_metrics
from tactile_vla.vla.v5_3_adjustment_end_model import AdjustmentEndModel
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_PROMPT_PROFILE
from tactile_vla.vla.v5_3_phase_change import h30_endpoint_indices


DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_rotation_v4")
DEFAULT_ACTION_INDEX = Path(
    "/data1/outputs/vla/rotation_v5_adjustment_v2/v5_prompt_training_index.json"
)
DEFAULT_CLASSIFICATION_INDEX = Path(
    "/data1/outputs/vla/rotation_v5_adjustment_end_v2/adjustment_end_training_index.json"
)
DEFAULT_NORM_DIR = Path("/data1/outputs/vla/assets/tactile_vla_rotation_v4")
DEFAULT_BACKBONE = Path(
    "/data1/outputs/vla/stage_a_action/pi05_delta_tac_rotation_phase_v5_2_1/15000"
)
DEFAULT_OUTPUT = Path("/data1/outputs/vla/adjustment_end_v5_3")
ACTION_DATA_PROFILE = "rotation_phase_v5_adjustment_v2"
ACTION_PROMPT_PROFILE = "phase_v2"
ACTION_EXPERIMENT_KIND = "phase_prompt_h30_terminal_hold"
TASK_CYCLE = ("action", "adjustment_end")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--action-index-file", type=Path, default=DEFAULT_ACTION_INDEX)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_CLASSIFICATION_INDEX)
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument("--stage-a-checkpoint", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-name",
        default="pi05_adjustment_end_rotation_v5_3_multitask_r10_p5",
    )
    parser.add_argument("--data-profile", default=DATA_PROFILE)
    parser.add_argument("--prompt-profile", default=PHASE_CHANGE_PROMPT_PROFILE)
    parser.add_argument("--experiment-kind", default="adjustment_end_action_multitask_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-period", type=int, default=1000)
    parser.add_argument("--action-max-token-len", type=int, default=200)
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
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_protocol(args: argparse.Namespace) -> None:
    required = {
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": "adjustment_end_action_multitask_v1",
        "batch_size": 8,
        "num_workers": 0,
        "num_steps": 8000,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "seed": 42,
        "save_interval": 1000,
        "action_max_token_len": 200,
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
        raise ValueError(f"V5.3 multitask fixed protocol mismatch: {mismatches}")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be a positive integer")


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
        "data_profile": ACTION_DATA_PROFILE,
        "prompt_profile": ACTION_PROMPT_PROFILE,
        "experiment_kind": ACTION_EXPERIMENT_KIND,
        "num_steps": 15000,
        "action_horizon": 30,
        "action_dim": 32,
        "state_history_len": 60,
        "state_history_dim": 7,
        "seed": 42,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in required.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V5.3 Stage A checkpoint config mismatch: {mismatches}")
    return config_path, config


class DeterministicUniformBatchSampler:
    """Uniform action replay with deterministic resume by update index."""

    def __init__(self, *, size: int, batch_size: int, num_batches: int, seed: int, start_batch: int):
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        if self.size < self.batch_size or self.batch_size <= 0:
            raise ValueError("Action replay dataset is smaller than one batch")
        if not 0 <= self.start_batch <= self.num_batches:
            raise ValueError("Invalid action replay start batch")

    def __len__(self) -> int:
        return self.num_batches - self.start_batch

    def __iter__(self):
        emitted = 0
        epoch = 0
        while emitted < self.num_batches:
            permutation = np.random.default_rng(self.seed + epoch).permutation(self.size)
            full_batches = self.size // self.batch_size
            for batch_index in range(full_batches):
                if emitted >= self.num_batches:
                    break
                start = batch_index * self.batch_size
                if emitted >= self.start_batch:
                    yield [int(value) for value in permutation[start : start + self.batch_size]]
                emitted += 1
            epoch += 1


def _model_config(args: argparse.Namespace, *, precision: str, max_token_len: int) -> Pi0Config:
    return Pi0Config(
        dtype=precision,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=max_token_len,
        pi05=True,
        use_state_history=True,
        state_history_len=args.state_history_len,
        state_history_dim=args.state_history_dim,
        history_hidden_dim=args.history_hidden_dim,
        pytorch_compile_mode=None,
    )


def _build_datasets(
    args: argparse.Namespace,
    *,
    action_config: Pi0Config,
    phase_config: Pi0Config,
    action_index: dict[str, Any],
    classification_index: dict[str, Any],
    manifest: dict[int, dict[str, Any]],
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    _, action_phase_lookup = validate_v5_adjustment_training_index(
        action_index,
        index_path=args.action_index_file,
        dataset_dir=args.dataset_dir,
    )
    if action_index["selection_hash"] != classification_index["selection_hash"]:
        raise ValueError("Action and adjustment_end indices use different selections")
    norm_stats = normalize.load(args.norm_stats_dir)
    shared = LeRobotDataset(
        "tactile_vla_rotation_v4",
        root=args.dataset_dir,
        delta_timestamps={
            "observation.state": [step / 30.0 for step in range(-59, 1)],
            "action": [step / 30.0 for step in range(args.action_horizon)],
        },
        download_videos=False,
        video_backend=args.video_backend,
    )
    action_raw = TactileVLAFrameDataset(
        dataset_dir=args.dataset_dir,
        indices=action_index["splits"]["train"]["execution_indices"],
        stage="execution",
        action_horizon=args.action_horizon,
        state_history_len=args.state_history_len,
        video_backend=args.video_backend,
        prompt_profile=ACTION_PROMPT_PROFILE,
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
            state_history_len=args.state_history_len,
        )
        classification[split] = TransformedAdjustmentEndDataset(raw, phase_transform)
    return action_dataset, classification


def _action_train_loader(dataset, args: argparse.Namespace, *, start_batch: int) -> DataLoader:
    sampler = DeterministicUniformBatchSampler(
        size=len(dataset),
        batch_size=args.batch_size,
        num_batches=args.num_steps // 2,
        seed=args.seed,
        start_batch=start_batch,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_numpy,
    )


def _classification_train_loader(dataset, args: argparse.Namespace, *, start_batch: int) -> DataLoader:
    labels = [bool(dataset.dataset.manifest[row]["adjustment_end"]) for row in dataset.dataset.row_indices]
    sampler = DeterministicOneToThreeBatchSampler(
        labels=labels,
        num_batches=args.num_steps // 2,
        seed=args.seed,
        start_batch=start_batch,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_numpy,
    )


def _eval_loader(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_numpy,
        drop_last=False,
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


def _train_action_step(filter_, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, actions = batch

    def loss_fn(module):
        return jnp.mean(module.backbone.compute_loss(rng, observation, actions, train=True))

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filter_))(model)
    state = _apply_gradients(state, model, grads, filter_)
    return state, {"loss": loss, "grad_norm": optax.global_norm(grads)}


def _train_classification_step(filter_, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, labels = batch

    def loss_fn(module):
        logits = module.adjustment_end_logits(observation, rng=rng, train=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filter_))(model)
    state = _apply_gradients(state, model, grads, filter_)
    return state, {"loss": loss, "grad_norm": optax.global_norm(grads)}


def _observation(local: dict[str, Any]) -> Observation:
    local["tokenized_prompt"] = np.asarray(local["tokenized_prompt"], dtype=np.int32)
    local["tokenized_prompt_mask"] = np.asarray(local["tokenized_prompt_mask"], dtype=np.bool_)
    return Observation.from_dict(local)


def _batch_to_jax(batch, task: str, data_sharding):
    local = jax.tree.map(np.asarray, batch)
    observation = _observation(local)
    target = (
        np.asarray(local["actions"], dtype=np.float32)
        if task == "action"
        else np.asarray(local["adjustment_end_label"], dtype=np.int32)
    )
    return jax.tree.map(
        lambda value: jax.make_array_from_process_local_data(data_sharding, value),
        (observation, target),
    )


def _predict(state, loader, data_sharding) -> list[dict[str, Any]]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    infer = nnx_utils.module_jit(model.adjustment_end_logits)
    output = []
    for raw in loader:
        observation, labels = _batch_to_jax(raw, "adjustment_end", data_sharding)
        probabilities = np.asarray(jax.device_get(jax.nn.softmax(infer(observation), axis=-1)))[:, 1]
        labels_np = np.asarray(jax.device_get(labels))
        for offset in range(len(labels_np)):
            output.append(
                {
                    "label": int(labels_np[offset]),
                    "probability": float(probabilities[offset]),
                    "episode_id": int(np.asarray(raw["episode_id"])[offset]),
                    "frame_index": int(np.asarray(raw["frame_index"])[offset]),
                    "rexecution_frame": int(np.asarray(raw["rexecution_frame"])[offset]),
                }
            )
    return output


def _h30_simulation(rows, threshold):
    by_episode: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], {})[row["frame_index"]] = row
    offsets = {}
    for start in range(30):
        counts = {"transition": 0, "early": 0, "miss": 0}
        for episode_rows in by_episode.values():
            rexecution_values = {int(row["rexecution_frame"]) for row in episode_rows.values()}
            if len(rexecution_values) != 1:
                raise ValueError("H30 simulation requires one explicit R per episode")
            rexecution = rexecution_values.pop()
            endpoints = h30_endpoint_indices(
                start_offset=start,
                stop_frame=rexecution + ADJUSTMENT_END_END_OFFSET,
            )
            triggers = [
                frame
                for frame in endpoints
                if frame in episode_rows and episode_rows[frame]["probability"] >= threshold
            ]
            if not triggers:
                counts["miss"] += 1
            elif triggers[0] < rexecution + ADJUSTMENT_END_START_OFFSET:
                counts["early"] += 1
            elif triggers[0] <= rexecution + ADJUSTMENT_END_END_OFFSET:
                counts["transition"] += 1
            else:
                counts["miss"] += 1
        total = len(by_episode)
        offsets[str(start)] = {
            **counts,
            "episode_count": total,
            "transition_recall": counts["transition"] / total,
            "early_transition_episode_rate": counts["early"] / total,
            "miss_rate": counts["miss"] / total,
        }
    return {
        "diagnostic_only": True,
        "acceptance_gate": None,
        "start_offsets": offsets,
        "actual_s0": offsets["0"],
    }


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


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _export_delta(path: Path, state, filter_, *, force: bool = False) -> None:
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, delta_params(state.params, filter_).to_pure_dict(), force=force)


def _evaluate_final(
    *,
    state,
    filter_,
    run_dir: Path,
    step: int,
    loaders,
    data_sharding,
    stage_a_backbone_sha: str,
    classification_index: dict[str, Any],
) -> dict[str, Any]:
    step_dir = run_dir / str(step)
    val_rows = _predict(state, loaders["val"], data_sharding)
    selection = select_max_recall_under_early_fpr(val_rows)
    selected = selection["selected"]
    threshold = float(selected["threshold"])
    test_rows = _predict(state, loaders["test"], data_sharding)
    trained_backbone_sha = parameter_tree_sha256(state.params["backbone"])
    final = {
        "official_step": step,
        "checkpoint_format": MULTITASK_CHECKPOINT_FORMAT,
        "adjustment_end_threshold": threshold,
        "threshold_policy": selection["threshold_policy"],
        "threshold_requirements": {
            "maximum_early_false_positive_rate": 0.01,
            "minimum_recall": None,
        },
        "validation_operating_point": selected,
        "validation_threshold_search": selection,
        "validation_relative_probability_profile": relative_probability_profile(val_rows),
        "validation_h30_simulation": _h30_simulation(val_rows, threshold),
        "test": threshold_metrics(test_rows, threshold),
        "test_relative_probability_profile": relative_probability_profile(test_rows),
        "acceptance_gates": [],
        "action_regression_gate": None,
        "accepted_for_robot": True,
        "experimental_override": False,
        "stage_a_backbone_parameter_tree_sha256": stage_a_backbone_sha,
        "trained_backbone_parameter_tree_sha256": trained_backbone_sha,
        "caption_source": classification_index["caption_source"],
        "prompt_helper": classification_index["prompt_helper"],
        "label_policy": classification_index["label_policy"],
    }
    _write_predictions(step_dir / "val_predictions.jsonl", val_rows)
    _write_predictions(step_dir / "test_predictions.jsonl", test_rows)
    (step_dir / "relative_probability_profile.json").write_text(
        json.dumps(
            {
                "validation": final["validation_relative_probability_profile"],
                "test": final["test_relative_probability_profile"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    _export_delta(step_dir / "delta_params", state, filter_, force=True)
    (step_dir / "adjustment_end_metadata.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False) + "\n"
    )
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False) + "\n"
    )
    return final


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    _validate_protocol(args)
    if args.batch_size % jax.device_count():
        raise ValueError("batch size must be divisible by JAX device count")
    classification_index = _load_json(args.index_file)
    if (
        classification_index.get("data_profile") != DATA_PROFILE
        or classification_index.get("prompt_profile") != PHASE_CHANGE_PROMPT_PROFILE
        or classification_index.get("experiment_kind") != EXPERIMENT_KIND
    ):
        raise ValueError("V5.3 classification index header mismatch")
    manifest_path = Path(str(classification_index["manifest_file"]))
    manifest = load_indexed_manifest_rows(index=classification_index, manifest_path=manifest_path)
    action_index = _load_json(args.action_index_file)
    stage_a_config_path, _ = _validate_stage_a(args.stage_a_checkpoint)
    if classification_index["source_files"]["backbone_config"]["sha256"] != sha256_file(
        stage_a_config_path
    ):
        raise ValueError("V5.3 data index references a different Stage A config")

    precision = args.precision
    if precision == "auto":
        precision = "bfloat16" if jax.default_backend() in {"gpu", "tpu"} else "float32"
    action_config = _model_config(args, precision=precision, max_token_len=args.action_max_token_len)
    phase_config = _model_config(
        args,
        precision=precision,
        max_token_len=args.phase_change_max_token_len,
    )
    action_dataset, classification_datasets = _build_datasets(
        args,
        action_config=action_config,
        phase_config=phase_config,
        action_index=action_index,
        classification_index=classification_index,
        manifest=manifest,
    )
    if args.dry_run:
        action = next(iter(_action_train_loader(action_dataset, args, start_batch=0)))
        classification = next(
            iter(_classification_train_loader(classification_datasets["train"], args, start_batch=0))
        )
        print(
            json.dumps(
                {
                    "task_cycle": list(TASK_CYCLE),
                    "updates_per_task": args.num_steps // 2,
                    "action_batch": {
                        key: list(np.asarray(value).shape)
                        for key, value in action.items()
                        if not isinstance(value, dict)
                    },
                    "classification_batch": {
                        key: list(np.asarray(value).shape)
                        for key, value in classification.items()
                        if not isinstance(value, dict)
                    },
                    "classification_positive_count": int(
                        np.asarray(classification["adjustment_end_label"]).sum()
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
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
        enable_async_checkpointing=False,
    )
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    filter_ = multitask_trainable_filter()
    state, state_sharding = _init_state(
        action_config,
        args,
        _optimizer(args),
        filter_,
        jax.random.key(args.seed),
        mesh,
    )
    rng = jax.random.key(args.seed + 1)
    jax.block_until_ready(state)
    stage_a_backbone_sha = parameter_tree_sha256(state.params["backbone"])
    if resuming:
        saved_config = _load_json(run_dir / "config.json")
        if saved_config.get("checkpoint_format") != MULTITASK_CHECKPOINT_FORMAT:
            raise ValueError("Cannot resume a non-multitask V5.3 checkpoint")
        resume_mismatches = {
            key: (saved_config.get(key), expected)
            for key, expected in {
                "data_profile": DATA_PROFILE,
                "label_policy": LABEL_POLICY,
                "training_index_sha256": sha256_file(args.index_file),
            }.items()
            if saved_config.get(key) != expected
        }
        if resume_mismatches:
            raise ValueError(
                "Cannot resume a checkpoint trained with a different adjustment_end label set: "
                f"{resume_mismatches}"
            )
        state, rng = _restore(manager, state, filter_, rng)
    start_step = int(jax.device_get(state.step))
    if start_step % 2:
        raise ValueError("V5.3 multitask resume checkpoint must end on a complete two-task cycle")
    if args.eval_only and start_step != args.num_steps:
        raise ValueError(f"--eval-only requires restored step {args.num_steps}, got {start_step}")

    config_payload = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "precision": precision,
        "checkpoint_format": MULTITASK_CHECKPOINT_FORMAT,
        "task_cycle": list(TASK_CYCLE),
        "updates_per_task": args.num_steps // 2,
        "loss_weights": {"action": 1.0, "adjustment_end": 1.0},
        "classification_sampling_ratio": {"positive": 1, "negative": 3},
        "label_policy": LABEL_POLICY,
        "trainable_components": ["paligemma_lora", "adjustment_end_head"],
        "frozen_components": [
            "action_expert_all_parameters",
            "paligemma_non_lora",
            "state_history_encoder",
            "action_projection_layers",
        ],
        "action_regression_gate": None,
        "minimum_recall_gate": None,
        "threshold_policy": "max_recall_subject_to_early_fpr_lte_0_01",
        "stage_a_config_sha256": sha256_file(stage_a_config_path),
        "stage_a_backbone_parameter_tree_sha256": stage_a_backbone_sha,
        "action_index_sha256": sha256_file(args.action_index_file),
        "training_index_sha256": sha256_file(args.index_file),
        "manifest_sha256": sha256_file(manifest_path),
        "caption_source": classification_index["caption_source"],
        "prompt_helper": classification_index["prompt_helper"],
        "optimizer": {
            "name": "AdamW",
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
            "schedule": "constant",
        },
    }
    if jax.process_index() == 0 and not resuming:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n"
        )

    val_loader = _eval_loader(classification_datasets["val"], args)
    test_loader = _eval_loader(classification_datasets["test"], args)
    loaders = {"val": val_loader, "test": test_loader}
    if args.eval_only:
        if jax.process_index() == 0:
            final = _evaluate_final(
                state=state,
                filter_=filter_,
                run_dir=run_dir,
                step=start_step,
                loaders=loaders,
                data_sharding=data_sharding,
                stage_a_backbone_sha=stage_a_backbone_sha,
                classification_index=classification_index,
            )
            print(json.dumps(final, indent=2, ensure_ascii=False))
        return

    completed_per_task = start_step // 2
    action_loader = _action_train_loader(action_dataset, args, start_batch=completed_per_task)
    classification_loader = _classification_train_loader(
        classification_datasets["train"], args, start_batch=completed_per_task
    )
    iterators = {"action": iter(action_loader), "adjustment_end": iter(classification_loader)}
    paction = jax.jit(
        lambda step_rng, train_state, batch: _train_action_step(
            filter_, step_rng, train_state, batch
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    pclassification = jax.jit(
        lambda step_rng, train_state, batch: _train_classification_step(
            filter_, step_rng, train_state, batch
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    train_functions = {"action": paction, "adjustment_end": pclassification}
    metrics_file = run_dir / "metrics.jsonl"
    recent = {task: [] for task in TASK_CYCLE}
    started = time.time()
    progress = tqdm.trange(
        start_step,
        args.num_steps,
        initial=start_step,
        total=args.num_steps,
        desc="V5.3 action+adjustment_end",
    )
    for _ in progress:
        current_step = int(jax.device_get(state.step))
        task = TASK_CYCLE[current_step % 2]
        numpy_batch = next(iterators[task])
        batch = _batch_to_jax(numpy_batch, task, data_sharding)
        rng, step_rng = jax.random.split(rng)
        with sharding.set_mesh(mesh):
            state, info = train_functions[task](step_rng, state, batch)
        info = jax.device_get(info)
        recent[task].append(float(info["loss"]))
        step = int(jax.device_get(state.step))
        if step % args.log_interval == 0:
            payload = {
                "step": step,
                "task": task,
                "elapsed_sec": time.time() - started,
                **{
                    f"{name}_loss": float(np.mean(values)) if values else None
                    for name, values in recent.items()
                },
            }
            progress.write(str(payload))
            if jax.process_index() == 0:
                with metrics_file.open("a") as stream:
                    stream.write(json.dumps(payload) + "\n")
            recent = {name: [] for name in TASK_CYCLE}
        if step % args.eval_interval == 0:
            val_rows = _predict(state, val_loader, data_sharding)
            monitor = {"step": step, **ranking_metrics(val_rows)}
            progress.write(json.dumps(monitor))
            if jax.process_index() == 0:
                with metrics_file.open("a") as stream:
                    stream.write(json.dumps(monitor) + "\n")
        if step % args.save_interval == 0 or step == args.num_steps:
            _save(manager, state, step, filter_, rng)
            manager.wait_until_finished()

    manager.wait_until_finished()
    if int(jax.device_get(state.step)) != args.num_steps:
        raise ValueError(f"Expected completed step {args.num_steps}")
    if jax.process_index() == 0:
        final = _evaluate_final(
            state=state,
            filter_=filter_,
            run_dir=run_dir,
            step=args.num_steps,
            loaders=loaders,
            data_sharding=data_sharding,
            stage_a_backbone_sha=stage_a_backbone_sha,
            classification_index=classification_index,
        )
        logging.info(
            "Selected threshold %.8f: val recall=%.4f early FPR=%.4f",
            final["adjustment_end_threshold"],
            final["validation_operating_point"]["recall"],
            final["validation_operating_point"]["early_false_positive_rate"],
        )


if __name__ == "__main__":
    main()
