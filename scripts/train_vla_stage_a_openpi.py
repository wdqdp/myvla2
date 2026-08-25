#!/usr/bin/env python3
"""Train VLA Stage A action generation with OpenPI pi05 in JAX/NNX."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))

DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_v3")
DEFAULT_PROFILE_DIR = Path("/data1/outputs/vla/rotation_moderately_success_v1")
DEFAULT_INDEX_FILE = DEFAULT_PROFILE_DIR / "vla_indices_v3.json"
DEFAULT_SPLIT_FILE = DEFAULT_PROFILE_DIR / "splits.json"
DEFAULT_NORM_STATS_DIR = Path(
    "/data1/outputs/vla/assets/tactile_vla_rotation_moderately_success_v1"
)
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/vla/stage_a_action")
DEFAULT_BASE_CHECKPOINT = Path.home() / ".cache/modelscope/hub/models/hairuoliu/pi05_base/params"

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.environ.setdefault("OPENPI_DATA_HOME", "/data1/outputs/openpi_cache")
# The JAX training entrypoint never uses TensorFlow.  Prevent Transformers
# from importing an unrelated TF installation during model module discovery.
os.environ.setdefault("USE_TF", "0")

from flax import nnx
from flax.training import common_utils
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from torch.utils.data import DataLoader
import tqdm

from openpi.models import model as openpi_model
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.shared import normalize
from openpi.training import checkpoints as openpi_checkpoints
from openpi.training import sharding
from openpi.training import utils as training_utils
from openpi.training import weight_loaders
from tactile_vla.vla.artifacts import artifact_identity
from tactile_vla.vla.artifacts import assert_identity_matches
from tactile_vla.vla.artifacts import checkpoint_artifact_identity
from tactile_vla.vla.artifacts import LEGACY_DATA_PROFILE
from tactile_vla.vla.artifacts import validate_norm_stats_identity
from tactile_vla.vla.data_profiles import ROTATION_MODERATELY_SUCCESS_V1
from tactile_vla.vla.data_profiles import EXPECTED_ACTION_COUNTS
from tactile_vla.vla.index import SplitConfig
from tactile_vla.vla.index import index_payload
from tactile_vla.vla.index import load_or_create_splits
from tactile_vla.vla.index import scan_lerobot_frames
from tactile_vla.vla.index import validate_index_action_horizon
from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset
from tactile_vla.vla.openpi_bridge import TransformedTactileVLADataset
from tactile_vla.vla.openpi_bridge import build_transform
from tactile_vla.vla.openpi_bridge import collate_numpy
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.prompts import resolve_prompt_profile
from tactile_vla.vla.v4_data import ROTATION_V4
from tactile_vla.vla.v4_data import V4_TRAINING_INDEX_SCHEMA
from tactile_vla.vla.v4_data import validate_v4_index_dataset
from tactile_vla.vla.v5_phase_data import PHASE_EXPERIMENT_KIND
from tactile_vla.vla.v5_phase_data import ROTATION_PHASE_V5
from tactile_vla.vla.v5_phase_data import V5_TRAINING_INDEX_SCHEMA
from tactile_vla.vla.v5_phase_data import validate_v5_training_index
from tactile_vla.vla.v5_adjustment_data import ROTATION_PHASE_V5_ADJUSTMENT_V2
from tactile_vla.vla.v5_adjustment_data import V2_EXPERIMENT_KIND
from tactile_vla.vla.v5_adjustment_data import V2_TRAINING_INDEX_SCHEMA
from tactile_vla.vla.v5_adjustment_data import validate_v5_adjustment_training_index


PHASE_DATA_PROFILES = {ROTATION_PHASE_V5, ROTATION_PHASE_V5_ADJUSTMENT_V2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_STATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="pi05_delta_tac_rotation_moderately_v1")
    parser.add_argument("--data-profile", default=ROTATION_MODERATELY_SUCCESS_V1)
    parser.add_argument("--prompt-profile", default=MINIMAL_PROMPT_PROFILE)
    parser.add_argument("--experiment-kind")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-final", type=float, default=5e-7)
    parser.add_argument("--lr-transition-steps", type=int, default=7000)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-period", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--state-history-len", type=int, default=60)
    parser.add_argument("--state-history-dim", type=int, default=7)
    parser.add_argument("--history-hidden-dim", type=int, default=256)
    parser.add_argument("--use-state-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--paligemma-variant", default="gemma_2b_lora")
    parser.add_argument("--action-expert-variant", default="gemma_300m_lora")
    parser.add_argument("--precision", default="auto", choices=("auto", "bfloat16", "float32"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--train-lora-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--dry-run", action="store_true", help="Only build one transformed batch and print shapes.")
    return parser.parse_args()


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_v4_args(args: argparse.Namespace) -> None:
    if args.data_profile != ROTATION_V4:
        return
    if args.prompt_profile != MINIMAL_PROMPT_PROFILE:
        raise ValueError("rotation_v4 Stage A requires prompt_profile='minimal_v1'")
    if args.no_norm:
        raise ValueError("rotation_v4 Stage A requires its train-index norm stats")


def validate_v5_args(args: argparse.Namespace) -> None:
    if args.data_profile not in PHASE_DATA_PROFILES:
        return
    expected_prompt = (
        PHASE_PROMPT_PROFILE_V2
        if args.data_profile == ROTATION_PHASE_V5_ADJUSTMENT_V2
        else PHASE_PROMPT_PROFILE
    )
    expected_experiment = (
        V2_EXPERIMENT_KIND
        if args.data_profile == ROTATION_PHASE_V5_ADJUSTMENT_V2
        else PHASE_EXPERIMENT_KIND
    )
    if args.prompt_profile != expected_prompt:
        raise ValueError(f"{args.data_profile} Stage A requires prompt_profile={expected_prompt!r}")
    if args.experiment_kind != expected_experiment:
        raise ValueError(f"{args.data_profile} Stage A requires experiment_kind={expected_experiment!r}")
    if args.no_norm:
        raise ValueError(f"{args.data_profile} Stage A must reuse the V4 norm stats")


V4_STAGE_A_PROTOCOL = {
    "split": "train",
    "batch_size": 8,
    "num_steps": 15_000,
    "lr": 5e-5,
    "lr_final": 5e-7,
    "lr_transition_steps": 7_000,
    "save_interval": 1_000,
    "keep_period": 5_000,
    "action_horizon": 30,
    "action_dim": 32,
    "use_state_history": True,
    "state_history_len": 60,
    "state_history_dim": 7,
    "history_hidden_dim": 256,
    "max_token_len": 200,
    "paligemma_variant": "gemma_2b_lora",
    "action_expert_variant": "gemma_300m_lora",
    "train_lora_only": True,
    "allow_random_init": False,
}
V5_STAGE_A_PROTOCOL = {**V4_STAGE_A_PROTOCOL, "seed": 42}


def validate_v4_training_protocol(args: argparse.Namespace) -> None:
    if args.data_profile not in {ROTATION_V4, *PHASE_DATA_PROFILES}:
        return
    protocol = V5_STAGE_A_PROTOCOL if args.data_profile in PHASE_DATA_PROFILES else V4_STAGE_A_PROTOCOL
    mismatches = {
        key: {"requested": getattr(args, key), "required": expected}
        for key, expected in protocol.items()
        if getattr(args, key) != expected
    }
    requested_checkpoint = Path(str(args.checkpoint)).expanduser().resolve()
    required_checkpoint = DEFAULT_BASE_CHECKPOINT.expanduser().resolve()
    if requested_checkpoint != required_checkpoint:
        mismatches["checkpoint"] = {
            "requested": str(requested_checkpoint),
            "required": str(required_checkpoint),
        }
    if mismatches:
        raise ValueError(f"{args.data_profile} Stage A protocol mismatch: {mismatches}")


def validate_v4_resume_config(saved: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data_profile not in {ROTATION_V4, *PHASE_DATA_PROFILES}:
        return
    protocol = V5_STAGE_A_PROTOCOL if args.data_profile in PHASE_DATA_PROFILES else V4_STAGE_A_PROTOCOL
    keys = (
        *protocol,
        "data_profile",
        "prompt_profile",
        *(("experiment_kind",) if args.data_profile in PHASE_DATA_PROFILES else ()),
        "weight_decay",
        "grad_clip",
        "log_interval",
        "seed",
        "precision",
        "ema_decay",
        "fsdp_devices",
        "video_backend",
    )
    mismatches = {
        key: {"saved": saved.get(key), "requested": getattr(args, key)}
        for key in keys
        if saved.get(key) != getattr(args, key)
    }
    saved_checkpoint = Path(str(saved.get("checkpoint", ""))).expanduser().resolve()
    requested_checkpoint = Path(str(args.checkpoint)).expanduser().resolve()
    if saved_checkpoint != requested_checkpoint:
        mismatches["checkpoint"] = {
            "saved": str(saved_checkpoint),
            "requested": str(requested_checkpoint),
        }
    if mismatches:
        raise ValueError(f"{args.data_profile} Stage A resume config mismatch: {mismatches}")


def ensure_index(args: argparse.Namespace) -> dict:
    if args.index_file.exists():
        payload = json.loads(args.index_file.read_text())
        validate_index_action_horizon(payload, args.action_horizon, index_path=args.index_file)
        if args.data_profile == ROTATION_V4:
            if payload.get("schema_version") != V4_TRAINING_INDEX_SCHEMA:
                raise ValueError("rotation_v4 requires the dedicated V4 unified training index")
            validate_v4_index_dataset(payload, args.dataset_dir)
        elif args.data_profile == ROTATION_PHASE_V5:
            if payload.get("schema_version") != V5_TRAINING_INDEX_SCHEMA:
                raise ValueError("rotation_phase_v5 requires the dedicated V5 prompt training index")
            _, lookup = validate_v5_training_index(
                payload,
                index_path=args.index_file,
                dataset_dir=args.dataset_dir,
            )
            args._v5_action_phase_lookup = lookup
        elif args.data_profile == ROTATION_PHASE_V5_ADJUSTMENT_V2:
            if payload.get("schema_version") != V2_TRAINING_INDEX_SCHEMA:
                raise ValueError("rotation_phase_v5_adjustment_v2 requires its dedicated V2 index")
            _, lookup = validate_v5_adjustment_training_index(
                payload,
                index_path=args.index_file,
                dataset_dir=args.dataset_dir,
            )
            args._v5_action_phase_lookup = lookup
        return payload
    if args.data_profile != LEGACY_DATA_PROFILE:
        raise FileNotFoundError(
            f"Versioned data profile index does not exist: {args.index_file}. "
            "Run the matching versioned profile/index preparation command first."
        )
    records = scan_lerobot_frames(args.dataset_dir)
    splits = load_or_create_splits(records, args.split_file, SplitConfig(seed=args.seed))
    payload = index_payload(
        records,
        splits,
        seed=args.seed,
        negative_ratio=3.0,
        action_horizon=args.action_horizon,
    )
    payload["dataset_dir"] = str(args.dataset_dir)
    args.index_file.parent.mkdir(parents=True, exist_ok=True)
    args.index_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    validate_index_action_horizon(payload, args.action_horizon, index_path=args.index_file)
    return payload


def build_loader(
    args: argparse.Namespace,
    model_config: Pi0Config,
    payload: dict[str, Any] | None = None,
) -> DataLoader:
    payload = payload or ensure_index(args)
    indices = payload["splits"][args.split]["execution_indices"]
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    norm_stats = None if args.no_norm else normalize.load(args.norm_stats_dir)
    phase_lookup = None
    if args.data_profile in PHASE_DATA_PROFILES:
        phase_lookup = getattr(args, "_v5_action_phase_lookup", None)
        if phase_lookup is None:
            validator = (
                validate_v5_adjustment_training_index
                if args.data_profile == ROTATION_PHASE_V5_ADJUSTMENT_V2
                else validate_v5_training_index
            )
            _, phase_lookup = validator(
                payload, index_path=args.index_file, dataset_dir=args.dataset_dir
            )
            args._v5_action_phase_lookup = phase_lookup
    dataset = TactileVLAFrameDataset(
        dataset_dir=args.dataset_dir,
        indices=indices,
        stage="execution",
        action_horizon=args.action_horizon,
        state_history_len=args.state_history_len if args.use_state_history else 0,
        video_backend=args.video_backend,
        prompt_profile=args.prompt_profile,
        action_phase_by_global_index=phase_lookup,
        dataset_repo_id=(
            "tactile_vla_rotation_v4"
            if args.data_profile in {ROTATION_V4, *PHASE_DATA_PROFILES}
            else "tactile_vla"
        ),
    )
    transformed = TransformedTactileVLADataset(
        dataset,
        build_transform(model_config, norm_stats=norm_stats, use_quantile_norm=not args.no_norm),
    )
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
    return DataLoader(
        transformed,
        batch_size=args.batch_size,
        shuffle=args.split == "train",
        num_workers=args.num_workers,
        collate_fn=collate_numpy,
        drop_last=True,
        **loader_kwargs,
    )


def print_batch_shapes(batch: dict) -> None:
    for key, value in batch.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"{key}.{sub_key}: shape={np.asarray(sub_value).shape} dtype={np.asarray(sub_value).dtype}")
        else:
            print(f"{key}: shape={np.asarray(value).shape} dtype={np.asarray(value).dtype}")


def print_dry_run_inputs(loader: DataLoader, batch: dict[str, Any]) -> None:
    raw_dataset = getattr(loader.dataset, "dataset", None)
    if raw_dataset is not None:
        print(f"prompt_profile={raw_dataset.prompt_profile}")
        phase_lookup = getattr(raw_dataset, "action_phase_by_global_index", None)
        if phase_lookup:
            shown: set[str] = set()
            positions = {global_index: position for position, global_index in enumerate(raw_dataset.indices)}
            for global_index in raw_dataset.indices:
                phase_row = phase_lookup[global_index]
                phase = str(phase_row["phase"])
                if phase in shown:
                    continue
                sample = raw_dataset[positions[global_index]]
                print(
                    f"phase={phase} chunk_phase_pure={phase_row['chunk_phase_pure']} "
                    f"text_prompt={sample['prompt']}"
                )
                shown.add(phase)
                if len(shown) == 3:
                    break
        else:
            sample = raw_dataset[0]
            print(f"text_prompt={sample['prompt']}")
    print_batch_shapes(batch)


def make_lr_schedule(args: argparse.Namespace) -> optax.Schedule:
    if args.lr_final is None:
        return optax.constant_schedule(args.lr)
    return optax.join_schedules(
        [optax.constant_schedule(args.lr), optax.constant_schedule(args.lr_final)],
        [args.lr_transition_steps],
    )


def make_optimizer(args: argparse.Namespace) -> optax.GradientTransformation:
    tx = optax.adamw(
        make_lr_schedule(args),
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    return optax.chain(optax.clip_by_global_norm(args.grad_clip), tx)


def load_weights_and_validate(loader: weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )


def count_state_params(state: nnx.State) -> int:
    total = 0
    for leaf in jax.tree.leaves(state):
        value = getattr(leaf, "value", leaf)
        if hasattr(value, "size"):
            total += int(value.size)
    return total


@at.typecheck
def init_train_state(
    *,
    model_config: Pi0Config,
    trainable_filter: nnx.filterlib.Filter,
    freeze_filter: nnx.filterlib.Filter,
    tx: optax.GradientTransformation,
    ema_decay: float | None,
    weight_loader: weight_loaders.WeightLoader,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        del rng
        model = model_config.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            opt_state=tx.init(params.filter(trainable_filter)),
            tx=tx,
            ema_decay=ema_decay,
            ema_params=None if ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if isinstance(weight_loader, weight_loaders.NoOpWeightLoader):
        train_state = jax.jit(init, out_shardings=state_sharding)(init_rng)
    else:
        partial_params = load_weights_and_validate(weight_loader, train_state_shape.params.to_pure_dict())
        train_state = jax.jit(
            init,
            donate_argnums=(1,),
            in_shardings=(replicated_sharding, replicated_sharding),
            out_shardings=state_sharding,
        )(init_rng, partial_params)
    return train_state, state_sharding


@at.typecheck
def train_step(
    trainable_filter: nnx.filterlib.Filter,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[openpi_model.Observation, openpi_model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: openpi_model.BaseModel,
        rng: at.KeyArrayLike,
        observation: openpi_model.Observation,
        actions: openpi_model.Actions,
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_full_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_full_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_full_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return new_state, {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }


def batch_to_jax(batch: dict, data_sharding: jax.sharding.Sharding) -> tuple[openpi_model.Observation, openpi_model.Actions]:
    local_batch = jax.tree.map(np.asarray, batch)
    if "tokenized_prompt" in local_batch:
        local_batch["tokenized_prompt"] = np.asarray(local_batch["tokenized_prompt"], dtype=np.int32)
    if "tokenized_prompt_mask" in local_batch:
        local_batch["tokenized_prompt_mask"] = np.asarray(local_batch["tokenized_prompt_mask"], dtype=np.bool_)
    observation = Observation.from_dict(local_batch)
    actions = np.asarray(local_batch["actions"], dtype=np.float32)
    return jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), (observation, actions))


def split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


def save_state(checkpoint_manager: ocp.CheckpointManager, state: training_utils.TrainState, step: int) -> None:
    with at.disable_typechecking():
        train_state, params = split_params(state)
    checkpoint_manager.save(step, {"train_state": train_state, "params": {"params": params}})


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    step: int | None = None,
) -> training_utils.TrainState:
    with at.disable_typechecking():
        train_state, params = split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={"train_state": train_state, "params": {"params": params}},
        )
    return merge_params(restored["train_state"], restored["params"])


def write_jsonl(path: Path, payload: dict) -> None:
    if jax.process_index() != 0:
        return
    with path.open("a") as file:
        file.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    init_logging()
    jax_cache_dir = PROJECT_ROOT / ".cache" / "jax"
    jax_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))
    logging.info("Running on: %s", platform.node())
    logging.info("JAX devices: %s", jax.devices())

    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size={args.batch_size} must be divisible by jax.device_count()={jax.device_count()}.")
    if args.use_state_history and args.state_history_dim != 7:
        raise ValueError("This dataset stores 7-D puppet qpos; --state-history-dim must be 7")
    if args.train_lora_only and "lora" not in args.paligemma_variant and "lora" not in args.action_expert_variant:
        raise ValueError("--train-lora-only requires LoRA model variants.")
    args.prompt_profile = resolve_prompt_profile(args.prompt_profile)
    validate_v4_args(args)
    validate_v5_args(args)
    validate_v4_training_protocol(args)
    if args.max_frames is not None and not args.dry_run:
        raise ValueError("--max-frames is only supported for --dry-run in profile-bound training")
    if args.data_profile == ROTATION_MODERATELY_SUCCESS_V1:
        pinned = {
            "batch_size": (args.batch_size, 8),
            "num_steps": (args.num_steps, 10_000),
            "lr": (args.lr, 5e-5),
            "lr_final": (args.lr_final, 5e-7),
            "lr_transition_steps": (args.lr_transition_steps, 7_000),
            "save_interval": (args.save_interval, 1_000),
            "keep_period": (args.keep_period, 5_000),
            "action_horizon": (args.action_horizon, 30),
            "action_dim": (args.action_dim, 32),
            "state_history_len": (args.state_history_len, 60),
            "state_history_dim": (args.state_history_dim, 7),
        }
        mismatches = {
            key: {"requested": actual, "required": expected}
            for key, (actual, expected) in pinned.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(f"Versioned Stage A protocol mismatch: {mismatches}")
        if args.allow_random_init:
            raise ValueError("Versioned Stage A must initialize from the raw pi05_base checkpoint")

    precision = args.precision
    if precision == "auto":
        precision = "bfloat16" if jax.default_backend() in {"gpu", "tpu"} else "float32"

    model_config = Pi0Config(
        dtype=precision,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        pi05=True,
        use_state_history=args.use_state_history,
        state_history_len=args.state_history_len,
        state_history_dim=args.state_history_dim,
        history_hidden_dim=args.history_hidden_dim,
        pytorch_compile_mode=None,
    )
    index = ensure_index(args)
    identity = artifact_identity(
        index,
        index_path=args.index_file,
        prompt_profile=args.prompt_profile,
        requested_data_profile=args.data_profile,
    )
    if (
        args.data_profile == ROTATION_MODERATELY_SUCCESS_V1
        and identity["action_indices_identity"]["all"]["count"]
        != EXPECTED_ACTION_COUNTS["all"]
    ):
        raise ValueError(
            "Versioned Stage A index does not contain the required 98,233 action starts"
        )
    if not args.no_norm and args.data_profile != LEGACY_DATA_PROFILE:
        norm_summary = validate_norm_stats_identity(
            args.norm_stats_dir / "summary.json",
            identity,
            context="Stage A norm stats",
        )
        if args.data_profile == ROTATION_V4:
            identity["norm_stats_sha256"] = norm_summary["norm_stats_sha256"]
    loader = build_loader(args, model_config, index)
    first_batch = next(iter(loader))
    if args.dry_run:
        print_dry_run_inputs(loader, first_batch)
        return

    run_dir = args.output_dir / args.run_name
    checkpoint_manager, resuming = openpi_checkpoints.initialize_checkpoint_dir(
        run_dir,
        keep_period=args.keep_period,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    if resuming:
        saved_config = json.loads((run_dir / "config.json").read_text())
        assert_identity_matches(
            checkpoint_artifact_identity(saved_config),
            identity,
            context="Stage A resume",
        )
        validate_v4_resume_config(saved_config, args)
    if jax.process_index() == 0 and not resuming:
        config_payload = vars(args) | {
            "artifact_identity": identity,
            "data_config_hash": identity["data_config_hash"],
            "action_frame_manifest_hash": identity["action_frame_manifest_hash"],
            "action_indices_identity": identity["action_indices_identity"],
            "index_sha256": identity["index_sha256"],
        }
        (run_dir / "config.json").write_text(
            json.dumps(config_payload, indent=2, default=str, ensure_ascii=False) + "\n"
        )

    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    freeze_filter = model_config.get_freeze_filter() if args.train_lora_only else nnx.Nothing
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    tx = make_optimizer(args)
    loader_config: weight_loaders.WeightLoader
    if args.allow_random_init:
        loader_config = weight_loaders.NoOpWeightLoader()
    else:
        missing_regex = ".*lora.*"
        if args.use_state_history:
            missing_regex = r"(?:.*lora.*|history_.*)"
        loader_config = weight_loaders.CheckpointWeightLoader(
            str(args.checkpoint),
            missing_regex=missing_regex,
        )

    train_state, train_state_sharding = init_train_state(
        model_config=model_config,
        trainable_filter=trainable_filter,
        freeze_filter=freeze_filter,
        tx=tx,
        ema_decay=args.ema_decay,
        weight_loader=loader_config,
        init_rng=init_rng,
        mesh=mesh,
        resume=resuming,
    )
    if resuming:
        train_state = restore_state(checkpoint_manager, train_state)
    jax.block_until_ready(train_state)

    trainable_params = count_state_params(train_state.params.filter(trainable_filter))
    total_params = count_state_params(train_state.params)
    logging.info("precision=%s trainable_params=%d total_params=%d", precision, trainable_params, total_params)

    ptrain_step = jax.jit(
        lambda rng, state, batch: train_step(trainable_filter, rng, state, batch),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    metrics_file = run_dir / "metrics.jsonl"
    data_iter = iter(loader)
    start_step = int(jax.device_get(train_state.step))
    start_time = time.time()
    infos: list[dict[str, at.Array]] = []

    pbar = tqdm.trange(start_step, args.num_steps, initial=start_step, total=args.num_steps, desc="Stage A action")
    for _ in pbar:
        try:
            numpy_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            numpy_batch = next(data_iter)

        batch = batch_to_jax(numpy_batch, data_sharding)
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        step = int(jax.device_get(train_state.step))

        if step % args.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            payload = {
                "step": step,
                "elapsed_sec": time.time() - start_time,
                **{key: float(value) for key, value in reduced_info.items()},
            }
            pbar.write(str(payload))
            write_jsonl(metrics_file, payload)
            infos.clear()

        if step % args.save_interval == 0 or step == args.num_steps:
            save_state(checkpoint_manager, train_state, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
