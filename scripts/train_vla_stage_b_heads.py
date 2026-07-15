#!/usr/bin/env python3
"""Train VLA Stage B auxiliary heads with a JAX/NNX pi05 backbone."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))

DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla")
DEFAULT_INDEX_FILE = Path("/data1/outputs/vla/indices/vla_indices_h30_state_memory.json")
DEFAULT_SPLIT_FILE = Path("/data1/outputs/vla/indices/splits_h30_state_memory.json")
DEFAULT_NORM_STATS_DIR = Path("/data1/outputs/vla/assets/tactile_vla_h30_state_memory")
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/vla/stage_b_heads")
DEFAULT_BACKBONE_CHECKPOINT = Path("/data1/outputs/vla/stage_a_action/pi05_delta_tac_h30_state_memory/10000")

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.environ.setdefault("OPENPI_DATA_HOME", "/data1/outputs/openpi_cache")

from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader
import tqdm

from openpi.models import gemma as openpi_gemma
from openpi.models import model as openpi_model
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.shared import normalize
from openpi.training import sharding
from openpi.training import weight_loaders
from tactile_vla.common.metrics import classification_report
from tactile_vla.vla import labels as vla_labels
from tactile_vla.vla import stage_b_jax
from tactile_vla.vla.index import SplitConfig
from tactile_vla.vla.index import index_payload
from tactile_vla.vla.index import load_or_create_splits
from tactile_vla.vla.index import scan_lerobot_frames
from tactile_vla.vla.index import validate_index_action_horizon
from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset
from tactile_vla.vla.openpi_bridge import TransformedTactileVLADataset
from tactile_vla.vla.openpi_bridge import build_transform
from tactile_vla.vla.openpi_bridge import collate_numpy


@struct.dataclass
class StageBState:
    step: jax.Array
    backbone_params: nnx.State
    head_params: dict[str, Any]
    opt_state: optax.OptState
    backbone_def: Any = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_STATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="pi05_heads_jax_h30_state_memory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    parser.add_argument(
        "--backbone-checkpoint",
        default=str(DEFAULT_BACKBONE_CHECKPOINT),
        help="OpenPI JAX params directory, or a Stage A step directory.",
    )
    parser.add_argument("--allow-random-backbone", action="store_true")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--need-loss-weight", type=float, default=1.0)
    parser.add_argument("--failure-loss-weight", type=float, default=1.0)
    parser.add_argument("--plan-loss-weight", type=float, default=1.0)
    parser.add_argument("--reasoning-augment-after-frames", type=int, default=10)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--max-status-frames", type=int)
    parser.add_argument("--max-reasoning-frames", type=int)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--dry-run", action="store_true", help="Only build one transformed status/reasoning batch.")
    return parser.parse_args()


def ensure_index(args: argparse.Namespace) -> dict:
    if args.index_file.exists():
        payload = json.loads(args.index_file.read_text())
        validate_index_action_horizon(payload, args.action_horizon, index_path=args.index_file)
        return payload
    records = scan_lerobot_frames(args.dataset_dir)
    splits = load_or_create_splits(records, args.split_file, SplitConfig(seed=args.seed))
    payload = index_payload(
        records,
        splits,
        seed=args.seed,
        negative_ratio=3.0,
        reasoning_augment_after_frames=args.reasoning_augment_after_frames,
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
    *,
    split: str,
    stage: str,
    shuffle: bool,
) -> DataLoader:
    payload = ensure_index(args)
    key = "status_indices" if stage == "status" else "reasoning_indices"
    indices = payload["splits"][split][key]
    reasoning_source_indices = None
    if stage == "reasoning":
        reasoning_source_indices = payload["splits"][split].get("reasoning_source_indices", indices)
    max_frames = args.max_status_frames if stage == "status" else args.max_reasoning_frames
    if max_frames is not None:
        indices = indices[:max_frames]
        if reasoning_source_indices is not None:
            reasoning_source_indices = reasoning_source_indices[:max_frames]
    norm_stats = None if args.no_norm else normalize.load(args.norm_stats_dir)
    dataset = TactileVLAFrameDataset(
        dataset_dir=args.dataset_dir,
        indices=indices,
        stage=stage,  # type: ignore[arg-type]
        reasoning_source_indices=reasoning_source_indices,
        action_horizon=args.action_horizon,
        state_history_len=args.state_history_len if args.use_state_history else 0,
        video_backend=args.video_backend,
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
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_numpy,
        drop_last=shuffle or jax.device_count() > 1,
        **loader_kwargs,
    )


def print_batch_shapes(name: str, batch: dict) -> None:
    print(f"[{name}]")
    for key, value in batch.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"{key}.{sub_key}: shape={np.asarray(sub_value).shape} dtype={np.asarray(sub_value).dtype}")
        else:
            print(f"{key}: shape={np.asarray(value).shape} dtype={np.asarray(value).dtype}")


def resolve_backbone_checkpoint(value: str | None) -> str | None:
    if value is None or value.startswith("gs://"):
        return value
    path = Path(value)
    if (path / "params").exists():
        return str(path / "params")
    return str(path)


def make_optimizer(args: argparse.Namespace) -> optax.GradientTransformation:
    tx = optax.adamw(
        args.lr,
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


def count_params(tree: Any) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        value = getattr(leaf, "value", leaf)
        if hasattr(value, "size"):
            total += int(value.size)
    return total


def init_stage_b_state(
    *,
    model_config: Pi0Config,
    head_config: stage_b_jax.AuxiliaryHeadConfig,
    tx: optax.GradientTransformation,
    weight_loader: weight_loaders.WeightLoader,
    init_rng: jax.Array,
    mesh: jax.sharding.Mesh,
) -> tuple[StageBState, Any]:
    input_dim = openpi_gemma.get_config(model_config.paligemma_variant).width

    def init(rng: jax.Array, partial_params: at.Params | None = None) -> StageBState:
        backbone_rng, head_rng = jax.random.split(rng)
        backbone = model_config.create(backbone_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(backbone)
            state.replace_by_pure_dict(partial_params)
            backbone = nnx.merge(graphdef, state)
        graphdef, backbone_params = nnx.split(backbone)
        head_params = stage_b_jax.init_head_params(head_rng, input_dim=input_dim, config=head_config)
        return StageBState(
            step=jnp.array(0, dtype=jnp.int32),
            backbone_params=backbone_params,
            backbone_def=graphdef,
            head_params=head_params,
            opt_state=tx.init(head_params),
            tx=tx,
        )

    state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(state_shape, mesh, log=True)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    if isinstance(weight_loader, weight_loaders.NoOpWeightLoader):
        state = jax.jit(init, out_shardings=state_sharding)(init_rng)
    else:
        partial_params = load_weights_and_validate(weight_loader, state_shape.backbone_params.to_pure_dict())
        state = jax.jit(
            init,
            donate_argnums=(1,),
            in_shardings=(replicated_sharding, replicated_sharding),
            out_shardings=state_sharding,
        )(init_rng, partial_params)
    return state, state_sharding


def batch_to_jax(batch: dict, data_sharding: jax.sharding.Sharding) -> tuple[openpi_model.Observation, dict[str, jax.Array]]:
    local = jax.tree.map(np.asarray, batch)
    local["tokenized_prompt"] = np.asarray(local["tokenized_prompt"], dtype=np.int32)
    local["tokenized_prompt_mask"] = np.asarray(local["tokenized_prompt_mask"], dtype=np.bool_)
    observation = Observation.from_dict(local)
    targets = {
        "need": np.asarray(local["need_recovery_label"], dtype=np.int32),
        "failure": np.asarray(local["failure_reason_label"], dtype=np.int32),
        "failure_mask": np.asarray(local["failure_reason_mask"], dtype=np.bool_),
        "plan": np.asarray(local["recovery_plan_label"], dtype=np.int32),
        "plan_mask": np.asarray(local["recovery_plan_mask"], dtype=np.bool_),
    }
    return jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), (observation, targets))


def masked_integer_ce(logits: jax.Array, labels: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(jnp.float32)
    safe_labels = jnp.where(mask > 0, labels, 0)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, safe_labels)
    return jnp.sum(losses * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def train_step(
    mode: str,
    head_config: stage_b_jax.AuxiliaryHeadConfig,
    loss_weights: dict[str, float],
    rng: jax.Array,
    state: StageBState,
    batch: tuple[openpi_model.Observation, dict[str, jax.Array]],
) -> tuple[StageBState, dict[str, jax.Array]]:
    observation, targets = batch
    step_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(head_params):
        backbone = nnx.merge(state.backbone_def, state.backbone_params)
        outputs = stage_b_jax.forward(
            backbone,
            head_params,
            observation,
            config=head_config,
            rng=step_rng,
            train=True,
        )
        need_loss = optax.softmax_cross_entropy_with_integer_labels(outputs["need_recovery"], targets["need"]).mean()
        failure_loss = masked_integer_ce(outputs["failure_reason"], targets["failure"], targets["failure_mask"])
        plan_loss = masked_integer_ce(outputs["recovery_plan"], targets["plan"], targets["plan_mask"])
        if mode == "status":
            loss = loss_weights["need"] * need_loss + loss_weights["failure"] * failure_loss
        elif mode == "reasoning":
            loss = loss_weights["plan"] * plan_loss
        else:
            raise ValueError(f"Unknown Stage B mode: {mode}")
        return loss, {
            "need_loss": need_loss,
            "failure_loss": failure_loss,
            "plan_loss": plan_loss,
        }

    (loss, detail), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.head_params)
    updates, opt_state = state.tx.update(grads, state.opt_state, state.head_params)
    head_params = optax.apply_updates(state.head_params, updates)
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        head_params=head_params,
        opt_state=opt_state,
    )
    return new_state, {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        **detail,
    }


def predict_step(
    head_config: stage_b_jax.AuxiliaryHeadConfig,
    state: StageBState,
    batch: tuple[openpi_model.Observation, dict[str, jax.Array]],
) -> dict[str, jax.Array]:
    observation, targets = batch
    backbone = nnx.merge(state.backbone_def, state.backbone_params)
    outputs = stage_b_jax.forward(
        backbone,
        state.head_params,
        observation,
        config=head_config,
        train=False,
    )
    return {
        "need_true": targets["need"],
        "need_pred": jnp.argmax(outputs["need_recovery"], axis=-1).astype(jnp.int32),
        "failure_true": targets["failure"],
        "failure_pred": jnp.argmax(outputs["failure_reason"], axis=-1).astype(jnp.int32),
        "failure_mask": targets["failure_mask"],
        "plan_true": targets["plan"],
        "plan_pred": jnp.argmax(outputs["recovery_plan"], axis=-1).astype(jnp.int32),
        "plan_mask": targets["plan_mask"],
    }


def evaluate(
    state: StageBState,
    loaders: dict[str, DataLoader],
    data_sharding: jax.sharding.Sharding,
    predict_fn,
) -> dict:
    need_true: list[int] = []
    need_pred: list[int] = []
    failure_true: list[int] = []
    failure_pred: list[int] = []
    plan_true: list[int] = []
    plan_pred: list[int] = []

    for batch in loaders["status"]:
        output = jax.device_get(predict_fn(state, batch_to_jax(batch, data_sharding)))
        need_true.extend(np.asarray(output["need_true"]).reshape(-1).tolist())
        need_pred.extend(np.asarray(output["need_pred"]).reshape(-1).tolist())
        failure_mask = np.asarray(output["failure_mask"]).reshape(-1).astype(bool)
        failure_true.extend(np.asarray(output["failure_true"]).reshape(-1)[failure_mask].tolist())
        failure_pred.extend(np.asarray(output["failure_pred"]).reshape(-1)[failure_mask].tolist())

    for batch in loaders["reasoning"]:
        output = jax.device_get(predict_fn(state, batch_to_jax(batch, data_sharding)))
        plan_mask = np.asarray(output["plan_mask"]).reshape(-1).astype(bool)
        plan_true.extend(np.asarray(output["plan_true"]).reshape(-1)[plan_mask].tolist())
        plan_pred.extend(np.asarray(output["plan_pred"]).reshape(-1)[plan_mask].tolist())

    return {
        "need_recovery": classification_report(need_true, need_pred, num_classes=2, class_names=["false", "true"]),
        "failure_reason": classification_report(
            failure_true,
            failure_pred,
            num_classes=len(vla_labels.FAILURE_REASONS),
            class_names=vla_labels.FAILURE_REASONS,
        ),
        "recovery_plan": classification_report(
            plan_true,
            plan_pred,
            num_classes=len(vla_labels.RECOVERY_PLANS),
            class_names=vla_labels.RECOVERY_PLANS,
        ),
    }


def save_head_checkpoint(state: StageBState, run_dir: Path, name: str, args: argparse.Namespace, metrics: dict) -> None:
    checkpoint_dir = run_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    flat = traverse_util.flatten_dict(jax.device_get(state.head_params), sep="/")
    np.savez(checkpoint_dir / "head_params.npz", **{key: np.asarray(value) for key, value in flat.items()})
    (checkpoint_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    (checkpoint_dir / "labels.json").write_text(json.dumps(vla_labels.label_map_payload(), indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.train_backbone:
        raise NotImplementedError("JAX Stage B currently trains auxiliary heads only; pi05 backbone stays frozen.")
    if not args.allow_random_backbone and args.backbone_checkpoint is None:
        raise ValueError("Stage B requires --backbone-checkpoint unless --allow-random-backbone is set.")
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size={args.batch_size} must be divisible by jax.device_count()={jax.device_count()}.")
    if args.use_state_history and args.state_history_dim != 7:
        raise ValueError("This dataset stores 7-D puppet qpos; --state-history-dim must be 7")

    jax_cache_dir = PROJECT_ROOT / ".cache" / "jax"
    jax_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))
    np.random.seed(args.seed)

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
    train_status = build_loader(args, model_config, split="train", stage="status", shuffle=True)
    train_reasoning = build_loader(args, model_config, split="train", stage="reasoning", shuffle=True)
    val_loaders = {
        "status": build_loader(args, model_config, split="val", stage="status", shuffle=False),
        "reasoning": build_loader(args, model_config, split="val", stage="reasoning", shuffle=False),
    }
    first_status = next(iter(train_status))
    first_reasoning = next(iter(train_reasoning))
    if args.dry_run:
        print_batch_shapes("status", first_status)
        print_batch_shapes("reasoning", first_reasoning)
        return

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str, ensure_ascii=False) + "\n")

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    tx = make_optimizer(args)
    head_config = stage_b_jax.AuxiliaryHeadConfig(hidden_dim=args.hidden_dim, dropout=args.dropout)
    checkpoint = resolve_backbone_checkpoint(args.backbone_checkpoint)
    loader_config: weight_loaders.WeightLoader
    if args.allow_random_backbone:
        loader_config = weight_loaders.NoOpWeightLoader()
    else:
        loader_config = weight_loaders.CheckpointWeightLoader(str(checkpoint))

    state, state_sharding = init_stage_b_state(
        model_config=model_config,
        head_config=head_config,
        tx=tx,
        weight_loader=loader_config,
        init_rng=jax.random.key(args.seed),
        mesh=mesh,
    )
    jax.block_until_ready(state)
    print(
        " ".join(
            [
                f"jax_backend={jax.default_backend()}",
                f"precision={precision}",
                f"head_params={count_params(state.head_params)}",
                f"backbone_params={count_params(state.backbone_params)}",
            ]
        )
    )

    loss_weights = {
        "need": float(args.need_loss_weight),
        "failure": float(args.failure_loss_weight),
        "plan": float(args.plan_loss_weight),
    }
    pstatus_step = jax.jit(
        lambda train_state, batch, rng: train_step("status", head_config, loss_weights, rng, train_state, batch),
        in_shardings=(state_sharding, data_sharding, replicated_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(0,),
    )
    preasoning_step = jax.jit(
        lambda train_state, batch, rng: train_step("reasoning", head_config, loss_weights, rng, train_state, batch),
        in_shardings=(state_sharding, data_sharding, replicated_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(0,),
    )
    ppredict = jax.jit(
        lambda train_state, batch: predict_step(head_config, train_state, batch),
        in_shardings=(state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    metrics_file = run_dir / "metrics.jsonl"
    rng = jax.random.key(args.seed + 1)
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        losses: list[float] = []
        detail: dict[str, list[float]] = {"need_loss": [], "failure_loss": [], "plan_loss": [], "grad_norm": []}

        for batch in tqdm.tqdm(train_status, desc=f"epoch {epoch} status"):
            rng, step_rng = jax.random.split(rng)
            state, info = pstatus_step(state, batch_to_jax(batch, data_sharding), step_rng)
            info = jax.device_get(info)
            losses.append(float(info["loss"]))
            for key in ("need_loss", "failure_loss", "grad_norm"):
                detail[key].append(float(info[key]))

        for batch in tqdm.tqdm(train_reasoning, desc=f"epoch {epoch} reasoning"):
            rng, step_rng = jax.random.split(rng)
            state, info = preasoning_step(state, batch_to_jax(batch, data_sharding), step_rng)
            info = jax.device_get(info)
            losses.append(float(info["loss"]))
            for key in ("plan_loss", "grad_norm"):
                detail[key].append(float(info[key]))

        val_metrics = evaluate(state, val_loaders, data_sharding, ppredict)
        score = (
            val_metrics["need_recovery"]["macro_f1"]
            + val_metrics["failure_reason"]["macro_f1"]
            + val_metrics["recovery_plan"]["macro_f1"]
        ) / 3.0
        payload = {
            "epoch": epoch,
            "step": int(jax.device_get(state.step)),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_detail": {key: float(np.mean(values)) if values else 0.0 for key, values in detail.items()},
            "val_score": score,
            "val": val_metrics,
            "elapsed_sec": time.time() - start,
        }
        with metrics_file.open("a") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        save_head_checkpoint(state, run_dir, "last", args, payload)
        if score > best_score:
            best_score = score
            save_head_checkpoint(state, run_dir, "best", args, payload)


if __name__ == "__main__":
    main()
