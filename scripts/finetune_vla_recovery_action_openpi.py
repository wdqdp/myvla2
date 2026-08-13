#!/usr/bin/env python3
"""Fine-tune Stage A on clipped recovery actions with balanced normal replay."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import asdict
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

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.environ.setdefault("OPENPI_DATA_HOME", "/data1/outputs/openpi_cache")

from flax import nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.shared import normalize
from openpi.training import checkpoints as openpi_checkpoints
from openpi.training import sharding
from openpi.training import weight_loaders
from torch.utils.data import DataLoader
import tqdm

import train_vla_stage_a_openpi as stage_a
from tactile_vla.vla.openpi_bridge import build_transform
from tactile_vla.vla.openpi_bridge import collate_numpy
from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset
from tactile_vla.vla.openpi_bridge import TransformedTactileVLADataset
from tactile_vla.vla.recovery_action_finetuning import FineTuneSelection
from tactile_vla.vla.recovery_action_finetuning import HierarchicalGroupSampler
from tactile_vla.vla.recovery_action_finetuning import load_selection_config
from tactile_vla.vla.recovery_action_finetuning import load_split_episode_ids
from tactile_vla.vla.recovery_action_finetuning import scan_finetune_frames
from tactile_vla.vla.recovery_action_finetuning import select_finetune_frames


DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_v3")
DEFAULT_SPLIT_FILE = Path("/data1/outputs/vla/indices/splits_v3.json")
DEFAULT_NORM_STATS_DIR = Path("/data1/outputs/vla/assets/tactile_vla_v3")
DEFAULT_SELECTION_CONFIG = PROJECT_ROOT / "configs/recovery_action_finetune/moderately_k15.json"
DEFAULT_STAGE_A_CHECKPOINT = Path(
    "/data1/outputs/vla/stage_a_action/pi05_delta_tac_v3_m60/15000/params"
)
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/vla/stage_a_recovery_action_finetune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--selection-config", type=Path, default=DEFAULT_SELECTION_CONFIG)
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_STATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="pi05_delta_tac_v3_m60_recovery_moderately_k15")
    parser.add_argument("--split", default="train", choices=("train",))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-final", type=float, default=1e-6)
    parser.add_argument("--lr-transition-steps", type=int, default=750)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--keep-period", type=int, default=1000)
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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE_A_CHECKPOINT)
    parser.add_argument("--train-lora-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--selection-only", action="store_true", help="Validate and print selection without video/model I/O.")
    parser.add_argument("--dry-run", action="store_true", help="Build and print one transformed batch without training.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_steps <= 0:
        raise ValueError("batch-size and num-steps must be positive")
    if args.samples_per_epoch is not None and args.samples_per_epoch < args.batch_size:
        raise ValueError("samples-per-epoch must be at least batch-size")
    if args.action_horizon <= 0 or args.action_dim <= 0:
        raise ValueError("action-horizon and action-dim must be positive")
    if args.state_history_len <= 0 or args.state_history_dim != 7:
        raise ValueError("This experiment requires positive 7-D state history")
    if args.lr <= 0 or args.lr_final <= 0 or args.lr_transition_steps < 0:
        raise ValueError("Learning-rate values must be positive and transition steps non-negative")
    if args.train_lora_only and "lora" not in args.paligemma_variant and "lora" not in args.action_expert_variant:
        raise ValueError("--train-lora-only requires at least one LoRA model variant")


def build_selection(args: argparse.Namespace) -> FineTuneSelection:
    config = load_selection_config(args.selection_config)
    frames = scan_finetune_frames(args.dataset_dir)
    split_ids = load_split_episode_ids(args.split_file, args.split)
    return select_finetune_frames(
        frames,
        split_episode_ids=split_ids,
        split=args.split,
        config=config,
        action_horizon=args.action_horizon,
    )


def build_loader(
    args: argparse.Namespace,
    model_config: Pi0Config,
    selection: FineTuneSelection,
) -> tuple[DataLoader, HierarchicalGroupSampler]:
    norm_stats = None if args.no_norm else normalize.load(args.norm_stats_dir)
    dataset = TactileVLAFrameDataset(
        dataset_dir=args.dataset_dir,
        indices=selection.indices,
        stage="execution",
        action_horizon=args.action_horizon,
        state_history_len=args.state_history_len if args.use_state_history else 0,
        video_backend=args.video_backend,
    )
    transformed = TransformedTactileVLADataset(
        dataset,
        build_transform(model_config, norm_stats=norm_stats, use_quantile_norm=not args.no_norm),
    )
    sampler = HierarchicalGroupSampler(
        selection.frames,
        group_weights=selection.group_weights,
        num_samples=args.samples_per_epoch,
        seed=args.seed,
    )
    loader_kwargs: dict[str, Any] = {}
    if args.num_workers > 0:
        loader_kwargs.update(multiprocessing_context="spawn", persistent_workers=True)
    loader = DataLoader(
        transformed,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_numpy,
        drop_last=True,
        **loader_kwargs,
    )
    return loader, sampler


def _write_run_metadata(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    selection: FineTuneSelection,
    sampler: HierarchicalGroupSampler,
) -> None:
    config_payload = {
        **vars(args),
        "selection_summary": selection.summary(),
        "sampler_num_samples": len(sampler),
        "sampler_planned_group_counts": sampler.planned_group_counts(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, default=str, ensure_ascii=False) + "\n"
    )
    manifest = selection.manifest(
        config_path=args.selection_config,
        dataset_dir=args.dataset_dir,
        split_file=args.split_file,
    )
    (run_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


def _validate_resume_selection(run_dir: Path, selection: FineTuneSelection) -> None:
    manifest_path = run_dir / "selection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Resume requires the saved selection manifest: {manifest_path}")
    saved = json.loads(manifest_path.read_text())
    if saved.get("summary") != selection.summary():
        raise ValueError("Current recovery fine-tuning selection differs from the saved resume selection")
    current_frames = [asdict(frame) for frame in selection.frames]
    if saved.get("frames") != current_frames:
        raise ValueError("Current recovery fine-tuning frame index differs from the saved resume selection")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    stage_a.init_logging()
    logging.info("Running on: %s", platform.node())
    selection = build_selection(args)
    summary = selection.summary()
    logging.info("Recovery fine-tuning selection:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))
    if args.selection_only:
        sampler = HierarchicalGroupSampler(
            selection.frames,
            group_weights=selection.group_weights,
            num_samples=args.samples_per_epoch,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    **summary,
                    "sampler_num_samples": len(sampler),
                    "sampler_planned_group_counts": sampler.planned_group_counts(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    jax_cache_dir = PROJECT_ROOT / ".cache" / "jax"
    jax_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))
    logging.info("JAX devices: %s", jax.devices())
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"batch_size={args.batch_size} must be divisible by jax.device_count()={jax.device_count()}"
        )

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
    loader, sampler = build_loader(args, model_config, selection)
    first_batch = next(iter(loader))
    if args.dry_run:
        stage_a.print_batch_shapes(first_batch)
        print(json.dumps({"selection": summary, "planned_sampling": sampler.planned_group_counts()}, indent=2))
        return

    run_dir = args.output_dir / args.run_name
    checkpoint_manager, resuming = openpi_checkpoints.initialize_checkpoint_dir(
        run_dir,
        keep_period=args.keep_period,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    if jax.process_index() == 0:
        if resuming:
            _validate_resume_selection(run_dir, selection)
        else:
            _write_run_metadata(run_dir, args=args, selection=selection, sampler=sampler)

    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    freeze_filter = model_config.get_freeze_filter() if args.train_lora_only else nnx.Nothing
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    tx = stage_a.make_optimizer(args)
    checkpoint_path = args.checkpoint / "params" if (args.checkpoint / "params").is_dir() else args.checkpoint
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Stage A checkpoint params do not exist: {checkpoint_path}")
    missing_regex = r"(?:.*lora.*|history_.*)" if args.use_state_history else ".*lora.*"
    loader_config: weight_loaders.WeightLoader = weight_loaders.CheckpointWeightLoader(
        str(checkpoint_path),
        missing_regex=missing_regex,
    )
    train_state, train_state_sharding = stage_a.init_train_state(
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
        train_state = stage_a.restore_state(checkpoint_manager, train_state)
    jax.block_until_ready(train_state)

    trainable_params = stage_a.count_state_params(train_state.params.filter(trainable_filter))
    total_params = stage_a.count_state_params(train_state.params)
    logging.info("precision=%s trainable_params=%d total_params=%d", precision, trainable_params, total_params)
    ptrain_step = jax.jit(
        lambda step_rng, state, batch: stage_a.train_step(trainable_filter, step_rng, state, batch),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    metrics_file = run_dir / "metrics.jsonl"
    data_iter = iter(loader)
    start_step = int(jax.device_get(train_state.step))
    start_time = time.time()
    infos: list[dict[str, at.Array]] = []
    pbar = tqdm.trange(
        start_step,
        args.num_steps,
        initial=start_step,
        total=args.num_steps,
        desc="Recovery action fine-tune",
    )
    for _ in pbar:
        try:
            numpy_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            numpy_batch = next(data_iter)
        batch = stage_a.batch_to_jax(numpy_batch, data_sharding)
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
            stage_a.write_jsonl(metrics_file, payload)
            infos.clear()
        if step % args.save_interval == 0 or step == args.num_steps:
            stage_a.save_state(checkpoint_manager, train_state, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
