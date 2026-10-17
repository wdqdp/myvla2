#!/usr/bin/env python3
"""Joint Stage A from pi05_base, balanced by physical task and action phase."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import train_vla_stage_a_openpi as base
from tactile_vla.vla.v12_1_data import (
    DATA_PROFILE,
    DEFAULT_INDEX,
    OUTPUT,
    ROOT,
    artifact_identity,
    read_json,
    validate_index,
)
from tactile_vla.vla.v12_1_runtime import build_loaders, check_args, evaluate_action_report, print_dry_run

RUN_NAME = "pi05_delta_tac_blackbox_book_stage_a_v12_1_no_history"
DEFAULT_OUTPUT = ROOT / "joint_blackbox_book/outputs/stage_a_action"


def inject_defaults(defaults):
    for flag, value in defaults.items():
        if not any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:]):
            sys.argv.extend([flag, str(value)])


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--action-eval-samples-per-phase", type=int, default=256)
    extra, rest = parser.parse_known_args()
    sys.argv[:] = [sys.argv[0], *rest]
    if any(a in ("-h", "--help") for a in rest):
        print("V12.1 options: --eval-interval 1000 --action-eval-samples-per-phase 256 (per domain and phase)")
    inject_defaults(
        {
            "--dataset-dir": ROOT / "joint_blackbox_book",
            "--split-file": DEFAULT_INDEX,
            "--index-file": DEFAULT_INDEX,
            "--norm-stats-dir": OUTPUT / "norm_stats",
            "--output-dir": DEFAULT_OUTPUT,
            "--run-name": RUN_NAME,
            "--data-profile": DATA_PROFILE,
            "--prompt-profile": "phase_v2",
            "--num-steps": 15000,
            "--batch-size": 8,
            "--state-history-len": 0,
            "--history-hidden-dim": 0,
            "--keep-period": 1000,
            "--save-interval": 1000,
            "--fsdp-devices": 1,
        }
    )
    if not any(a in ("--use-state-history", "--no-use-state-history") for a in sys.argv):
        sys.argv.append("--no-use-state-history")
    args = base.parse_args()
    args.eval_interval = extra.eval_interval
    args.action_eval_samples_per_phase = extra.action_eval_samples_per_phase
    args.experiment_version = "v12_1"
    check_args(args)
    if args.data_profile != DATA_PROFILE or args.prompt_profile != "phase_v2":
        raise ValueError("Wrong joint Stage A protocol")
    if not args.train_lora_only or args.allow_random_init or args.split != "train":
        raise ValueError("V12.1 uses pi05_base LoRA initialization and train split only")
    if Path(args.checkpoint).resolve() != base.DEFAULT_BASE_CHECKPOINT.resolve():
        raise ValueError("V12.1 Stage A must initialize from the raw pi05_base, not an old task model")
    if args.action_eval_samples_per_phase <= 0 or args.lr_transition_steps < 0:
        raise ValueError("Evaluation support must be positive; LR transition must be nonnegative")
    return args


def main():
    args = parse_args()
    base.init_logging()
    cache = PROJECT_ROOT / ".cache/jax"
    cache.mkdir(parents=True, exist_ok=True)
    base.jax.config.update("jax_compilation_cache_dir", str(cache))
    index = read_json(args.index_file)
    validate_index(index)
    if args.norm_stats_dir.resolve() != Path(index["norm_stats_dir"]).resolve():
        raise ValueError("Index and shared norm stats differ")
    identity = artifact_identity(index, args.index_file)
    if args.batch_size % base.jax.device_count():
        raise ValueError("Global batch must divide across all visible devices")
    precision = (
        args.precision
        if args.precision != "auto"
        else ("bfloat16" if base.jax.default_backend() in ("gpu", "tpu") else "float32")
    )
    config = base.Pi0Config(
        dtype=precision,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_dim=32,
        action_horizon=30,
        max_token_len=args.max_token_len,
        pi05=True,
        use_state_history=False,
        state_history_len=0,
        history_hidden_dim=0,
        pytorch_compile_mode=None,
    )
    loaders = build_loaders(args, config, index, action_only=True)
    if args.dry_run:
        print_dry_run(loaders["train"])
        print(json.dumps({"action_eval_selection": loaders["action_eval_selection"]}))
        return
    run_dir = args.output_dir / args.run_name
    manager, resuming = base.openpi_checkpoints.initialize_checkpoint_dir(
        run_dir, keep_period=args.keep_period, overwrite=False, resume=args.resume, enable_async_checkpointing=False
    )
    if resuming:
        saved = read_json(run_dir / "config.json")
        if saved["artifact_identity"] != identity:
            raise ValueError("Resume requires identical joint data and normalization")
        for key in (
            "checkpoint",
            "fsdp_devices",
            "lr",
            "lr_final",
            "lr_transition_steps",
            "paligemma_variant",
            "action_expert_variant",
            "max_token_len",
        ):
            if str(saved[key]) != str(getattr(args, key)):
                raise ValueError(f"Resume configuration differs: {key}")
    else:
        payload = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        payload.update(
            {
                "artifact_identity": identity,
                "precision": precision,
                "stage_a_protocol": "v12_1_joint_lora_no_history",
                "action_eval_selection": loaders["action_eval_selection"],
                "task_sampling": "1:1",
                "phase_sampling": "1:1",
            }
        )
        (run_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    mesh = base.sharding.make_mesh(args.fsdp_devices)
    data_sharding = base.jax.sharding.NamedSharding(mesh, base.jax.sharding.PartitionSpec(base.sharding.DATA_AXIS))
    replicated = base.jax.sharding.NamedSharding(mesh, base.jax.sharding.PartitionSpec())
    frozen = config.get_freeze_filter()
    train_filter = base.nnx.All(base.nnx.Param, base.nnx.Not(frozen))
    train_rng, init_rng = base.jax.random.split(base.jax.random.key(args.seed))
    state, state_sharding = base.init_train_state(
        model_config=config,
        trainable_filter=train_filter,
        freeze_filter=frozen,
        tx=base.make_optimizer(args),
        ema_decay=args.ema_decay,
        weight_loader=base.weight_loaders.CheckpointWeightLoader(args.checkpoint, missing_regex=".*lora.*"),
        init_rng=init_rng,
        mesh=mesh,
        resume=resuming,
    )
    if resuming:
        state = base.restore_state(manager, state)
    base.jax.block_until_ready(state)
    base.logging.info(
        "trainable_params=%d total_params=%d",
        base.count_state_params(state.params.filter(train_filter)),
        base.count_state_params(state.params),
    )
    train = base.jax.jit(
        lambda rng, s, batch: base.train_step(train_filter, rng, s, batch),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    iterator = iter(loaders["train"]["action"])
    start = int(base.jax.device_get(state.step))
    started = time.time()
    recent = []
    best = (
        read_json(run_dir / "best_step.json")["loss"]
        if resuming and (run_dir / "best_step.json").exists()
        else float("inf")
    )
    progress = base.tqdm.trange(start, args.num_steps, initial=start, total=args.num_steps, desc="Joint Stage A V12.1")
    for _ in progress:
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loaders["train"]["action"])
            raw = next(iterator)
        batch = base.batch_to_jax(raw, data_sharding)
        with base.sharding.set_mesh(mesh):
            state, info = train(train_rng, state, batch)
        step = int(base.jax.device_get(state.step))
        recent.append({k: float(v) for k, v in base.jax.device_get(info).items()})
        if step % args.log_interval == 0:
            payload = {
                "step": step,
                **{k: float(base.np.mean([r[k] for r in recent])) for k in recent[0]},
                "elapsed_sec": time.time() - started,
            }
            base.write_jsonl(run_dir / "metrics.jsonl", payload)
            progress.write(str(payload))
            recent.clear()
        if step % args.save_interval == 0 or step == args.num_steps:
            base.save_state(manager, state, step)
            manager.wait_until_finished()
        if step % args.eval_interval == 0 or step == args.num_steps:
            report = evaluate_action_report(state, loaders["val"], data_sharding, seed=args.seed + 100)
            base.write_jsonl(run_dir / "metrics.jsonl", {"step": step, "val_action": report})
            progress.write(json.dumps({"step": step, "val_action": report}))
            if report["loss"] < best and (step % args.save_interval == 0 or step == args.num_steps):
                best = report["loss"]
                # Reference a retained numeric checkpoint; do not duplicate model weights.
                (run_dir / "best_step.json").write_text(json.dumps({"step": step, **report}, indent=2) + "\n")
    manager.wait_until_finished()


if __name__ == "__main__":
    main()
