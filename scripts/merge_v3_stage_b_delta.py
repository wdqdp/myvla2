#!/usr/bin/env python3
"""Merge a V3 Stage B delta checkpoint with its frozen Stage A backbone."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))

os.environ.setdefault("OPENPI_DATA_HOME", "/data1/outputs/openpi_cache")

from flax import nnx
from flax import traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import gemma as openpi_gemma
from openpi.models import model as openpi_model
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.training import weight_loaders
from tactile_vla.vla.artifacts import assert_identity_matches
from tactile_vla.vla.artifacts import checkpoint_artifact_identity
from tactile_vla.vla.data_profiles import ROTATION_MODERATELY_SUCCESS_V1
from tactile_vla.vla.stage_b_v3_checkpoint import CHECKPOINT_FORMAT
from tactile_vla.vla.stage_b_v3_checkpoint import cast_frozen_params
from tactile_vla.vla.stage_b_v3_checkpoint import delta_params
from tactile_vla.vla.stage_b_v3_checkpoint import merge_delta_params
from tactile_vla.vla.stage_b_v3_checkpoint import trainable_filter
from tactile_vla.vla.stage_b_v3_model import StageBV3Model


MERGED_CHECKPOINT_FORMAT = "stage_b_v3_merged_full_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--stage-b-delta",
        type=Path,
        required=True,
        help="Stage B best directory or its delta_params directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output checkpoint root; merged parameters are written under output/params",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _params_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path / "params" if (path / "params").is_dir() else path


def _delta_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidate = path / "delta_params"
    if candidate.is_dir():
        return candidate
    if path.name == "delta_params" and path.is_dir():
        return path
    raise FileNotFoundError(
        f"Cannot find delta_params under {path}. Pass the Stage B best directory "
        "or the delta_params directory itself."
    )


def _find_config(path: Path) -> tuple[Path, dict[str, Any]]:
    current = path.expanduser().resolve()
    candidates = [current / "config.json"]
    candidates.extend(parent / "config.json" for parent in list(current.parents)[:4])
    for candidate in candidates:
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text())
    raise FileNotFoundError(f"Cannot find Stage B config.json near {path}")


def _model_config(config: dict[str, Any]) -> Pi0Config:
    precision = str(config.get("precision", "auto"))
    if precision == "auto":
        precision = "bfloat16" if jax.default_backend() in {"gpu", "tpu"} else "float32"
    return Pi0Config(
        dtype=precision,
        paligemma_variant=str(config.get("paligemma_variant", "gemma_2b_lora")),
        action_expert_variant=str(config.get("action_expert_variant", "gemma_300m_lora")),
        action_dim=int(config.get("action_dim", 32)),
        action_horizon=int(config.get("action_horizon", 30)),
        max_token_len=int(config.get("max_token_len", 200)),
        pi05=True,
        use_state_history=bool(config.get("use_state_history", False)),
        state_history_len=int(config.get("state_history_len", 60)),
        state_history_dim=int(config.get("state_history_dim", 7)),
        history_hidden_dim=int(config.get("history_hidden_dim", 256)),
        pytorch_compile_mode=None,
    )


def _load_weights_and_validate(
    loader: weight_loaders.WeightLoader,
    params_reference: at.Params,
) -> at.Params:
    loaded = loader.load(params_reference)
    at.check_pytree_equality(
        expected=params_reference,
        got=loaded,
        check_shapes=True,
        check_dtypes=True,
    )
    return traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(loaded).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )


def _build_stage_a_initialized_state(
    model_config: Pi0Config,
    config: dict[str, Any],
    stage_a_checkpoint: Path,
) -> nnx.State:
    backbone_rng, head_rng = jax.random.split(jax.random.key(int(config.get("seed", 42))))
    backbone = model_config.create(backbone_rng)
    graphdef, backbone_state = nnx.split(backbone)
    loader = weight_loaders.CheckpointWeightLoader(str(_params_dir(stage_a_checkpoint)))
    loaded = _load_weights_and_validate(loader, backbone_state.to_pure_dict())
    backbone_state.replace_by_pure_dict(loaded)
    backbone = nnx.merge(graphdef, backbone_state)

    model = StageBV3Model(
        backbone,
        paligemma_width=openpi_gemma.get_config(model_config.paligemma_variant).width,
        need_hidden_dim=int(config.get("need_hidden_dim", 512)),
        need_dropout=float(config.get("need_dropout", 0.1)),
        rngs=nnx.Rngs(head_rng),
    )
    state = nnx.state(model)
    train_filter = trainable_filter()
    frozen_filter = nnx.All(nnx.Param, nnx.Not(train_filter))
    state = cast_frozen_params(state, frozen_filter)
    del loaded, backbone, model
    gc.collect()
    return state


def _load_delta(base_state: nnx.State, delta_dir: Path) -> nnx.State:
    template = delta_params(base_state, trainable_filter())
    restored = openpi_model.restore_params(delta_dir, restore_type=np.ndarray)
    at.check_pytree_equality(
        expected=template.to_pure_dict(),
        got=restored,
        check_shapes=True,
        check_dtypes=False,
    )
    template.replace_by_pure_dict(restored)
    return template


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    delta_dir = _delta_dir(args.stage_b_delta)
    config_path, config = _find_config(delta_dir)
    checkpoint_format = config.get("checkpoint_format")
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported Stage B checkpoint_format={checkpoint_format!r}; "
            f"expected {CHECKPOINT_FORMAT!r}"
        )

    configured_stage_a = Path(str(config.get("stage_a_checkpoint", args.stage_a_checkpoint))).expanduser().resolve()
    requested_stage_a = args.stage_a_checkpoint.expanduser().resolve()
    if configured_stage_a != requested_stage_a:
        raise ValueError(
            "Stage A checkpoint does not match the Stage B training config: "
            f"requested={requested_stage_a}, configured={configured_stage_a}"
        )
    _, stage_a_config = _find_config(requested_stage_a)
    assert_identity_matches(
        checkpoint_artifact_identity(stage_a_config),
        checkpoint_artifact_identity(config),
        context="Stage B merge Stage A checkpoint",
    )
    if config.get("data_profile") == ROTATION_MODERATELY_SUCCESS_V1:
        if delta_dir.parent.name != "best":
            raise ValueError(
                "rotation_moderately_success_v1 may merge only the selected best delta"
            )
        if args.output.name != "merged_best":
            raise ValueError(
                "rotation_moderately_success_v1 merged output must be an independent merged_best directory"
            )
    output = args.output.expanduser().resolve()
    if not args.dry_run and output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")

    logging.info("Building V3 model from Stage A: %s", requested_stage_a)
    model_config = _model_config(config)
    base_state = _build_stage_a_initialized_state(model_config, config, requested_stage_a)
    logging.info("Loading Stage B delta: %s", delta_dir)
    stage_b_delta = _load_delta(base_state, delta_dir)
    merged_state = merge_delta_params(base_state, stage_b_delta)
    at.check_pytree_equality(
        expected=base_state.to_pure_dict(),
        got=merged_state.to_pure_dict(),
        check_shapes=True,
        check_dtypes=False,
    )

    delta_count = sum(
        int(getattr(leaf, "value", leaf).size)
        for leaf in jax.tree.leaves(stage_b_delta)
        if hasattr(getattr(leaf, "value", leaf), "size")
    )
    total_count = sum(
        int(getattr(leaf, "value", leaf).size)
        for leaf in jax.tree.leaves(merged_state)
        if hasattr(getattr(leaf, "value", leaf), "size")
    )
    logging.info("Validated delta_params=%d total_params=%d", delta_count, total_count)
    if args.dry_run:
        logging.info("Dry-run complete; no merged checkpoint was written")
        return

    params_dir = output / "params"
    output.mkdir(parents=True, exist_ok=True)
    logging.info("Writing full merged checkpoint to %s", params_dir)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(params_dir, {"params": merged_state}, force=args.overwrite)

    merged_config = dict(config)
    merged_config.update(
        checkpoint_format=MERGED_CHECKPOINT_FORMAT,
        training_checkpoint_format=CHECKPOINT_FORMAT,
        stage_a_checkpoint=str(requested_stage_a),
        stage_b_delta=str(delta_dir),
    )
    (output / "config.json").write_text(
        json.dumps(merged_config, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    manifest = {
        "checkpoint_format": MERGED_CHECKPOINT_FORMAT,
        "stage_a_checkpoint": str(requested_stage_a),
        "stage_b_delta": str(delta_dir),
        "source_config": str(config_path),
        "output_params": str(params_dir),
        "delta_parameter_count": delta_count,
        "total_parameter_count": total_count,
    }
    (output / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    metrics_path = delta_dir.parent / "metrics.json"
    if metrics_path.is_file():
        shutil.copy2(metrics_path, output / "metrics.json")
    logging.info("Merge complete: %s", output)


if __name__ == "__main__":
    main()
