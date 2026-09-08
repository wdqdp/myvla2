#!/usr/bin/env python3
"""Serve the V7 no-history action policy and adjustment_end classifier."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
os.environ.setdefault("USE_TF", "0")

from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import gemma as openpi_gemma
from openpi.models import model as openpi_model
from openpi.models.pi0_config import Pi0Config
from openpi.serving import websocket_policy_server
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.shared import normalize
from tactile_vla.vla.stage_b_v3_checkpoint import cast_frozen_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import delta_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import merge_delta_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import multitask_trainable_filter
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v5_3_adjustment_end_model import AdjustmentEndModel
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v5_3_phase_change import QPOS_BIN_COUNT
from tactile_vla.vla.v5_3_phase_change import QPOS_SAMPLE_OFFSETS
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v5_3_phase_change import build_adjustment_end_prompt
from tactile_vla.vla.v7_adjustment_data import ROTATION_PHASE_V7_ADJUSTMENT
from tactile_vla.vla.v7_adjustment_data import V7_EXPERIMENT_KIND
from tactile_vla.vla.v7_adjustment_end_data import DATA_PROFILE
from tactile_vla.vla.v7_adjustment_end_data import EXPERIMENT_KIND
from tactile_vla.vla.v7_adjustment_end_data import LABEL_POLICY
from scripts import serve_tactile_vla_v5_3 as v53
from scripts.train_vla_adjustment_end_multitask_v7 import MULTITASK_CHECKPOINT_FORMAT


ACTION_HORIZON = 30
ACTION_DIM = 32
OUTPUT_ACTION_DIM = 7
PHASE_CHANGE_TIMEOUT_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--adjustment-end-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-load-mode", choices=("full", "delta"), default="full")
    parser.add_argument("--stage-a-checkpoint", type=Path)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--captioner-checkpoint-sha256")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--adjustment-end-threshold-override", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _find_config(step_dir: Path) -> tuple[Path, dict]:
    for candidate in (step_dir / "config.json", step_dir.parent / "config.json", step_dir.parent.parent / "config.json"):
        if candidate.is_file():
            return candidate.resolve(), json.loads(candidate.read_text())
    raise FileNotFoundError(f"Cannot find config.json near {step_dir}")


def _model_config(config: dict, *, max_token_len: int, precision: str) -> Pi0Config:
    actual_precision = precision
    if actual_precision == "auto":
        configured = str(config.get("precision", "auto"))
        actual_precision = configured if configured != "auto" else (
            "bfloat16" if jax.default_backend() in {"gpu", "tpu"} else "float32"
        )
    result = Pi0Config(
        dtype=actual_precision,
        paligemma_variant=str(config["paligemma_variant"]),
        action_expert_variant=str(config["action_expert_variant"]),
        action_dim=int(config["action_dim"]),
        action_horizon=int(config["action_horizon"]),
        max_token_len=max_token_len,
        pi05=True,
        use_state_history=False,
        state_history_len=0,
        state_history_dim=7,
        history_hidden_dim=0,
        pytorch_compile_mode=None,
    )
    if (result.action_horizon, result.action_dim) != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError("V7 requires action shape [30,32]")
    return result


def _validate_config(args, config: dict, metadata: dict) -> tuple[float, str]:
    required = {
        "data_profile": DATA_PROFILE,
        "prompt_profile": "phase_change_v1",
        "experiment_kind": EXPERIMENT_KIND,
        "checkpoint_format": MULTITASK_CHECKPOINT_FORMAT,
        "num_steps": 8000,
        "phase_change_max_token_len": 512,
        "use_state_history": False,
        "state_history_len": 0,
        "history_hidden_dim": 0,
        "label_policy": LABEL_POLICY,
    }
    mismatch = {key: (config.get(key), value) for key, value in required.items() if config.get(key) != value}
    if mismatch:
        raise ValueError(f"V7 adjustment_end config mismatch: {mismatch}")
    if metadata.get("checkpoint_format") != MULTITASK_CHECKPOINT_FORMAT:
        raise ValueError("V7 adjustment_end metadata checkpoint format mismatch")
    if metadata.get("label_policy") != LABEL_POLICY or int(metadata.get("official_step", -1)) != 8000:
        raise ValueError("V7 adjustment_end metadata label/step mismatch")
    if args.checkpoint_load_mode == "delta" and args.stage_a_checkpoint is None:
        raise ValueError("--checkpoint-load-mode=delta requires --stage-a-checkpoint")
    threshold = float(metadata.get("adjustment_end_threshold", -1.0))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("V7 adjustment_end threshold is invalid")
    stored_sha = config["caption_source"]["checkpoint"].get("sha256")
    runtime_sha = args.captioner_checkpoint_sha256 or stored_sha
    if not isinstance(runtime_sha, str) or len(runtime_sha) != 64:
        raise ValueError(
            "Captioner identity was unavailable during annotation; pass "
            "--captioner-checkpoint-sha256 with the deployed checkpoint SHA256"
        )
    if stored_sha is not None and runtime_sha != stored_sha:
        raise ValueError("Runtime Captioner SHA differs from the training annotation identity")
    return threshold, runtime_sha


def _restore_tree(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(path)
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(path.resolve())
    if not isinstance(restored, dict) or not restored:
        raise ValueError(f"Invalid V7 parameter tree: {path}")
    return restored


class V7Policy(v53.V53Policy):
    def __init__(self, *, args, config_path: Path, config: dict):
        step_dir = args.adjustment_end_checkpoint.resolve()
        metadata_path = step_dir / "adjustment_end_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        final_metadata = json.loads(metadata_path.read_text())
        checkpoint_threshold, captioner_sha = _validate_config(args, config, final_metadata)
        self._checkpoint_format = MULTITASK_CHECKPOINT_FORMAT
        self._checkpoint_threshold = checkpoint_threshold
        self._threshold, self._manual_threshold_override = v53._resolve_runtime_threshold(
            checkpoint_threshold, args.adjustment_end_threshold_override
        )
        self._experimental_override = bool(final_metadata.get("experimental_override", False) or self._manual_threshold_override)
        self._accepted_for_robot = bool(final_metadata.get("accepted_for_robot", False) and not self._manual_threshold_override)
        self._action_config = _model_config(config, max_token_len=int(config.get("action_max_token_len", 200)), precision=args.precision)
        self._phase_config = _model_config(config, max_token_len=PHASE_CHANGE_MAX_TOKEN_LEN, precision=args.precision)
        norm_stats = normalize.load(args.norm_stats_dir)
        self._action_input = v53.build_transform(
            self._action_config, norm_stats=norm_stats, use_quantile_norm=True, use_delta_actions=True, delta_action_dims=7
        )
        self._phase_input = v53.build_transform(
            self._phase_config, norm_stats=norm_stats, use_quantile_norm=True, use_delta_actions=False
        )
        self._action_output = v53.build_action_output_transform(
            norm_stats=norm_stats, use_quantile_norm=True, use_delta_actions=True, delta_action_dims=7
        )

        wrapper = AdjustmentEndModel(
            self._action_config.create(jax.random.key(1)),
            paligemma_width=openpi_gemma.get_config(self._action_config.paligemma_variant).width,
            rngs=nnx.Rngs(jax.random.key(2)),
        )
        graphdef, template = nnx.split(wrapper)
        trainable = multitask_trainable_filter()
        template = cast_frozen_params(template, nnx.All(nnx.Param, nnx.Not(trainable)))
        if args.checkpoint_load_mode == "full":
            restored = _restore_tree(step_dir / "full_params")
            at.check_pytree_equality(expected=template.to_pure_dict(), got=restored, check_shapes=True, check_dtypes=False)
            template.replace_by_pure_dict(restored)
            wrapper = nnx.merge(graphdef, template)
        else:
            stage_a = args.stage_a_checkpoint.resolve()
            backbone_params = openpi_model.restore_params(stage_a / "params")
            backbone = self._action_config.load(backbone_params)
            wrapper = AdjustmentEndModel(
                backbone,
                paligemma_width=openpi_gemma.get_config(self._action_config.paligemma_variant).width,
                rngs=nnx.Rngs(jax.random.key(2)),
            )
            graphdef, state = nnx.split(wrapper)
            state = cast_frozen_params(state, nnx.All(nnx.Param, nnx.Not(trainable)))
            overlay = delta_params(state, trainable)
            restored = _restore_tree(step_dir / "delta_params")
            at.check_pytree_equality(expected=overlay.to_pure_dict(), got=restored, check_shapes=True, check_dtypes=False)
            overlay.replace_by_pure_dict(restored)
            wrapper = nnx.merge(graphdef, merge_delta_params(state, overlay))
        wrapper.eval()
        actual_sha = parameter_tree_sha256(nnx.state(wrapper))
        expected_sha = final_metadata.get("parameter_exports", {}).get("full_params", {}).get("parameter_tree_sha256")
        if expected_sha and actual_sha != expected_sha:
            raise ValueError("Restored V7 full model parameter-tree SHA mismatch")

        self._sample_actions = nnx_utils.module_jit(wrapper.backbone.sample_actions)
        self._adjustment_end_logits = nnx_utils.module_jit(wrapper.adjustment_end_logits)
        self._sample_rng = jax.random.key(3)
        self._num_inference_steps = int(args.num_inference_steps)
        if self._num_inference_steps <= 0:
            raise ValueError("num inference steps must be positive")
        self._metadata = {
            "name": "tactile_vla_v7",
            "checkpoint_kind": "full-v7-action-plus-adjustment-end" if args.checkpoint_load_mode == "full" else "stage-a-plus-v7-delta",
            "checkpoint": str(step_dir),
            "checkpoint_load_mode": args.checkpoint_load_mode,
            "prompt_profile": "phase_v2",
            "data_profile": ROTATION_PHASE_V7_ADJUSTMENT,
            "experiment_kind": V7_EXPERIMENT_KIND,
            "stage_a_protocol": "v7_no_state_history",
            "action_only_ablation": False,
            "supports_action_noise": True,
            "requires_action_noise": True,
            "supports_adjustment_end": True,
            "adjustment_end_threshold": self._threshold,
            "adjustment_end_checkpoint_threshold": checkpoint_threshold,
            "adjustment_end_manual_threshold_override": self._manual_threshold_override,
            "adjustment_end_experimental_override": self._experimental_override,
            "adjustment_end_accepted_for_robot": self._accepted_for_robot,
            "adjustment_end_threshold_policy": final_metadata["threshold_policy"],
            "adjustment_end_checkpoint_format": MULTITASK_CHECKPOINT_FORMAT,
            "adjustment_end_data_profile": DATA_PROFILE,
            "adjustment_end_label_policy": LABEL_POLICY,
            "phase_change_prompt_profile": "phase_change_v1",
            "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
            "qpos_h30_sample_offsets": list(QPOS_SAMPLE_OFFSETS),
            "qpos_bin_count": QPOS_BIN_COUNT,
            "qpos_discretization_extra_clip": False,
            "captioner_checkpoint_sha256": captioner_sha,
            "captioner_window_size": 30,
            "phase_change_timeout_seconds": PHASE_CHANGE_TIMEOUT_SECONDS,
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "action_noise_shape": [ACTION_HORIZON, ACTION_DIM],
            "output_action_dim": OUTPUT_ACTION_DIM,
            "state_history_len": 0,
            "state_history_dim": 7,
            "use_state_history": False,
            "inference_parameter_tree_sha256": actual_sha,
            "adjustment_end_checkpoint": str(step_dir),
        }

    @staticmethod
    def _clean_inputs(request):
        forbidden = {"observation/state_history", "observation/state_history_mask"} & set(request)
        if forbidden:
            raise ValueError(f"V7 no-history request contains forbidden fields: {sorted(forbidden)}")
        return v53.V53Policy._clean_inputs(request)


def warm_up(policy: V7Policy) -> dict:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    common = {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/state": np.zeros(7, dtype=np.float32),
    }
    action = policy.infer({
        **common,
        "mode": "execution",
        "prompt": v53.build_phase_prompt(phase="execution", instruction="dry run", recovery_plan="move left", prompt_profile="phase_v2"),
        "action_noise": np.zeros((30, 32), dtype=np.float32),
    })
    prompt, _ = build_adjustment_end_prompt(
        instruction="dry run",
        tactile_caption="Touch[area=none; Fx=near_zero; Fy=near_zero; Fz=near_zero; rotation=none]",
        recovery_plan="move left",
        qpos_h30=np.zeros((30, 7), dtype=np.float32),
        stats=StateQuantileStats(q01=np.zeros(7), q99=np.ones(7)),
    )
    phase = policy.infer({**common, "mode": "adjustment_end", "prompt": prompt})
    return {
        "action_shape": list(np.asarray(action["actions"]).shape),
        "adjustment_end": bool(phase["adjustment_end"]),
        "adjustment_end_probs": np.asarray(phase["adjustment_end_probs"]).tolist(),
        "use_state_history": False,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    config_path, config = _find_config(args.adjustment_end_checkpoint.resolve())
    policy = V7Policy(args=args, config_path=config_path, config=config)
    summary = warm_up(policy)
    logging.info("V7 action and adjustment_end warm-up complete: %s", summary)
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    logging.info("Serving V7 on %s (%s):%d", socket.gethostname(), args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
