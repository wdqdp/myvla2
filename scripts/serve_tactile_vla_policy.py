#!/usr/bin/env python3
"""Serve the tactile VLA Stage A action policy plus Stage B auxiliary heads."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, traverse_util

from openpi.models import model as openpi_model
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.serving import websocket_policy_server
from openpi.shared import nnx_utils, normalize
from tactile_vla.runtime.state_history import DEFAULT_STATE_HISTORY_FPS
from tactile_vla.vla import labels as vla_labels
from tactile_vla.vla import stage_b_jax
from tactile_vla.vla.openpi_bridge import build_action_output_transform, build_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-checkpoint", type=Path, required=True, help="Stage A step dir, or its params dir.")
    parser.add_argument("--stage-b-heads", type=Path, required=True, help="Stage B head_params.npz, or containing dir.")
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--output-action-dim", type=int, default=7)
    parser.add_argument("--use-state-history", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--state-history-len", type=int)
    parser.add_argument("--state-history-dim", type=int)
    parser.add_argument("--state-history-fps", type=float)
    parser.add_argument("--history-hidden-dim", type=int)
    parser.add_argument("--max-token-len", type=int)
    parser.add_argument("--paligemma-variant")
    parser.add_argument("--action-expert-variant")
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"))
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--need-recovery-threshold", type=float, default=0.5)
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Warm up with one execution and reasoning request, then exit.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _find_config_near(path: Path) -> dict[str, Any]:
    path = path.resolve()
    candidates = []
    if path.is_dir():
        candidates.extend([path / "config.json", path.parent / "config.json", path.parent.parent / "config.json"])
    else:
        candidates.extend([path.parent / "config.json", path.parent.parent / "config.json"])
    for candidate in candidates:
        if candidate.exists():
            return _load_json(candidate)
    return {}


def _resolve_params_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    if checkpoint.name == "params" and checkpoint.exists():
        return checkpoint
    if (checkpoint / "params").exists():
        return checkpoint / "params"
    return checkpoint


def _resolve_heads_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        path = path / "head_params.npz"
    if not path.exists():
        raise FileNotFoundError(f"Stage B head params not found: {path}")
    return path


def _resolve_precision(value: str | None) -> str:
    if value in {None, "auto"}:
        return "bfloat16" if jax.default_backend() in {"gpu", "tpu"} else "float32"
    return value


def _npz_to_tree(path: Path) -> dict[str, Any]:
    arrays = np.load(path)
    flat = {tuple(key.split("/")): jnp.asarray(arrays[key]) for key in arrays.files}
    return traverse_util.unflatten_dict(flat)


def _softmax_probs(logits: Any) -> np.ndarray:
    return np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1), dtype=np.float32)


class TactileVLAPolicy:
    """Policy interface consumed by OpenPI's websocket server."""

    def __init__(
        self,
        *,
        model_config: Pi0Config,
        params_dir: Path,
        head_params_path: Path,
        norm_stats: dict | None,
        use_quantile_norm: bool,
        output_action_dim: int,
        num_inference_steps: int,
        need_recovery_threshold: float,
        metadata: dict[str, Any],
    ) -> None:
        self._model_config = model_config
        self._input_transform = build_transform(
            model_config,
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
            use_delta_actions=True,
            delta_action_dims=output_action_dim,
        )
        self._output_transform = build_action_output_transform(
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
            use_delta_actions=True,
            delta_action_dims=output_action_dim,
        )
        dtype = jnp.bfloat16 if model_config.dtype == "bfloat16" else jnp.float32
        logging.info("Loading Stage A params from %s", params_dir)
        params = openpi_model.restore_params(params_dir, dtype=dtype)
        flat_param_paths = traverse_util.flatten_dict(params)
        checkpoint_has_history = any(path and str(path[0]).startswith("history_") for path in flat_param_paths)
        if checkpoint_has_history != model_config.use_state_history:
            raise ValueError(
                "Stage A checkpoint/config state-history mismatch: "
                f"checkpoint_has_history={checkpoint_has_history}, "
                f"use_state_history={model_config.use_state_history}. "
                "Use the config.json saved with this checkpoint or pass matching history arguments."
            )
        self._model = model_config.load(params)
        self._model.eval()
        self._sample_actions = nnx_utils.module_jit(self._model.sample_actions)
        self._rng = jax.random.key(0)
        self._sample_kwargs = {"num_steps": int(num_inference_steps)}
        self._output_action_dim = int(output_action_dim)
        self._need_recovery_threshold = float(need_recovery_threshold)

        logging.info("Loading Stage B heads from %s", head_params_path)
        self._head_params = _npz_to_tree(head_params_path)
        self._head_config = stage_b_jax.AuxiliaryHeadConfig(
            hidden_dim=int(self._head_params["shared"]["bias"].shape[0]),
            dropout=0.0,
            num_failure_reasons=len(vla_labels.FAILURE_REASONS),
            num_recovery_plans=len(vla_labels.RECOVERY_PLANS),
        )
        graphdef, backbone_state = nnx.split(self._model)
        self._backbone_state = backbone_state

        def head_forward(backbone_state, head_params, observation):
            backbone = nnx.merge(graphdef, backbone_state)
            return stage_b_jax.forward(backbone, head_params, observation, config=self._head_config, train=False)

        self._head_forward = jax.jit(head_forward)
        self._metadata = metadata

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _prepare_observation(self, request: dict[str, Any]) -> tuple[dict[str, Any], Observation]:
        inputs = dict(request)
        inputs.pop("mode", None)
        transformed = self._input_transform(inputs)
        transformed = jax.tree.map(np.asarray, transformed)
        batched = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], transformed)
        return transformed, Observation.from_dict(batched)

    def _predict_heads(self, observation: Observation) -> dict[str, np.ndarray]:
        outputs = self._head_forward(self._backbone_state, self._head_params, observation)
        return {key: np.asarray(value[0]) for key, value in outputs.items()}

    def _sample_absolute_actions(self, transformed: dict[str, Any], observation: Observation) -> np.ndarray:
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self._sample_actions(sample_rng, observation, **self._sample_kwargs)
        outputs = {
            "state": transformed["state"],
            "actions": np.asarray(actions[0]),
        }
        restored = self._output_transform(outputs)
        actions = np.asarray(restored["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < self._output_action_dim:
            raise ValueError(f"Expected restored actions [T,{self._output_action_dim}+], got {actions.shape}")
        return actions[:, : self._output_action_dim]

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = str(request.get("mode", "execution"))
        if mode not in {"execution", "reasoning"}:
            raise ValueError(f"Unknown tactile VLA mode: {mode!r}")

        start_time = time.monotonic()
        transformed, observation = self._prepare_observation(request)
        head_logits = self._predict_heads(observation)

        if mode == "reasoning":
            plan_probs = _softmax_probs(head_logits["recovery_plan"])
            plan_id = int(np.argmax(plan_probs))
            return {
                "recovery_plan": vla_labels.id_to_recovery_plan(plan_id),
                "recovery_plan_probs": plan_probs.tolist(),
                "policy_timing": {"infer_ms": (time.monotonic() - start_time) * 1000.0},
            }

        actions = self._sample_absolute_actions(transformed, observation)
        need_probs = _softmax_probs(head_logits["need_recovery"])
        failure_probs = _softmax_probs(head_logits["failure_reason"])
        need_recovery = bool(float(need_probs[1]) >= self._need_recovery_threshold)
        failure_id = int(np.argmax(failure_probs))
        return {
            "actions": actions,
            "need_recovery": need_recovery,
            "need_recovery_probs": need_probs.tolist(),
            "failure_reason": vla_labels.id_to_failure_reason(failure_id),
            "failure_reason_probs": failure_probs.tolist(),
            "policy_timing": {"infer_ms": (time.monotonic() - start_time) * 1000.0},
        }


def build_model_config(args: argparse.Namespace, stage_a_config: dict[str, Any]) -> Pi0Config:
    precision = _resolve_precision(args.precision or stage_a_config.get("precision"))
    use_state_history = (
        bool(args.use_state_history)
        if args.use_state_history is not None
        else bool(stage_a_config.get("use_state_history", False))
    )
    return Pi0Config(
        dtype=precision,
        paligemma_variant=args.paligemma_variant or stage_a_config.get("paligemma_variant", "gemma_2b_lora"),
        action_expert_variant=args.action_expert_variant
        or stage_a_config.get("action_expert_variant", "gemma_300m_lora"),
        action_dim=int(args.action_dim or stage_a_config.get("action_dim", 32)),
        action_horizon=int(args.action_horizon or stage_a_config.get("action_horizon", 30)),
        max_token_len=int(args.max_token_len or stage_a_config.get("max_token_len", 200)),
        pi05=True,
        use_state_history=use_state_history,
        state_history_len=int(args.state_history_len or stage_a_config.get("state_history_len", 60)),
        state_history_dim=int(args.state_history_dim or stage_a_config.get("state_history_dim", 7)),
        history_hidden_dim=int(args.history_hidden_dim or stage_a_config.get("history_hidden_dim", 256)),
        pytorch_compile_mode=None,
    )


def warm_up_policy(policy: TactileVLAPolicy) -> dict[str, Any]:
    """Compile both inference paths and validate their basic output contract."""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros((7,), dtype=np.float32)
    history = np.zeros(
        (policy._model_config.state_history_len, policy._model_config.state_history_dim),  # noqa: SLF001
        dtype=np.float32,
    )
    history_mask = np.ones((policy._model_config.state_history_len,), dtype=np.bool_)  # noqa: SLF001
    history_inputs = (
        {
            "observation/state_history": history,
            "observation/state_history_mask": history_mask,
        }
        if policy._model_config.use_state_history  # noqa: SLF001
        else {}
    )
    execution = policy.infer(
        {
            "mode": "execution",
            "observation/image": image,
            "observation/wrist_image": image,
            "observation/state": state,
            **history_inputs,
            "prompt": (
                "Mode: execution. Task: dry run. Tactile: no rotation. "
                "Recovery plan: none. Output the next robot action chunk and monitor whether recovery is needed."
            ),
        }
    )
    reasoning = policy.infer(
        {
            "mode": "reasoning",
            "observation/image": image,
            "observation/wrist_image": image,
            "observation/state": state,
            **history_inputs,
            "prompt": (
                "Mode: reasoning. Task: dry run. Tactile: no rotation. "
                "Failure-recovery memory: none. Choose the next recovery plan."
            ),
        }
    )
    actions = np.asarray(execution["actions"])
    output_action_dim = int(policy.metadata["output_action_dim"])
    if actions.ndim != 2 or actions.shape[1] != output_action_dim:
        raise ValueError(f"Warm-up expected actions [T,{output_action_dim}], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Warm-up produced actions containing NaN or Inf")
    return {
        "execution": {
            "actions_shape": list(actions.shape),
            "need_recovery": bool(execution["need_recovery"]),
            "failure_reason": execution["failure_reason"],
        },
        "reasoning": {
            "recovery_plan": reasoning["recovery_plan"],
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    stage_a_config = _find_config_near(args.stage_a_checkpoint)
    stage_b_config = _find_config_near(args.stage_b_heads)
    model_config = build_model_config(args, stage_a_config)
    params_dir = _resolve_params_dir(args.stage_a_checkpoint)
    head_params_path = _resolve_heads_path(args.stage_b_heads)
    norm_stats = None if args.no_norm else normalize.load(args.norm_stats_dir)
    state_history_fps = float(
        args.state_history_fps
        if args.state_history_fps is not None
        else stage_a_config.get("state_history_fps", DEFAULT_STATE_HISTORY_FPS)
    )
    if state_history_fps <= 0:
        raise ValueError(f"state_history_fps must be positive, got {state_history_fps}")
    metadata = {
        "name": "tactile_vla_policy",
        "stage_a_checkpoint": str(args.stage_a_checkpoint),
        "stage_b_heads": str(head_params_path),
        "action_horizon": model_config.action_horizon,
        "action_dim": model_config.action_dim,
        "output_action_dim": args.output_action_dim,
        "use_state_history": model_config.use_state_history,
        "state_history_len": model_config.state_history_len,
        "state_history_dim": model_config.state_history_dim,
        "state_history_fps": state_history_fps,
        "history_hidden_dim": model_config.history_hidden_dim,
        "need_recovery_threshold": args.need_recovery_threshold,
        "failure_reasons": list(vla_labels.FAILURE_REASONS),
        "recovery_plans": list(vla_labels.RECOVERY_PLANS),
        "stage_a_config": stage_a_config,
        "stage_b_config": stage_b_config,
    }
    policy = TactileVLAPolicy(
        model_config=model_config,
        params_dir=params_dir,
        head_params_path=head_params_path,
        norm_stats=norm_stats,
        use_quantile_norm=not args.no_norm,
        output_action_dim=args.output_action_dim,
        num_inference_steps=args.num_inference_steps,
        need_recovery_threshold=args.need_recovery_threshold,
        metadata=metadata,
    )
    logging.info("Warming up execution and reasoning inference paths")
    warm_up_summary = warm_up_policy(policy)
    if args.dry_run:
        print(json.dumps(warm_up_summary, indent=2, ensure_ascii=False))
        return
    logging.info("Policy warm-up complete: %s", json.dumps(warm_up_summary, ensure_ascii=False))

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating tactile VLA server (host: %s, ip: %s, bind: %s:%d)", hostname, local_ip, args.host, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
