#!/usr/bin/env python3
"""Serve Stage A or merged Stage B as a deterministic action-only ablation policy."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
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
os.environ.setdefault("USE_TF", "0")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import gemma as openpi_gemma
from openpi.models import model as openpi_model
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.serving import websocket_policy_server
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.shared import normalize
from tactile_vla.vla.openpi_bridge import build_action_output_transform
from tactile_vla.vla.openpi_bridge import build_transform
from tactile_vla.vla.stage_b_v3_model import StageBV3Model
from tactile_vla.vla.artifacts import assert_identity_matches
from tactile_vla.vla.artifacts import checkpoint_artifact_identity
from tactile_vla.vla.artifacts import checkpoint_step_number
from tactile_vla.vla.artifacts import validate_merged_best_metrics
from tactile_vla.vla.artifacts import validate_norm_stats_identity
from tactile_vla.vla.prompts import build_execution_prompt
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE_V2
from tactile_vla.vla.prompts import resolve_prompt_profile
from tactile_vla.vla.v4_data import ROTATION_V4
from tactile_vla.vla.v5_adjustment_data import ROTATION_PHASE_V5_ADJUSTMENT_V2
from tactile_vla.vla.v5_adjustment_data import V2_EXPERIMENT_KIND
from tactile_vla.vla.v5_phase_data import PHASE_EXPERIMENT_KIND
from tactile_vla.vla.v5_phase_data import ROTATION_PHASE_V5


ACTION_HORIZON = 30
ACTION_DIM = 32
OUTPUT_ACTION_DIM = 7

PHASE_ACTION_PROFILES = {
    ROTATION_PHASE_V5: (PHASE_PROMPT_PROFILE, PHASE_EXPERIMENT_KIND),
    ROTATION_PHASE_V5_ADJUSTMENT_V2: (
        PHASE_PROMPT_PROFILE_V2,
        V2_EXPERIMENT_KIND,
    ),
}
VERSIONED_ACTION_PROFILES = {
    ROTATION_V4: (MINIMAL_PROMPT_PROFILE, None),
    **PHASE_ACTION_PROFILES,
}

_METADATA_CONFIG_KEYS = (
    "run_name",
    "data_profile",
    "prompt_profile",
    "experiment_kind",
    "checkpoint_format",
    "num_steps",
    "seed",
    "action_horizon",
    "action_dim",
    "use_state_history",
    "state_history_len",
    "state_history_dim",
    "state_history_fps",
    "history_hidden_dim",
    "max_token_len",
    "paligemma_variant",
    "action_expert_variant",
    "precision",
    "train_lora_only",
)
_METADATA_IDENTITY_KEYS = (
    "selection_hash",
    "training_data_hash",
    "index_sha256",
    "v4_norm_stats_sha256",
    "norm_stats_sha256",
    "data_config_hash",
    "v4_profile_config_hash",
    "action_frame_manifest_hash",
)


def metadata_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Keep websocket metadata small even when training config has per-frame sidecars."""

    summary = {key: config[key] for key in _METADATA_CONFIG_KEYS if key in config}
    identity = checkpoint_artifact_identity(config)
    compact_identity = {key: identity[key] for key in _METADATA_IDENTITY_KEYS if key in identity}
    if compact_identity:
        summary["artifact_identity"] = compact_identity
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-kind", choices=("stage-a", "stage-b"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage A step/params or merged Stage B directory.")
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _find_config(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    candidates = [path / "config.json", path.parent / "config.json", path.parent.parent / "config.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text())
    raise FileNotFoundError(f"Cannot find config.json near checkpoint: {path}")


def _params_dir(path: Path) -> Path:
    path = path.resolve()
    params_dir = path / "params" if (path / "params").is_dir() else path
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {params_dir}")
    return params_dir


def validate_v4_norm_artifacts(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    data_profile = config.get("data_profile")
    if data_profile not in VERSIONED_ACTION_PROFILES:
        return None
    expected_prompt, expected_experiment = VERSIONED_ACTION_PROFILES[data_profile]
    if resolve_prompt_profile(config.get("prompt_profile")) != expected_prompt:
        raise ValueError(
            f"{data_profile} action serving requires prompt_profile={expected_prompt!r}"
        )
    if data_profile in PHASE_ACTION_PROFILES:
        if config.get("experiment_kind") != expected_experiment:
            raise ValueError(
                f"{data_profile} serving requires experiment_kind={expected_experiment!r}"
            )
        if args.checkpoint_kind != "stage-a":
            raise ValueError(f"{data_profile} phase validation has no Stage B checkpoint")
    checkpoint_root = args.checkpoint.expanduser().resolve()
    if checkpoint_root.name == "params":
        checkpoint_root = checkpoint_root.parent
    identity = checkpoint_artifact_identity(config)
    if args.checkpoint_kind == "stage-a":
        checkpoint_step_number(
            checkpoint_root,
            context=f"{data_profile} Stage A action serving checkpoint",
        )
    else:
        if checkpoint_root.name != "merged_best":
            raise ValueError("rotation_v4 Stage B action serving requires merged_best")
        stage_a_checkpoint = Path(str(config.get("stage_a_checkpoint", ""))).expanduser().resolve()
        stage_a_step = checkpoint_step_number(
            stage_a_checkpoint,
            context="rotation_v4 merged action serving Stage A checkpoint",
        )
        if config.get("stage_a_checkpoint_step") != stage_a_step:
            raise ValueError(
                "rotation_v4 merged config Stage A checkpoint step mismatch: "
                f"recorded={config.get('stage_a_checkpoint_step')!r}, actual={stage_a_step}"
            )
        manifest_path = checkpoint_root / "merge_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("stage_a_checkpoint") != str(stage_a_checkpoint):
            raise ValueError("rotation_v4 merge manifest Stage A checkpoint mismatch")
        if manifest.get("stage_a_checkpoint_step") != stage_a_step:
            raise ValueError("rotation_v4 merge manifest Stage A checkpoint step mismatch")
        delta_dir = Path(str(config.get("stage_b_delta", ""))).expanduser().resolve()
        if manifest.get("stage_b_delta") != str(delta_dir):
            raise ValueError("rotation_v4 merge manifest Stage B delta mismatch")
        assert_identity_matches(
            manifest.get("artifact_identity", {}),
            identity,
            context="rotation_v4 action serve merged identity",
        )
        if manifest.get("norm_stats_sha256") != identity.get("norm_stats_sha256"):
            raise ValueError("rotation_v4 merge manifest norm_stats_sha256 mismatch")
        validate_merged_best_metrics(
            checkpoint_root,
            manifest,
            config,
            context="rotation_v4 action serve",
        )
    summary = validate_norm_stats_identity(
        args.norm_stats_dir / "summary.json",
        identity,
        context=f"{data_profile} action serve norm stats",
    )
    expected_norm_sha = str(
        identity.get(
            "v4_norm_stats_sha256"
            if data_profile in PHASE_ACTION_PROFILES
            else "norm_stats_sha256",
            "",
        )
    )
    if len(expected_norm_sha) != 64 or summary.get("norm_stats_sha256") != expected_norm_sha:
        raise ValueError(
            f"{data_profile} action serve norm_stats_sha256 does not match checkpoint identity"
        )
    return summary


def _precision(value: str | None, config: dict[str, Any]) -> str:
    value = value or str(config.get("precision", "auto"))
    if value == "auto":
        return "bfloat16" if jax.default_backend() in {"gpu", "tpu"} else "float32"
    return value


def _model_config(args: argparse.Namespace, config: dict[str, Any]) -> Pi0Config:
    model_config = Pi0Config(
        dtype=_precision(args.precision, config),
        paligemma_variant=str(config.get("paligemma_variant", "gemma_2b_lora")),
        action_expert_variant=str(config.get("action_expert_variant", "gemma_300m_lora")),
        action_dim=int(config.get("action_dim", ACTION_DIM)),
        action_horizon=int(config.get("action_horizon", ACTION_HORIZON)),
        max_token_len=int(config.get("max_token_len", 200)),
        pi05=True,
        use_state_history=bool(config.get("use_state_history", False)),
        state_history_len=int(config.get("state_history_len", 60)),
        state_history_dim=int(config.get("state_history_dim", OUTPUT_ACTION_DIM)),
        history_hidden_dim=int(config.get("history_hidden_dim", 256)),
        pytorch_compile_mode=None,
    )
    if (model_config.action_horizon, model_config.action_dim) != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            "Forced-phase ablation requires action noise [30,32], but checkpoint config has "
            f"[{model_config.action_horizon},{model_config.action_dim}]"
        )
    if not model_config.use_state_history:
        raise ValueError("Forced-recovery ablation requires the V3 state-history checkpoint")
    if (model_config.state_history_len, model_config.state_history_dim) != (60, OUTPUT_ACTION_DIM):
        raise ValueError(
            "Forced-recovery ablation requires state history [60,7], but checkpoint config has "
            f"[{model_config.state_history_len},{model_config.state_history_dim}]"
        )
    return model_config


def validate_action_noise(value: Any, *, action_horizon: int, action_dim: int) -> np.ndarray:
    if value is None:
        raise ValueError("Action-only ablation requests must include action_noise")
    noise = np.asarray(value, dtype=np.float32)
    expected = (int(action_horizon), int(action_dim))
    if noise.shape != expected:
        raise ValueError(f"Expected action_noise with shape {expected}, got {noise.shape}")
    if not np.isfinite(noise).all():
        raise ValueError("action_noise must contain only finite values")
    return noise


class ActionOnlyAblationPolicy:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        config_path: Path,
        config: dict[str, Any],
        model_config: Pi0Config,
        norm_stats: dict[str, Any],
    ) -> None:
        self._config = model_config
        self._input_transform = build_transform(
            model_config,
            norm_stats=norm_stats,
            use_quantile_norm=True,
            use_delta_actions=True,
            delta_action_dims=OUTPUT_ACTION_DIM,
        )
        self._output_transform = build_action_output_transform(
            norm_stats=norm_stats,
            use_quantile_norm=True,
            use_delta_actions=True,
            delta_action_dims=OUTPUT_ACTION_DIM,
        )

        dtype = jnp.bfloat16 if model_config.dtype == "bfloat16" else jnp.float32
        params_dir = _params_dir(args.checkpoint)
        logging.info("Loading %s params from %s", args.checkpoint_kind, params_dir)
        params = openpi_model.restore_params(params_dir, dtype=dtype)
        if args.checkpoint_kind == "stage-a":
            if str(config.get("checkpoint_format", "")).startswith("stage_b_v3"):
                raise ValueError("--checkpoint-kind=stage-a cannot load a Stage B checkpoint")
            backbone = model_config.load(params)
        else:
            checkpoint_format = str(config.get("checkpoint_format", ""))
            if not checkpoint_format.startswith("stage_b_v3_merged_full"):
                raise ValueError(
                    "--checkpoint-kind=stage-b requires a merged full Stage B checkpoint; "
                    f"got checkpoint_format={checkpoint_format!r}"
                )
            wrapper = StageBV3Model(
                model_config.create(jax.random.key(0)),
                paligemma_width=openpi_gemma.get_config(model_config.paligemma_variant).width,
                need_hidden_dim=int(config.get("need_hidden_dim", 512)),
                need_dropout=float(config.get("need_dropout", 0.1)),
                rngs=nnx.Rngs(jax.random.key(1)),
            )
            graphdef, state = nnx.split(wrapper)
            at.check_pytree_equality(
                expected=state.to_pure_dict(),
                got=params,
                check_shapes=True,
                check_dtypes=False,
            )
            state.replace_by_pure_dict(params)
            wrapper = nnx.merge(graphdef, state)
            wrapper.eval()
            backbone = wrapper.backbone

        backbone.eval()
        self._sample_actions = nnx_utils.module_jit(backbone.sample_actions)
        self._sample_rng = jax.random.key(2)
        self._num_inference_steps = int(args.num_inference_steps)
        if self._num_inference_steps <= 0:
            raise ValueError("--num-inference-steps must be positive")

        state_history_fps = float(config.get("state_history_fps", 30.0))
        if state_history_fps <= 0:
            raise ValueError(f"Checkpoint state_history_fps must be positive, got {state_history_fps}")
        self._metadata = {
            "name": "tactile_vla_action_ablation",
            "action_only_ablation": True,
            "checkpoint_kind": args.checkpoint_kind,
            "checkpoint": str(args.checkpoint.resolve()),
            "prompt_profile": resolve_prompt_profile(config.get("prompt_profile")),
            "data_profile": str(config.get("data_profile", "legacy")),
            "experiment_kind": config.get("experiment_kind"),
            "norm_stats_sha256": (
                checkpoint_artifact_identity(config).get("norm_stats_sha256")
                or checkpoint_artifact_identity(config).get("v4_norm_stats_sha256")
            ),
            "params_dir": str(params_dir),
            "config_path": str(config_path),
            "action_horizon": model_config.action_horizon,
            "action_dim": model_config.action_dim,
            "output_action_dim": OUTPUT_ACTION_DIM,
            "num_inference_steps": self._num_inference_steps,
            "use_state_history": model_config.use_state_history,
            "state_history_len": model_config.state_history_len,
            "state_history_dim": model_config.state_history_dim,
            "state_history_fps": state_history_fps,
            "supports_action_noise": True,
            "requires_action_noise": True,
            "action_noise_shape": [model_config.action_horizon, model_config.action_dim],
            # V5 training config contains a 405k-entry phase lookup and is about
            # 178 MB.  Sending it in the websocket handshake also duplicates it
            # into every trial log, while none of those rows are inference inputs.
            "config": metadata_config_summary(config),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @staticmethod
    def _observation(transformed: dict[str, Any]) -> Observation:
        arrays = jax.tree.map(np.asarray, transformed)
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], arrays)
        return Observation.from_dict(batched)

    def _prepare(self, request: dict[str, Any]) -> tuple[dict[str, Any], Observation, np.ndarray]:
        inputs = dict(request)
        inputs.pop("mode", None)
        noise = validate_action_noise(
            inputs.pop("action_noise", None),
            action_horizon=self._config.action_horizon,
            action_dim=self._config.action_dim,
        )
        # These fields are useful to the client log but are not model inputs.
        inputs.pop("noise_seed", None)
        inputs.pop("noise_phase", None)
        inputs.pop("noise_index", None)
        transformed = self._input_transform(inputs)
        return transformed, self._observation(transformed), noise

    def _actions(
        self,
        transformed: dict[str, Any],
        observation: Observation,
        noise: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sampled = self._sample_actions(
            self._sample_rng,
            observation,
            num_steps=self._num_inference_steps,
            noise=jnp.asarray(noise)[None, ...],
        )
        raw_actions = np.asarray(sampled[0], dtype=np.float32)
        if raw_actions.shape != (self._config.action_horizon, self._config.action_dim):
            raise ValueError(f"Expected raw model actions [30,32], got {raw_actions.shape}")
        if not np.isfinite(raw_actions).all():
            raise ValueError("Model produced non-finite raw actions")
        restored = self._output_transform(
            {"state": transformed["state"], "actions": raw_actions.copy()}
        )
        actions = np.asarray(restored["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape != (self._config.action_horizon, self._config.action_dim):
            raise ValueError(f"Expected transformed actions [30,32], got {actions.shape}")
        actions = actions[:, :OUTPUT_ACTION_DIM]
        if not np.isfinite(actions).all():
            raise ValueError("Output transform produced non-finite actions")
        return raw_actions, actions

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = str(request.get("mode", "execution"))
        if mode not in {"action", "execution"}:
            raise ValueError(f"Action-only ablation server does not support mode={mode!r}")
        started = time.monotonic()
        transformed, observation, noise = self._prepare(request)
        raw_actions, actions = self._actions(transformed, observation, noise)
        return {
            "raw_model_actions": raw_actions,
            "actions": actions,
            "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000.0},
        }


def warm_up(policy: ActionOnlyAblationPolicy) -> dict[str, Any]:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros((OUTPUT_ACTION_DIM,), dtype=np.float32)
    response = policy.infer(
        {
            "mode": "execution",
            "observation/image": image,
            "observation/wrist_image": image,
            "observation/state": state,
            "observation/state_history": np.zeros(
                (policy._config.state_history_len, policy._config.state_history_dim),  # noqa: SLF001
                dtype=np.float32,
            ),
            "observation/state_history_mask": np.ones(
                (policy._config.state_history_len,),  # noqa: SLF001
                dtype=np.bool_,
            ),
            "prompt": build_execution_prompt(
                instruction="dry run",
                tactile_caption=(
                    "Touch[area=none; Fx=near_zero; Fy=near_zero; "
                    "Fz=near_zero; rotation=none]"
                ),
                prompt_profile=policy.metadata["prompt_profile"],
            ),
            "action_noise": np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32),
        }
    )
    return {
        "checkpoint_kind": policy.metadata["checkpoint_kind"],
        "raw_model_actions_shape": list(np.asarray(response["raw_model_actions"]).shape),
        "actions_shape": list(np.asarray(response["actions"]).shape),
        "action_noise_shape": policy.metadata["action_noise_shape"],
        "state_history_shape": [
            policy.metadata["state_history_len"],
            policy.metadata["state_history_dim"],
        ],
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    config_path, config = _find_config(args.checkpoint)
    validate_v4_norm_artifacts(args, config)
    model_config = _model_config(args, config)
    norm_stats = normalize.load(args.norm_stats_dir)
    policy = ActionOnlyAblationPolicy(
        args=args,
        config_path=config_path,
        config=config,
        model_config=model_config,
        norm_stats=norm_stats,
    )
    # The V5 training config owns a large per-frame lookup.  The policy keeps
    # only its compact metadata summary, so release the parsed training config
    # before warm-up instead of pinning it for the full server lifetime.
    del config
    logging.info("Warming up deterministic action-only inference")
    summary = warm_up(policy)
    logging.info("Action-only warm-up complete: %s", json.dumps(summary, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    hostname = socket.gethostname()
    logging.info("Serving action-only ablation on %s (%s):%d", hostname, args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
