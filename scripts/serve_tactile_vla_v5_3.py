#!/usr/bin/env python3
"""Serve V5.2 Stage A actions plus the V5.3 adjustment-end classifier."""

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
os.environ.setdefault("USE_TF", "0")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import gemma as openpi_gemma
from openpi.models import model as openpi_model
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.serving import websocket_policy_server
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.shared import normalize
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.openpi_bridge import build_action_output_transform
from tactile_vla.vla.openpi_bridge import build_transform
from tactile_vla.vla.prompts import build_phase_prompt
from tactile_vla.vla.stage_b_v3_checkpoint import cast_frozen_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import CHECKPOINT_FORMAT
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import delta_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import merge_delta_params
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import MULTITASK_CHECKPOINT_FORMAT
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import multitask_trainable_filter
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import parameter_tree_sha256
from tactile_vla.vla.v5_3_adjustment_end_checkpoint import trainable_filter
from tactile_vla.vla.v5_3_adjustment_end_data import DATA_PROFILE as ADJUSTMENT_END_DATA_PROFILE
from tactile_vla.vla.v5_3_adjustment_end_data import LABEL_POLICY
from tactile_vla.vla.v5_3_adjustment_end_model import AdjustmentEndModel
from tactile_vla.vla.v5_3_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
from tactile_vla.vla.v5_3_phase_change import QPOS_BIN_COUNT
from tactile_vla.vla.v5_3_phase_change import QPOS_SAMPLE_OFFSETS
from tactile_vla.vla.v5_3_phase_change import build_adjustment_end_prompt
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats


ACTION_HORIZON = 30
ACTION_DIM = 32
OUTPUT_ACTION_DIM = 7
PHASE_CHANGE_TIMEOUT_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--stage-a-checkpoint", type=Path, required=True)
    parser.add_argument("--adjustment-end-checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument(
        "--adjustment-end-threshold-override",
        type=float,
        help=(
            "Manual robot-probe threshold. The server advertises this as an experimental "
            "override and does not modify checkpoint metadata."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _find_config(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    for candidate in (path / "config.json", path.parent / "config.json", path.parent.parent / "config.json"):
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text())
    raise FileNotFoundError(f"Cannot find config.json near {path}")


def _params_dir(path: Path) -> Path:
    resolved = path.resolve()
    return resolved / "params" if (resolved / "params").is_dir() else resolved


def _restore_delta_params(path: Path):
    """Restore a directly exported V5.3 delta tree (no top-level params wrapper)."""

    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(path.resolve())
    if not isinstance(restored, dict) or not restored:
        raise ValueError(f"V5.3 delta checkpoint is not a non-empty parameter tree: {path}")
    return restored


def _model_config(config: dict[str, Any], *, max_token_len: int, precision: str) -> Pi0Config:
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
        use_state_history=True,
        state_history_len=int(config["state_history_len"]),
        state_history_dim=int(config["state_history_dim"]),
        history_hidden_dim=int(config["history_hidden_dim"]),
        pytorch_compile_mode=None,
    )
    if (result.action_horizon, result.action_dim) != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError("V5.3 requires action shape [30,32]")
    if (result.state_history_len, result.state_history_dim) != (60, 7):
        raise ValueError("V5.3 requires continuous state history [60,7]")
    return result


def _validate_configs(args, stage_a_path, stage_a, head_path, head, metadata):
    if args.stage_a_checkpoint.resolve().name != "15000":
        raise ValueError("V5.3 server requires the fixed Stage A step 15000")
    stage_required = {
        "data_profile": "rotation_phase_v5_adjustment_v2",
        "prompt_profile": "phase_v2",
        "experiment_kind": "phase_prompt_h30_terminal_hold",
        "num_steps": 15000,
        "seed": 42,
    }
    checkpoint_format = str(head.get("checkpoint_format", ""))
    if checkpoint_format == CHECKPOINT_FORMAT:
        head_required = {
            "prompt_profile": "phase_change_v1",
            "experiment_kind": "adjustment_end_qpos_h30_text",
            "num_steps": 4000,
            "phase_change_max_token_len": 512,
            "checkpoint_format": CHECKPOINT_FORMAT,
        }
    elif checkpoint_format == MULTITASK_CHECKPOINT_FORMAT:
        head_required = {
            "prompt_profile": "phase_change_v1",
            "experiment_kind": "adjustment_end_action_multitask_v1",
            "num_steps": 8000,
            "phase_change_max_token_len": 512,
            "checkpoint_format": MULTITASK_CHECKPOINT_FORMAT,
        }
    else:
        raise ValueError(f"Unsupported V5.3 checkpoint format: {checkpoint_format!r}")
    adjustment_end_data_profile = str(head.get("data_profile", ""))
    if adjustment_end_data_profile not in {
        "rotation_phase_v5_adjustment_end_v1",
        ADJUSTMENT_END_DATA_PROFILE,
    }:
        raise ValueError(
            f"Unsupported V5.3 adjustment_end data profile: {adjustment_end_data_profile!r}"
        )
    if adjustment_end_data_profile == ADJUSTMENT_END_DATA_PROFILE:
        if head.get("label_policy") != LABEL_POLICY:
            raise ValueError("V5.3 R-10..R+5 checkpoint label policy mismatch")
        if metadata.get("label_policy") != LABEL_POLICY:
            raise ValueError("V5.3 R-10..R+5 metadata label policy mismatch")
    for config, expected, name in (
        (stage_a, stage_required, "Stage A"),
        (head, head_required, "adjustment_end"),
    ):
        mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
        if mismatch:
            raise ValueError(f"V5.3 {name} config mismatch: {mismatch}")
    if Path(str(head["stage_a_checkpoint"])).resolve() != args.stage_a_checkpoint.resolve():
        raise ValueError("V5.3 head was trained with a different Stage A checkpoint")
    if head["stage_a_config_sha256"] != sha256_file(stage_a_path):
        raise ValueError("V5.3 head Stage A config SHA mismatch")
    if int(metadata.get("official_step", -1)) != int(head_required["num_steps"]):
        raise ValueError(
            "V5.3 metadata/checkpoint step mismatch: "
            f"metadata={metadata.get('official_step')}, config={head_required['num_steps']}"
        )
    threshold = float(metadata.get("adjustment_end_threshold", -1.0))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("V5.3 adjustment_end threshold is invalid")
    return threshold, checkpoint_format


def _resolve_runtime_threshold(
    checkpoint_threshold: float,
    manual_override: float | None,
) -> tuple[float, bool]:
    if manual_override is None:
        return float(checkpoint_threshold), False
    if not 0.0 <= manual_override <= 1.0:
        raise ValueError("--adjustment-end-threshold-override must be in [0,1]")
    return float(manual_override), True


class V53Policy:
    def __init__(self, *, args, stage_a_config_path, stage_a_config, head_config_path, head_config):
        step_dir = args.adjustment_end_checkpoint.resolve()
        metadata_path = step_dir / "adjustment_end_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        final_metadata = json.loads(metadata_path.read_text())
        checkpoint_threshold, self._checkpoint_format = _validate_configs(
            args, stage_a_config_path, stage_a_config, head_config_path, head_config, final_metadata
        )
        self._checkpoint_threshold = checkpoint_threshold
        self._threshold, self._manual_threshold_override = _resolve_runtime_threshold(
            checkpoint_threshold,
            args.adjustment_end_threshold_override,
        )
        self._experimental_override = bool(
            final_metadata.get("experimental_override", False)
            or self._manual_threshold_override
        )
        self._accepted_for_robot = bool(
            final_metadata.get("accepted_for_robot", False)
            and not self._manual_threshold_override
        )
        if self._experimental_override:
            logging.warning(
                "Loading NON-OFFICIAL adjustment_end robot probe: accepted_for_robot=%s "
                "h30_gate_passed=%s threshold_policy=%s checkpoint_threshold=%.8f "
                "runtime_threshold=%.8f manual_override=%s",
                self._accepted_for_robot,
                final_metadata.get("h30_gate_passed"),
                final_metadata.get("threshold_policy"),
                self._checkpoint_threshold,
                self._threshold,
                self._manual_threshold_override,
            )
        self._action_config = _model_config(
            stage_a_config,
            max_token_len=int(stage_a_config.get("max_token_len", 200)),
            precision=args.precision,
        )
        self._phase_config = _model_config(
            stage_a_config,
            max_token_len=PHASE_CHANGE_MAX_TOKEN_LEN,
            precision=args.precision,
        )
        norm_stats = normalize.load(args.norm_stats_dir)
        self._action_input = build_transform(
            self._action_config,
            norm_stats=norm_stats,
            use_quantile_norm=True,
            use_delta_actions=True,
            delta_action_dims=OUTPUT_ACTION_DIM,
        )
        self._phase_input = build_transform(
            self._phase_config,
            norm_stats=norm_stats,
            use_quantile_norm=True,
            use_delta_actions=False,
        )
        self._action_output = build_action_output_transform(
            norm_stats=norm_stats,
            use_quantile_norm=True,
            use_delta_actions=True,
            delta_action_dims=OUTPUT_ACTION_DIM,
        )

        # Restore the checkpoint's original mixed dtypes, then reproduce the
        # exact training-time frozen-parameter cast below.  Casting every
        # floating-point leaf here would incorrectly convert the trainable
        # PaliGemma LoRA and change the audited backbone SHA.
        backbone_params = openpi_model.restore_params(_params_dir(args.stage_a_checkpoint))
        backbone = self._action_config.load(backbone_params)
        wrapper = AdjustmentEndModel(
            backbone,
            paligemma_width=openpi_gemma.get_config(self._action_config.paligemma_variant).width,
            rngs=nnx.Rngs(jax.random.key(1)),
        )
        graphdef, base_state = nnx.split(wrapper)
        if self._checkpoint_format == MULTITASK_CHECKPOINT_FORMAT:
            overlay_filter = multitask_trainable_filter()
            overlay_dir = step_dir / "delta_params"
        else:
            overlay_filter = trainable_filter()
            overlay_dir = step_dir / "head_params"
        frozen_filter = nnx.All(nnx.Param, nnx.Not(overlay_filter))
        base_state = cast_frozen_params(base_state, frozen_filter)
        initial_backbone_sha = parameter_tree_sha256(base_state["backbone"])
        expected_initial_sha = str(head_config["stage_a_backbone_parameter_tree_sha256"])
        if initial_backbone_sha != expected_initial_sha:
            raise ValueError("Restored V5.2 Stage A backbone parameter-tree SHA mismatch")
        overlay_template = delta_params(base_state, overlay_filter)
        if not overlay_dir.is_dir():
            raise FileNotFoundError(overlay_dir)
        restored_head = _restore_delta_params(overlay_dir)
        at.check_pytree_equality(
            expected=overlay_template.to_pure_dict(),
            got=restored_head,
            check_shapes=True,
            check_dtypes=False,
        )
        overlay_template.replace_by_pure_dict(restored_head)
        wrapper = nnx.merge(graphdef, merge_delta_params(base_state, overlay_template))
        wrapper.eval()
        actual_backbone_sha = parameter_tree_sha256(nnx.state(wrapper)["backbone"])
        expected_backbone_sha = str(
            final_metadata.get(
                "trained_backbone_parameter_tree_sha256",
                head_config["stage_a_backbone_parameter_tree_sha256"],
            )
        )
        if actual_backbone_sha != expected_backbone_sha:
            raise ValueError("Restored V5.3 inference backbone parameter-tree SHA mismatch")

        self._sample_actions = nnx_utils.module_jit(wrapper.backbone.sample_actions)
        self._adjustment_end_logits = nnx_utils.module_jit(wrapper.adjustment_end_logits)
        self._sample_rng = jax.random.key(2)
        self._num_inference_steps = int(args.num_inference_steps)
        if self._num_inference_steps <= 0:
            raise ValueError("num inference steps must be positive")
        caption_source = head_config["caption_source"]
        self._metadata = {
            "name": "tactile_vla_v5_3",
            "checkpoint_kind": (
                "stage-a-plus-adjustment-end-paligemma-lora"
                if self._checkpoint_format == MULTITASK_CHECKPOINT_FORMAT
                else "stage-a-plus-adjustment-end-head"
            ),
            "checkpoint": str(args.stage_a_checkpoint.resolve()),
            "prompt_profile": "phase_v2",
            "data_profile": "rotation_phase_v5_adjustment_v2",
            "experiment_kind": "phase_prompt_h30_terminal_hold",
            "action_only_ablation": False,
            "supports_action_noise": True,
            "requires_action_noise": True,
            "supports_adjustment_end": True,
            "adjustment_end_threshold": self._threshold,
            "adjustment_end_checkpoint_threshold": self._checkpoint_threshold,
            "adjustment_end_manual_threshold_override": self._manual_threshold_override,
            "adjustment_end_experimental_override": self._experimental_override,
            "adjustment_end_accepted_for_robot": self._accepted_for_robot,
            "adjustment_end_h30_gate_passed": bool(final_metadata.get("h30_gate_passed", False)),
            "adjustment_end_threshold_policy": (
                "manual_robot_probe_override"
                if self._manual_threshold_override
                else final_metadata.get("threshold_policy", "official_v5_3")
            ),
            "adjustment_end_checkpoint_format": self._checkpoint_format,
            "adjustment_end_data_profile": head_config["data_profile"],
            "adjustment_end_label_policy": head_config.get(
                "label_policy",
                {
                    "boundary": "rexecution_frame",
                    "positive_start_offset_inclusive": -5,
                    "positive_end_offset_inclusive": 0,
                    "valid_end_offset_inclusive": 0,
                },
            ),
            "phase_change_prompt_profile": "phase_change_v1",
            "phase_change_max_token_len": PHASE_CHANGE_MAX_TOKEN_LEN,
            "qpos_h30_sample_offsets": list(QPOS_SAMPLE_OFFSETS),
            "qpos_bin_count": QPOS_BIN_COUNT,
            "qpos_discretization_extra_clip": False,
            "captioner_checkpoint_sha256": caption_source["checkpoint"]["sha256"],
            "captioner_window_size": int(caption_source["window_size"]),
            "phase_change_timeout_seconds": PHASE_CHANGE_TIMEOUT_SECONDS,
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "action_noise_shape": [ACTION_HORIZON, ACTION_DIM],
            "output_action_dim": OUTPUT_ACTION_DIM,
            "state_history_len": 60,
            "state_history_dim": 7,
            "state_history_fps": 30.0,
            "use_state_history": True,
            "stage_a_checkpoint": str(args.stage_a_checkpoint.resolve()),
            "adjustment_end_checkpoint": str(step_dir),
            "stage_a_backbone_parameter_tree_sha256": initial_backbone_sha,
            "inference_backbone_parameter_tree_sha256": actual_backbone_sha,
        }

    @property
    def metadata(self):
        return self._metadata

    @staticmethod
    def _observation(transformed):
        arrays = jax.tree.map(np.asarray, transformed)
        return Observation.from_dict(jax.tree.map(lambda value: jnp.asarray(value)[None, ...], arrays))

    @staticmethod
    def _clean_inputs(request):
        inputs = dict(request)
        inputs.pop("mode", None)
        for key in ("noise_seed", "noise_phase", "noise_index"):
            inputs.pop(key, None)
        return inputs

    def _infer_action(self, request):
        inputs = self._clean_inputs(request)
        noise = np.asarray(inputs.pop("action_noise", None), dtype=np.float32)
        if noise.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(noise).all():
            raise ValueError("execution request requires finite action_noise [30,32]")
        transformed = self._action_input(inputs)
        observation = self._observation(transformed)
        raw = np.asarray(self._sample_actions(
            self._sample_rng,
            observation,
            num_steps=self._num_inference_steps,
            noise=jnp.asarray(noise)[None, ...],
        )[0], dtype=np.float32)
        restored = self._action_output({"state": transformed["state"], "actions": raw.copy()})
        actions = np.asarray(restored["actions"], dtype=np.float32)[:, :OUTPUT_ACTION_DIM]
        if raw.shape != (ACTION_HORIZON, ACTION_DIM) or actions.shape != (ACTION_HORIZON, 7):
            raise ValueError("V5.3 action output shape mismatch")
        return {"raw_model_actions": raw, "actions": actions}

    def _infer_adjustment_end(self, request):
        inputs = self._clean_inputs(request)
        inputs.pop("action_noise", None)
        prompt = str(inputs.get("prompt", ""))
        if not prompt.startswith("Mode: adjustment.\n") or "\nqpos_h30:[[" not in prompt:
            raise ValueError("adjustment_end request does not use the phase_change_v1 prompt")
        transformed = self._phase_input(inputs)
        logits = self._adjustment_end_logits(self._observation(transformed))
        probabilities = np.asarray(jax.device_get(jax.nn.softmax(logits, axis=-1))[0], dtype=np.float32)
        return {
            "adjustment_end": bool(float(probabilities[1]) >= self._threshold),
            "adjustment_end_probs": probabilities,
        }

    def infer(self, request):
        started = time.monotonic()
        mode = str(request.get("mode", "execution"))
        if mode in {"execution", "action"}:
            result = self._infer_action(request)
        elif mode == "adjustment_end":
            result = self._infer_adjustment_end(request)
        else:
            raise ValueError(f"V5.3 server does not support mode={mode!r}")
        result["policy_timing"] = {"infer_ms": (time.monotonic() - started) * 1000.0}
        return result


def warm_up(policy: V53Policy) -> dict[str, Any]:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros(7, dtype=np.float32)
    history = np.zeros((60, 7), dtype=np.float32)
    common = {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/state": state,
        "observation/state_history": history,
        "observation/state_history_mask": np.ones(60, dtype=np.bool_),
    }
    action = policy.infer({
        **common,
        "mode": "execution",
        "prompt": build_phase_prompt(
            phase="execution", instruction="dry run", recovery_plan="move left", prompt_profile="phase_v2"
        ),
        "action_noise": np.zeros((30, 32), dtype=np.float32),
    })
    phase_prompt, _ = build_adjustment_end_prompt(
        instruction="dry run",
        tactile_caption="Touch[area=none; Fx=near_zero; Fy=near_zero; Fz=near_zero; rotation=none]",
        recovery_plan="move left",
        qpos_h30=np.zeros((30, 7), dtype=np.float32),
        stats=StateQuantileStats(q01=np.zeros(7), q99=np.ones(7)),
    )
    phase = policy.infer({**common, "mode": "adjustment_end", "prompt": phase_prompt})
    return {
        "action_shape": list(np.asarray(action["actions"]).shape),
        "adjustment_end": bool(phase["adjustment_end"]),
        "adjustment_end_probs": np.asarray(phase["adjustment_end_probs"]).tolist(),
        "warm_modes": ["execution", "adjustment_end"],
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    stage_a_path, stage_a_config = _find_config(args.stage_a_checkpoint)
    head_path, head_config = _find_config(args.adjustment_end_checkpoint)
    policy = V53Policy(
        args=args,
        stage_a_config_path=stage_a_path,
        stage_a_config=stage_a_config,
        head_config_path=head_path,
        head_config=head_config,
    )
    summary = warm_up(policy)
    logging.info("V5.3 execution and adjustment_end warm-up complete: %s", summary)
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    logging.info("Serving V5.3 on %s (%s):%d", socket.gethostname(), args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
