#!/usr/bin/env python3
"""Serve the unified tactile-VLA V3 action, monitor, diagnosis, and planner."""

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
from openpi.models import tokenizer as openpi_tokenizer
from tactile_vla.vla.openpi_bridge import build_action_output_transform
from tactile_vla.vla.openpi_bridge import build_structured_inference_transform
from tactile_vla.vla.openpi_bridge import build_transform
from tactile_vla.vla.stage_b_v3_model import StageBV3Model
from tactile_vla.vla.structured_generation import constrained_greedy_generate
from tactile_vla.vla.structured_generation import constrained_greedy_generate_from_prefill
from tactile_vla.vla.structured_text import failure_grammar
from tactile_vla.vla.structured_text import recovery_grammar
from tactile_vla.vla.structured_text import ConstrainedTokenGrammar
from tactile_vla.vla.structured_text import legal_failure_reasons
from tactile_vla.vla.structured_text import legal_recovery_plans
from tactile_vla.vla.prompts import build_assessment_prompt
from tactile_vla.vla.prompts import build_execution_prompt
from tactile_vla.vla.prompts import build_failure_prompt
from tactile_vla.vla.prompts import build_reasoning_prompt
from tactile_vla.vla.artifacts import load_checkpoint_prompt_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="V3 Stage B step directory or params directory")
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-action-dim", type=int, default=7)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--need-recovery-threshold", type=float, default=0.5)
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"))
    parser.add_argument("--reasoning-max-token-len", type=int)
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _find_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    candidates = [
        path / "config.json",
        path.parent / "config.json",
        path.parent.parent / "config.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(f"Cannot find config.json near V3 checkpoint: {path}")


def _params_dir(path: Path) -> Path:
    path = path.resolve()
    return path / "params" if (path / "params").exists() else path


def _precision(value: str | None, config: dict[str, Any]) -> str:
    value = value or str(config.get("precision", "auto"))
    if value == "auto":
        return "bfloat16" if jax.default_backend() in {"gpu", "tpu"} else "float32"
    return value


def _model_config(args: argparse.Namespace, config: dict[str, Any]) -> Pi0Config:
    return Pi0Config(
        dtype=_precision(args.precision, config),
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


class TactileVLAPolicyV3:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        config: dict[str, Any],
        model_config: Pi0Config,
        norm_stats: dict[str, Any] | None,
    ) -> None:
        self._config = model_config
        state_history_fps = float(config.get("state_history_fps", 30.0))
        if state_history_fps <= 0:
            raise ValueError(
                f"Checkpoint state_history_fps must be positive, got {state_history_fps}"
            )
        reasoning_max_len = int(
            args.reasoning_max_token_len
            or config.get("reasoning_max_token_len", 320)
        )
        self._tokenizer = openpi_tokenizer.PaligemmaTokenizer(reasoning_max_len)
        def encode(text: str) -> list[int]:
            return self._tokenizer.encode_text(text, add_eos=True)
        configured_failure = config.get("failure_grammar")
        configured_plan = config.get("recovery_grammar")
        self._failure_grammar = (
            ConstrainedTokenGrammar(configured_failure, encode=encode)
            if configured_failure is not None
            else failure_grammar(encode)
        )
        self._plan_grammar = (
            ConstrainedTokenGrammar(configured_plan, encode=encode)
            if configured_plan is not None
            else recovery_grammar(encode)
        )
        grammar_profile = str(config.get("grammar_profile", "v3_full_v1"))
        if grammar_profile != "v3_full_v1":
            raise ValueError(f"Unsupported checkpoint grammar_profile={grammar_profile!r}")
        if (
            self._failure_grammar.texts != legal_failure_reasons()
            or self._plan_grammar.texts != legal_recovery_plans()
        ):
            raise ValueError("Checkpoint does not retain the complete V3 grammar")
        self._prompt_profile = load_checkpoint_prompt_profile(config)
        self._regular_transform = build_transform(
            model_config,
            norm_stats=norm_stats,
            use_quantile_norm=not args.no_norm,
            delta_action_dims=args.output_action_dim,
        )
        self._assessment_transform = build_structured_inference_transform(
            model_config,
            tokenizer=self._tokenizer,
            max_len=model_config.max_token_len,
            norm_stats=norm_stats,
            use_quantile_norm=not args.no_norm,
        )
        self._reasoning_transform = build_structured_inference_transform(
            model_config,
            tokenizer=self._tokenizer,
            max_len=reasoning_max_len,
            norm_stats=norm_stats,
            use_quantile_norm=not args.no_norm,
        )
        self._output_transform = build_action_output_transform(
            norm_stats=norm_stats,
            use_quantile_norm=not args.no_norm,
            delta_action_dims=args.output_action_dim,
        )

        backbone = model_config.create(jax.random.key(0))
        model = StageBV3Model(
            backbone,
            paligemma_width=openpi_gemma.get_config(model_config.paligemma_variant).width,
            need_hidden_dim=int(config.get("need_hidden_dim", 512)),
            need_dropout=float(config.get("need_dropout", 0.1)),
            rngs=nnx.Rngs(jax.random.key(1)),
        )
        graphdef, state = nnx.split(model)
        dtype = jnp.bfloat16 if model_config.dtype == "bfloat16" else jnp.float32
        params = openpi_model.restore_params(_params_dir(args.checkpoint), dtype=dtype)
        at.check_pytree_equality(
            expected=state.to_pure_dict(),
            got=params,
            check_shapes=True,
            check_dtypes=False,
        )
        state.replace_by_pure_dict(params)
        self._model = nnx.merge(graphdef, state)
        self._model.eval()
        self._sample_actions = nnx_utils.module_jit(self._model.backbone.sample_actions)
        self._need_logits = nnx_utils.module_jit(self._model.need_recovery_logits)
        self._assessment_prefill = nnx_utils.module_jit(self._model.assessment_prefill)
        self._generation_prefill = nnx_utils.module_jit(self._model.generation_prefill)
        self._generation_step = nnx_utils.module_jit(self._model.generation_step)
        self._rng = jax.random.key(2)
        self._num_inference_steps = int(args.num_inference_steps)
        self._need_threshold = float(args.need_recovery_threshold)
        self._output_action_dim = int(args.output_action_dim)
        self._metadata = {
            "name": "tactile_vla_policy_v3",
            "stage_b_version": "v3_autoregressive",
            "prompt_profile": self._prompt_profile,
            "grammar_profile": grammar_profile,
            "checkpoint": str(args.checkpoint),
            "action_horizon": model_config.action_horizon,
            "action_dim": model_config.action_dim,
            "output_action_dim": self._output_action_dim,
            "use_state_history": model_config.use_state_history,
            "state_history_len": model_config.state_history_len,
            "state_history_dim": model_config.state_history_dim,
            "state_history_fps": state_history_fps,
            "need_recovery_threshold": self._need_threshold,
            "supports_step_monitor": True,
            "supports_shared_assessment": True,
            "supports_failure_generation": True,
            "supports_recovery_generation": True,
            "reasoning_max_token_len": reasoning_max_len,
            "reasoning_window_frames": int(config.get("reasoning_window_frames", 15)),
            "training_target_coverage": config.get("training_target_coverage"),
            "config": config,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @staticmethod
    def _observation(transformed: dict[str, Any]) -> Observation:
        arrays = jax.tree.map(np.asarray, transformed)
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], arrays)
        return Observation.from_dict(batched)

    def _prepare(self, request: dict[str, Any], *, mode: str) -> tuple[dict[str, Any], Observation]:
        inputs = dict(request)
        inputs.pop("mode", None)
        if mode in {"assessment", "monitor", "failure"}:
            transform = self._assessment_transform
        elif mode == "reasoning":
            transform = self._reasoning_transform
        else:
            transform = self._regular_transform
        transformed = transform(inputs)
        return transformed, self._observation(transformed)

    def _need_response_from_logits(self, logits: jax.Array) -> dict[str, Any]:
        probabilities = np.asarray(jax.nn.softmax(logits[0]), dtype=np.float32)
        return {
            "need_recovery": bool(float(probabilities[1]) >= self._need_threshold),
            "need_recovery_probs": probabilities.tolist(),
        }

    def _need_response(self, observation: Observation) -> dict[str, Any]:
        return self._need_response_from_logits(self._need_logits(observation))

    def _actions(self, transformed: dict[str, Any], observation: Observation) -> np.ndarray:
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self._sample_actions(
            sample_rng,
            observation,
            num_steps=self._num_inference_steps,
        )
        restored = self._output_transform(
            {"state": transformed["state"], "actions": np.asarray(actions[0])}
        )
        result = np.asarray(restored["actions"], dtype=np.float32)
        return result[:, : self._output_action_dim]

    def _generate(self, observation: Observation, *, failure: bool) -> str:
        grammar = self._failure_grammar if failure else self._plan_grammar
        return constrained_greedy_generate(
            self._model.backbone,
            observation,
            grammar,
            prefill_fn=lambda _backbone, obs, compact_ids: self._generation_prefill(
                obs, compact_ids
            ),
            step_fn=lambda _backbone, *values: self._generation_step(*values),
        )

    def _assessment(self, observation: Observation) -> dict[str, Any]:
        """Classify and conditionally diagnose from one shared VLM prefix."""

        compact_ids = jnp.asarray(self._failure_grammar.compact_token_ids, dtype=jnp.int32)
        need_logits, text_logits, kv_cache, prefix_mask, semantic_position = (
            self._assessment_prefill(observation, compact_ids)
        )
        response = self._need_response_from_logits(need_logits)
        if response["need_recovery"]:
            response["failure_reason"] = constrained_greedy_generate_from_prefill(
                self._model.backbone,
                self._failure_grammar,
                logits=text_logits,
                kv_cache=kv_cache,
                prefix_mask=prefix_mask,
                semantic_position=semantic_position,
                step_fn=lambda _backbone, *values: self._generation_step(*values),
            )
        return response

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = str(request.get("mode", "execution"))
        if mode not in {"execution", "assessment", "monitor", "failure", "reasoning"}:
            raise ValueError(f"Unknown V3 inference mode: {mode!r}")
        started = time.monotonic()
        transformed, observation = self._prepare(request, mode=mode)
        if mode == "assessment":
            response = self._assessment(observation)
        elif mode == "failure":
            response = {"failure_reason": self._generate(observation, failure=True)}
        elif mode == "reasoning":
            response = {"recovery_plan": self._generate(observation, failure=False)}
        elif mode == "monitor":
            # Compatibility path for older V3 clients. New clients use the
            # shared ``assessment`` request instead.
            response = self._need_response(observation)
        else:
            # The asynchronous assessment stream is the sole recovery signal.
            # Avoid a redundant prefix forward in the action-chunk request.
            response = {"actions": self._actions(transformed, observation)}
        response["policy_timing"] = {"infer_ms": (time.monotonic() - started) * 1000.0}
        return response


def warm_up(policy: TactileVLAPolicyV3) -> dict[str, Any]:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros((7,), dtype=np.float32)
    base = {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/state": state,
    }
    if policy._config.use_state_history:  # noqa: SLF001
        base.update(
            {
                "observation/state_history": np.zeros(
                    (policy._config.state_history_len, policy._config.state_history_dim),  # noqa: SLF001
                    dtype=np.float32,
                ),
                "observation/state_history_mask": np.ones(
                    (policy._config.state_history_len,),  # noqa: SLF001
                    dtype=np.bool_,
                ),
            }
        )
    execution = policy.infer(
        base
        | {
            "mode": "execution",
            "prompt": build_execution_prompt(
                instruction="dry run",
                tactile_caption="Touch[rotation=none]",
                prompt_profile=policy._prompt_profile,  # noqa: SLF001
            ),
        }
    )
    assessment = policy.infer(
        base
        | {
            "mode": "assessment",
            "prompt": build_assessment_prompt(
                instruction="dry run",
                tactile_caption="Touch[rotation=none]",
                prompt_profile=policy._prompt_profile,  # noqa: SLF001
            ),
        }
    )
    failure = policy.infer(
        base
        | {
            "mode": "failure",
            "prompt": build_failure_prompt(
                instruction="dry run",
                tactile_caption="Touch[rotation=clockwise]",
                prompt_profile=policy._prompt_profile,  # noqa: SLF001
            ),
        }
    )
    reasoning = policy.infer(
        base
        | {
            "mode": "reasoning",
            "prompt": build_reasoning_prompt(
                instruction="dry run",
                failed_tactile_caption="Touch[rotation=clockwise]",
                failure_recovery_memory=[
                    {
                        "recovery_plan": "initial plan",
                        "failure_reason": "failure_reason=rotate right,grasp appropriate.",
                    }
                ],
                prompt_profile=policy._prompt_profile,  # noqa: SLF001
            ),
        }
    )
    return {
        "actions_shape": list(np.asarray(execution["actions"]).shape),
        "prompt_profile": policy.metadata["prompt_profile"],
        "grammar_profile": policy.metadata["grammar_profile"],
        "use_state_history": policy.metadata["use_state_history"],
        "state_history_shape": (
            [
                policy.metadata["state_history_len"],
                policy.metadata["state_history_dim"],
            ]
            if policy.metadata["use_state_history"]
            else None
        ),
        "need_recovery": bool(assessment["need_recovery"]),
        "assessment_failure_reason": assessment.get("failure_reason"),
        "failure_reason": failure["failure_reason"],
        "recovery_plan": reasoning["recovery_plan"],
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    config = _find_config(args.checkpoint)
    model_config = _model_config(args, config)
    norm_stats = None if args.no_norm else normalize.load(args.norm_stats_dir)
    policy = TactileVLAPolicyV3(
        args=args,
        config=config,
        model_config=model_config,
        norm_stats=norm_stats,
    )
    logging.info("Warming up all V3 inference paths")
    summary = warm_up(policy)
    logging.info("V3 warm-up complete: %s", json.dumps(summary, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    hostname = socket.gethostname()
    logging.info("Serving V3 policy on %s (%s):%d", hostname, args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
