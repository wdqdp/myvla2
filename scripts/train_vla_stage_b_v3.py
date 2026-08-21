#!/usr/bin/env python3
"""Train unified tactile-VLA V3 action, monitor, diagnosis, and planning tasks."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator
import dataclasses
import json
import logging
import os
from pathlib import Path
import platform
import re
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
os.environ.setdefault("USE_TF", "0")

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from torch.utils.data import DataLoader
import tqdm

from openpi.models import gemma as openpi_gemma
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.shared import normalize
from openpi.training import checkpoints as openpi_checkpoints
from openpi.training import sharding
from openpi.training import utils as training_utils
from openpi.training import weight_loaders
from openpi.models import tokenizer as openpi_tokenizer
from tactile_vla.common.metrics import classification_report
from tactile_vla.vla.artifacts import artifact_identity
from tactile_vla.vla.artifacts import assert_identity_matches
from tactile_vla.vla.artifacts import checkpoint_artifact_identity
from tactile_vla.vla.artifacts import checkpoint_step_number
from tactile_vla.vla.artifacts import LEGACY_DATA_PROFILE
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.artifacts import validate_norm_stats_identity
from tactile_vla.vla.data_profiles import ROTATION_MODERATELY_SUCCESS_V1
from tactile_vla.vla.data_profiles import EXPECTED_ACTION_COUNTS
from tactile_vla.vla.data_profiles import select_profile_records
from tactile_vla.vla import stage_b_v3_jax
from tactile_vla.vla.stage_b_v3_checkpoint import CHECKPOINT_FORMAT
from tactile_vla.vla.stage_b_v3_checkpoint import cast_frozen_params
from tactile_vla.vla.stage_b_v3_checkpoint import delta_params
from tactile_vla.vla.stage_b_v3_checkpoint import merge_delta_params
from tactile_vla.vla.stage_b_v3_checkpoint import resume_state
from tactile_vla.vla.stage_b_v3_checkpoint import trainable_filter
from tactile_vla.vla.index import FrameRecord
from tactile_vla.vla.index import load_or_create_splits
from tactile_vla.vla.index import scan_lerobot_frames
from tactile_vla.vla.index import SplitConfig
from tactile_vla.vla.index import v3_index_payload
from tactile_vla.vla.openpi_bridge import build_structured_inference_transform
from tactile_vla.vla.openpi_bridge import build_structured_text_transform
from tactile_vla.vla.openpi_bridge import build_transform
from tactile_vla.vla.openpi_bridge import collate_numpy
from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset
from tactile_vla.vla.openpi_bridge import TransformedTactileVLADataset
from tactile_vla.vla.openpi_bridge import V3RecoveryManifestDataset
from tactile_vla.vla.openpi_bridge import V3StageBFrameDataset
from tactile_vla.vla.openpi_bridge import V4DirectManifestDataset
from tactile_vla.vla.stage_b_v3_model import StageBV3Model
from tactile_vla.vla.structured_generation import constrained_greedy_generate_full_forward
from tactile_vla.vla.structured_text import ConstrainedTokenGrammar
from tactile_vla.vla.structured_text import failure_grammar
from tactile_vla.vla.structured_text import recovery_grammar
from tactile_vla.vla.structured_text import legal_failure_reasons
from tactile_vla.vla.structured_text import legal_recovery_plans
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE
from tactile_vla.vla.prompts import resolve_prompt_profile
from tactile_vla.vla.v4_data import ROTATION_V4
from tactile_vla.vla.v4_data import V4_TRAINING_INDEX_SCHEMA
from tactile_vla.vla.v4_data import validate_v4_index_dataset


DEFAULT_DATASET_DIR = Path("/data1/tac_data/lerobot_data/tactile_vla_v3")
DEFAULT_PROFILE_DIR = Path("/data1/outputs/vla/rotation_moderately_success_v1")
DEFAULT_SPLIT_FILE = DEFAULT_PROFILE_DIR / "splits.json"
DEFAULT_INDEX_FILE = DEFAULT_PROFILE_DIR / "vla_indices_v3.json"
DEFAULT_MANIFEST_DIR = DEFAULT_PROFILE_DIR / "reasoning"
DEFAULT_NORM_STATS_DIR = Path(
    "/data1/outputs/vla/assets/tactile_vla_rotation_moderately_success_v1"
)
DEFAULT_OUTPUT_DIR = Path("/data1/outputs/vla/stage_b_v3")
DEFAULT_STAGE_A_CHECKPOINT = Path(
    "/data1/outputs/vla/stage_a_action/pi05_delta_tac_rotation_moderately_v1/15000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--reasoning-manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_STATS_DIR)
    parser.add_argument("--stage-a-checkpoint", type=Path, default=DEFAULT_STAGE_A_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="pi05_stage_b_rotation_moderately_v1")
    parser.add_argument("--data-profile", default=ROTATION_MODERATELY_SUCCESS_V1)
    parser.add_argument("--prompt-profile", default=MINIMAL_PROMPT_PROFILE)
    parser.add_argument("--grammar-profile", default="v3_full_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--state-history-len", type=int, default=60)
    parser.add_argument("--state-history-dim", type=int, default=7)
    parser.add_argument("--state-history-fps", type=float, default=30.0)
    parser.add_argument("--history-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--use-state-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition only the action expert on dense proprioceptive history.",
    )
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--reasoning-max-token-len", type=int, default=320)
    parser.add_argument("--reasoning-window-frames", type=int, default=15)
    parser.add_argument("--status-negative-ratio", type=float, default=3.0)
    parser.add_argument("--paligemma-variant", default="gemma_2b_lora")
    parser.add_argument("--action-expert-variant", default="gemma_300m_lora")
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--need-hidden-dim", type=int, default=512)
    parser.add_argument("--need-dropout", type=float, default=0.1)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--need-loss-weight", type=float, default=1.0)
    parser.add_argument("--failure-loss-weight", type=float, default=1.0)
    parser.add_argument("--plan-loss-weight", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--keep-period", type=int, default=1000)
    parser.add_argument("--eval-max-need-samples", type=int, default=2048)
    parser.add_argument("--eval-max-text-samples", type=int, default=32)
    parser.add_argument("--action-loss-degradation-limit", type=float, default=0.10)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--overwrite-index", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def validate_v4_args(args: argparse.Namespace) -> None:
    if args.data_profile != ROTATION_V4:
        return
    if args.prompt_profile != MINIMAL_PROMPT_PROFILE:
        raise ValueError("rotation_v4 Stage B requires prompt_profile='minimal_v1'")
    if args.no_norm:
        raise ValueError("rotation_v4 Stage B requires its train-index norm stats")


V4_STAGE_B_PROTOCOL = {
    "batch_size": 8,
    "num_steps": 4_000,
    "lr": 1e-4,
    "eval_interval": 500,
    "save_interval": 1_000,
    "keep_period": 1_000,
    "action_horizon": 30,
    "action_dim": 32,
    "use_state_history": True,
    "state_history_len": 60,
    "state_history_dim": 7,
    "state_history_fps": 30.0,
    "history_hidden_dim": 256,
    "max_token_len": 200,
    "reasoning_max_token_len": 320,
    "reasoning_window_frames": 15,
    "status_negative_ratio": 3.0,
    "need_hidden_dim": 512,
    "need_dropout": 0.1,
    "action_loss_weight": 1.0,
    "need_loss_weight": 1.0,
    "failure_loss_weight": 1.0,
    "plan_loss_weight": 1.0,
    "action_loss_degradation_limit": 0.10,
    "paligemma_variant": "gemma_2b_lora",
    "action_expert_variant": "gemma_300m_lora",
    "grammar_profile": "v3_full_v1",
}
V4_STAGE_B_TRAINABLE_COMPONENTS = ("paligemma_lora", "need_head")
V4_STAGE_B_FROZEN_COMPONENTS = ("action_expert", "paligemma_non_lora")


def validate_v4_training_protocol(args: argparse.Namespace) -> None:
    if args.data_profile != ROTATION_V4:
        return
    mismatches = {
        key: {"requested": getattr(args, key), "required": required}
        for key, required in V4_STAGE_B_PROTOCOL.items()
        if getattr(args, key) != required
    }
    if mismatches:
        raise ValueError(f"rotation_v4 Stage B protocol mismatch: {mismatches}")


def validate_v4_resume_config(
    saved: dict[str, Any],
    args: argparse.Namespace,
    *,
    actual_precision: str,
) -> None:
    if args.data_profile != ROTATION_V4:
        return
    keys = (
        *V4_STAGE_B_PROTOCOL,
        "data_profile",
        "prompt_profile",
        "weight_decay",
        "grad_clip",
        "seed",
        "log_interval",
        "eval_max_need_samples",
        "fsdp_devices",
        "video_backend",
    )
    mismatches = {
        key: {"saved": saved.get(key), "requested": getattr(args, key)}
        for key in keys
        if saved.get(key) != getattr(args, key)
    }
    if saved.get("precision") != actual_precision:
        mismatches["precision"] = {
            "saved": saved.get("precision"),
            "requested": actual_precision,
        }
    expected_trainable = list(V4_STAGE_B_TRAINABLE_COMPONENTS)
    expected_frozen = list(V4_STAGE_B_FROZEN_COMPONENTS)
    if saved.get("trainable_components") != expected_trainable:
        mismatches["trainable_components"] = {
            "saved": saved.get("trainable_components"),
            "requested": expected_trainable,
        }
    if saved.get("frozen_components") != expected_frozen:
        mismatches["frozen_components"] = {
            "saved": saved.get("frozen_components"),
            "requested": expected_frozen,
        }
    if mismatches:
        raise ValueError(f"rotation_v4 Stage B resume config mismatch: {mismatches}")


def validate_stage_a_checkpoint_step(
    data_profile: str,
    checkpoint: Path,
) -> int | None:
    if data_profile not in {ROTATION_MODERATELY_SUCCESS_V1, ROTATION_V4}:
        return None
    context = f"{data_profile} Stage B Stage A checkpoint"
    return checkpoint_step_number(checkpoint, context=context)


def resolve_params_dir(path: Path) -> Path:
    path = path.resolve()
    return path / "params" if (path / "params").exists() else path


def ensure_v3_index(args: argparse.Namespace) -> tuple[dict[str, Any], list[Any]]:
    if args.data_profile == ROTATION_V4:
        if args.overwrite_index:
            raise ValueError("rotation_v4 unified indices are immutable and cannot be rebuilt by training")
        if not args.index_file.is_file():
            raise FileNotFoundError(
                f"V4 unified index does not exist: {args.index_file}. "
                "Run scripts/prepare_v4_training_index.py first."
            )
        payload = json.loads(args.index_file.read_text())
        expected = {
            "reasoning_window_frames": args.reasoning_window_frames,
            "action_horizon": args.action_horizon,
            "status_negative_ratio": args.status_negative_ratio,
        }
        actual_values = {
            "reasoning_window_frames": payload.get("failure_window_length"),
            "action_horizon": payload.get("action_horizon"),
            "status_negative_ratio": payload.get("status_negative_ratio"),
        }
        for key, expected_value in expected.items():
            actual = actual_values[key]
            if float(actual) != float(expected_value):
                raise ValueError(
                    f"V4 index {key}={actual!r}, expected {expected_value!r}; regenerate the V4 index"
                )
        if payload.get("schema_version") != V4_TRAINING_INDEX_SCHEMA:
            raise ValueError("rotation_v4 requires the dedicated V4 unified training index")
        frames, _ = validate_v4_index_dataset(payload, args.dataset_dir)
        return payload, frames

    all_records = scan_lerobot_frames(args.dataset_dir)
    records = (
        select_profile_records(all_records)
        if args.data_profile == ROTATION_MODERATELY_SUCCESS_V1
        else all_records
    )
    if args.index_file.exists() and not args.overwrite_index:
        payload = json.loads(args.index_file.read_text())
        expected = {
            "reasoning_window_frames": args.reasoning_window_frames,
            "action_horizon": args.action_horizon,
        }
        for key, value in expected.items():
            if int(payload.get(key, -1)) != value:
                raise ValueError(
                    f"V3 index {key}={payload.get(key)!r}, expected {value}; "
                    "use --overwrite-index to rebuild it"
                )
        indexed_profile = str(payload.get("data_profile", LEGACY_DATA_PROFILE))
        if indexed_profile != args.data_profile:
            raise ValueError(
                f"V3 index data_profile={indexed_profile!r}, expected {args.data_profile!r}"
            )
        return payload, records

    if args.data_profile != LEGACY_DATA_PROFILE:
        if args.overwrite_index:
            raise ValueError(
                "Versioned profile indices are generated only by "
                "scripts/prepare_rotation_moderately_profile.py"
            )
        raise FileNotFoundError(args.index_file)

    splits = load_or_create_splits(records, args.split_file, SplitConfig(seed=args.seed))
    payload = v3_index_payload(
        records,
        splits,
        seed=args.seed,
        negative_ratio=args.status_negative_ratio,
        reasoning_window_frames=args.reasoning_window_frames,
        action_horizon=args.action_horizon,
    )
    payload["dataset_dir"] = str(args.dataset_dir)
    payload["split_file"] = str(args.split_file)
    args.index_file.parent.mkdir(parents=True, exist_ok=True)
    args.index_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload, records


def _find_checkpoint_config(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.expanduser().resolve()
    candidates = [path / "config.json", path.parent / "config.json", path.parent.parent / "config.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text())
    raise FileNotFoundError(f"Cannot find checkpoint config.json near {path}")


def validate_reasoning_manifests(
    manifest_dir: Path,
    *,
    require_single_memory: bool,
) -> dict[str, Any]:
    identity: dict[str, Any] = {"splits": {}}
    directions: set[str] = set()
    direction_pattern = re.compile(r"move horizontally (left|right|front|back) moderately")
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if require_single_memory:
            invalid = [
                row.get("sample_id", offset)
                for offset, row in enumerate(rows)
                if int(row.get("memory_length", -1)) != 1
                or len(row.get("failure_recovery_memory", [])) != 1
                or bool(row.get("donor_episode_ids", []))
            ]
            if invalid:
                raise ValueError(
                    f"{path} contains non-single-round reasoning samples: {invalid[:10]}"
                )
        split_directions = []
        for row in rows:
            match = direction_pattern.search(str(row.get("target_recovery_plan", "")))
            if match is None:
                raise ValueError(
                    f"Reasoning target is outside rotation-moderately coverage: "
                    f"{row.get('target_recovery_plan')!r}"
                )
            split_directions.append(match.group(1))
            directions.add(match.group(1))
        identity["splits"][split] = {
            "count": len(rows),
            "sha256": sha256_file(path),
            "directions": dict(Counter(split_directions)),
        }
    if require_single_memory and directions != {"left", "right", "front", "back"}:
        raise ValueError(f"Reasoning manifests do not cover all four directions: {sorted(directions)}")
    return identity


def _loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    if len(dataset) == 0:
        raise ValueError("V3 Stage B dataset stream is empty")
    if shuffle and len(dataset) < batch_size:
        raise ValueError(
            f"Training stream has {len(dataset)} samples, fewer than batch_size={batch_size}"
        )
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_numpy,
        drop_last=shuffle,
        **kwargs,
    )


def build_loaders(
    args: argparse.Namespace,
    model_config: Pi0Config,
    index: dict[str, Any],
    records: list[Any],
    tokenizer: openpi_tokenizer.PaligemmaTokenizer,
    failure_codec: ConstrainedTokenGrammar,
    plan_codec: ConstrainedTokenGrammar,
) -> dict[str, dict[str, DataLoader]]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    norm_stats = None if args.no_norm else normalize.load(args.norm_stats_dir)
    regular_transform = build_transform(
        model_config,
        norm_stats=norm_stats,
        use_quantile_norm=not args.no_norm,
    )
    failure_transform = build_structured_text_transform(
        model_config,
        tokenizer=tokenizer,
        grammar=failure_codec,
        max_len=model_config.max_token_len,
        norm_stats=norm_stats,
        use_quantile_norm=not args.no_norm,
    )
    assessment_transform = build_structured_inference_transform(
        model_config,
        tokenizer=tokenizer,
        max_len=model_config.max_token_len,
        norm_stats=norm_stats,
        use_quantile_norm=not args.no_norm,
    )
    plan_transform = build_structured_text_transform(
        model_config,
        tokenizer=tokenizer,
        grammar=plan_codec,
        max_len=args.reasoning_max_token_len,
        norm_stats=norm_stats,
        use_quantile_norm=not args.no_norm,
    )
    frame_lookup = (
        {
            (record.original_episode_id, record.attempt_id, record.frame_index): record.global_index
            for record in records
        }
        if args.data_profile != ROTATION_V4
        else {}
    )
    structured_repo_id = (
        "tactile_vla_rotation_v4"
        if args.data_profile == ROTATION_V4
        else "tactile_vla_v3"
    )
    shared_structured_dataset = LeRobotDataset(
        structured_repo_id,
        root=args.dataset_dir,
        download_videos=False,
        video_backend=args.video_backend,
    )
    action_delta_timestamps: dict[str, list[float]] = {
        "action": [step / args.state_history_fps for step in range(args.action_horizon)]
    }
    if args.use_state_history:
        action_delta_timestamps["observation.state"] = [
            step / args.state_history_fps
            for step in range(-(args.state_history_len - 1), 1)
        ]
    shared_action_dataset = LeRobotDataset(
        (
            "tactile_vla_rotation_v4"
            if args.data_profile == ROTATION_V4
            else "tactile_vla"
        ),
        root=args.dataset_dir,
        delta_timestamps=action_delta_timestamps,
        download_videos=False,
        video_backend=args.video_backend,
    )

    result: dict[str, dict[str, DataLoader]] = {}
    for split in ("train", "val"):
        split_index = index["splits"][split]
        training = split == "train"
        action_dataset = TransformedTactileVLADataset(
            TactileVLAFrameDataset(
                dataset_dir=args.dataset_dir,
                indices=split_index["execution_indices"],
                stage="execution",
                action_horizon=args.action_horizon,
                state_history_len=(
                    args.state_history_len if args.use_state_history else 0
                ),
                fps=args.state_history_fps,
                video_backend=args.video_backend,
                prompt_profile=args.prompt_profile,
                dataset_repo_id=(
                    "tactile_vla_rotation_v4"
                    if args.data_profile == ROTATION_V4
                    else "tactile_vla"
                ),
                lerobot_dataset=shared_action_dataset,
            ),
            regular_transform,
        )
        # Need recovery and failure diagnosis intentionally use the same raw
        # prompt and the same structured ``Answer:`` prefix. This keeps the
        # training representation identical to the shared assessment request
        # used by the deployment server.
        if args.data_profile == ROTATION_V4:
            need_raw = V4DirectManifestDataset(
                dataset_dir=args.dataset_dir,
                manifest_file=split_index["status_manifest_file"],
                manifest_row_indices=split_index["status_manifest_row_indices"],
                global_indices=split_index["status_indices"],
                task="need",
                expected_manifest_sha256=split_index["status_manifest_sha256"],
                expected_split=split,
                video_backend=args.video_backend,
                prompt_profile=args.prompt_profile,
                dataset_repo_id=structured_repo_id,
                lerobot_dataset=shared_structured_dataset,
            )
            failure_raw = V4DirectManifestDataset(
                dataset_dir=args.dataset_dir,
                manifest_file=split_index["failure_reason_manifest_file"],
                manifest_row_indices=split_index["failure_reason_manifest_row_indices"],
                global_indices=split_index["failure_reason_indices"],
                task="failure",
                expected_manifest_sha256=split_index["failure_reason_manifest_sha256"],
                expected_split=split,
                video_backend=args.video_backend,
                prompt_profile=args.prompt_profile,
                dataset_repo_id=structured_repo_id,
                lerobot_dataset=shared_structured_dataset,
            )
            plan_raw = V4DirectManifestDataset(
                dataset_dir=args.dataset_dir,
                manifest_file=split_index["reasoning_manifest_file"],
                manifest_row_indices=split_index["reasoning_manifest_row_indices"],
                global_indices=split_index["reasoning_indices"],
                task="plan",
                expected_manifest_sha256=split_index["reasoning_manifest_sha256"],
                expected_split=split,
                video_backend=args.video_backend,
                prompt_profile=args.prompt_profile,
                dataset_repo_id=structured_repo_id,
                lerobot_dataset=shared_structured_dataset,
            )
        else:
            need_raw = V3StageBFrameDataset(
                dataset_dir=args.dataset_dir,
                indices=split_index["status_indices"],
                task="need_recovery",
                video_backend=args.video_backend,
                prompt_profile=args.prompt_profile,
                lerobot_dataset=shared_structured_dataset,
            )
            failure_raw = V3StageBFrameDataset(
                dataset_dir=args.dataset_dir,
                indices=split_index["failure_reason_indices"],
                task="failure_reason",
                video_backend=args.video_backend,
                prompt_profile=args.prompt_profile,
                lerobot_dataset=shared_structured_dataset,
            )
            plan_manifest = args.reasoning_manifest_dir / f"{split}.jsonl"
            if not plan_manifest.is_file():
                raise FileNotFoundError(plan_manifest)
            plan_raw = V3RecoveryManifestDataset(
                dataset_dir=args.dataset_dir,
                manifest_file=plan_manifest,
                frame_lookup=frame_lookup,
                reasoning_window_frames=args.reasoning_window_frames,
                training=training,
                video_backend=args.video_backend,
                prompt_profile=args.prompt_profile,
                lerobot_dataset=shared_structured_dataset,
            )
        need_dataset = TransformedTactileVLADataset(need_raw, assessment_transform)
        failure_dataset = TransformedTactileVLADataset(failure_raw, failure_transform)
        plan_dataset = TransformedTactileVLADataset(plan_raw, plan_transform)
        result[split] = {
            "action": _loader(
                action_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=training,
            ),
            "need": _loader(
                need_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=training,
            ),
            "failure": _loader(
                failure_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=training,
            ),
            "plan": _loader(
                plan_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=training,
            ),
        }
    return result


def print_batch_shapes(task: str, batch: dict[str, Any]) -> None:
    print(f"[{task}]")
    for key, value in batch.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"{key}.{sub_key}: shape={np.asarray(sub_value).shape} dtype={np.asarray(sub_value).dtype}")
        else:
            print(f"{key}: shape={np.asarray(value).shape} dtype={np.asarray(value).dtype}")


def _shape_and_dtype(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def print_dry_run_vlm_inputs(loaders: dict[str, DataLoader]) -> None:
    """Print one raw, human-readable VLM input example for every V3 task."""

    task_names = {
        "action": "execution / action chunk",
        "need": "inference / shared assessment (need_recovery)",
        "failure": "inference / shared assessment (failure_reason)",
        "plan": "inference / recovery planning",
    }
    print("\n===== Dry-run VLM input examples (before tokenization) =====")
    for task in ("action", "need", "failure", "plan"):
        transformed_dataset = loaders[task].dataset
        raw_dataset = getattr(transformed_dataset, "dataset", None)
        if raw_dataset is None:
            raise TypeError(
                f"Cannot inspect raw VLM input for task={task}: "
                f"unexpected dataset type {type(transformed_dataset).__name__}"
            )
        sample = raw_dataset[0]
        metadata = {
            key: int(np.asarray(sample[key]).item())
            for key in ("global_index", "episode_id", "attempt_id", "frame_index")
            if key in sample
        }
        payload: dict[str, Any] = {
            "task": task_names[task],
            "sample_metadata_not_in_prompt": metadata,
            "text_prompt": str(sample["prompt"]),
            "visual_inputs": {
                "front": _shape_and_dtype(sample["observation/image"]),
                "left_wrist": _shape_and_dtype(sample["observation/wrist_image"]),
                "right_wrist": "all-zero placeholder, created by the input transform",
            },
            "state_input": _shape_and_dtype(sample["observation/state"]),
        }
        if task == "action":
            if "observation/state_history" in sample:
                history_mask = np.asarray(
                    sample["observation/state_history_mask"],
                    dtype=np.bool_,
                )
                payload["action_expert_state_memory"] = {
                    "history": _shape_and_dtype(sample["observation/state_history"]),
                    "mask": _shape_and_dtype(history_mask),
                    "valid_frames": int(history_mask.sum()),
                }
            payload["training_target"] = {
                "action_chunk": _shape_and_dtype(sample["actions"]),
            }
        elif task == "need":
            payload["training_target"] = {
                "need_recovery": bool(np.asarray(sample["need_recovery_label"]).item()),
            }
        else:
            payload["training_target"] = {
                "structured_text": str(sample["target_text"]),
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def make_optimizer(args: argparse.Namespace) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(
            args.lr,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=args.weight_decay,
        ),
    )


def count_state_params(state: nnx.State) -> int:
    return sum(
        int(getattr(leaf, "value", leaf).size)
        for leaf in jax.tree.leaves(state)
        if hasattr(getattr(leaf, "value", leaf), "size")
    )


def load_weights_and_validate(loader: weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(loaded).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )


def init_train_state(
    *,
    model_config: Pi0Config,
    args: argparse.Namespace,
    tx: optax.GradientTransformation,
    train_filter: nnx.filterlib.Filter,
    init_rng: jax.Array,
    mesh: jax.sharding.Mesh,
) -> tuple[training_utils.TrainState, Any]:
    paligemma_width = openpi_gemma.get_config(model_config.paligemma_variant).width

    def init(rng: jax.Array, backbone_params: at.Params | None = None) -> training_utils.TrainState:
        backbone_rng, head_rng = jax.random.split(rng)
        backbone = model_config.create(backbone_rng)
        if backbone_params is not None:
            graphdef, state = nnx.split(backbone)
            state.replace_by_pure_dict(backbone_params)
            backbone = nnx.merge(graphdef, state)
        model = StageBV3Model(
            backbone,
            paligemma_width=paligemma_width,
            need_hidden_dim=args.need_hidden_dim,
            need_dropout=args.need_dropout,
            rngs=nnx.Rngs(head_rng),
        )
        params = nnx.state(model)
        frozen_filter = nnx.All(nnx.Param, nnx.Not(train_filter))
        params = cast_frozen_params(params, frozen_filter)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            opt_state=tx.init(params.filter(train_filter)),
            tx=tx,
            ema_decay=None,
            ema_params=None,
        )

    state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(state_shape, mesh, log=True)
    backbone_reference = state_shape.params["backbone"].to_pure_dict()
    loader = weight_loaders.CheckpointWeightLoader(str(resolve_params_dir(args.stage_a_checkpoint)))
    backbone_params = load_weights_and_validate(loader, backbone_reference)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=(replicated, replicated),
        out_shardings=state_sharding,
    )(init_rng, backbone_params)
    return state, state_sharding


def _apply_gradients(
    state: training_utils.TrainState,
    model: StageBV3Model,
    grads: nnx.State,
    train_filter: nnx.filterlib.Filter,
) -> training_utils.TrainState:
    parameters = state.params.filter(train_filter)
    updates, optimizer_state = state.tx.update(grads, state.opt_state, parameters)
    updated = optax.apply_updates(parameters, updates)
    nnx.update(model, updated)
    return dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=optimizer_state,
    )


def train_action_step(
    train_filter: nnx.filterlib.Filter,
    loss_weight: float,
    rng: jax.Array,
    state: training_utils.TrainState,
    batch: tuple[Observation, jax.Array],
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, actions = batch

    def loss_fn(module: StageBV3Model) -> jax.Array:
        return loss_weight * jnp.mean(
            module.backbone.compute_loss(rng, observation, actions, train=True)
        )

    loss, grads = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, train_filter),
    )(model)
    state = _apply_gradients(state, model, grads, train_filter)
    return state, {"loss": loss, "grad_norm": optax.global_norm(grads)}


def train_need_step(
    train_filter: nnx.filterlib.Filter,
    loss_weight: float,
    rng: jax.Array,
    state: training_utils.TrainState,
    batch: tuple[Observation, dict[str, jax.Array]],
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, targets = batch

    def loss_fn(module: StageBV3Model) -> jax.Array:
        logits = module.need_recovery_logits(observation, rng=rng, train=True)
        return loss_weight * optax.softmax_cross_entropy_with_integer_labels(
            logits,
            targets["need"],
        ).mean()

    loss, grads = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, train_filter),
    )(model)
    state = _apply_gradients(state, model, grads, train_filter)
    return state, {"loss": loss, "grad_norm": optax.global_norm(grads)}


def train_text_step(
    train_filter: nnx.filterlib.Filter,
    loss_weight: float,
    compact_token_ids: jax.Array,
    rng: jax.Array,
    state: training_utils.TrainState,
    batch: tuple[Observation, dict[str, jax.Array]],
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, targets = batch

    def loss_fn(module: StageBV3Model) -> jax.Array:
        logits = module.structured_token_logits(
            observation,
            compact_token_ids,
            rng=rng,
            train=True,
        )
        return loss_weight * stage_b_v3_jax.constrained_token_cross_entropy(
            logits,
            targets["target_ids"],
            targets["allowed"],
        )

    loss, grads = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, train_filter),
    )(model)
    state = _apply_gradients(state, model, grads, train_filter)
    return state, {"loss": loss, "grad_norm": optax.global_norm(grads)}


def batch_to_jax(
    batch: dict[str, Any],
    task: str,
    data_sharding: jax.sharding.Sharding,
) -> Any:
    local = jax.tree.map(np.asarray, batch)
    for key in ("tokenized_prompt", "token_ar_mask", "structured_target_compact_ids"):
        if key in local:
            local[key] = np.asarray(local[key], dtype=np.int32)
    for key in (
        "tokenized_prompt_mask",
        "token_loss_mask",
        "structured_allowed_token_mask",
    ):
        if key in local:
            local[key] = np.asarray(local[key], dtype=np.bool_)
    observation = Observation.from_dict(local)
    if task == "action":
        payload = (observation, np.asarray(local["actions"], dtype=np.float32))
    elif task == "need":
        payload = (observation, {"need": np.asarray(local["need_recovery_label"], dtype=np.int32)})
    else:
        payload = (
            observation,
            {
                "target_ids": np.asarray(local["structured_target_compact_ids"], dtype=np.int32),
                "allowed": np.asarray(local["structured_allowed_token_mask"], dtype=np.bool_),
                "text_index": np.asarray(local["structured_target_text_index"], dtype=np.int32),
            },
        )
    return jax.tree.map(
        lambda value: jax.make_array_from_process_local_data(data_sharding, value),
        payload,
    )


def cycling(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _split_checkpoint_state(
    state: training_utils.TrainState,
    train_filter: nnx.filterlib.Filter,
    rng: jax.Array,
) -> tuple[dict[str, Any], nnx.State]:
    train_state = dataclasses.replace(state, params=nnx.State({}))
    training_payload = {
        "state": train_state,
        # Store raw uint32 data rather than key<fry> so Orbax restores it
        # without dtype conversion ambiguity.
        "rng_key_data": jax.random.key_data(rng),
    }
    return training_payload, resume_state(state.params, train_filter)


def _merge_checkpoint_state(
    base_state: training_utils.TrainState,
    training_payload: dict[str, Any],
    params: dict[str, nnx.State],
) -> tuple[training_utils.TrainState, jax.Array]:
    restored_state = training_payload["state"]
    restored_state = dataclasses.replace(
        restored_state,
        params=merge_delta_params(base_state.params, params["params"]),
    )
    rng = jax.random.wrap_key_data(training_payload["rng_key_data"])
    return restored_state, rng


def save_state(
    manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    step: int,
    train_filter: nnx.filterlib.Filter,
    rng: jax.Array,
) -> None:
    with at.disable_typechecking():
        training_payload, params = _split_checkpoint_state(state, train_filter, rng)
    manager.save(
        step,
        {"train_state": training_payload, "params": {"params": params}},
    )


def restore_state(
    manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    train_filter: nnx.filterlib.Filter,
    rng: jax.Array,
) -> tuple[training_utils.TrainState, jax.Array]:
    with at.disable_typechecking():
        training_payload, params = _split_checkpoint_state(state, train_filter, rng)
        restored = manager.restore(
            None,
            items={
                "train_state": training_payload,
                "params": {"params": params},
            },
        )
    return _merge_checkpoint_state(state, restored["train_state"], restored["params"])


def save_best_checkpoint(
    run_dir: Path,
    state: training_utils.TrainState,
    metrics: dict[str, Any],
    train_filter: nnx.filterlib.Filter,
) -> None:
    """Save only the Stage B overlay; merge it after training for inference."""

    if jax.process_index() != 0:
        return
    best_dir = run_dir / "best"
    params_dir = best_dir / "delta_params"
    best_dir.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(
            params_dir,
            {"params": delta_params(state.params, train_filter)},
            force=True,
        )
    (best_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )


def _strip_teacher_forcing(observation: Observation) -> Observation:
    if observation.token_loss_mask is None:
        raise ValueError("Text evaluation requires token_loss_mask")
    answer_mask = observation.token_loss_mask.astype(jnp.bool_)
    return dataclasses.replace(
        observation,
        tokenized_prompt=jnp.where(answer_mask, 0, observation.tokenized_prompt),
        tokenized_prompt_mask=jnp.logical_and(
            observation.tokenized_prompt_mask,
            jnp.logical_not(answer_mask),
        ),
        token_ar_mask=jnp.zeros_like(observation.token_ar_mask),
        token_loss_mask=jnp.zeros_like(observation.token_loss_mask),
    )


def evaluate_action_loss(
    state: training_utils.TrainState,
    loader: DataLoader,
    data_sharding: jax.sharding.Sharding,
    *,
    seed: int,
    max_batches: int = 32,
) -> float:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    compute_loss = nnx_utils.module_jit(model.backbone.compute_loss)
    values: list[float] = []
    rng = jax.random.key(seed)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        observation, actions = batch_to_jax(batch, "action", data_sharding)
        rng, step_rng = jax.random.split(rng)
        loss = compute_loss(step_rng, observation, actions).mean()
        values.append(float(jax.device_get(loss)))
    if not values:
        raise ValueError("Action validation loader is empty")
    return float(np.mean(values))


def evaluate_need(
    state: training_utils.TrainState,
    loader: DataLoader,
    data_sharding: jax.sharding.Sharding,
    *,
    max_samples: int,
) -> dict[str, Any]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    need_logits = nnx_utils.module_jit(model.need_recovery_logits)
    true: list[int] = []
    pred: list[int] = []
    for batch in loader:
        observation, targets = batch_to_jax(batch, "need", data_sharding)
        logits = need_logits(observation)
        batch_true = np.asarray(jax.device_get(targets["need"])).reshape(-1)
        batch_pred = np.asarray(jax.device_get(jnp.argmax(logits, axis=-1))).reshape(-1)
        remaining = max_samples - len(true)
        true.extend(batch_true[:remaining].tolist())
        pred.extend(batch_pred[:remaining].tolist())
        if len(true) >= max_samples:
            break
    return classification_report(true, pred, num_classes=2, class_names=["false", "true"])


def evaluate_text(
    state: training_utils.TrainState,
    loader: DataLoader,
    task: str,
    grammar: ConstrainedTokenGrammar,
    data_sharding: jax.sharding.Sharding,
    *,
    max_samples: int | None,
) -> dict[str, Any]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    structured_logits = nnx_utils.module_jit(model.structured_token_logits)
    correct = 0
    total = 0
    direction_correct: Counter[str] = Counter()
    direction_total: Counter[str] = Counter()
    direction_magnitude_correct: Counter[str] = Counter()
    direction_magnitude_total: Counter[str] = Counter()

    def direction_for(text: str) -> str:
        pattern = (
            r"failure_reason=rotate (left|right|front|back|none),"
            if task == "failure"
            else r"move horizontally (left|right|front|back|none) "
        )
        match = re.search(pattern, text)
        return match.group(1) if match else "unknown"

    def direction_magnitude_for(text: str) -> str:
        if task != "plan":
            return ""
        match = re.search(
            r"move horizontally (left|right|front|back|none) "
            r"(slightly|moderately|significantly)",
            text,
        )
        return f"{match.group(1)}/{match.group(2)}" if match else "unknown"

    def result() -> dict[str, Any]:
        return {
            "exact_match": correct / total if total else 0.0,
            "num_samples": total,
            "by_direction": {
                direction: {
                    "exact_match": direction_correct[direction] / support,
                    "support": support,
                }
                for direction, support in sorted(direction_total.items())
            },
            "by_direction_magnitude": {
                combination: {
                    "exact_match": direction_magnitude_correct[combination] / support,
                    "support": support,
                }
                for combination, support in sorted(direction_magnitude_total.items())
            },
        }

    for batch in loader:
        observation, targets = batch_to_jax(batch, task, data_sharding)
        text_indices = np.asarray(jax.device_get(targets["text_index"])).reshape(-1)
        batch_size = int(observation.state.shape[0])
        for batch_index in range(batch_size):
            single = jax.tree.map(
                lambda value: None if value is None else value[batch_index : batch_index + 1],
                observation,
            )
            prompt_only = _strip_teacher_forcing(single)
            prediction = constrained_greedy_generate_full_forward(
                prompt_only,
                grammar,
                lambda obs, compact_ids: structured_logits(obs, compact_ids),
            )
            target = grammar.texts[int(text_indices[batch_index])]
            matched = int(prediction == target)
            direction = direction_for(target)
            direction_magnitude = direction_magnitude_for(target)
            correct += matched
            total += 1
            direction_correct[direction] += matched
            direction_total[direction] += 1
            if direction_magnitude:
                direction_magnitude_correct[direction_magnitude] += matched
                direction_magnitude_total[direction_magnitude] += 1
            if max_samples is not None and total >= max_samples:
                return result()
    return result()


def text_eval_sample_limit(data_profile: str, requested_limit: int) -> int | None:
    """V4 validates every terminal manifest row; older profiles keep their cap."""

    return None if data_profile == ROTATION_V4 else requested_limit


def validate_text_eval_coverage(
    data_profile: str,
    failure_metrics: dict[str, Any],
    plan_metrics: dict[str, Any],
) -> None:
    if data_profile not in {ROTATION_MODERATELY_SUCCESS_V1, ROTATION_V4}:
        return
    required_directions = {"left", "right", "front", "back"}
    for task_name, task_metrics in (
        ("failure", failure_metrics),
        ("plan", plan_metrics),
    ):
        covered = {
            direction
            for direction, values in task_metrics["by_direction"].items()
            if int(values["support"]) > 0
        }
        if covered != required_directions:
            raise ValueError(
                f"{task_name} validation direction coverage mismatch: {sorted(covered)}"
            )
    if data_profile == ROTATION_V4:
        required_combinations = {
            f"{direction}/{magnitude}"
            for direction in required_directions
            for magnitude in ("moderately", "slightly")
        }
        covered_combinations = {
            combination
            for combination, values in plan_metrics["by_direction_magnitude"].items()
            if int(values["support"]) > 0
        }
        if covered_combinations != required_combinations:
            raise ValueError(
                "plan validation direction/magnitude coverage mismatch: "
                f"{sorted(covered_combinations)}"
            )


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    if jax.process_index() == 0:
        with path.open("a") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    init_logging()
    args.prompt_profile = resolve_prompt_profile(args.prompt_profile)
    validate_v4_args(args)
    validate_v4_training_protocol(args)
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.reasoning_window_frames != 15:
        logging.warning(
            "The confirmed V3 protocol uses 15 frames; overriding it with %d",
            args.reasoning_window_frames,
        )
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"batch_size={args.batch_size} must be divisible by device_count={jax.device_count()}"
        )
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if args.num_steps % 4 != 0:
        raise ValueError("Stage B total updates must be divisible by the four-task cycle")
    if args.grammar_profile != "v3_full_v1":
        raise ValueError("This training pipeline requires grammar_profile='v3_full_v1'")
    if args.data_profile == ROTATION_MODERATELY_SUCCESS_V1:
        pinned = {
            "batch_size": (args.batch_size, 8),
            "num_steps": (args.num_steps, 4000),
            "lr": (args.lr, 1e-4),
            "eval_interval": (args.eval_interval, 500),
            "save_interval": (args.save_interval, 1000),
            "keep_period": (args.keep_period, 1000),
            "action_horizon": (args.action_horizon, 30),
            "reasoning_window_frames": (args.reasoning_window_frames, 15),
            "status_negative_ratio": (args.status_negative_ratio, 3.0),
        }
        mismatches = {
            key: {"requested": actual, "required": expected}
            for key, (actual, expected) in pinned.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(f"Versioned Stage B protocol mismatch: {mismatches}")
    if args.eval_max_need_samples <= 0:
        raise ValueError("--eval-max-need-samples must be positive")
    if (
        args.data_profile == ROTATION_MODERATELY_SUCCESS_V1
        and args.eval_max_text_samples < 4
    ):
        raise ValueError(
            "Rotation profile validation needs at least four text samples to cover every direction"
        )
    if args.use_state_history:
        if args.state_history_dim != 7:
            raise ValueError(
                "V3 LeRobot demonstrations store 7-D puppet qpos; "
                "--state-history-dim must be 7"
            )
        if min(
            args.state_history_len,
            args.state_history_dim,
            args.state_history_fps,
            args.history_hidden_dim,
        ) <= 0:
            raise ValueError("State-history length, dimension, fps, and hidden dimension must be positive")

    cache_dir = PROJECT_ROOT / ".cache" / "jax"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    logging.info("Running on %s with devices=%s", platform.node(), jax.devices())
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
    tokenizer = openpi_tokenizer.PaligemmaTokenizer(args.reasoning_max_token_len)
    failure_codec = failure_grammar(
        lambda text: tokenizer.encode_text(text, add_eos=True)
    )
    plan_codec = recovery_grammar(
        lambda text: tokenizer.encode_text(text, add_eos=True)
    )
    if failure_codec.texts != legal_failure_reasons() or plan_codec.texts != legal_recovery_plans():
        raise AssertionError("V3 full grammar was unexpectedly narrowed")
    index, records = ensure_v3_index(args)
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
        raise ValueError("Stage B action replay index must contain exactly 98,233 action starts")
    if args.data_profile == ROTATION_V4:
        reasoning_identity = identity["reasoning_manifest_identity"]
    else:
        reasoning_identity = validate_reasoning_manifests(
            args.reasoning_manifest_dir,
            require_single_memory=args.data_profile == ROTATION_MODERATELY_SUCCESS_V1,
        )
        identity["reasoning_manifest_identity"] = reasoning_identity

    if not args.no_norm:
        norm_summary = validate_norm_stats_identity(
            args.norm_stats_dir / "summary.json",
            identity,
            context="Stage B norm stats",
        )
        if args.data_profile == ROTATION_V4:
            identity["norm_stats_sha256"] = norm_summary["norm_stats_sha256"]

    stage_a_checkpoint_step = validate_stage_a_checkpoint_step(
        args.data_profile,
        args.stage_a_checkpoint,
    )
    if not args.dry_run:
        stage_a_config_path, stage_a_config = _find_checkpoint_config(args.stage_a_checkpoint)
        assert_identity_matches(
            checkpoint_artifact_identity(stage_a_config),
            identity,
            context=f"Stage B Stage A checkpoint ({stage_a_config_path})",
        )
    loaders = build_loaders(
        args,
        model_config,
        index,
        records,
        tokenizer,
        failure_codec,
        plan_codec,
    )
    first_batches = {task: next(iter(loader)) for task, loader in loaders["train"].items()}
    if args.dry_run:
        print_dry_run_vlm_inputs(loaders["train"])
        print("\n===== Dry-run transformed batch shapes =====")
        for task, batch in first_batches.items():
            print_batch_shapes(task, batch)
        print(
            json.dumps(
                {
                    "failure_compact_vocab": len(failure_codec.compact_token_ids),
                    "failure_max_target_tokens": failure_codec.max_target_tokens,
                    "plan_compact_vocab": len(plan_codec.compact_token_ids),
                    "plan_max_target_tokens": plan_codec.max_target_tokens,
                },
                indent=2,
            )
        )
        return

    run_dir = args.output_dir / args.run_name
    manager, resuming = openpi_checkpoints.initialize_checkpoint_dir(
        run_dir,
        keep_period=args.keep_period,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    if resuming:
        saved_config = json.loads((run_dir / "config.json").read_text())
        saved_format = saved_config.get("checkpoint_format")
        if saved_format != CHECKPOINT_FORMAT:
            raise ValueError(
                "Cannot resume this run with delta checkpointing: "
                f"config checkpoint_format={saved_format!r}, expected {CHECKPOINT_FORMAT!r}. "
                "Start a new run with --overwrite."
            )
        saved_stage_a = Path(str(saved_config["stage_a_checkpoint"])).expanduser().resolve()
        requested_stage_a = args.stage_a_checkpoint.expanduser().resolve()
        if saved_stage_a != requested_stage_a:
            raise ValueError(
                "Delta resume requires the same frozen Stage A checkpoint: "
                f"saved={saved_stage_a}, requested={requested_stage_a}"
            )
        assert_identity_matches(
            checkpoint_artifact_identity(saved_config),
            identity,
            context="Stage B resume",
        )
        saved_reasoning_identity = saved_config.get("artifact_identity", {}).get(
            "reasoning_manifest_identity"
        )
        if saved_reasoning_identity != reasoning_identity:
            raise ValueError(
                "Stage B resume reasoning manifest identity mismatch: "
                f"saved={saved_reasoning_identity}, requested={reasoning_identity}"
            )
        resume_model_values = {
            "paligemma_variant": args.paligemma_variant,
            "action_expert_variant": args.action_expert_variant,
            "action_dim": args.action_dim,
            "action_horizon": args.action_horizon,
            "max_token_len": args.max_token_len,
            "use_state_history": args.use_state_history,
            "state_history_len": args.state_history_len,
            "state_history_dim": args.state_history_dim,
            "state_history_fps": args.state_history_fps,
            "history_hidden_dim": args.history_hidden_dim,
            "need_hidden_dim": args.need_hidden_dim,
            "need_dropout": args.need_dropout,
            "precision": precision,
        }
        for key, requested in resume_model_values.items():
            saved = saved_config.get(key)
            if saved != requested:
                raise ValueError(
                    f"Delta resume model config mismatch for {key}: "
                    f"saved={saved!r}, requested={requested!r}"
                )
        validate_v4_resume_config(saved_config, args, actual_precision=precision)
    if jax.process_index() == 0 and not resuming:
        run_dir.mkdir(parents=True, exist_ok=True)
        config_payload = vars(args) | {
            "precision": precision,
            "task_cycle": ["action", "need", "failure", "plan"],
            "updates_per_task": args.num_steps // 4,
            "failure_grammar": list(failure_codec.texts),
            "recovery_grammar": list(plan_codec.texts),
            "grammar_profile": args.grammar_profile,
            "training_target_coverage": {
                "failure_reason": [
                    f"failure_reason=rotate {direction},grasp appropriate."
                    for direction in ("right", "left", "front", "back")
                ],
                "recovery_plan": [
                    f"recovery_plan=move horizontally {direction} {magnitude}, "
                    "move vertically none moderately."
                    for direction in ("right", "left", "front", "back")
                    for magnitude in (
                        ("moderately", "slightly")
                        if args.data_profile == ROTATION_V4
                        else ("moderately",)
                    )
                ],
            },
            "artifact_identity": identity,
            "data_config_hash": identity["data_config_hash"],
            "action_frame_manifest_hash": identity["action_frame_manifest_hash"],
            "action_indices_identity": identity["action_indices_identity"],
            "index_sha256": identity["index_sha256"],
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_contents": [
                "paligemma_lora",
                "need_head",
                "optimizer_state",
                "step",
                "loop_rng",
                "model_rng_state",
            ],
            "trainable_components": list(V4_STAGE_B_TRAINABLE_COMPONENTS),
            "frozen_components": list(V4_STAGE_B_FROZEN_COMPONENTS),
            "stage_a_checkpoint_step": stage_a_checkpoint_step,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config_payload, indent=2, default=str, ensure_ascii=False) + "\n"
        )

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS),
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    filter_ = trainable_filter()
    state, state_sharding = init_train_state(
        model_config=model_config,
        args=args,
        tx=make_optimizer(args),
        train_filter=filter_,
        init_rng=jax.random.key(args.seed),
        mesh=mesh,
    )
    rng = jax.random.key(args.seed + 1)
    jax.block_until_ready(state)
    baseline_action_loss = evaluate_action_loss(
        state,
        loaders["val"]["action"],
        data_sharding,
        seed=args.seed + 100,
    )
    logging.info("Stage A validation action loss baseline=%.6f", baseline_action_loss)
    if resuming:
        state, rng = restore_state(manager, state, filter_, rng)
    jax.block_until_ready(state)
    logging.info(
        "precision=%s trainable_params=%d total_params=%d",
        precision,
        count_state_params(state.params.filter(filter_)),
        count_state_params(state.params),
    )

    failure_ids = jnp.asarray(failure_codec.compact_token_ids, dtype=jnp.int32)
    plan_ids = jnp.asarray(plan_codec.compact_token_ids, dtype=jnp.int32)
    paction = jax.jit(
        lambda rng, train_state, batch: train_action_step(
            filter_, args.action_loss_weight, rng, train_state, batch
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    pneed = jax.jit(
        lambda rng, train_state, batch: train_need_step(
            filter_, args.need_loss_weight, rng, train_state, batch
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    pfailure = jax.jit(
        lambda rng, train_state, batch: train_text_step(
            filter_, args.failure_loss_weight, failure_ids, rng, train_state, batch
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    pplan = jax.jit(
        lambda rng, train_state, batch: train_text_step(
            filter_, args.plan_loss_weight, plan_ids, rng, train_state, batch
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    train_functions = {
        "action": paction,
        "need": pneed,
        "failure": pfailure,
        "plan": pplan,
    }
    iterators = {
        task: iter(cycling(loader))
        for task, loader in loaders["train"].items()
    }
    task_cycle = ("action", "need", "failure", "plan")
    best_metadata_path = run_dir / "best" / "metrics.json"
    best_score = (
        float(json.loads(best_metadata_path.read_text()).get("val_score", -1.0))
        if resuming and best_metadata_path.is_file()
        else -1.0
    )
    metrics_file = run_dir / "metrics.jsonl"
    start_step = int(jax.device_get(state.step))
    recent: dict[str, list[float]] = {task: [] for task in task_cycle}
    start_time = time.time()
    progress = tqdm.trange(start_step, args.num_steps, initial=start_step, total=args.num_steps, desc="Stage B V3")
    for _ in progress:
        current_step = int(jax.device_get(state.step))
        task = task_cycle[current_step % len(task_cycle)]
        numpy_batch = next(iterators[task])
        batch = batch_to_jax(numpy_batch, task, data_sharding)
        rng, step_rng = jax.random.split(rng)
        with sharding.set_mesh(mesh):
            state, info = train_functions[task](step_rng, state, batch)
        info = jax.device_get(info)
        recent[task].append(float(info["loss"]))
        step = int(jax.device_get(state.step))

        if step % args.log_interval == 0:
            payload = {
                "step": step,
                "task": task,
                "elapsed_sec": time.time() - start_time,
                **{
                    f"{name}_loss": float(np.mean(values)) if values else None
                    for name, values in recent.items()
                },
            }
            progress.write(str(payload))
            write_jsonl(metrics_file, payload)
            recent = {name: [] for name in task_cycle}

        should_evaluate = step % args.eval_interval == 0 or step == args.num_steps
        if should_evaluate:
            action_loss = evaluate_action_loss(
                state,
                loaders["val"]["action"],
                data_sharding,
                seed=args.seed + 100,
            )
            need_metrics = evaluate_need(
                state,
                loaders["val"]["need"],
                data_sharding,
                max_samples=args.eval_max_need_samples,
            )
            need_support = {
                name: int(values["support"])
                for name, values in need_metrics["per_class"].items()
            }
            if any(support <= 0 for support in need_support.values()):
                raise ValueError(
                    f"Stratified need validation lost class support: {need_support}"
                )
            failure_metrics = evaluate_text(
                state,
                loaders["val"]["failure"],
                "failure",
                failure_codec,
                data_sharding,
                max_samples=text_eval_sample_limit(
                    args.data_profile,
                    args.eval_max_text_samples,
                ),
            )
            plan_metrics = evaluate_text(
                state,
                loaders["val"]["plan"],
                "plan",
                plan_codec,
                data_sharding,
                max_samples=text_eval_sample_limit(
                    args.data_profile,
                    args.eval_max_text_samples,
                ),
            )
            validate_text_eval_coverage(
                args.data_profile,
                failure_metrics,
                plan_metrics,
            )
            degradation = action_loss / baseline_action_loss - 1.0
            score = float(
                (
                    need_metrics["macro_f1"]
                    + failure_metrics["exact_match"]
                    + plan_metrics["exact_match"]
                )
                / 3.0
            )
            metrics = {
                "step": step,
                "val_score": score,
                "action_loss": action_loss,
                "action_loss_baseline": baseline_action_loss,
                "action_loss_degradation": degradation,
                "action_gate_passed": degradation <= args.action_loss_degradation_limit,
                "need_recovery": need_metrics,
                "failure_reason": failure_metrics,
                "recovery_plan": plan_metrics,
            }
            progress.write(json.dumps(metrics, ensure_ascii=False))
            write_jsonl(metrics_file, metrics)
            if metrics["action_gate_passed"] and score > best_score:
                best_score = score
                save_best_checkpoint(run_dir, state, metrics, filter_)

        if step % args.save_interval == 0 or step == args.num_steps:
            save_state(manager, state, step, filter_, rng)

    logging.info("Waiting for V3 checkpoint writes to finish")
    manager.wait_until_finished()


if __name__ == "__main__":
    main()
