"""Book V9.2 factual adjustment-end data, with raw H100 history."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_stage_a_data import native_reexecution_events, validate_training_index
from tactile_vla.vla.v4_data import SPLITS, file_identity, validate_v4_index_dataset
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles, scan_selected_qpos
from tactile_vla.vla.v7_5_phase_change import (
    PHASE_CHANGE_MAX_TOKEN_LEN, PHASE_CHANGE_PROMPT_PROFILE,
    build_adjustment_end_prompt, helper_identity, normalize_state_qpos,
    pi05_phase_change_token_length,
)
from tactile_vla.vla.v7_5_runtime_history import RUNTIME_SAMPLE_OFFSETS


DATA_PROFILE = "book_adjustment_end_v9_2_h100"
EXPERIMENT_KIND = "book_adjustment_end_action_multitask_v9_2_factual_1to2_h100"
MANIFEST_SCHEMA = "book_v9_2_adjustment_end_manifest_v1"
INDEX_SCHEMA = "book_v9_2_adjustment_end_training_index_v1"
LABEL_POLICY = {
    "sample_range": "inclusive_[A,R]", "A": "attempt2_frame_0",
    "R": "native_v4_reexecution_frame_index",
    "positive_range": "inclusive_[R-10,R]", "negative_range": "inclusive_[A,R-11]",
}
HISTORY_POLICY = {
    "window": "raw_chronological_inclusive_[p-99,p]",
    "sample_offsets": list(RUNTIME_SAMPLE_OFFSETS),
    "early_attempt2_prefix": "same_episode_attempt1_tail",
    "idle_compression": False, "counterfactual": False,
}
TRAIN_SAMPLING_POLICY = {
    "positive": "all_11_per_attempt2",
    "negative_hard": "all_10_inclusive_[R-20,R-11]",
    "negative_middle": "random_8_without_replacement_inclusive_[R-40,R-21]",
    "negative_early": "random_4_without_replacement_inclusive_[A,R-41]",
    "seed": 42, "positive_to_negative": "1:2",
    "val_test": "all_factual_frames_inclusive_[A,R]",
}
EXPECTED_COUNTS = {
    "train": (24, 264, 528), "val": (3, 33, 474), "test": (3, 33, 438),
}


def adjustment_end_label(frame_index: int, reexecution_frame: int) -> bool:
    p, r = int(frame_index), int(reexecution_frame)
    if not 0 <= p <= r:
        raise ValueError("Book V9.2 classification frame must lie in inclusive [A,R]")
    return p >= r - 10


def selected_train_frames(reexecution_frame: int, *, episode_id: int, seed: int = 42) -> set[int]:
    """Preserve all boundary-near examples; thin earlier negatives by distance."""
    r = int(reexecution_frame)
    if r < 44:
        raise ValueError(f"R={r} cannot support the agreed three negative buckets")
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(episode_id)]))
    hard = set(range(r - 20, r - 10))
    middle = set(int(value) for value in rng.choice(np.arange(r - 40, r - 20), 8, replace=False))
    early = set(int(value) for value in rng.choice(np.arange(0, r - 40), 4, replace=False))
    positive = set(range(r - 10, r + 1))
    result = hard | middle | early | positive
    if len(result) != 33 or sum(value >= r - 10 for value in result) != 11:
        raise AssertionError("Book V9.2 bucket selection is not 11 positive / 22 negative")
    return result


def build_artifacts(
    *, dataset_dir: Path, v4_index_file: Path, action_index_file: Path,
    norm_stats_file: Path, stage_a_checkpoint: Path | None, tokenizer: Any, seed: int = 42,
    expected_counts: Mapping[str, tuple[int, int, int]] = EXPECTED_COUNTS,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    v4_index = json.loads(v4_index_file.read_text())
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    action_index = json.loads(action_index_file.read_text())
    validate_training_index(action_index, index_path=action_index_file, dataset_dir=dataset_dir)
    if action_index["selection_hash"] != v4_index["selection_hash"]:
        raise ValueError("Book Stage A and V4 selections differ")
    if action_index["v4_norm_stats_sha256"] != sha256_file(norm_stats_file):
        raise ValueError("Book Stage A and classifier norm stats differ")
    events, timing_identity = native_reexecution_events(v4_index=v4_index, frames=frames)
    if timing_identity != action_index["native_reexecution_timing_identity"]:
        raise ValueError("Native rexecution event identity changed")
    # V9.2 still passes a checkpoint; V9.4 constructs model-independent data.
    model_identity, model_sources = {}, {}
    if stage_a_checkpoint is not None:
        stage_a_checkpoint = stage_a_checkpoint.expanduser().resolve()
        if stage_a_checkpoint.name != "15000" or not (stage_a_checkpoint / "params/_METADATA").is_file():
            raise ValueError("Book V9.2 requires the Stage A step 15000 checkpoint")
        stage_a_config = stage_a_checkpoint.parent / "config.json"
        config = json.loads(stage_a_config.read_text())
        if (config.get("data_profile"), config.get("num_steps"), config.get("use_state_history")) != (
            "book_stage_a_v1", 15000, False,
        ):
            raise ValueError("Stage A config is not the book V9.1 no-history model")
        model_identity = {"stage_a_checkpoint": {"path": str(stage_a_checkpoint), "step": 15000}}
        model_sources = {
            "backbone_config": file_identity(stage_a_config),
            "stage_a_params_metadata": file_identity(stage_a_checkpoint / "params/_METADATA"),
        }

    groups: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for frame in frames:
        groups[frame.attempt_key].append(frame)
    for key, group in groups.items():
        group.sort(key=lambda frame: frame.frame_index)
        if [frame.frame_index for frame in group] != list(range(len(group))):
            raise ValueError(f"Non-contiguous attempt frames: {key}")
    split_by_global = {
        int(global_index): split for split in SPLITS
        for global_index in v4_index["splits"][split]["execution_indices"]
    }
    episode_split: dict[int, str] = {}
    for frame in frames:
        split = split_by_global.get(frame.global_index)
        if split is not None:
            prior = episode_split.setdefault(frame.episode_id, split)
            if prior != split:
                raise ValueError(f"Episode {frame.episode_id} crosses splits")
    qpos = scan_selected_qpos(dataset_dir=dataset_dir, selected_episode_ids={key[0] for key in events})
    stats = load_state_quantiles(norm_stats_file)
    manifest: list[dict[str, Any]] = []
    split_rows = {split: [] for split in SPLITS}
    split_globals = {split: [] for split in SPLITS}
    selected_counts = {split: Counter() for split in SPLITS}
    full_counts = {split: Counter() for split in SPLITS}
    attempt_counts = Counter()
    token_lengths = []
    for (episode_id, attempt_id), r in sorted(events.items()):
        if attempt_id != 2:
            raise ValueError("Unexpected rexecution event on attempt1")
        split = episode_split[episode_id]
        attempt_counts[split] += 1
        attempt1 = groups[(episode_id, 1)]
        attempt2 = groups[(episode_id, 2)]
        timeline = attempt1 + attempt2
        if len(attempt1) < 99:
            raise ValueError(f"Attempt1 too short for raw H100: {episode_id}")
        selected = selected_train_frames(r, episode_id=episode_id, seed=seed) if split == "train" else set(range(r + 1))
        for p in range(r + 1):
            frame = attempt2[p]
            position = len(attempt1) + p
            sampled_frames = [timeline[position - 99 + offset] for offset in RUNTIME_SAMPLE_OFFSETS]
            sampled_globals = [item.global_index for item in sampled_frames]
            sampled_qpos = np.stack([qpos[index] for index in sampled_globals])
            current_qpos = qpos[frame.global_index]
            prompt, discrete = build_adjustment_end_prompt(
                instruction=frame.instruction, tactile_caption=frame.tactile_caption,
                recovery_plan=frame.input_recovery_plan, sampled_qpos=sampled_qpos, stats=stats,
            )
            token_length = pi05_phase_change_token_length(
                tokenizer=tokenizer, prompt=prompt,
                normalized_current_qpos=normalize_state_qpos(current_qpos, stats),
            )
            if token_length > PHASE_CHANGE_MAX_TOKEN_LEN:
                raise ValueError(f"Book V9.2 prompt truncated at global_index={frame.global_index}")
            positive = adjustment_end_label(p, r)
            row_index = len(manifest)
            manifest.append({
                "schema_version": MANIFEST_SCHEMA, "data_profile": DATA_PROFILE,
                "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE, "experiment_kind": EXPERIMENT_KIND,
                "split": split, "episode_id": episode_id, "attempt_id": 2,
                "frame_index": p, "current_global_index": frame.global_index,
                "rexecution_frame": r, "history_global_indices": sampled_globals,
                "history_attempt_ids": [item.attempt_id for item in sampled_frames],
                "history_crosses_attempt": any(item.attempt_id == 1 for item in sampled_frames),
                "qpos_h100_11_discrete": discrete.tolist(), "adjustment_end": positive,
                "classification_sample_valid": p in selected,
                "phase_change_token_len": token_length, "prompt": prompt,
            })
            full_counts[split]["positive" if positive else "negative"] += 1
            token_lengths.append(token_length)
            if p in selected:
                split_rows[split].append(row_index)
                split_globals[split].append(frame.global_index)
                selected_counts[split]["positive" if positive else "negative"] += 1
    splits = {}
    for split in SPLITS:
        actual = (attempt_counts[split], selected_counts[split]["positive"], selected_counts[split]["negative"])
        if actual != expected_counts[split]:
            raise ValueError(f"Book adjustment {split} counts changed: {actual} != {expected_counts[split]}")
        splits[split] = {
            "manifest_row_indices": split_rows[split], "global_indices": split_globals[split],
            "sample_count": len(split_rows[split]), "positive_count": actual[1],
            "negative_count": actual[2], "attempt2_count": actual[0],
            "full_factual_count": sum(full_counts[split].values()),
        }
    token_array = np.asarray(token_lengths)
    index = {
        "schema_version": INDEX_SCHEMA, "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE, "experiment_kind": EXPERIMENT_KIND,
        "dataset_dir": str(dataset_dir), "selection_hash": v4_index["selection_hash"],
        "action_training_data_hash": action_index["training_data_hash"],
        "label_policy": LABEL_POLICY, "history_policy": HISTORY_POLICY,
        "train_sampling_policy": {**TRAIN_SAMPLING_POLICY, "seed": seed},
        "splits": splits, "manifest_identity": {"count": len(manifest), "content_sha256": sha256_json(manifest)},
        "prompt_helper": helper_identity(),
        "prompt_token_lengths": {"min": int(token_array.min()), "max": int(token_array.max()),
                                 "p99": int(np.percentile(token_array, 99)), "over_limit_count": 0},
        "state_norm": {"method": "q01_q99_pi05", "norm_stats_sha256": sha256_file(norm_stats_file),
                       "q01": stats.q01.tolist(), "q99": stats.q99.tolist()},
        "caption_source": {"field": "tactile_caption", "source": "book_v4_lerobot"},
        **model_identity,
        "source_files": {
            "v4_training_index": file_identity(v4_index_file),
            "book_stage_a_index": file_identity(action_index_file),
            "v4_norm_stats": file_identity(norm_stats_file),
            **model_sources,
        },
    }
    summary = {
        "schema_version": "book_v9_2_adjustment_end_summary_v1",
        "data_profile": DATA_PROFILE, "attempt2_count": sum(attempt_counts.values()),
        "manifest_count": len(manifest), "splits": splits,
        "full_factual_counts": {split: dict(full_counts[split]) for split in SPLITS},
        "history_crosses_attempt_count": sum(bool(row["history_crosses_attempt"]) for row in manifest),
        "train_sampling_policy": index["train_sampling_policy"],
    }
    return manifest, index, summary


def load_indexed_manifest_rows(*, index: Mapping[str, Any], manifest_path: Path) -> dict[int, dict[str, Any]]:
    if index.get("schema_version") != INDEX_SCHEMA or index.get("label_policy") != LABEL_POLICY:
        raise ValueError("Book V9.2 classification index protocol mismatch")
    if index.get("training_data_hash") != sha256_json({key: value for key, value in index.items() if key != "training_data_hash"}):
        raise ValueError("Book V9.2 training_data_hash mismatch")
    if sha256_file(manifest_path) != index["manifest_identity"].get("file_sha256"):
        raise ValueError("Book V9.2 manifest file SHA256 mismatch")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if len(rows) != index["manifest_identity"]["count"] or sha256_json(rows) != index["manifest_identity"]["content_sha256"]:
        raise ValueError("Book V9.2 manifest content mismatch")
    wanted = {}
    for split in SPLITS:
        selected = index["splits"][split]
        if len(selected["manifest_row_indices"]) != selected["sample_count"]:
            raise ValueError(f"Book V9.2 {split} sample count mismatch")
        for row_index, global_index in zip(selected["manifest_row_indices"], selected["global_indices"], strict=True):
            row = rows[row_index]
            if row["split"] != split or row["current_global_index"] != global_index or not row["classification_sample_valid"]:
                raise ValueError(f"Book V9.2 manifest row {row_index} identity mismatch")
            wanted[row_index] = row
    return wanted


class AdjustmentEndManifestDataset:
    def __init__(self, *, manifest, manifest_row_indices, global_indices, lerobot_dataset, state_history_len=0):
        if state_history_len != 0:
            raise ValueError("Book V9.2 classifier uses prompt H100 only, not continuous state history")
        self.manifest = manifest
        self.row_indices = [int(value) for value in manifest_row_indices]
        self.global_indices = [int(value) for value in global_indices]
        self.dataset = lerobot_dataset

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, index):
        row_index = self.row_indices[index]
        row = self.manifest[row_index]
        global_index = self.global_indices[index]
        item = self.dataset[global_index]
        identity = tuple(int(item[key]) for key in ("index", "episode_id", "attempt_id", "frame_index"))
        expected = (global_index, row["episode_id"], 2, row["frame_index"])
        if identity != expected:
            raise ValueError(f"Book V9.2 LeRobot identity mismatch: {identity} != {expected}")
        state = np.asarray(item["observation.state"], dtype=np.float32)
        if state.shape != (7,):
            raise ValueError("Book V9.2 state must be seven-dimensional")
        return {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": state, "prompt": row["prompt"],
            "adjustment_end_label": int(row["adjustment_end"]),
            "global_index": global_index, "episode_id": row["episode_id"],
            "attempt_id": 2, "frame_index": row["frame_index"],
            "rexecution_frame": row["rexecution_frame"], "manifest_row_index": row_index,
        }


class TransformedAdjustmentEndDataset:
    def __init__(self, dataset, transform):
        self.dataset, self.transform = dataset, transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raw = self.dataset[index]
        identity = {key: np.asarray(raw[key], dtype=np.int64) for key in (
            "global_index", "episode_id", "attempt_id", "frame_index", "rexecution_frame", "manifest_row_index",
        )}
        transformed = self.transform(raw)
        transformed["adjustment_end_label"] = np.asarray(raw["adjustment_end_label"], dtype=np.int32)
        transformed.update(identity)
        return transformed
