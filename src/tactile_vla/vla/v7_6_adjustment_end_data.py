"""V7.6 magnitude-counterfactual adjustment-end data.

V7.6 keeps the V7.5 H100 observation for every physical sample and changes
only the recovery-plan magnitude in its counterfactual copy.  This makes the
desired travel distance identifiable from the prompt instead of from motion
length alone.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
from itertools import pairwise
import json
from pathlib import Path
from typing import Any

import numpy as np

from tactile_vla.vla.artifacts import canonical_json_bytes, sha256_json
from tactile_vla.vla.structured_text import recovery_plan_text
from tactile_vla.vla.v4_data import SPLITS, validate_v4_index_dataset
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles, scan_selected_qpos
from tactile_vla.vla.v7_5_adjustment_end_data import (
    AdjustmentEndManifestDataset as _V75AdjustmentEndManifestDataset,
)
from tactile_vla.vla.v7_5_adjustment_end_data import (
    TransformedAdjustmentEndDataset as _V75TransformedAdjustmentEndDataset,
)
from tactile_vla.vla.v7_5_adjustment_end_data import build_adjustment_end_artifacts as build_v7_5_adjustment_end_artifacts
from tactile_vla.vla.v4_data import file_identity
from tactile_vla.vla.v7_5_adjustment_end_data import sample_history_frame_indices
from tactile_vla.vla.v7_5_phase_change import (
    PHASE_CHANGE_MAX_TOKEN_LEN,
    PHASE_CHANGE_PROMPT_PROFILE,
    normalize_state_qpos,
    pi05_phase_change_token_length,
)


DATA_PROFILE = "rotation_phase_v7_6_adjustment_end_counterfactual_h100"
EXPERIMENT_KIND = "adjustment_end_action_multitask_v7_6_counterfactual_1to2_h100_idle_1721"
MANIFEST_SCHEMA = "tactile_vla_v7_6_adjustment_end_manifest_v1"
TRAINING_INDEX_SCHEMA = "tactile_vla_v7_6_adjustment_end_training_index_v1"
SUMMARY_SCHEMA = "tactile_vla_v7_6_adjustment_end_summary_v1"
ARTIFACT_HASH_SCHEMA = "tactile_vla_v7_6_adjustment_end_artifact_hashes_v1"
HISTORY_SEED = 42
TRAIN_SLIGHT_COUNTERFACTUAL_KEEP = 975
EXPECTED_ATTEMPT2_COUNTS = {"train": 144, "val": 18, "test": 18}
EXPECTED_POSITIVE_COUNTS = {"train": 5245, "val": 660, "test": 629}
EXPECTED_NEGATIVE_COUNTS = {"train": 10490, "val": 1950, "test": 1565}
SAMPLE_VARIANT_IDS = {
    "factual": 0,
    "moderately_to_slightly": 1,
    "slightly_to_moderately": 2,
}
MAGNITUDE_IDS = {"slightly": 0, "moderately": 1}

LABEL_POLICY = {
    "factual": {
        "sample_range": "inclusive_[A,S+10]",
        "negative_range": "inclusive_[A,S-11]",
        "positive_range": "inclusive_[S-10,S+10]",
    },
    "moderately_to_slightly": {
        "sample_range": "inclusive_[A,S]",
        "slightly_stop": "floor((A+S)/2)",
        "negative_range": "inclusive_[A,S_s-1]",
        "positive_range": "inclusive_[S_s,S]",
    },
    "slightly_to_moderately": {
        "sample_range": "inclusive_[A,S]",
        "label": False,
        "reason": "observed_slightly_trajectory_has_not_reached_moderately_target",
    },
}
HISTORY_POLICY = {
    "window": "inclusive_[p-99,p]",
    "window_frames": 100,
    "sampled_qpos_points": 11,
    "idle_interval": "strictly_between_gripper_motion_stop_and_arm_adjustment_start",
    "idle_keep_k_ratio": {"0": 17, "1": 2, "2": 1},
    "eligibility": "overlap_length_l > k*10",
    "ratio_scope": "per_split_over_unique_physical_histories_with_l_gt_0_before_cloning",
    "ratio_rounding": "largest_remainder_k0_then_k1_then_k2",
    "sampling": "retain_k_random_idle_frames_then_uniformly_sample_non_idle_timeline",
    "counterfactual_pair_history": "identical_to_factual",
    "seed": HISTORY_SEED,
}
COUNTERFACTUAL_SELECTION_POLICY = {
    "train_slightly_to_moderately_keep_count": TRAIN_SLIGHT_COUNTERFACTUAL_KEEP,
    "hard_negative_range": "inclusive_[S-10,S]",
    "hard_negative_selection": "retain_all",
    "additional_selection_pool": "inclusive_[A,S-11]",
    "additional_allocation": "episode_capacity_proportional_largest_remainder",
    "within_episode_priority": "sha256(seed,episode_id,frame_index)",
    "validation_and_test": "retain_all",
    "seed": HISTORY_SEED,
}


def counterfactual_slightly_stop(arm_adjustment_start: int, arm_adjustment_stop: int) -> int:
    return (int(arm_adjustment_start) + int(arm_adjustment_stop)) // 2


def counterfactual_label(
    *, frame_index: int, source_magnitude: str, arm_adjustment_start: int, arm_adjustment_stop: int
) -> bool:
    frame = int(frame_index)
    start, stop = int(arm_adjustment_start), int(arm_adjustment_stop)
    if not start <= frame <= stop:
        raise ValueError("Counterfactual samples must lie in inclusive [A,S]")
    if source_magnitude == "moderately":
        return frame >= counterfactual_slightly_stop(start, stop)
    if source_magnitude == "slightly":
        return False
    raise ValueError(f"Unsupported source magnitude: {source_magnitude!r}")


def _stable_priority(seed: int, *values: int) -> int:
    joined = ":".join(str(int(value)) for value in (seed, *values))
    return int.from_bytes(hashlib.sha256(joined.encode()).digest()[:8], "big")


def _proportional_largest_remainder(capacities: Mapping[int, int], total: int, seed: int) -> dict[int, int]:
    available = sum(int(value) for value in capacities.values())
    if total < 0 or total > available:
        raise ValueError(f"Cannot select {total} rows from capacity {available}")
    if available == 0:
        return {int(key): 0 for key in capacities}
    quotas = {int(key): total * int(value) / available for key, value in capacities.items()}
    result = {key: int(np.floor(quota)) for key, quota in quotas.items()}
    remainder = total - sum(result.values())
    order = sorted(
        result,
        key=lambda key: (-(quotas[key] - result[key]), _stable_priority(seed, key), key),
    )
    for key in order[:remainder]:
        result[key] += 1
    if sum(result.values()) != total or any(result[key] > capacities[key] for key in result):
        raise AssertionError("Largest-remainder allocation failed")
    return result


def select_train_slight_counterfactual_pairs(
    rows: Sequence[Mapping[str, Any]], *, keep_count: int = TRAIN_SLIGHT_COUNTERFACTUAL_KEEP, seed: int = HISTORY_SEED
) -> tuple[set[int], dict[str, Any]]:
    """Select source-slightly counterfactual physical rows for the train split."""

    eligible = [
        row for row in rows
        if row["split"] == "train"
        and row["source_magnitude"] == "slightly"
        and int(row["frame_index"]) <= int(row["arm_adjustment_stop_frame"])
    ]
    hard = [
        row for row in eligible
        if int(row["frame_index"]) >= int(row["arm_adjustment_stop_frame"]) - 10
    ]
    if len(hard) > keep_count:
        raise ValueError("Hard-negative set is larger than the requested retained set")
    hard_pairs = {int(row["pair_id"]) for row in hard}
    early_by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        if int(row["pair_id"]) not in hard_pairs:
            early_by_episode[int(row["episode_id"])].append(row)
    additional_count = keep_count - len(hard_pairs)
    allocation = _proportional_largest_remainder(
        {episode: len(values) for episode, values in early_by_episode.items()},
        additional_count,
        seed,
    )
    additional_pairs: set[int] = set()
    per_episode = {}
    for episode, values in sorted(early_by_episode.items()):
        selected = sorted(
            values,
            key=lambda row: (
                _stable_priority(seed, int(row["episode_id"]), int(row["frame_index"])),
                int(row["frame_index"]),
            ),
        )[: allocation[episode]]
        additional_pairs.update(int(row["pair_id"]) for row in selected)
        per_episode[str(episode)] = len(selected)
    selected_pairs = hard_pairs | additional_pairs
    if len(selected_pairs) != keep_count:
        raise AssertionError("V7.6 retained counterfactual count is not exact")
    return selected_pairs, {
        "eligible_count": len(eligible),
        "retained_count": len(selected_pairs),
        "removed_count": len(eligible) - len(selected_pairs),
        "hard_negative_retained_count": len(hard_pairs),
        "additional_retained_count": len(additional_pairs),
        "per_episode_additional_counts": per_episode,
    }


def _replace_recovery_plan(prompt: str, plan: str) -> str:
    lines = str(prompt).splitlines()
    indices = [index for index, line in enumerate(lines) if line.startswith("Recovery plan: ")]
    if len(indices) != 1:
        raise ValueError("Prompt must contain exactly one Recovery plan line")
    lines[indices[0]] = f"Recovery plan: {plan}"
    return "\n".join(lines)


def _token_summary(lengths: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=np.int32)
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "p50": int(np.percentile(values, 50)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
        "max": int(values.max()),
        "over_limit_count": int(np.count_nonzero(values > PHASE_CHANGE_MAX_TOKEN_LEN)),
    }


def _factual_label(frame: int, stop: int) -> bool:
    return int(stop) - 10 <= int(frame) <= int(stop) + 10


def build_adjustment_end_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    action_index_file: Path,
    boundary_filter_file: Path,
    norm_stats_file: Path,
    caption_summary_file: Path,
    stage_a_checkpoint: Path,
    stage_a_config_file: Path,
    tokenizer: Any,
    seed: int = HISTORY_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    factual_rows, factual_index, _ = build_v7_5_adjustment_end_artifacts(
        dataset_dir=dataset_dir,
        v4_index_file=v4_index_file,
        action_index_file=action_index_file,
        boundary_filter_file=boundary_filter_file,
        norm_stats_file=norm_stats_file,
        caption_summary_file=caption_summary_file,
        stage_a_checkpoint=stage_a_checkpoint,
        stage_a_config_file=stage_a_config_file,
        tokenizer=tokenizer,
        seed=seed,
        idle_keep_weights=(17, 2, 1),
    )
    v4_index = json.loads(v4_index_file.read_text())
    _, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir.expanduser().resolve())
    selected_episode_ids = {int(row["episode_id"]) for row in factual_rows}
    qpos_lookup = scan_selected_qpos(dataset_dir=dataset_dir, selected_episode_ids=selected_episode_ids)
    stats = load_state_quantiles(norm_stats_file)

    prepared_factual: list[dict[str, Any]] = []
    for source in factual_rows:
        row = dict(source)
        frame = global_lookup[int(row["current_global_index"])]
        magnitude = str(frame.horizontal_magnitude)
        if magnitude not in MAGNITUDE_IDS:
            raise ValueError(f"Unsupported factual magnitude {magnitude!r}")
        expected_plan = recovery_plan_text(frame.horizontal_direction, magnitude, "none", "moderately")
        if f"Recovery plan: {expected_plan}" not in str(row["prompt"]).splitlines():
            raise ValueError(f"Unexpected factual recovery plan at global index {frame.global_index}")
        row.update({
            "schema_version": MANIFEST_SCHEMA,
            "data_profile": DATA_PROFILE,
            "experiment_kind": EXPERIMENT_KIND,
            "sample_variant": "factual",
            "sample_variant_id": SAMPLE_VARIANT_IDS["factual"],
            "pair_id": int(row["current_global_index"]),
            "source_direction": str(frame.horizontal_direction),
            "source_magnitude": magnitude,
            "source_magnitude_id": MAGNITUDE_IDS[magnitude],
            "prompt_magnitude": magnitude,
            "prompt_magnitude_id": MAGNITUDE_IDS[magnitude],
            "target_stop_frame": int(row["arm_adjustment_stop_frame"]),
        })
        prepared_factual.append(row)

    selected_train_pairs, selection_summary = select_train_slight_counterfactual_pairs(
        prepared_factual, keep_count=TRAIN_SLIGHT_COUNTERFACTUAL_KEEP, seed=seed
    )
    manifest: list[dict[str, Any]] = []
    split_rows: dict[str, list[int]] = {split: [] for split in SPLITS}
    split_globals: dict[str, list[int]] = {split: [] for split in SPLITS}
    token_lengths: list[int] = []
    variant_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}

    def append(row: dict[str, Any]) -> None:
        split = str(row["split"])
        row_index = len(manifest)
        manifest.append(row)
        split_rows[split].append(row_index)
        split_globals[split].append(int(row["current_global_index"]))
        token_lengths.append(int(row["phase_change_token_len"]))
        variant_counts[split][str(row["sample_variant"])] += 1

    for factual in prepared_factual:
        append(factual)
        frame_index = int(factual["frame_index"])
        stop = int(factual["arm_adjustment_stop_frame"])
        if frame_index > stop:
            continue
        source_magnitude = str(factual["source_magnitude"])
        if (
            source_magnitude == "slightly"
            and factual["split"] == "train"
            and int(factual["pair_id"]) not in selected_train_pairs
        ):
            continue
        prompt_magnitude = "slightly" if source_magnitude == "moderately" else "moderately"
        variant = f"{source_magnitude}_to_{prompt_magnitude}"
        plan = recovery_plan_text(
            str(factual["source_direction"]), prompt_magnitude, "none", "moderately"
        )
        prompt = _replace_recovery_plan(str(factual["prompt"]), plan)
        current_qpos = qpos_lookup[int(factual["current_global_index"])]
        token_length = pi05_phase_change_token_length(
            tokenizer=tokenizer,
            prompt=prompt,
            normalized_current_qpos=normalize_state_qpos(current_qpos, stats),
        )
        if token_length > PHASE_CHANGE_MAX_TOKEN_LEN:
            raise ValueError(f"V7.6 prompt truncation at pair_id={factual['pair_id']} variant={variant}")
        target_stop = (
            counterfactual_slightly_stop(int(factual["arm_adjustment_start_frame"]), stop)
            if source_magnitude == "moderately"
            else -1
        )
        counterfactual = {
            **factual,
            "sample_variant": variant,
            "sample_variant_id": SAMPLE_VARIANT_IDS[variant],
            "prompt_magnitude": prompt_magnitude,
            "prompt_magnitude_id": MAGNITUDE_IDS[prompt_magnitude],
            "target_stop_frame": target_stop,
            "adjustment_end": counterfactual_label(
                frame_index=frame_index,
                source_magnitude=source_magnitude,
                arm_adjustment_start=int(factual["arm_adjustment_start_frame"]),
                arm_adjustment_stop=stop,
            ),
            "phase_change_token_len": token_length,
            "prompt": prompt,
        }
        append(counterfactual)

    splits = {}
    for split in SPLITS:
        selected = [manifest[index] for index in split_rows[split]]
        positive = sum(bool(row["adjustment_end"]) for row in selected)
        negative = len(selected) - positive
        expected = (EXPECTED_POSITIVE_COUNTS[split], EXPECTED_NEGATIVE_COUNTS[split])
        if (positive, negative) != expected:
            raise ValueError(f"V7.6 {split} counts changed: {(positive, negative)} != {expected}")
        factual_split = factual_index["splits"][split]
        splits[split] = {
            "manifest_row_indices": split_rows[split],
            "global_indices": split_globals[split],
            "sample_count": len(selected),
            "positive_count": positive,
            "negative_count": negative,
            "attempt2_count": EXPECTED_ATTEMPT2_COUNTS[split],
            "variant_counts": dict(sorted(variant_counts[split].items())),
            "idle_keep_ratio_unique_physical_histories": factual_split["idle_keep_ratio"],
        }

    index: dict[str, Any] = {
        **factual_index,
        "schema_version": TRAINING_INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "label_policy": LABEL_POLICY,
        "history_policy": HISTORY_POLICY,
        "counterfactual_selection_policy": COUNTERFACTUAL_SELECTION_POLICY,
        "counterfactual_selection_summary": selection_summary,
        "classification_sampling_policy": {
            "strategy": "deterministic_natural_manifest_stream",
            "train_positive": EXPECTED_POSITIVE_COUNTS["train"],
            "train_negative": EXPECTED_NEGATIVE_COUNTS["train"],
            "positive_to_negative_ratio": "1:2",
            "seed": seed,
            "tail_policy": "carry_into_next_epoch",
        },
        "splits": splits,
        "manifest_identity": {"count": len(manifest), "content_sha256": sha256_json(manifest)},
        "prompt_token_lengths": _token_summary(token_lengths),
    }
    index.pop("training_data_hash", None)
    index["training_data_hash"] = sha256_json(index)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "data_profile": DATA_PROFILE,
        "label_policy": LABEL_POLICY,
        "history_policy": HISTORY_POLICY,
        "counterfactual_selection_policy": COUNTERFACTUAL_SELECTION_POLICY,
        "counterfactual_selection_summary": selection_summary,
        "valid_sample_count": len(manifest),
        "positive_count": sum(EXPECTED_POSITIVE_COUNTS.values()),
        "negative_count": sum(EXPECTED_NEGATIVE_COUNTS.values()),
        "split_summary": {
            split: {
                key: value for key, value in splits[split].items()
                if key not in {"manifest_row_indices", "global_indices"}
            }
            for split in SPLITS
        },
        "prompt_token_lengths": index["prompt_token_lengths"],
        "training_data_hash": index["training_data_hash"],
    }
    return manifest, index, summary


def validate_adjustment_end_artifacts(*, index: Mapping[str, Any], manifest: list[dict[str, Any]]) -> None:
    expected = {
        "schema_version": TRAINING_INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": EXPERIMENT_KIND,
        "label_policy": LABEL_POLICY,
        "history_policy": HISTORY_POLICY,
        "counterfactual_selection_policy": COUNTERFACTUAL_SELECTION_POLICY,
    }
    if any(index.get(key) != value for key, value in expected.items()):
        raise ValueError("V7.6 adjustment_end index header mismatch")
    payload = {key: value for key, value in index.items() if key != "training_data_hash"}
    if index.get("training_data_hash") != sha256_json(payload):
        raise ValueError("V7.6 adjustment_end training_data_hash mismatch")
    identity = index["manifest_identity"]
    if int(identity["count"]) != len(manifest) or identity["content_sha256"] != sha256_json(manifest):
        raise ValueError("V7.6 manifest identity mismatch")
    factual_by_pair = {
        int(row["pair_id"]): row for row in manifest if row["sample_variant"] == "factual"
    }
    for row in manifest:
        frame = int(row["frame_index"])
        start, stop = int(row["arm_adjustment_start_frame"]), int(row["arm_adjustment_stop_frame"])
        variant = str(row["sample_variant"])
        if int(row["sample_variant_id"]) != SAMPLE_VARIANT_IDS.get(variant, -1):
            raise ValueError("V7.6 sample variant id mismatch")
        source, prompt_magnitude = str(row["source_magnitude"]), str(row["prompt_magnitude"])
        if int(row["source_magnitude_id"]) != MAGNITUDE_IDS.get(source, -1):
            raise ValueError("V7.6 source magnitude id mismatch")
        if int(row["prompt_magnitude_id"]) != MAGNITUDE_IDS.get(prompt_magnitude, -1):
            raise ValueError("V7.6 prompt magnitude id mismatch")
        if variant == "factual":
            if prompt_magnitude != source or not start <= frame <= stop + 10:
                raise ValueError("Invalid V7.6 factual range or magnitude")
            expected_label = _factual_label(frame, stop)
        else:
            if not start <= frame <= stop or source == prompt_magnitude:
                raise ValueError("Invalid V7.6 counterfactual range or magnitude")
            expected_label = counterfactual_label(
                frame_index=frame, source_magnitude=source,
                arm_adjustment_start=start, arm_adjustment_stop=stop,
            )
            factual = factual_by_pair.get(int(row["pair_id"]))
            if factual is None:
                raise ValueError("Counterfactual row lacks its factual pair")
            invariant_keys = (
                "current_global_index", "sampled_history_frame_indices", "history_global_indices",
                "qpos_h100_11_discrete", "idle_keep_count", "retained_idle_frame_indices",
            )
            if any(row[key] != factual[key] for key in invariant_keys):
                raise ValueError("Counterfactual pair changed its physical observation/history")
            factual_lines, counterfactual_lines = str(factual["prompt"]).splitlines(), str(row["prompt"]).splitlines()
            differences = [i for i, pair in enumerate(zip(factual_lines, counterfactual_lines, strict=True)) if pair[0] != pair[1]]
            if len(differences) != 1 or not factual_lines[differences[0]].startswith("Recovery plan: "):
                raise ValueError("Counterfactual prompt changed more than the recovery plan")
        if bool(row["adjustment_end"]) != expected_label:
            raise ValueError("V7.6 label mismatch")
        sampled = [int(value) for value in row["sampled_history_frame_indices"]]
        if len(sampled) != 11 or any(a >= b for a, b in pairwise(sampled)) or sampled[-1] != frame:
            raise ValueError("V7.6 sampled history is invalid")
        k, overlap = int(row["idle_keep_count"]), int(row["idle_overlap_length"])
        if overlap > 0 and overlap <= k * 10:
            raise ValueError("V7.6 idle retention violates l > k*10")
        if int(row["phase_change_token_len"]) > PHASE_CHANGE_MAX_TOKEN_LEN:
            raise ValueError("V7.6 prompt exceeds token limit")
    for split in SPLITS:
        indices = [int(value) for value in index["splits"][split]["manifest_row_indices"]]
        rows = [manifest[value] for value in indices]
        positive = sum(bool(row["adjustment_end"]) for row in rows)
        if (positive, len(rows) - positive) != (
            EXPECTED_POSITIVE_COUNTS[split], EXPECTED_NEGATIVE_COUNTS[split]
        ):
            raise ValueError(f"V7.6 {split} class counts mismatch")


def load_indexed_manifest_rows(*, index: Mapping[str, Any], manifest_path: Path) -> dict[int, dict[str, Any]]:
    expected_hash = sha256_json({key: value for key, value in index.items() if key != "training_data_hash"})
    if index.get("training_data_hash") != expected_hash or index.get("label_policy") != LABEL_POLICY:
        raise ValueError("V7.6 adjustment_end index identity mismatch")
    wanted = {
        int(row_index): (split, int(global_index))
        for split in SPLITS
        for row_index, global_index in zip(
            index["splits"][split]["manifest_row_indices"],
            index["splits"][split]["global_indices"], strict=True,
        )
    }
    file_digest, content_digest = hashlib.sha256(), hashlib.sha256()
    content_digest.update(b"[")
    selected: dict[int, dict[str, Any]] = {}
    row_count = 0
    with manifest_path.open("rb") as stream:
        for raw in stream:
            file_digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row_count:
                content_digest.update(b",")
            content_digest.update(canonical_json_bytes(row))
            if row_count in wanted:
                split, global_index = wanted[row_count]
                if row["split"] != split or int(row["current_global_index"]) != global_index:
                    raise ValueError(f"V7.6 indexed manifest row {row_count} identity mismatch")
                selected[row_count] = row
            row_count += 1
    content_digest.update(b"]")
    identity = index["manifest_identity"]
    if row_count != int(identity["count"]) or content_digest.hexdigest() != identity["content_sha256"]:
        raise ValueError("V7.6 manifest content mismatch")
    if file_digest.hexdigest() != str(identity.get("file_sha256", "")):
        raise ValueError("V7.6 manifest file SHA mismatch")
    if set(selected) != set(wanted):
        raise ValueError("V7.6 manifest lacks indexed rows")
    return selected


def artifact_hash_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_HASH_SCHEMA,
        "artifacts": {name: file_identity(path) for name, path in sorted(paths.items())},
    }


class DeterministicNaturalBatchSampler:
    """Shuffle the reduced manifest naturally, carrying epoch tails forward."""

    BATCH_SIZE = 8

    def __init__(self, *, labels: list[bool], num_batches: int, seed: int, start_batch: int = 0):
        self.size = len(labels)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        if self.size < self.BATCH_SIZE:
            raise ValueError("Classification dataset is smaller than one batch")
        if self.num_batches <= 0 or not 0 <= self.start_batch <= self.num_batches:
            raise ValueError("Invalid sampler batch range")
        positive = sum(bool(value) for value in labels)
        negative = self.size - positive
        if negative != positive * 2:
            raise ValueError(f"V7.6 train manifest must be exactly 1:2, got {positive}:{negative}")

    def __len__(self) -> int:
        return self.num_batches - self.start_batch

    def __iter__(self):
        emitted = 0
        epoch = 0
        buffer: list[int] = []
        while emitted < self.num_batches:
            permutation = np.random.default_rng(self.seed + epoch).permutation(self.size)
            buffer.extend(int(value) for value in permutation)
            cursor = 0
            while len(buffer) - cursor >= self.BATCH_SIZE and emitted < self.num_batches:
                batch = buffer[cursor : cursor + self.BATCH_SIZE]
                cursor += self.BATCH_SIZE
                if emitted >= self.start_batch:
                    yield batch
                emitted += 1
            buffer = buffer[cursor:]
            epoch += 1


class AdjustmentEndManifestDataset(_V75AdjustmentEndManifestDataset):
    def __getitem__(self, dataset_index: int) -> dict[str, Any]:
        output = super().__getitem__(dataset_index)
        row_index = self.row_indices[dataset_index]
        row = self.manifest[row_index]
        output.update({
            "manifest_row_index": row_index,
            "sample_variant_id": int(row["sample_variant_id"]),
            "source_magnitude_id": int(row["source_magnitude_id"]),
            "prompt_magnitude_id": int(row["prompt_magnitude_id"]),
            "pair_id": int(row["pair_id"]),
            "target_stop_frame": int(row["target_stop_frame"]),
        })
        return output


class TransformedAdjustmentEndDataset(_V75TransformedAdjustmentEndDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.dataset[index]
        identity_keys = (
            "global_index", "episode_id", "attempt_id", "frame_index", "rexecution_frame",
            "arm_adjustment_stop_frame", "manifest_row_index", "sample_variant_id",
            "source_magnitude_id", "prompt_magnitude_id", "pair_id", "target_stop_frame",
        )
        identity = {key: np.asarray(raw[key], dtype=np.int64) for key in identity_keys}
        transformed = self.transform(raw)
        transformed["adjustment_end_label"] = np.asarray(raw["adjustment_end_label"], dtype=np.int32)
        transformed.update(identity)
        return transformed


__all__ = [
    "AdjustmentEndManifestDataset", "ARTIFACT_HASH_SCHEMA", "COUNTERFACTUAL_SELECTION_POLICY",
    "DATA_PROFILE", "DeterministicNaturalBatchSampler", "EXPECTED_NEGATIVE_COUNTS",
    "EXPECTED_POSITIVE_COUNTS", "EXPERIMENT_KIND", "HISTORY_POLICY", "LABEL_POLICY",
    "MAGNITUDE_IDS", "MANIFEST_SCHEMA", "SAMPLE_VARIANT_IDS", "SUMMARY_SCHEMA",
    "TRAINING_INDEX_SCHEMA", "TransformedAdjustmentEndDataset", "artifact_hash_payload",
    "build_adjustment_end_artifacts", "counterfactual_label", "counterfactual_slightly_stop",
    "load_indexed_manifest_rows", "sample_history_frame_indices",
    "select_train_slight_counterfactual_pairs", "validate_adjustment_end_artifacts",
]
