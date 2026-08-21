"""Strict V4 rotation data artifacts and LeRobot frame identities.

The V4 raw/profile pipeline deliberately stops before training.  This module
joins those immutable sidecars to the independently converted LeRobot rows;
it never infers task semantics from episode-number ranges.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.structured_text import legal_failure_reasons, legal_recovery_plans


ROTATION_V4 = "rotation_v4"
V4_PROFILE_SCHEMA = "tactile_vla_v4_rotation_profile_v1"
V4_SPLIT_SCHEMA = "tactile_vla_v4_rotation_split_v1"
V4_SELECTION_SCHEMA = "tactile_vla_v4_rotation_selection_v1"
V4_REASONING_SCHEMA = "tactile_vla_v4_synthetic_reasoning_v1"
V4_ATTEMPT_SCHEMA = "tactile_vla_v4_rotation_attempt_v1"
V4_TRAINING_INDEX_SCHEMA = "tactile_vla_v4_training_index_v1"
V4_NEED_SCHEMA = "tactile_vla_v4_need_manifest_v1"
SPLITS = ("train", "val", "test")
ROTATION_DIRECTIONS = ("right", "left", "front", "back")
V4_REQUIRED_SOURCE_FILES = {
    "selection",
    "profile",
    "splits",
    "action_frame_manifest",
    "reasoning_summary",
    *(f"need_{split}" for split in SPLITS),
    *(f"failure_reason_{split}" for split in SPLITS),
    *(f"reasoning_{split}" for split in SPLITS),
    "lerobot_parquet",
}


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar(value[0])
    return value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


@dataclass(frozen=True)
class V4Frame:
    global_index: int
    lerobot_episode_index: int
    episode_id: int
    attempt_id: int
    frame_index: int
    ros_timestamp: float
    schema_version: str
    result: str
    rotation_direction: str
    grasp_position: str
    horizontal_direction: str
    horizontal_magnitude: str
    valid: bool
    stage_a_eligible: bool
    execution_eligible: bool
    action_chunk_valid: bool
    tactile_caption: str
    instruction: str
    input_recovery_plan: str

    @property
    def key(self) -> tuple[int, int, int]:
        return self.episode_id, self.attempt_id, self.frame_index

    @property
    def attempt_key(self) -> tuple[int, int]:
        return self.episode_id, self.attempt_id


_LEROBOT_COLUMNS = (
    "index",
    "episode_index",
    "episode_id",
    "attempt_id",
    "frame_index",
    "ros_timestamp",
    "schema_version",
    "result",
    "rotation_direction",
    "grasp_position",
    "horizontal_direction",
    "horizontal_magnitude",
    "valid",
    "stage_a_eligible",
    "execution_eligible",
    "action_chunk_valid",
    "tactile_caption",
    "instruction",
    "input_recovery_plan",
)


def scan_v4_lerobot_frames(dataset_dir: Path) -> list[V4Frame]:
    # Serving imports V4 schema constants through ``openpi_bridge`` but never
    # reads Parquet.  Importing PyArrow after the OpenPI/JAX native stack can
    # crash some CUDA environments, so keep this data-only dependency lazy.
    import pyarrow.parquet as pq

    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files under {dataset_dir / 'data'}")
    frames: list[V4Frame] = []
    for path in parquet_files:
        schema = set(pq.read_schema(path).names)
        missing = sorted(set(_LEROBOT_COLUMNS) - schema)
        if missing:
            raise ValueError(f"{path}: missing V4 identity columns {missing}")
        data = pq.read_table(path, columns=list(_LEROBOT_COLUMNS)).to_pydict()
        for offset in range(len(data["index"])):
            value = {name: _scalar(data[name][offset]) for name in _LEROBOT_COLUMNS}
            frames.append(
                V4Frame(
                    global_index=int(value["index"]),
                    lerobot_episode_index=int(value["episode_index"]),
                    episode_id=int(value["episode_id"]),
                    attempt_id=int(value["attempt_id"]),
                    frame_index=int(value["frame_index"]),
                    ros_timestamp=float(value["ros_timestamp"]),
                    schema_version=str(value["schema_version"]),
                    result=str(value["result"]),
                    rotation_direction=str(value["rotation_direction"]),
                    grasp_position=str(value["grasp_position"]),
                    horizontal_direction=str(value["horizontal_direction"]),
                    horizontal_magnitude=str(value["horizontal_magnitude"]),
                    valid=bool(value["valid"]),
                    stage_a_eligible=bool(value["stage_a_eligible"]),
                    execution_eligible=bool(value["execution_eligible"]),
                    action_chunk_valid=bool(value["action_chunk_valid"]),
                    tactile_caption=str(value["tactile_caption"]),
                    instruction=str(value["instruction"]),
                    input_recovery_plan=str(value["input_recovery_plan"]),
                )
            )
    frames.sort(key=lambda row: row.global_index)
    return frames


def load_v4_sources(
    *,
    selection_file: Path,
    profile_file: Path,
    split_file: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = _load_json(selection_file)
    profile = _load_json(profile_file)
    splits = _load_json(split_file)
    if selection.get("schema_version") != V4_SELECTION_SCHEMA:
        raise ValueError("Unsupported V4 selection schema")
    stored_selection_hash = str(selection.get("selection_hash", ""))
    calculated_selection_hash = sha256_json(
        {key: value for key, value in selection.items() if key != "selection_hash"}
    )
    if not stored_selection_hash or stored_selection_hash != calculated_selection_hash:
        raise ValueError("V4 selection hash is invalid")
    if profile.get("schema_version") != V4_PROFILE_SCHEMA or profile.get("data_profile") != ROTATION_V4:
        raise ValueError("Unsupported V4 profile schema/data_profile")
    if splits.get("schema_version") != V4_SPLIT_SCHEMA:
        raise ValueError("Unsupported V4 split schema")
    selection_hash = stored_selection_hash
    if not selection_hash or profile.get("selection_hash") != selection_hash:
        raise ValueError("Selection/profile hash mismatch")
    if splits.get("selection_hash") != selection_hash:
        raise ValueError("Selection/split hash mismatch")
    profile_hash = str(profile.get("profile_config_hash", ""))
    if not profile_hash or splits.get("profile_config_hash") != profile_hash:
        raise ValueError("Profile/split config hash mismatch")

    selected_pairs = {
        (int(row["episode_id"]), int(row["attempt_id"])) for row in selection.get("attempts", [])
    }
    profile_pairs = {
        (int(row["episode_id"]), int(row["attempt_id"])) for row in profile.get("attempts", [])
    }
    if not selected_pairs or selected_pairs != profile_pairs:
        raise ValueError("Selection/profile attempt sets differ")
    split_ids = {
        split: {int(value) for value in splits["original_episode_ids"][split]}
        for split in SPLITS
    }
    all_ids = set().union(*split_ids.values())
    if sum(map(len, split_ids.values())) != len(all_ids):
        raise ValueError("An episode occurs in multiple V4 splits")
    if all_ids != {pair[0] for pair in profile_pairs}:
        raise ValueError("V4 split does not exactly cover profile episodes")
    return selection, profile, splits


def validate_v4_lerobot_frames(
    frames: Sequence[V4Frame],
    profile: Mapping[str, Any],
) -> tuple[dict[tuple[int, int, int], V4Frame], dict[tuple[int, int], list[V4Frame]]]:
    if not frames:
        raise ValueError("V4 LeRobot dataset is empty")
    by_key: dict[tuple[int, int, int], V4Frame] = {}
    by_attempt: dict[tuple[int, int], list[V4Frame]] = defaultdict(list)
    global_indices: set[int] = set()
    for frame in frames:
        if frame.global_index in global_indices:
            raise ValueError(f"Duplicate LeRobot global index: {frame.global_index}")
        if frame.key in by_key:
            raise ValueError(f"Duplicate V4 frame identity: {frame.key}")
        if frame.schema_version != V4_ATTEMPT_SCHEMA:
            raise ValueError(f"Frame {frame.key} has non-V4 schema {frame.schema_version!r}")
        global_indices.add(frame.global_index)
        by_key[frame.key] = frame
        by_attempt[frame.attempt_key].append(frame)

    if global_indices != set(range(len(frames))):
        raise ValueError("V4 LeRobot global indices must be the exact contiguous range 0..N-1")

    profile_attempts = {
        (int(row["episode_id"]), int(row["attempt_id"])): row
        for row in profile["attempts"]
    }
    if set(by_attempt) != set(profile_attempts):
        raise ValueError(
            "LeRobot/profile attempt set mismatch: "
            f"missing={sorted(set(profile_attempts) - set(by_attempt))[:20]}, "
            f"extra={sorted(set(by_attempt) - set(profile_attempts))[:20]}"
        )
    episode_indices: dict[int, tuple[int, int]] = {}
    action_horizon = int(profile["config"]["action_horizon"])
    for pair, attempt_frames in sorted(by_attempt.items()):
        attempt_frames.sort(key=lambda row: row.frame_index)
        profile_row = profile_attempts[pair]
        frame_count = int(profile_row["frame_count"])
        if [row.frame_index for row in attempt_frames] != list(range(frame_count)):
            raise ValueError(f"Attempt {pair} does not contain exact local frames 0..{frame_count - 1}")
        lerobot_ids = {row.lerobot_episode_index for row in attempt_frames}
        if len(lerobot_ids) != 1:
            raise ValueError(f"Attempt {pair} spans multiple LeRobot episode_index values")
        lerobot_id = next(iter(lerobot_ids))
        previous = episode_indices.setdefault(lerobot_id, pair)
        if previous != pair:
            raise ValueError(f"LeRobot episode_index {lerobot_id} aliases {previous} and {pair}")
        expected_stage = bool(profile_row["stage_a_eligible"])
        expected_valid = bool(profile_row["valid"])
        constant_fields = {
            "result": str(profile_row["result"]),
            "rotation_direction": str(profile_row["rotation_direction"]),
            "grasp_position": str(profile_row["grasp_position"]),
            "horizontal_direction": str(profile_row["horizontal_direction"]),
            "horizontal_magnitude": str(profile_row["horizontal_magnitude"]),
        }
        for row in attempt_frames:
            for field, expected in constant_fields.items():
                if getattr(row, field) != expected:
                    raise ValueError(f"Frame {row.key} {field} does not match profile")
            expected_chunk = row.frame_index + action_horizon <= frame_count
            if row.stage_a_eligible != expected_stage:
                raise ValueError(f"Frame {row.key} stage_a_eligible does not match profile")
            if row.valid != expected_valid:
                raise ValueError(f"Frame {row.key} valid does not match profile")
            if row.action_chunk_valid != expected_chunk:
                raise ValueError(f"Frame {row.key} action_chunk_valid does not match H{action_horizon}")
            if row.execution_eligible != (expected_valid and expected_chunk):
                raise ValueError(f"Frame {row.key} execution_eligible is not valid&&full-H{action_horizon}")
    return by_key, dict(by_attempt)


def map_action_manifest(
    action_rows: Sequence[Mapping[str, Any]],
    *,
    frame_lookup: Mapping[tuple[int, int, int], V4Frame],
    profile: Mapping[str, Any],
    splits: Mapping[str, Any],
) -> dict[str, list[int]]:
    result = {split: [] for split in SPLITS}
    seen: set[tuple[int, int, int]] = set()
    split_ids = {
        split: {int(value) for value in splits["original_episode_ids"][split]} for split in SPLITS
    }
    horizon = int(profile["config"]["action_horizon"])
    profile_attempts = {
        (int(row["episode_id"]), int(row["attempt_id"])): row
        for row in profile["attempts"]
    }
    for row_number, row in enumerate(action_rows):
        key = (int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"]))
        if key in seen:
            raise ValueError(f"Duplicate action local frame in manifest: {key}")
        seen.add(key)
        frame = frame_lookup.get(key)
        if frame is None:
            raise ValueError(f"Action manifest row {row_number} has no LeRobot frame: {key}")
        split = str(row["split"])
        if split not in SPLITS or key[0] not in split_ids[split]:
            raise ValueError(f"Action manifest row {row_number} has inconsistent split")
        if int(row.get("action_horizon", -1)) != horizon:
            raise ValueError(f"Action manifest row {row_number} has wrong horizon")
        expected_hdf5 = str(profile_attempts[key[:2]]["hdf5_path"])
        if str(row.get("hdf5_path", "")) != expected_hdf5:
            raise ValueError(f"Action manifest row {row_number} has wrong HDF5 attempt identity")
        if not bool(row.get("stage_a_eligible")) or not bool(row.get("execution_eligible")):
            raise ValueError(f"Action manifest row {row_number} does not carry both gates")
        if not frame.stage_a_eligible or not frame.execution_eligible or not frame.action_chunk_valid:
            raise ValueError(f"Action manifest row {row_number} maps to an ineligible LeRobot frame")
        result[split].append(frame.global_index)
    expected = {
        key
        for key, frame in frame_lookup.items()
        if frame.stage_a_eligible and frame.execution_eligible and frame.action_chunk_valid
    }
    if seen != expected:
        raise ValueError(
            "Action manifest is not an exact gate-derived frame set: "
            f"missing={len(expected - seen)}, extra={len(seen - expected)}"
        )
    return result


def _interleave_classes(positive: list[dict[str, Any]], negative: list[dict[str, Any]], ratio: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cursor = 0
    per_positive = max(1, int(round(ratio)))
    for row in positive:
        result.append(row)
        result.extend(negative[cursor : cursor + per_positive])
        cursor += per_positive
    result.extend(negative[cursor:])
    return result


def build_need_rows(
    *,
    profile: Mapping[str, Any],
    splits: Mapping[str, Any],
    by_attempt: Mapping[tuple[int, int], Sequence[V4Frame]],
    negative_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if negative_ratio <= 0:
        raise ValueError("V4 need negative ratio must be positive")
    profile_attempts = {
        (int(row["episode_id"]), int(row["attempt_id"])): row for row in profile["attempts"]
    }
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {"positive": [], "hard_negative": [], "easy_negative": []} for split in SPLITS
    }
    for pair, frames in sorted(by_attempt.items()):
        meta = profile_attempts[pair]
        split = str(meta["split"])
        task = str(meta["task"])
        failed = str(meta["result"]) == "failure"
        start_value = meta.get("failure_window_start")
        for frame in frames:
            source: str | None = None
            label: bool | None = None
            if failed:
                if start_value is None:
                    raise ValueError(f"Failed attempt {pair} lacks failure_window_start")
                if frame.frame_index >= int(start_value):
                    source, label = "failure_active", True
                else:
                    source, label = "pre_failure_hard_negative", False
            elif task == "one_success":
                source, label = "one_success_easy_negative", False
            elif str(meta["result"]) == "success" and int(meta["attempt_id"]) > 1:
                source, label = "successful_recovery_easy_negative", False
            if source is None or label is None:
                continue
            row = {
                "schema_version": V4_NEED_SCHEMA,
                "split": split,
                "global_index": frame.global_index,
                "episode_id": frame.episode_id,
                "attempt_id": frame.attempt_id,
                "frame_index": frame.frame_index,
                "need_recovery": label,
                "source": source,
            }
            category = "positive" if label else ("hard_negative" if "hard_negative" in source else "easy_negative")
            candidates[split][category].append(row)

    outputs: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {"schema_version": V4_NEED_SCHEMA, "splits": {}}
    for split in SPLITS:
        values = candidates[split]
        rng = random.Random(sha256_json([seed, split, "need-v1"]))
        positive = list(values["positive"])
        hard = list(values["hard_negative"])
        easy = list(values["easy_negative"])
        rng.shuffle(positive)
        rng.shuffle(hard)
        rng.shuffle(easy)
        if not positive or not hard + easy:
            raise ValueError(
                f"V4 need split {split} requires both classes: positive={len(positive)}, negative={len(hard)+len(easy)}"
            )
        keep = min(len(hard) + len(easy), int(round(len(positive) * negative_ratio)))
        negative = hard[:keep]
        if len(negative) < keep:
            negative.extend(easy[: keep - len(negative)])
        rows = (
            positive + negative
            if split == "train"
            else _interleave_classes(positive, negative, negative_ratio)
        )
        outputs[split] = rows
        source_counts = Counter(str(row["source"]) for row in rows)
        summary["splits"][split] = {
            "count": len(rows),
            "positive": len(positive),
            "negative": len(negative),
            "source_counts": dict(sorted(source_counts.items())),
            "candidate_counts": {name: len(items) for name, items in values.items()},
        }
    return outputs, summary


_PLAN_RE = re.compile(
    r"^recovery_plan=move horizontally (right|left|front|back) (slightly|moderately), "
    r"move vertically none moderately\.$"
)
_FAILURE_RE = re.compile(r"^failure_reason=rotate (right|left|front|back),grasp appropriate\.$")


def validate_direct_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    task: str,
    frame_lookup: Mapping[tuple[int, int, int], V4Frame],
    profile_attempts: Mapping[tuple[int, int], Mapping[str, Any]],
    failure_window_length: int,
) -> list[dict[str, Any]]:
    if task not in {"failure", "plan"}:
        raise ValueError(f"Unsupported V4 manifest task: {task}")
    output: list[dict[str, Any]] = []
    for row_number, original in enumerate(rows):
        row = dict(original)
        if row.get("schema_version") != V4_REASONING_SCHEMA or row.get("split") != split:
            raise ValueError(f"{task} row {row_number} has wrong schema/split")
        expected_type = "failure_reason" if task == "failure" else "recovery_plan"
        if row.get("sample_type") != expected_type:
            raise ValueError(f"{task} row {row_number} has wrong sample_type")
        observation = row.get("current_observation")
        if not isinstance(observation, Mapping) or observation.get("source_type") != "real":
            raise ValueError(f"{task} row {row_number} lacks real observation provenance")
        key = (
            int(observation["episode_id"]),
            int(observation["attempt_id"]),
            int(row["frame_index"]),
        )
        frame = frame_lookup.get(key)
        if frame is None:
            raise ValueError(f"{task} row {row_number} has no LeRobot frame {key}")
        meta = profile_attempts.get(key[:2])
        if meta is None or meta.get("split") != split or meta.get("result") != "failure":
            raise ValueError(f"{task} row {row_number} does not reference a failed attempt in split")
        offset = int(row["frame_offset"])
        start = int(row["window_start"])
        if not 0 <= offset < failure_window_length or key[2] != start + offset:
            raise ValueError(f"{task} row {row_number} has invalid per-frame window identity")
        if int(meta["failure_window_start"]) != start:
            raise ValueError(f"{task} row {row_number} window start differs from profile")
        if int(observation["frame_index"]) != key[2] or int(observation["frame_offset"]) != offset:
            raise ValueError(f"{task} row {row_number} duplicates inconsistent frame fields")
        if str(observation["tactile_caption"]) != frame.tactile_caption:
            raise ValueError(f"{task} row {row_number} caption differs from LeRobot")
        if abs(float(observation["ros_timestamp"]) - frame.ros_timestamp) > 1e-9:
            raise ValueError(f"{task} row {row_number} timestamp differs from LeRobot")
        if str(observation.get("hdf5_path", "")) != str(meta["hdf5_path"]):
            raise ValueError(f"{task} row {row_number} HDF5 attempt identity differs from profile")
        real_failure = str(observation["failure_reason"])
        if _FAILURE_RE.fullmatch(real_failure) is None or real_failure not in legal_failure_reasons():
            raise ValueError(f"{task} row {row_number} has failure outside full V3 grammar")
        if task == "failure":
            if str(row.get("target_failure_reason", "")) != real_failure:
                raise ValueError(f"failure row {row_number} target differs from real failure")
        else:
            target = str(row.get("target_recovery_plan", ""))
            match = _PLAN_RE.fullmatch(target)
            if match is None or target not in legal_recovery_plans():
                raise ValueError(f"plan row {row_number} target outside V4/full-V3 grammar")
            memory = row.get("failure_recovery_memory")
            length = int(row.get("memory_length", -1))
            if not isinstance(memory, list) or len(memory) != length or not 1 <= length <= 4:
                raise ValueError(f"plan row {row_number} has invalid memory length")
            magnitude = match.group(2)
            if (magnitude == "moderately" and length != 1) or (
                magnitude == "slightly" and length not in {2, 3, 4}
            ):
                raise ValueError(f"plan row {row_number} target/memory length mismatch")
            variant_id = str(row.get("variant_id", ""))
            rule_version = str(row.get("rule_version", ""))
            seed = int(row.get("seed", -1))
            if not variant_id or not rule_version or seed < 0:
                raise ValueError(f"plan row {row_number} lacks synthetic provenance")
            for pair_index, pair in enumerate(memory):
                if not isinstance(pair, Mapping) or pair.get("source_type") != "synthetic":
                    raise ValueError(f"plan row {row_number} contains non-synthetic memory")
                if {"donor_episode_id", "donor_attempt_id"} & set(pair):
                    raise ValueError(f"plan row {row_number} contains fake donor provenance")
                if (
                    pair.get("variant_id") != variant_id
                    or pair.get("rule_version") != rule_version
                    or int(pair.get("seed", -1)) != seed
                    or int(pair.get("pair_index", -1)) != pair_index
                ):
                    raise ValueError(f"plan row {row_number} pair provenance mismatch")
                if str(pair.get("failure_reason", "")) not in legal_failure_reasons():
                    raise ValueError(f"plan row {row_number} pair failure outside full V3 grammar")
                plan_text = str(pair.get("recovery_plan", ""))
                if pair_index == 0:
                    if plan_text != "initial plan":
                        raise ValueError(f"plan row {row_number} first pair is not initial plan")
                elif plan_text not in legal_recovery_plans():
                    raise ValueError(f"plan row {row_number} pair plan outside full V3 grammar")
            failure_directions = []
            for pair in memory:
                failure_match = _FAILURE_RE.fullmatch(str(pair["failure_reason"]))
                assert failure_match is not None
                failure_directions.append(failure_match.group(1))
            for pair_index in range(1, length):
                pair_plan_match = _PLAN_RE.fullmatch(str(memory[pair_index]["recovery_plan"]))
                if pair_plan_match is None:
                    raise ValueError(f"plan row {row_number} pair plan is outside V4 rotation subset")
                expected_magnitude = "moderately" if pair_index == 1 else "slightly"
                if (
                    pair_plan_match.group(1) != failure_directions[pair_index - 1]
                    or pair_plan_match.group(2) != expected_magnitude
                ):
                    raise ValueError(f"plan row {row_number} synthetic memory chain is incompatible")
            if str(memory[-1]["failure_reason"]) != real_failure:
                raise ValueError(f"plan row {row_number} terminal failure differs from observation")
            if match.group(1) != failure_directions[-1]:
                raise ValueError(f"plan row {row_number} target direction differs from terminal failure")
            source = row.get("target_source")
            if not isinstance(source, Mapping) or source.get("source_type") != "real":
                raise ValueError(f"plan row {row_number} target is not real")
            target_pair = (int(source["episode_id"]), int(source["plan_attempt_id"]))
            if target_pair not in profile_attempts or target_pair[0] != key[0] or target_pair[1] != key[1] + 1:
                raise ValueError(f"plan row {row_number} target is not the adjacent real attempt")
            target_meta = profile_attempts[target_pair]
            expected_target = (
                f"recovery_plan=move horizontally {target_meta['horizontal_direction']} "
                f"{target_meta['horizontal_magnitude']}, move vertically "
                f"{target_meta['vertical_direction']} {target_meta['vertical_magnitude']}."
            )
            if target != expected_target:
                raise ValueError(f"plan row {row_number} target differs from adjacent real attempt")
        row["global_index"] = frame.global_index
        output.append(row)
    return output


def file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.expanduser().resolve()), "sha256": sha256_file(path)}


def v4_action_identity(splits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    all_indices: list[int] = []
    for split in SPLITS:
        indices = [int(value) for value in splits[split]["execution_indices"]]
        identity[split] = {"count": len(indices), "sha256": sha256_json(indices)}
        all_indices.extend(indices)
    identity["all"] = {"count": len(all_indices), "sha256": sha256_json(all_indices)}
    return identity


def validate_v4_index_dataset(
    payload: Mapping[str, Any],
    dataset_dir: Path,
) -> tuple[list[V4Frame], dict[int, V4Frame]]:
    """Revalidate a persisted V4 training index against its LeRobot rows."""

    if payload.get("schema_version") != V4_TRAINING_INDEX_SCHEMA or payload.get("data_profile") != ROTATION_V4:
        raise ValueError("rotation_v4 training requires the dedicated V4 unified index")
    source_files = payload.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValueError("V4 unified index lacks source_files")
    missing_sources = sorted(V4_REQUIRED_SOURCE_FILES - set(source_files))
    if missing_sources:
        raise ValueError(f"V4 unified index lacks required source files: {missing_sources}")
    for source_name, identity in sorted(source_files.items()):
        if source_name == "lerobot_parquet":
            continue
        if not isinstance(identity, Mapping) or not identity.get("path") or not identity.get("sha256"):
            raise ValueError(f"V4 source {source_name!r} lacks path/sha256 identity")
        source_path = Path(str(identity["path"]))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        actual_source_hash = sha256_file(source_path)
        if actual_source_hash != str(identity["sha256"]):
            raise ValueError(
                f"V4 source file hash mismatch for {source_name}: "
                f"stored={identity['sha256']}, actual={actual_source_hash}, path={source_path}"
            )

    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    stored_parquet_hashes = source_files.get("lerobot_parquet")
    actual_parquet_hashes = {
        path.relative_to(dataset_dir).as_posix(): sha256_file(path) for path in parquet_files
    }
    if stored_parquet_hashes != actual_parquet_hashes:
        raise ValueError("V4 LeRobot parquet file hashes differ from the unified index")
    frames = scan_v4_lerobot_frames(dataset_dir)
    global_lookup = {frame.global_index: frame for frame in frames}
    if set(global_lookup) != set(range(len(frames))):
        raise ValueError("V4 LeRobot global indices must be the exact contiguous range 0..N-1")
    attempts = {frame.attempt_key for frame in frames}
    actual_lerobot_identity = {
        "frame_count": len(frames),
        "attempt_count": len(attempts),
        "frame_key_sha256": sha256_json(
            [
                [frame.episode_id, frame.attempt_id, frame.frame_index, frame.global_index]
                for frame in frames
            ]
        ),
    }
    if payload.get("lerobot_identity") != actual_lerobot_identity:
        raise ValueError(
            "V4 LeRobot dataset identity differs from unified index: "
            f"stored={payload.get('lerobot_identity')}, actual={actual_lerobot_identity}"
        )
    horizon = int(payload.get("action_horizon", -1))
    if horizon <= 0:
        raise ValueError("V4 unified index has an invalid action horizon")
    attempt_counts = Counter(frame.attempt_key for frame in frames)
    for frame in frames:
        expected_chunk = frame.frame_index + horizon <= attempt_counts[frame.attempt_key]
        if frame.action_chunk_valid != expected_chunk:
            raise ValueError(f"V4 frame {frame.key} action_chunk_valid does not match H{horizon}")
        if frame.execution_eligible != (frame.valid and expected_chunk):
            raise ValueError(f"V4 frame {frame.key} execution_eligible is not valid&&full-H{horizon}")
    indexed: list[int] = []
    for split in SPLITS:
        split_indices = [int(value) for value in payload["splits"][split]["execution_indices"]]
        if len(split_indices) != len(set(split_indices)):
            raise ValueError(f"V4 {split} action index contains duplicate global indices")
        for global_index in split_indices:
            frame = global_lookup.get(global_index)
            if frame is None:
                raise ValueError(f"V4 {split} action index references missing global index {global_index}")
            if not frame.stage_a_eligible or not frame.execution_eligible or not frame.action_chunk_valid:
                raise ValueError(f"V4 action global index {global_index} does not satisfy both gates and H{horizon}")
        indexed.extend(split_indices)
    if len(indexed) != len(set(indexed)):
        raise ValueError("V4 action global index occurs in more than one split")
    eligible = {
        frame.global_index
        for frame in frames
        if frame.stage_a_eligible and frame.execution_eligible and frame.action_chunk_valid
    }
    if set(indexed) != eligible:
        raise ValueError(
            "V4 action index is not the exact eligible LeRobot frame set: "
            f"missing={len(eligible - set(indexed))}, extra={len(set(indexed) - eligible)}"
        )
    stored_action_identity = payload.get("action_indices_identity")
    actual_action_identity = v4_action_identity(payload["splits"])
    if stored_action_identity != actual_action_identity:
        raise ValueError("V4 action_indices_identity does not match execution_indices")
    return frames, global_lookup
