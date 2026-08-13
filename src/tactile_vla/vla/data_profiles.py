"""Versioned, sidecar-only data selection for rotation recovery training."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Any

from tactile_vla.vla.artifacts import sha256_json
from tactile_vla.vla.index import FrameRecord
from tactile_vla.vla.index import execution_indices


ROTATION_MODERATELY_SUCCESS_V1 = "rotation_moderately_success_v1"
PROFILE_SCHEMA_VERSION = "tactile_vla_data_profile_v1"
EXPECTED_ACTION_COUNTS = {
    "once_success_attempt1": 35_295,
    "rotation_failed_attempt1": 23_105,
    "recovery_success_attempt2": 39_833,
    "all": 98_233,
}


@dataclass(frozen=True)
class EpisodeGroup:
    name: str
    episode_ids: tuple[int, ...]
    split_counts: tuple[int, int, int]
    attempts: tuple[int, ...]
    rotation_direction: str | None = None


ROTATION_MODERATELY_GROUPS = (
    EpisodeGroup("once_success_horizontal", tuple(range(1, 21)), (16, 2, 2), (1,)),
    EpisodeGroup("once_success_vertical", tuple(range(21, 41)), (16, 2, 2), (1,)),
    EpisodeGroup("recovery_right", tuple(range(41, 53)), (10, 1, 1), (1, 2), "right"),
    EpisodeGroup("recovery_left", tuple(range(61, 73)), (10, 1, 1), (1, 2), "left"),
    EpisodeGroup("recovery_front", tuple(range(81, 93)), (10, 1, 1), (1, 2), "front"),
    EpisodeGroup("recovery_back", tuple(range(101, 113)), (10, 1, 1), (1, 2), "back"),
)


def profile_config(*, seed: int = 42, action_horizon: int = 30) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "data_profile": ROTATION_MODERATELY_SUCCESS_V1,
        "seed": seed,
        "action_horizon": action_horizon,
        "shuffle": "uniform",
        "frame_filter": "execution_eligible",
        "caption_area_filter": False,
        "reasoning": {
            "samples_per_recovery_episode": 1,
            "memory_length": 1,
            "donors": False,
            "window_frames": 15,
        },
        "groups": [asdict(group) for group in ROTATION_MODERATELY_GROUPS],
        "expected_action_counts": EXPECTED_ACTION_COUNTS,
    }


def data_config_hash(*, seed: int = 42, action_horizon: int = 30) -> str:
    return sha256_json(profile_config(seed=seed, action_horizon=action_horizon))


def selected_episode_ids() -> set[int]:
    return {episode_id for group in ROTATION_MODERATELY_GROUPS for episode_id in group.episode_ids}


def _group_seed(seed: int, group_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group_name}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_profile_splits(*, seed: int = 42) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Split each semantic group with an independent stable random stream."""

    splits = {"train": [], "val": [], "test": []}
    episode_groups: dict[str, str] = {}
    for group in ROTATION_MODERATELY_GROUPS:
        episode_ids = list(group.episode_ids)
        random.Random(_group_seed(seed, group.name)).shuffle(episode_ids)
        train_count, val_count, test_count = group.split_counts
        if train_count + val_count + test_count != len(episode_ids):
            raise AssertionError(f"Invalid split counts for group {group.name}")
        boundaries = (train_count, train_count + val_count)
        allocations = {
            "train": episode_ids[: boundaries[0]],
            "val": episode_ids[boundaries[0] : boundaries[1]],
            "test": episode_ids[boundaries[1] :],
        }
        for split, values in allocations.items():
            splits[split].extend(values)
        episode_groups.update({str(episode_id): group.name for episode_id in group.episode_ids})
    return ({name: sorted(values) for name, values in splits.items()}, episode_groups)


def select_profile_records(records: Iterable[FrameRecord]) -> list[FrameRecord]:
    expected_attempts = {
        episode_id: set(group.attempts)
        for group in ROTATION_MODERATELY_GROUPS
        for episode_id in group.episode_ids
    }
    selected = [
        record
        for record in records
        if record.original_episode_id in expected_attempts
        and record.attempt_id in expected_attempts[record.original_episode_id]
    ]
    found_episode_ids = {record.original_episode_id for record in selected}
    missing = selected_episode_ids() - found_episode_ids
    if missing:
        raise ValueError(f"Data profile is missing selected episodes: {sorted(missing)}")
    for episode_id, attempts in expected_attempts.items():
        found = {record.attempt_id for record in selected if record.original_episode_id == episode_id}
        if found != attempts:
            raise ValueError(
                f"Episode {episode_id} attempts mismatch: expected={sorted(attempts)}, found={sorted(found)}"
            )
    return sorted(selected, key=lambda record: record.global_index)


def _expected_failure(direction: str) -> str:
    return f"failure_reason=rotate {direction},grasp appropriate."


def _expected_plan(direction: str) -> str:
    return (
        f"recovery_plan=move horizontally {direction} moderately, "
        "move vertically none moderately."
    )


def validate_profile_metadata(records: Iterable[FrameRecord]) -> None:
    by_episode_attempt: dict[tuple[int, int], list[FrameRecord]] = {}
    for record in records:
        by_episode_attempt.setdefault(
            (record.original_episode_id, record.attempt_id), []
        ).append(record)

    for group in ROTATION_MODERATELY_GROUPS:
        for episode_id in group.episode_ids:
            if group.rotation_direction is None:
                attempt = by_episode_attempt[(episode_id, 1)]
                if any(record.need_recovery or record.failure_reason.strip() for record in attempt):
                    raise ValueError(f"Once-success episode {episode_id} contains a failure label")
                continue

            failed = by_episode_attempt[(episode_id, 1)]
            recovered = by_episode_attempt[(episode_id, 2)]
            reasons = {record.failure_reason for record in failed if record.need_recovery}
            expected_reason = _expected_failure(group.rotation_direction)
            if reasons != {expected_reason}:
                raise ValueError(
                    f"Episode {episode_id} attempt1 failure mismatch: "
                    f"expected={expected_reason!r}, found={sorted(reasons)}"
                )
            expected_plan = _expected_plan(group.rotation_direction)
            plans = {record.input_recovery_plan for record in recovered}
            if plans != {expected_plan}:
                raise ValueError(
                    f"Episode {episode_id} attempt2 plan mismatch: "
                    f"expected={expected_plan!r}, found={sorted(plans)}"
                )
            if any(record.need_recovery or record.failure_reason.strip() for record in recovered):
                raise ValueError(f"Episode {episode_id} attempt2 is not a successful recovery")


def action_count_summary(
    records: Iterable[FrameRecord],
    *,
    action_horizon: int = 30,
) -> dict[str, int]:
    records = list(records)
    categories = {
        "once_success_attempt1": [
            record
            for record in records
            if record.original_episode_id <= 40 and record.attempt_id == 1
        ],
        "rotation_failed_attempt1": [
            record
            for record in records
            if record.original_episode_id > 40 and record.attempt_id == 1
        ],
        "recovery_success_attempt2": [
            record
            for record in records
            if record.original_episode_id > 40 and record.attempt_id == 2
        ],
    }
    result = {
        name: len(execution_indices(values, action_horizon=action_horizon))
        for name, values in categories.items()
    }
    result["all"] = sum(result.values())
    return result


def action_attempt_coverage(records: Iterable[FrameRecord]) -> dict[str, int]:
    records = list(records)
    categories = {
        "once_success_attempt1": {
            record.original_episode_id
            for record in records
            if record.original_episode_id <= 40 and record.attempt_id == 1
        },
        "rotation_failed_attempt1": {
            record.original_episode_id
            for record in records
            if record.original_episode_id > 40 and record.attempt_id == 1
        },
        "recovery_success_attempt2": {
            record.original_episode_id
            for record in records
            if record.original_episode_id > 40 and record.attempt_id == 2
        },
    }
    return {name: len(episode_ids) for name, episode_ids in categories.items()}


def validate_expected_action_counts(
    records: Iterable[FrameRecord],
    *,
    action_horizon: int = 30,
) -> dict[str, int]:
    counts = action_count_summary(records, action_horizon=action_horizon)
    if counts != EXPECTED_ACTION_COUNTS:
        raise ValueError(
            f"Data profile action counts mismatch: expected={EXPECTED_ACTION_COUNTS}, found={counts}"
        )
    return counts


def direction_by_episode() -> dict[int, str]:
    return {
        episode_id: str(group.rotation_direction)
        for group in ROTATION_MODERATELY_GROUPS
        if group.rotation_direction is not None
        for episode_id in group.episode_ids
    }


def build_single_round_reasoning_samples(
    records: Iterable[FrameRecord],
    splits: Mapping[str, Iterable[int]],
) -> dict[str, list[dict[str, Any]]]:
    records = list(records)
    directions = direction_by_episode()
    samples = {"train": [], "val": [], "test": []}
    split_by_episode = {
        int(episode_id): split
        for split, episode_ids in splits.items()
        for episode_id in episode_ids
    }
    for episode_id, direction in sorted(directions.items()):
        failed = sorted(
            (
                record
                for record in records
                if record.original_episode_id == episode_id
                and record.attempt_id == 1
                and record.need_recovery
            ),
            key=lambda record: record.frame_index,
        )
        recovered = [
            record
            for record in records
            if record.original_episode_id == episode_id and record.attempt_id == 2
        ]
        if not failed or not recovered:
            raise ValueError(f"Episode {episode_id} has no complete attempt1->attempt2 pair")
        trigger = failed[0]
        target_plan = _expected_plan(direction)
        memory = [
            {
                "recovery_plan": "initial plan",
                "failure_reason": _expected_failure(direction),
            }
        ]
        split = split_by_episode[episode_id]
        samples[split].append(
            {
                "split": split,
                "memory_length": 1,
                "failure_recovery_memory": memory,
                "current_observation": {
                    "episode_id": episode_id,
                    "attempt_id": 1,
                    "frame_index": trigger.frame_index,
                    "tactile_caption": trigger.tactile_caption,
                    "failure_reason": trigger.failure_reason,
                },
                "target_recovery_plan": target_plan,
                "target_recovery_plan_mask": True,
                "target_source": {
                    "episode_id": episode_id,
                    "failed_attempt_id": 1,
                    "plan_attempt_id": 2,
                    "transition_id": f"episode{episode_id}:attempt1->attempt2",
                },
                "memory_provenance": [
                    {
                        "pair_index": 0,
                        "episode_id": episode_id,
                        "attempt_id": 1,
                        "transition_id": "initial",
                    }
                ],
                "donor_episode_ids": [],
            }
        )
    return samples


def selection_summary(records: Iterable[FrameRecord]) -> dict[str, Any]:
    records = list(records)
    return {
        "episodes": len({record.original_episode_id for record in records}),
        "lerobot_episodes": len({record.lerobot_episode_index for record in records}),
        "frames": len(records),
        "attempt_frames": dict(Counter(str(record.attempt_id) for record in records)),
        "selected_original_episode_ids": sorted(
            {record.original_episode_id for record in records}
        ),
    }
