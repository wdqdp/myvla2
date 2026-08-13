"""Dataset selection and balanced sampling for recovery-action fine-tuning.

The selector is intentionally driven by a JSON data-group configuration.  The
current experiment enables only the moderately recovery group, while a later
slightly group can be added without changing selection or sampling code.
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Sampler


CONFIG_SCHEMA_VERSION = "tactile_vla_recovery_action_finetune_v1"
MANIFEST_SCHEMA_VERSION = "tactile_vla_recovery_action_selection_v1"
RECOVERY_MAGNITUDES = frozenset({"slightly", "moderately", "significantly"})
HORIZONTAL_DIRECTIONS = frozenset({"left", "right", "front", "back"})


@dataclass(frozen=True)
class FineTuneFrame:
    global_index: int
    lerobot_episode_index: int
    original_episode_id: int
    attempt_id: int
    frame_index: int
    ros_timestamp: float
    result: str
    horizontal_direction: str
    horizontal_magnitude: str
    input_recovery_plan: str
    execution_eligible: bool


@dataclass(frozen=True)
class SelectedFrame:
    global_index: int
    source_group: str
    source_kind: str
    balance_key: str
    original_episode_id: int
    attempt_id: int
    frame_index: int
    relative_timestamp: float
    horizontal_direction: str
    horizontal_magnitude: str


@dataclass(frozen=True)
class RecoveryGroupConfig:
    name: str
    enabled: bool
    attempt_id: int
    result: str
    horizontal_magnitude: str
    clip_seconds: float
    sampling_weight: float
    episode_ids_by_direction: Mapping[str, tuple[int, ...]]

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return tuple(
            episode_id
            for direction in self.episode_ids_by_direction
            for episode_id in self.episode_ids_by_direction[direction]
        )


@dataclass(frozen=True)
class NormalStratumConfig:
    name: str
    episode_ids: tuple[int, ...]
    select_count: int


@dataclass(frozen=True)
class NormalSuccessGroupConfig:
    name: str
    attempt_id: int
    result: str
    sampling_weight: float
    strata: tuple[NormalStratumConfig, ...]


@dataclass(frozen=True)
class FineTuneSelectionConfig:
    selection_seed: int
    forbidden_recovery_magnitudes: frozenset[str]
    normal_success_group: NormalSuccessGroupConfig
    recovery_groups: tuple[RecoveryGroupConfig, ...]


@dataclass(frozen=True)
class FineTuneSelection:
    frames: tuple[SelectedFrame, ...]
    group_weights: Mapping[str, float]
    selected_episode_ids: Mapping[str, tuple[int, ...]]
    excluded_by_split: Mapping[str, tuple[int, ...]]
    action_horizon: int
    split: str
    selection_seed: int

    @property
    def indices(self) -> list[int]:
        return [frame.global_index for frame in self.frames]

    def summary(self) -> dict[str, Any]:
        group_counts = Counter(frame.source_group for frame in self.frames)
        magnitude_counts = Counter(
            frame.horizontal_magnitude for frame in self.frames if frame.source_kind == "recovery"
        )
        direction_counts: dict[str, dict[str, int]] = {}
        for group in self.group_weights:
            direction_counts[group] = dict(
                sorted(
                    Counter(
                        frame.horizontal_direction
                        for frame in self.frames
                        if frame.source_group == group and frame.source_kind == "recovery"
                    ).items()
                )
            )
        return {
            "split": self.split,
            "selection_seed": self.selection_seed,
            "action_horizon": self.action_horizon,
            "total_frames": len(self.frames),
            "group_frame_counts": dict(sorted(group_counts.items())),
            "recovery_magnitude_frame_counts": dict(sorted(magnitude_counts.items())),
            "recovery_direction_frame_counts": direction_counts,
            "selected_episode_ids": {
                group: list(values) for group, values in self.selected_episode_ids.items()
            },
            "excluded_by_split": {
                group: list(values) for group, values in self.excluded_by_split.items()
            },
            "group_sampling_weights": dict(self.group_weights),
        }

    def manifest(self, *, config_path: str | Path, dataset_dir: str | Path, split_file: str | Path) -> dict:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "config_path": str(Path(config_path).resolve()),
            "dataset_dir": str(Path(dataset_dir).resolve()),
            "split_file": str(Path(split_file).resolve()),
            "summary": self.summary(),
            "frames": [asdict(frame) for frame in self.frames],
        }


def _parse_ranges(value: Any, *, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list of inclusive [start, end] ranges")
    result: list[int] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{location}[{index}] must be an inclusive [start, end] pair")
        start, end = (int(pair[0]), int(pair[1]))
        if start < 0 or end < start:
            raise ValueError(f"Invalid episode range {pair!r} at {location}[{index}]")
        result.extend(range(start, end + 1))
    if len(result) != len(set(result)):
        raise ValueError(f"{location} contains duplicate episode IDs")
    return tuple(result)


def load_selection_config(path: str | Path) -> FineTuneSelectionConfig:
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported selection config schema {payload.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )

    forbidden = frozenset(str(value).strip().lower() for value in payload.get("forbidden_recovery_magnitudes", []))
    unknown_forbidden = forbidden - RECOVERY_MAGNITUDES
    if unknown_forbidden:
        raise ValueError(f"Unknown forbidden recovery magnitudes: {sorted(unknown_forbidden)}")

    normal_raw = payload.get("normal_success_group")
    if not isinstance(normal_raw, dict):
        raise ValueError("normal_success_group must be an object")
    strata: list[NormalStratumConfig] = []
    normal_episode_ids: set[int] = set()
    for index, raw in enumerate(normal_raw.get("strata", [])):
        location = f"normal_success_group.strata[{index}]"
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError(f"{location}.name must be non-empty")
        episode_ids = _parse_ranges(raw.get("episode_ranges"), location=f"{location}.episode_ranges")
        duplicate_ids = normal_episode_ids.intersection(episode_ids)
        if duplicate_ids:
            raise ValueError(f"Normal-success strata overlap on episode IDs {sorted(duplicate_ids)}")
        normal_episode_ids.update(episode_ids)
        select_count = int(raw.get("select_count", 0))
        if select_count <= 0:
            raise ValueError(f"{location}.select_count must be positive")
        strata.append(NormalStratumConfig(name=name, episode_ids=episode_ids, select_count=select_count))
    if not strata:
        raise ValueError("normal_success_group.strata must be non-empty")
    normal_group = NormalSuccessGroupConfig(
        name=str(normal_raw.get("name", "")).strip(),
        attempt_id=int(normal_raw.get("attempt_id", 1)),
        result=str(normal_raw.get("result", "")).strip(),
        sampling_weight=float(normal_raw.get("sampling_weight", 0.0)),
        strata=tuple(strata),
    )
    if not normal_group.name or normal_group.sampling_weight <= 0:
        raise ValueError("normal_success_group requires a non-empty name and positive sampling_weight")

    groups: list[RecoveryGroupConfig] = []
    names = {normal_group.name}
    configured_recovery_ids: set[int] = set()
    for index, raw in enumerate(payload.get("recovery_groups", [])):
        location = f"recovery_groups[{index}]"
        name = str(raw.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"{location}.name must be non-empty and unique, got {name!r}")
        names.add(name)
        direction_ranges = raw.get("episode_ranges_by_direction")
        if not isinstance(direction_ranges, dict) or not direction_ranges:
            raise ValueError(f"{location}.episode_ranges_by_direction must be a non-empty object")
        ids_by_direction: dict[str, tuple[int, ...]] = {}
        for raw_direction, ranges in direction_ranges.items():
            direction = str(raw_direction).strip().lower()
            if direction not in HORIZONTAL_DIRECTIONS:
                raise ValueError(f"Unknown horizontal direction {direction!r} at {location}")
            episode_ids = _parse_ranges(
                ranges,
                location=f"{location}.episode_ranges_by_direction.{direction}",
            )
            duplicate_ids = configured_recovery_ids.intersection(episode_ids)
            if duplicate_ids:
                raise ValueError(f"Recovery groups overlap on episode IDs {sorted(duplicate_ids)}")
            configured_recovery_ids.update(episode_ids)
            ids_by_direction[direction] = episode_ids

        magnitude = str(raw.get("horizontal_magnitude", "")).strip().lower()
        if magnitude not in RECOVERY_MAGNITUDES:
            raise ValueError(f"Unknown horizontal magnitude {magnitude!r} at {location}")
        enabled = bool(raw.get("enabled", True))
        if enabled and magnitude in forbidden:
            raise ValueError(
                f"Enabled recovery group {name!r} uses forbidden magnitude {magnitude!r}"
            )
        group = RecoveryGroupConfig(
            name=name,
            enabled=enabled,
            attempt_id=int(raw.get("attempt_id", 2)),
            result=str(raw.get("result", "")).strip(),
            horizontal_magnitude=magnitude,
            clip_seconds=float(raw.get("clip_seconds", 0.0)),
            sampling_weight=float(raw.get("sampling_weight", 0.0)),
            episode_ids_by_direction=ids_by_direction,
        )
        if group.enabled and (group.clip_seconds <= 0 or group.sampling_weight <= 0):
            raise ValueError(f"Enabled recovery group {name!r} requires positive clip_seconds and sampling_weight")
        groups.append(group)
    if not any(group.enabled for group in groups):
        raise ValueError("At least one recovery group must be enabled")

    return FineTuneSelectionConfig(
        selection_seed=int(payload.get("selection_seed", 42)),
        forbidden_recovery_magnitudes=forbidden,
        normal_success_group=normal_group,
        recovery_groups=tuple(groups),
    )


def load_split_episode_ids(path: str | Path, split: str) -> tuple[int, ...]:
    payload = json.loads(Path(path).read_text())
    try:
        values = payload["original_episode_ids"][split]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Split file does not contain original_episode_ids.{split}") from exc
    return tuple(int(value) for value in values)


def scan_finetune_frames(dataset_dir: str | Path) -> list[FineTuneFrame]:
    dataset_dir = Path(dataset_dir)
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")
    columns = [
        "index",
        "episode_index",
        "episode_id",
        "attempt_id",
        "frame_index",
        "ros_timestamp",
        "result",
        "horizontal_direction",
        "horizontal_magnitude",
        "input_recovery_plan",
        "execution_eligible",
    ]
    frames: list[FineTuneFrame] = []
    for parquet_file in parquet_files:
        data = pq.read_table(parquet_file, columns=columns).to_pydict()
        for row_index in range(len(data["index"])):
            frame = FineTuneFrame(
                global_index=int(data["index"][row_index]),
                lerobot_episode_index=int(data["episode_index"][row_index]),
                original_episode_id=int(data["episode_id"][row_index]),
                attempt_id=int(data["attempt_id"][row_index]),
                frame_index=int(data["frame_index"][row_index]),
                ros_timestamp=float(data["ros_timestamp"][row_index]),
                result=str(data["result"][row_index]),
                horizontal_direction=str(data["horizontal_direction"][row_index]),
                horizontal_magnitude=str(data["horizontal_magnitude"][row_index]),
                input_recovery_plan=str(data["input_recovery_plan"][row_index]),
                execution_eligible=bool(data["execution_eligible"][row_index]),
            )
            if not math.isfinite(frame.ros_timestamp):
                raise ValueError(f"Non-finite ros_timestamp at global_index={frame.global_index}")
            frames.append(frame)
    frames.sort(key=lambda frame: frame.global_index)
    global_indices = [frame.global_index for frame in frames]
    if len(global_indices) != len(set(global_indices)):
        raise ValueError("LeRobot dataset contains duplicate global frame indices")
    return frames


def _attempt_rows(
    frames_by_attempt: Mapping[tuple[int, int], list[FineTuneFrame]],
    *,
    episode_id: int,
    attempt_id: int,
) -> list[FineTuneFrame]:
    rows = frames_by_attempt.get((episode_id, attempt_id))
    if not rows:
        raise ValueError(f"Missing episode{episode_id}/attempt{attempt_id} in LeRobot dataset")
    expected = list(range(len(rows)))
    actual = [row.frame_index for row in rows]
    if actual != expected:
        raise ValueError(
            f"episode{episode_id}/attempt{attempt_id} frame indices are not contiguous from zero"
        )
    timestamps = np.asarray([row.ros_timestamp for row in rows], dtype=np.float64)
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"episode{episode_id}/attempt{attempt_id} timestamps are not strictly increasing")
    return rows


def _eligible_rows(
    rows: Sequence[FineTuneFrame],
    *,
    action_horizon: int,
    segment_end_timestamp: float | None,
) -> list[FineTuneFrame]:
    selected: list[FineTuneFrame] = []
    for row in rows:
        if not row.execution_eligible:
            continue
        endpoint_index = row.frame_index + action_horizon - 1
        if endpoint_index >= len(rows):
            raise ValueError(
                f"execution_eligible is true but H{action_horizon} is incomplete at global_index={row.global_index}"
            )
        if segment_end_timestamp is not None and rows[endpoint_index].ros_timestamp > segment_end_timestamp + 1e-6:
            continue
        selected.append(row)
    return selected


def _validate_constant_metadata(
    rows: Sequence[FineTuneFrame],
    *,
    episode_id: int,
    attempt_id: int,
    result: str,
) -> None:
    bad = [row.global_index for row in rows if row.attempt_id != attempt_id or row.result != result]
    if bad:
        raise ValueError(
            f"episode{episode_id}/attempt{attempt_id} does not consistently have result={result!r}; "
            f"first bad global index={bad[0]}"
        )


def select_finetune_frames(
    frames: Sequence[FineTuneFrame],
    *,
    split_episode_ids: Sequence[int],
    split: str,
    config: FineTuneSelectionConfig,
    action_horizon: int,
) -> FineTuneSelection:
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    split_ids = set(int(value) for value in split_episode_ids)
    frames_by_attempt: dict[tuple[int, int], list[FineTuneFrame]] = defaultdict(list)
    for frame in frames:
        frames_by_attempt[(frame.original_episode_id, frame.attempt_id)].append(frame)
    for rows in frames_by_attempt.values():
        rows.sort(key=lambda frame: frame.frame_index)

    rng = random.Random(config.selection_seed)
    selected: list[SelectedFrame] = []
    selected_episode_ids: dict[str, tuple[int, ...]] = {}
    excluded_by_split: dict[str, tuple[int, ...]] = {}
    group_weights: dict[str, float] = {}

    normal = config.normal_success_group
    selected_normal_ids: list[int] = []
    normal_stratum_by_episode: dict[int, str] = {}
    all_normal_ids = [episode_id for stratum in normal.strata for episode_id in stratum.episode_ids]
    excluded_by_split[normal.name] = tuple(sorted(set(all_normal_ids) - split_ids))
    for stratum in normal.strata:
        candidates = sorted(set(stratum.episode_ids).intersection(split_ids))
        if len(candidates) < stratum.select_count:
            raise ValueError(
                f"Normal-success stratum {stratum.name!r} has {len(candidates)} episodes in split {split!r}, "
                f"but select_count={stratum.select_count}"
            )
        chosen = sorted(rng.sample(candidates, stratum.select_count))
        selected_normal_ids.extend(chosen)
        normal_stratum_by_episode.update({episode_id: stratum.name for episode_id in chosen})
    selected_normal_ids.sort()
    selected_episode_ids[normal.name] = tuple(selected_normal_ids)
    group_weights[normal.name] = normal.sampling_weight
    for episode_id in selected_normal_ids:
        rows = _attempt_rows(
            frames_by_attempt,
            episode_id=episode_id,
            attempt_id=normal.attempt_id,
        )
        _validate_constant_metadata(
            rows,
            episode_id=episode_id,
            attempt_id=normal.attempt_id,
            result=normal.result,
        )
        eligible = _eligible_rows(rows, action_horizon=action_horizon, segment_end_timestamp=None)
        if not eligible:
            raise ValueError(f"Normal-success episode{episode_id} has no eligible H{action_horizon} frames")
        start_timestamp = rows[0].ros_timestamp
        for row in eligible:
            selected.append(
                SelectedFrame(
                    global_index=row.global_index,
                    source_group=normal.name,
                    source_kind="normal_success",
                    balance_key=normal_stratum_by_episode[episode_id],
                    original_episode_id=episode_id,
                    attempt_id=row.attempt_id,
                    frame_index=row.frame_index,
                    relative_timestamp=row.ros_timestamp - start_timestamp,
                    horizontal_direction="none",
                    horizontal_magnitude="none",
                )
            )

    for group in config.recovery_groups:
        if not group.enabled:
            continue
        requested_ids = set(group.episode_ids)
        group_ids = sorted(requested_ids.intersection(split_ids))
        excluded_by_split[group.name] = tuple(sorted(requested_ids - split_ids))
        selected_episode_ids[group.name] = tuple(group_ids)
        group_weights[group.name] = group.sampling_weight
        expected_direction = {
            episode_id: direction
            for direction, episode_ids in group.episode_ids_by_direction.items()
            for episode_id in episode_ids
        }
        for episode_id in group_ids:
            rows = _attempt_rows(
                frames_by_attempt,
                episode_id=episode_id,
                attempt_id=group.attempt_id,
            )
            _validate_constant_metadata(
                rows,
                episode_id=episode_id,
                attempt_id=group.attempt_id,
                result=group.result,
            )
            direction = expected_direction[episode_id]
            bad = [
                row
                for row in rows
                if row.horizontal_direction != direction
                or row.horizontal_magnitude != group.horizontal_magnitude
            ]
            if bad:
                row = bad[0]
                raise ValueError(
                    f"Configured recovery metadata mismatch at global_index={row.global_index}: "
                    f"expected direction={direction!r}, magnitude={group.horizontal_magnitude!r}; "
                    f"got direction={row.horizontal_direction!r}, magnitude={row.horizontal_magnitude!r}"
                )
            expected_prompt_fragment = f"move horizontally {direction} {group.horizontal_magnitude}"
            if any(expected_prompt_fragment not in row.input_recovery_plan for row in rows):
                raise ValueError(
                    f"episode{episode_id}/attempt{group.attempt_id} input_recovery_plan does not contain "
                    f"{expected_prompt_fragment!r}"
                )
            start_timestamp = rows[0].ros_timestamp
            segment_end = start_timestamp + group.clip_seconds
            eligible = _eligible_rows(
                rows,
                action_horizon=action_horizon,
                segment_end_timestamp=segment_end,
            )
            if not eligible:
                raise ValueError(
                    f"Recovery episode{episode_id}/attempt{group.attempt_id} has no eligible "
                    f"H{action_horizon} frames inside {group.clip_seconds:g}s"
                )
            for row in eligible:
                selected.append(
                    SelectedFrame(
                        global_index=row.global_index,
                        source_group=group.name,
                        source_kind="recovery",
                        balance_key=direction,
                        original_episode_id=episode_id,
                        attempt_id=row.attempt_id,
                        frame_index=row.frame_index,
                        relative_timestamp=row.ros_timestamp - start_timestamp,
                        horizontal_direction=direction,
                        horizontal_magnitude=group.horizontal_magnitude,
                    )
                )

    forbidden_selected = sorted(
        {
            frame.horizontal_magnitude
            for frame in selected
            if frame.source_kind == "recovery"
            and frame.horizontal_magnitude in config.forbidden_recovery_magnitudes
        }
    )
    if forbidden_selected:
        raise ValueError(f"Selected recovery data contains forbidden magnitudes: {forbidden_selected}")
    selected.sort(key=lambda frame: (frame.source_group, frame.original_episode_id, frame.frame_index))
    global_indices = [frame.global_index for frame in selected]
    if len(global_indices) != len(set(global_indices)):
        raise ValueError("Fine-tuning groups selected duplicate global frame indices")
    return FineTuneSelection(
        frames=tuple(selected),
        group_weights=group_weights,
        selected_episode_ids=selected_episode_ids,
        excluded_by_split=excluded_by_split,
        action_horizon=action_horizon,
        split=split,
        selection_seed=config.selection_seed,
    )


def _proportional_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if not weights or any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("Sampling weights must be non-empty, finite, and positive")
    weight_sum = float(sum(weights.values()))
    raw = {name: total * float(weight) / weight_sum for name, weight in weights.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _equal_counts(total: int, names: Sequence[str]) -> dict[str, int]:
    return _proportional_counts(total, {name: 1.0 for name in names})


class HierarchicalGroupSampler(Sampler[int]):
    """Sample exact group ratios, then balance keys and episodes hierarchically."""

    def __init__(
        self,
        frames: Sequence[SelectedFrame],
        *,
        group_weights: Mapping[str, float],
        num_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.frames = tuple(frames)
        self.group_weights = dict(group_weights)
        self.num_samples = len(self.frames) if num_samples is None else int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")

        buckets: dict[str, dict[str, dict[int, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for position, frame in enumerate(self.frames):
            buckets[frame.source_group][frame.balance_key][frame.original_episode_id].append(position)
        missing = set(self.group_weights) - set(buckets)
        unknown = set(buckets) - set(self.group_weights)
        if missing or unknown:
            raise ValueError(
                f"Sampler group mismatch: groups without frames={sorted(missing)}, "
                f"frames without weights={sorted(unknown)}"
            )
        self.buckets = {
            group: {
                key: {episode_id: tuple(values) for episode_id, values in episodes.items()}
                for key, episodes in keys.items()
            }
            for group, keys in buckets.items()
        }

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _episode_schedule(episode_ids: Sequence[int], count: int, rng: random.Random) -> list[int]:
        result: list[int] = []
        while len(result) < count:
            cycle = list(episode_ids)
            rng.shuffle(cycle)
            result.extend(cycle)
        return result[:count]

    def planned_group_counts(self) -> dict[str, int]:
        return _proportional_counts(self.num_samples, self.group_weights)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        sampled: list[int] = []
        for group, group_count in self.planned_group_counts().items():
            balance_buckets = self.buckets[group]
            balance_counts = _equal_counts(group_count, sorted(balance_buckets))
            for balance_key, balance_count in balance_counts.items():
                episode_buckets = balance_buckets[balance_key]
                episode_ids = sorted(episode_buckets)
                for episode_id in self._episode_schedule(episode_ids, balance_count, rng):
                    sampled.append(rng.choice(episode_buckets[episode_id]))
        if len(sampled) != self.num_samples:
            raise RuntimeError(f"Sampler generated {len(sampled)} indices, expected {self.num_samples}")
        rng.shuffle(sampled)
        return iter(sampled)
