"""V5 prompt-only phase labels layered on immutable V4 action data.

V5 deliberately does not rebuild LeRobot or action targets.  This module
derives phase sidecars from the raw gripper/Fz streams, then proves that the
selected episodes, splits, action starts, H30 targets and norm statistics are
still the V4 artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

import h5py
import numpy as np

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.prompts import ACTION_PHASES, PHASE_PROMPT_PROFILE
from tactile_vla.vla.v4_data import SPLITS, V4Frame, load_jsonl, validate_v4_index_dataset


ROTATION_PHASE_V5 = "rotation_phase_v5"
PHASE_EXPERIMENT_KIND = "phase_prompt_only"
V5_PHASE_BOUNDARY_SCHEMA = "tactile_vla_v5_phase_boundaries_v1"
V5_ACTION_PHASE_SCHEMA = "tactile_vla_v5_action_phase_manifest_v1"
V5_OVERRIDE_SCHEMA = "tactile_vla_v5_phase_boundary_overrides_v1"
V5_TRAINING_INDEX_SCHEMA = "tactile_vla_v5_prompt_training_index_v1"
V5_SUMMARY_SCHEMA = "tactile_vla_v5_phase_label_summary_v1"
EXPECTED_EPISODE_COUNT = 448
EXPECTED_ATTEMPT_COUNT = 776
ALLOWED_TASKS = frozenset({"one_success", "moderate_lift", "slightly_lift"})
Phase = Literal["execution", "reposition", "adjustment"]

_AREA_RE = re.compile(r"\barea=(none|small|medium|large)\b")

_RAW_DATASETS = {
    "puppet_gripper": (
        "arm/jointStatePosition/puppetRight",
        "puppetRight",
    ),
    "master_gripper": (
        "arm/jointStatePosition/masterRight",
        "masterRight",
    ),
    "fz_left": (
        "tactile/force_resultant/left",
        "force_resultant/left",
    ),
    "fz_right": (
        "tactile/force_resultant/right",
        "force_resultant/right",
    ),
    "timestamp": ("timestamp",),
}


class PhaseBoundaryError(ValueError):
    """Raised when an attempt must not receive an automatic phase label."""


@dataclass(frozen=True)
class ForceThresholds:
    left_baseline: float
    right_baseline: float
    noise_median: float
    noise_mad: float
    noise_sigma: float
    force_off_threshold: float
    force_on_threshold: float
    contact_reference: float
    separation_score: float
    open_baseline_frame_count: int


@dataclass(frozen=True)
class DetectedBoundary:
    release_frame: int
    regrasp_frame: int
    thresholds: ForceThresholds
    release_opening_progress: float
    regrasp_closing_progress: float
    initial_contact_run: tuple[int, int]
    release_low_force_run: tuple[int, int]
    regrasp_high_force_run: tuple[int, int]
    caption_support: dict[str, float | int | None]


@dataclass(frozen=True)
class RawAttemptStreams:
    puppet_gripper: np.ndarray
    master_gripper: np.ndarray
    fz_left: np.ndarray
    fz_right: np.ndarray
    timestamp: np.ndarray
    dataset_names: dict[str, str]
    source_sha256: str


def _require_dataset(handle: h5py.File, logical_name: str) -> tuple[str, h5py.Dataset]:
    for name in _RAW_DATASETS[logical_name]:
        if name in handle:
            value = handle[name]
            if not isinstance(value, h5py.Dataset):
                raise PhaseBoundaryError(f"Raw HDF5 object {name!r} is not a dataset")
            return name, value
    raise PhaseBoundaryError(
        f"Raw HDF5 lacks required {logical_name} dataset; tried {_RAW_DATASETS[logical_name]}"
    )


def _selected_stream_hash(values: Mapping[str, tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for logical_name in sorted(values):
        dataset_name, array = values[logical_name]
        contiguous = np.ascontiguousarray(array)
        header = json.dumps(
            {
                "logical_name": logical_name,
                "dataset_name": dataset_name,
                "shape": list(contiguous.shape),
                "dtype": contiguous.dtype.str,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def load_raw_attempt_streams(path: Path) -> RawAttemptStreams:
    """Load only the five V5 label sources and hash their exact stored values."""

    if not path.is_file():
        raise FileNotFoundError(path)
    loaded: dict[str, tuple[str, np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        for logical_name in _RAW_DATASETS:
            dataset_name, dataset = _require_dataset(handle, logical_name)
            loaded[logical_name] = (dataset_name, np.asarray(dataset))

    puppet = loaded["puppet_gripper"][1]
    master = loaded["master_gripper"][1]
    left = loaded["fz_left"][1]
    right = loaded["fz_right"][1]
    timestamp = loaded["timestamp"][1]
    if puppet.ndim != 2 or puppet.shape[1] < 7:
        raise PhaseBoundaryError(f"puppetRight must have shape [T,>=7], got {puppet.shape}")
    if master.ndim != 2 or master.shape[1] < 7:
        raise PhaseBoundaryError(f"masterRight must have shape [T,>=7], got {master.shape}")
    if left.ndim != 2 or left.shape[1] < 3:
        raise PhaseBoundaryError(f"left force_resultant must have shape [T,>=3], got {left.shape}")
    if right.ndim != 2 or right.shape[1] < 3:
        raise PhaseBoundaryError(f"right force_resultant must have shape [T,>=3], got {right.shape}")
    if timestamp.ndim != 1:
        raise PhaseBoundaryError(f"timestamp must have shape [T], got {timestamp.shape}")
    lengths = {puppet.shape[0], master.shape[0], left.shape[0], right.shape[0], timestamp.shape[0]}
    if len(lengths) != 1:
        raise PhaseBoundaryError(f"Raw V5 label streams have different frame counts: {sorted(lengths)}")
    selected = {
        "puppet_gripper": (loaded["puppet_gripper"][0], np.asarray(puppet[:, 6])),
        "master_gripper": (loaded["master_gripper"][0], np.asarray(master[:, 6])),
        "fz_left": (loaded["fz_left"][0], np.asarray(left[:, 2])),
        "fz_right": (loaded["fz_right"][0], np.asarray(right[:, 2])),
        "timestamp": (loaded["timestamp"][0], np.asarray(timestamp)),
    }
    arrays = {name: np.asarray(value[1], dtype=np.float64) for name, value in selected.items()}
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise PhaseBoundaryError(f"Raw V5 label stream {name} contains non-finite values")
    return RawAttemptStreams(
        puppet_gripper=arrays["puppet_gripper"],
        master_gripper=arrays["master_gripper"],
        fz_left=arrays["fz_left"],
        fz_right=arrays["fz_right"],
        timestamp=arrays["timestamp"],
        dataset_names={name: value[0] for name, value in selected.items()},
        source_sha256=_selected_stream_hash(selected),
    )


def _relative_progress(values: np.ndarray, *, name: str) -> tuple[np.ndarray, float, float]:
    p5, p95 = np.percentile(values, (5.0, 95.0))
    span = float(p95 - p5)
    scale = max(abs(float(p5)), abs(float(p95)), 1.0)
    if not math.isfinite(span) or span <= np.finfo(np.float64).eps * scale * 100:
        raise PhaseBoundaryError(f"{name} P5/P95 cannot define relative gripper progress")
    return np.clip((values - p5) / span, 0.0, 1.0), float(p5), float(p95)


def relative_gripper_progress(
    puppet_gripper: Sequence[float],
    master_gripper: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return puppet/master and combined P5/P95 relative opening progress."""

    puppet = np.asarray(puppet_gripper, dtype=np.float64)
    master = np.asarray(master_gripper, dtype=np.float64)
    if puppet.ndim != 1 or master.ndim != 1 or puppet.shape != master.shape:
        raise PhaseBoundaryError(
            f"Gripper streams must be same-length vectors, got {puppet.shape} and {master.shape}"
        )
    if len(puppet) < 30:
        raise PhaseBoundaryError("Attempt is too short for 30-frame gripper trend checks")
    puppet_progress, _, _ = _relative_progress(puppet, name="puppet gripper")
    master_progress, _, _ = _relative_progress(master, name="master gripper")
    combined = (puppet_progress + master_progress) * 0.5
    return puppet_progress, master_progress, combined


def adaptive_contact_force(
    *,
    puppet_gripper: Sequence[float],
    master_gripper: Sequence[float],
    fz_left: Sequence[float],
    fz_right: Sequence[float],
) -> tuple[np.ndarray, ForceThresholds, np.ndarray, np.ndarray]:
    """Estimate per-attempt Fz baselines and hysteresis thresholds.

    No absolute gripper opening or force threshold is used.  If robust noise
    and contact quantiles do not separate, the attempt is rejected instead of
    falling back to a gripper-distance label.
    """

    puppet_progress, master_progress, combined = relative_gripper_progress(
        puppet_gripper, master_gripper
    )
    left = np.asarray(fz_left, dtype=np.float64)
    right = np.asarray(fz_right, dtype=np.float64)
    if left.shape != combined.shape or right.shape != combined.shape:
        raise PhaseBoundaryError("Fz and gripper streams have different frame counts")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise PhaseBoundaryError("Fz streams contain non-finite values")

    baseline_count = max(3, int(math.ceil(0.10 * len(combined))))
    baseline_indices = np.argsort(combined, kind="stable")[-baseline_count:]
    left_baseline = float(np.median(left[baseline_indices]))
    right_baseline = float(np.median(right[baseline_indices]))
    contact_force = np.abs(left - left_baseline) + np.abs(right - right_baseline)
    no_contact = contact_force[baseline_indices]
    noise_median = float(np.median(no_contact))
    noise_mad = float(np.median(np.abs(no_contact - noise_median)))
    noise_sigma = max(1.4826 * noise_mad, np.finfo(np.float64).eps)
    force_off = noise_median + 6.0 * noise_sigma
    contact_reference = float(np.quantile(contact_force, 0.75))
    separation = (contact_reference - force_off) / noise_sigma
    if not math.isfinite(separation) or contact_reference <= force_off or separation < 6.0:
        raise PhaseBoundaryError(
            "No-contact and contact Fz distributions cannot be reliably separated: "
            f"noise_median={noise_median:.6g}, noise_mad={noise_mad:.6g}, "
            f"force_off={force_off:.6g}, contact_q75={contact_reference:.6g}, "
            f"separation={separation:.3f}"
        )
    # Keep the on threshold close enough to first contact that the required
    # 30-frame closing trend is still observable, while retaining hysteresis
    # above the robust no-contact envelope.
    force_on = force_off + max(
        0.10 * (contact_reference - force_off),
        6.0 * noise_sigma,
    )
    if force_on >= contact_reference:
        raise PhaseBoundaryError("Adaptive force hysteresis leaves no stable contact range")
    thresholds = ForceThresholds(
        left_baseline=left_baseline,
        right_baseline=right_baseline,
        noise_median=noise_median,
        noise_mad=noise_mad,
        noise_sigma=noise_sigma,
        force_off_threshold=float(force_off),
        force_on_threshold=float(force_on),
        contact_reference=contact_reference,
        separation_score=float(separation),
        open_baseline_frame_count=baseline_count,
    )
    return contact_force, thresholds, puppet_progress, master_progress


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(mask, dtype=np.bool_)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge_short_gaps(
    runs: Sequence[tuple[int, int]],
    *,
    max_gap: int,
) -> list[tuple[int, int]]:
    """Debounce isolated collision spikes without manufacturing a boundary."""

    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _trend(progress: np.ndarray, boundary: int, *, opening: bool) -> float:
    start = max(0, boundary - 30)
    window = progress[start : boundary + 1]
    if len(window) < 6:
        return 0.0
    edge = max(2, min(5, len(window) // 3))
    change = float(np.median(window[-edge:]) - np.median(window[:edge]))
    return change if opening else -change


def _caption_support(captions: Sequence[str] | None, release: int, regrasp: int) -> dict[str, Any]:
    if captions is None:
        return {
            "available": False,
            "release_non_none_fraction": None,
            "regrasp_non_none_fraction": None,
        }
    areas: list[str | None] = []
    for caption in captions:
        match = _AREA_RE.search(str(caption))
        areas.append(match.group(1) if match else None)

    def fraction(start: int, end: int) -> float | None:
        values = [area for area in areas[max(0, start) : min(len(areas), end)] if area is not None]
        if not values:
            return None
        return sum(area != "none" for area in values) / len(values)

    return {
        "available": any(area is not None for area in areas),
        "parsed_frame_count": sum(area is not None for area in areas),
        "release_non_none_fraction": fraction(release, release + 10),
        "regrasp_non_none_fraction": fraction(regrasp - 14, regrasp + 1),
    }


def detect_phase_boundaries(
    *,
    puppet_gripper: Sequence[float],
    master_gripper: Sequence[float],
    fz_left: Sequence[float],
    fz_right: Sequence[float],
    tactile_captions: Sequence[str] | None = None,
    force_off_frames: int = 10,
    force_on_frames: int = 15,
    trend_min_progress: float = 0.10,
) -> DetectedBoundary:
    """Detect release/regrasp with stable Fz runs and relative gripper trends."""

    if force_off_frames < 10 or force_on_frames < 15:
        raise ValueError("V5 requires at least 10 off frames and 15 on frames")
    contact_force, thresholds, puppet_progress, master_progress = adaptive_contact_force(
        puppet_gripper=puppet_gripper,
        master_gripper=master_gripper,
        fz_left=fz_left,
        fz_right=fz_right,
    )
    high_runs = [
        run for run in _true_runs(contact_force > thresholds.force_on_threshold)
        if run[1] - run[0] + 1 >= force_on_frames
    ]
    initial_runs = [run for run in high_runs if run[0] <= 15 and run[1] >= force_on_frames - 1]
    if not initial_runs:
        raise PhaseBoundaryError("Attempt2 does not begin with stable tactile contact")
    initial_run = initial_runs[0]

    low_runs = [
        run for run in _merge_short_gaps(
            _true_runs(contact_force < thresholds.force_off_threshold),
            max_gap=2,
        )
        if run[1] - run[0] + 1 >= force_off_frames
    ]
    release_candidates: list[tuple[tuple[int, int], float]] = []
    for run in low_runs:
        start = run[0]
        puppet_trend = _trend(puppet_progress, start, opening=True)
        master_trend = _trend(master_progress, start, opening=True)
        trend = max(puppet_trend, master_trend)
        if start > initial_run[0] and trend >= trend_min_progress:
            release_candidates.append((run, trend))
    if not release_candidates:
        raise PhaseBoundaryError("No stable force-off window with a preceding opening trend")
    release_run, opening_progress = release_candidates[0]
    release_frame = release_run[0]

    regrasp_candidates: list[tuple[tuple[int, int], float, int]] = []
    for run in high_runs:
        start = max(run[0], release_frame + 1)
        confirmed_end = start + force_on_frames - 1
        if confirmed_end > run[1]:
            continue
        puppet_trend = _trend(puppet_progress, start, opening=False)
        master_trend = _trend(master_progress, start, opening=False)
        trend = max(puppet_trend, master_trend)
        if trend >= trend_min_progress:
            regrasp_candidates.append((run, trend, confirmed_end))
    if not regrasp_candidates:
        raise PhaseBoundaryError("No stable force-on window with a preceding closing trend after release")
    regrasp_run, closing_progress, regrasp_frame = regrasp_candidates[0]
    if release_frame >= regrasp_frame:
        raise PhaseBoundaryError(
            f"Detected phase boundaries are reversed: release={release_frame}, regrasp={regrasp_frame}"
        )
    return DetectedBoundary(
        release_frame=release_frame,
        regrasp_frame=regrasp_frame,
        thresholds=thresholds,
        release_opening_progress=float(opening_progress),
        regrasp_closing_progress=float(closing_progress),
        initial_contact_run=initial_run,
        release_low_force_run=release_run,
        regrasp_high_force_run=regrasp_run,
        caption_support=_caption_support(tactile_captions, release_frame, regrasp_frame),
    )


def phase_for_frame(attempt_id: int, frame_index: int, release_frame: int | None, regrasp_frame: int | None) -> Phase:
    if attempt_id == 1:
        return "execution"
    if release_frame is None or regrasp_frame is None:
        raise ValueError("attempt2 phase lookup requires both boundaries")
    if frame_index <= release_frame:
        return "reposition"
    if frame_index <= regrasp_frame:
        return "adjustment"
    return "execution"


def load_phase_overrides(path: Path) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Versioned V5 override file is required (an empty list is valid): {path}"
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != V5_OVERRIDE_SCHEMA:
        raise ValueError("Unsupported V5 phase-boundary override schema")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError("V5 phase-boundary overrides require a positive integer version")
    rows = payload.get("overrides")
    if not isinstance(rows, list):
        raise ValueError("V5 phase-boundary overrides must contain an overrides list")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for row_number, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"Override row {row_number} is not an object")
        key = (int(raw["episode_id"]), int(raw["attempt_id"]))
        if key in result:
            raise ValueError(f"Duplicate phase-boundary override for {key}")
        if key[1] != 2:
            raise ValueError(f"Only attempt2 may have phase-boundary overrides, got {key}")
        release = raw["release_frame"]
        regrasp = raw["regrasp_frame"]
        for name, value in (("release_frame", release), ("regrasp_frame", regrasp)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Override {key} {name} must be a frame integer or timestamp")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Override {key} {name} must be finite and non-negative")
        # Integer values retain the original attempt-local frame semantics.
        # Floats are approximate raw timestamps and are resolved only after the
        # corresponding attempt's timestamp stream has been loaded.
        if isinstance(release, int) and isinstance(regrasp, int) and regrasp <= release:
            raise ValueError(f"Override {key} has invalid boundary order")
        if isinstance(release, float) and isinstance(regrasp, float) and regrasp <= release:
            raise ValueError(f"Override {key} has invalid timestamp order")
        result[key] = dict(raw)
    return payload, result


def resolve_phase_override(
    override: Mapping[str, Any],
    timestamps: Sequence[float],
    *,
    attempt_key: tuple[int, int],
) -> tuple[int, int, dict[str, Any]]:
    """Resolve legacy frame integers or approximate timestamps to local frames.

    A JSON integer inside the attempt frame range is interpreted as a legacy
    zero-based local frame. A float, or an integer outside that range, is
    interpreted as an approximate raw timestamp and mapped to the nearest
    timestamp in the attempt. Timestamp values must still lie within this
    attempt so a typo cannot silently select an endpoint from another attempt.
    """

    attempt_timestamps = np.asarray(timestamps, dtype=np.float64)
    if attempt_timestamps.ndim != 1 or len(attempt_timestamps) == 0:
        raise ValueError(f"Override {attempt_key} cannot resolve against empty timestamps")
    if not np.isfinite(attempt_timestamps).all():
        raise ValueError(f"Override {attempt_key} cannot resolve against non-finite timestamps")

    resolution: dict[str, Any] = {}

    def resolve(name: str) -> int:
        value = override[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Override {attempt_key} {name} must be a frame integer or timestamp"
            )
        if isinstance(value, int) and 0 <= value < len(attempt_timestamps):
            resolution[f"{name}_input_kind"] = "attempt_local_frame"
            resolution[f"resolved_{name}"] = value
            resolution[f"resolved_{name}_timestamp"] = float(attempt_timestamps[value])
            return value

        requested_timestamp = float(value)
        first_timestamp = float(np.min(attempt_timestamps))
        last_timestamp = float(np.max(attempt_timestamps))
        if not first_timestamp <= requested_timestamp <= last_timestamp:
            raise ValueError(
                f"Override {attempt_key} {name} timestamp {requested_timestamp:.9f} "
                f"is outside attempt range [{first_timestamp:.9f}, {last_timestamp:.9f}]"
            )
        frame = int(np.argmin(np.abs(attempt_timestamps - requested_timestamp)))
        matched_timestamp = float(attempt_timestamps[frame])
        resolution[f"{name}_input_kind"] = "approximate_timestamp"
        resolution[f"resolved_{name}"] = frame
        resolution[f"resolved_{name}_timestamp"] = matched_timestamp
        resolution[f"{name}_timestamp_error_seconds"] = abs(
            matched_timestamp - requested_timestamp
        )
        return frame

    release = resolve("release_frame")
    regrasp = resolve("regrasp_frame")
    if release >= regrasp:
        raise ValueError(
            f"Override {attempt_key} resolves to invalid boundary order: "
            f"release={release}, regrasp={regrasp}"
        )
    return release, regrasp, resolution


def _validate_attempt_alignment(
    meta: Mapping[str, Any],
    frames: Sequence[V4Frame],
    streams: RawAttemptStreams,
) -> None:
    expected_count = int(meta["frame_count"])
    if len(frames) != expected_count or len(streams.timestamp) != expected_count:
        raise PhaseBoundaryError(
            f"Attempt {(meta['episode_id'], meta['attempt_id'])} frame count mismatch: "
            f"profile={expected_count}, LeRobot={len(frames)}, raw={len(streams.timestamp)}"
        )
    ordered = sorted(frames, key=lambda frame: frame.frame_index)
    if [frame.frame_index for frame in ordered] != list(range(expected_count)):
        raise PhaseBoundaryError("LeRobot frame_index does not exactly cover raw HDF5 frames")
    lerobot_timestamps = np.asarray([frame.ros_timestamp for frame in ordered], dtype=np.float64)
    if not np.allclose(lerobot_timestamps, streams.timestamp, rtol=0.0, atol=1e-9):
        mismatch = int(np.flatnonzero(np.abs(lerobot_timestamps - streams.timestamp) > 1e-9)[0])
        raise PhaseBoundaryError(
            f"LeRobot/raw timestamp mismatch at frame {mismatch}: "
            f"lerobot={lerobot_timestamps[mismatch]!r}, raw={streams.timestamp[mismatch]!r}"
        )


def _file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.expanduser().resolve()), "sha256": sha256_file(path)}


def build_v5_phase_artifacts(
    *,
    dataset_dir: Path,
    v4_index_file: Path,
    overrides_file: Path,
    v4_norm_stats_dir: Path,
    expected_episode_count: int | None = EXPECTED_EPISODE_COUNT,
    expected_attempt_count: int | None = EXPECTED_ATTEMPT_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build boundary rows, action rows, V5 index skeleton and summary.

    Persisted JSONL file hashes are filled by the CLI after atomic writes.
    """

    v4_index = json.loads(v4_index_file.read_text())
    frames, global_lookup = validate_v4_index_dataset(v4_index, dataset_dir)
    source_files = v4_index["source_files"]
    selection_path = Path(str(source_files["selection"]["path"]))
    profile_path = Path(str(source_files["profile"]["path"]))
    split_path = Path(str(source_files["splits"]["path"]))
    selection = json.loads(selection_path.read_text())
    profile = json.loads(profile_path.read_text())
    splits = json.loads(split_path.read_text())
    selection_attempts = selection.get("attempts", [])
    profile_attempts = profile.get("attempts", [])
    episode_ids = {int(row["episode_id"]) for row in selection_attempts}
    if expected_episode_count is not None and len(episode_ids) != expected_episode_count:
        raise ValueError(
            f"V5 requires exactly {expected_episode_count} V4 episodes, got {len(episode_ids)}"
        )
    if expected_attempt_count is not None and len(selection_attempts) != expected_attempt_count:
        raise ValueError(
            f"V5 requires exactly {expected_attempt_count} V4 attempts, got {len(selection_attempts)}"
        )
    selected_tasks = {str(row["task"]) for row in selection_attempts}
    if selected_tasks != ALLOWED_TASKS:
        raise ValueError(
            f"V5 selection must contain exactly {sorted(ALLOWED_TASKS)}, got {sorted(selected_tasks)}"
        )
    forbidden = [
        row for row in selection_attempts
        if str(row.get("task")) in {"no_grasp", "small_grasp"}
        or str(row.get("grasp_position", "")) == "small"
    ]
    if forbidden:
        raise ValueError("V5 selection contains no_grasp/small_grasp samples")
    if v4_index.get("selection_hash") != selection.get("selection_hash"):
        raise ValueError("V5 selection hash differs from the V4 unified index")

    split_ids = {
        split: [int(value) for value in splits["original_episode_ids"][split]]
        for split in SPLITS
    }
    profile_by_attempt = {
        (int(row["episode_id"]), int(row["attempt_id"])): row for row in profile_attempts
    }
    frames_by_attempt: dict[tuple[int, int], list[V4Frame]] = defaultdict(list)
    for frame in frames:
        frames_by_attempt[frame.attempt_key].append(frame)
    if set(frames_by_attempt) != set(profile_by_attempt):
        raise ValueError("V5 raw labeling attempt set differs from V4 LeRobot/profile")

    override_payload, overrides = load_phase_overrides(overrides_file)
    extra_overrides = sorted(set(overrides) - set(profile_by_attempt))
    if extra_overrides:
        raise ValueError(f"Overrides reference attempts outside V4 selection: {extra_overrides[:20]}")
    raw_root = Path(str(profile["raw_data_dir"]))
    boundary_rows: list[dict[str, Any]] = []
    raw_identities: dict[str, Any] = {}
    unresolved: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    label_sources: Counter[str] = Counter()

    for key in sorted(profile_by_attempt):
        meta = profile_by_attempt[key]
        raw_path = Path(str(meta["hdf5_path"]))
        if not raw_path.is_absolute():
            raw_path = raw_root / raw_path
        streams = load_raw_attempt_streams(raw_path)
        attempt_frames = sorted(frames_by_attempt[key], key=lambda frame: frame.frame_index)
        _validate_attempt_alignment(meta, attempt_frames, streams)
        raw_identities[f"{key[0]}/{key[1]}"] = {
            "path": str(raw_path.resolve()),
            "sha256": streams.source_sha256,
            "dataset_names": streams.dataset_names,
            "frame_count": len(streams.timestamp),
        }

        common = {
            "schema_version": V5_PHASE_BOUNDARY_SCHEMA,
            "data_profile": ROTATION_PHASE_V5,
            "prompt_profile": PHASE_PROMPT_PROFILE,
            "experiment_kind": PHASE_EXPERIMENT_KIND,
            "episode_id": key[0],
            "attempt_id": key[1],
            "split": str(meta["split"]),
            "task": str(meta["task"]),
            "frame_count": len(attempt_frames),
            "raw_hdf5_path": str(raw_path.resolve()),
            "raw_label_source_sha256": streams.source_sha256,
            "raw_dataset_names": streams.dataset_names,
            "first_global_index": attempt_frames[0].global_index,
            "last_global_index": attempt_frames[-1].global_index,
        }
        if key[1] == 1:
            row = {
                **common,
                "release_frame": None,
                "regrasp_frame": None,
                "label_source": "attempt1_execution",
                "force_thresholds": None,
                "automatic_detection_error": None,
                "override": None,
            }
            boundary_rows.append(row)
            label_sources[row["label_source"]] += 1
            phase_counts["execution"] += len(attempt_frames)
            continue

        override = overrides.get(key)
        automatic_error: str | None = None
        detected: DetectedBoundary | None = None
        try:
            detected = detect_phase_boundaries(
                puppet_gripper=streams.puppet_gripper,
                master_gripper=streams.master_gripper,
                fz_left=streams.fz_left,
                fz_right=streams.fz_right,
                tactile_captions=[frame.tactile_caption for frame in attempt_frames],
            )
        except PhaseBoundaryError as exc:
            automatic_error = str(exc)
        if override is None and detected is None:
            unresolved.append({"episode_id": key[0], "attempt_id": key[1], "error": automatic_error})
            continue
        override_resolution: dict[str, Any] | None = None
        if override is not None:
            release, regrasp, override_resolution = resolve_phase_override(
                override,
                streams.timestamp,
                attempt_key=key,
            )
            label_source = "override"
        else:
            assert detected is not None
            release = detected.release_frame
            regrasp = detected.regrasp_frame
            label_source = "automatic"
        if not (0 <= release < regrasp < len(attempt_frames)):
            raise ValueError(
                f"Phase boundaries for {key} fall outside the attempt: "
                f"release={release}, regrasp={regrasp}, frames={len(attempt_frames)}"
            )
        row = {
            **common,
            "release_frame": release,
            "regrasp_frame": regrasp,
            "label_source": label_source,
            "force_thresholds": asdict(detected.thresholds) if detected is not None else None,
            "automatic_detection": (
                {
                    key: value
                    for key, value in asdict(detected).items()
                    if key not in {"thresholds", "release_frame", "regrasp_frame"}
                }
                if detected is not None
                else None
            ),
            "automatic_detection_error": automatic_error,
            "override": (
                {**dict(override), **override_resolution}
                if override is not None and override_resolution is not None
                else None
            ),
        }
        boundary_rows.append(row)
        label_sources[label_source] += 1
        phase_counts["reposition"] += release + 1
        phase_counts["adjustment"] += regrasp - release
        phase_counts["execution"] += len(attempt_frames) - regrasp - 1

    if unresolved:
        preview = "; ".join(
            f"{row['episode_id']}/{row['attempt_id']}: {row['error']}" for row in unresolved[:20]
        )
        raise PhaseBoundaryError(
            f"{len(unresolved)} attempt2 rows require phase_boundary_overrides.json; {preview}"
        )
    boundary_by_attempt = {
        (int(row["episode_id"]), int(row["attempt_id"])): row for row in boundary_rows
    }
    if len(boundary_by_attempt) != len(profile_by_attempt):
        raise AssertionError("V5 boundary manifest does not cover every V4 attempt")

    action_rows: list[dict[str, Any]] = []
    split_entries: dict[str, Any] = {}
    all_v4_indices: list[int] = []
    horizon = int(v4_index["action_horizon"])
    for split in SPLITS:
        indices = [int(value) for value in v4_index["splits"][split]["execution_indices"]]
        manifest_rows: list[int] = []
        for global_index in indices:
            frame = global_lookup[global_index]
            boundary = boundary_by_attempt[frame.attempt_key]
            release = boundary["release_frame"]
            regrasp = boundary["regrasp_frame"]
            phase = phase_for_frame(frame.attempt_id, frame.frame_index, release, regrasp)
            chunk_phases = {
                phase_for_frame(frame.attempt_id, offset, release, regrasp)
                for offset in range(frame.frame_index, frame.frame_index + horizon)
            }
            row_index = len(action_rows)
            action_rows.append(
                {
                    "schema_version": V5_ACTION_PHASE_SCHEMA,
                    "data_profile": ROTATION_PHASE_V5,
                    "prompt_profile": PHASE_PROMPT_PROFILE,
                    "experiment_kind": PHASE_EXPERIMENT_KIND,
                    "split": split,
                    "global_index": global_index,
                    "episode_id": frame.episode_id,
                    "attempt_id": frame.attempt_id,
                    "frame_index": frame.frame_index,
                    "phase": phase,
                    "chunk_phase_pure": len(chunk_phases) == 1,
                    "chunk_end_frame": frame.frame_index + horizon - 1,
                    "action_horizon": horizon,
                }
            )
            manifest_rows.append(row_index)
        split_entries[split] = {
            "execution_indices": indices,
            "action_phase_manifest_row_indices": manifest_rows,
            "summary": {
                "action_count": len(indices),
                "phase_counts": dict(Counter(action_rows[i]["phase"] for i in manifest_rows)),
                "chunk_phase_pure": sum(bool(action_rows[i]["chunk_phase_pure"]) for i in manifest_rows),
                "chunk_phase_crossing": sum(not bool(action_rows[i]["chunk_phase_pure"]) for i in manifest_rows),
            },
        }
        all_v4_indices.extend(indices)

    v4_norm_summary_path = v4_norm_stats_dir / "summary.json"
    v4_norm_stats_path = v4_norm_stats_dir / "norm_stats.json"
    if not v4_norm_summary_path.is_file() or not v4_norm_stats_path.is_file():
        raise FileNotFoundError("V5 must reuse the persisted V4 norm summary and norm_stats.json")
    norm_summary = json.loads(v4_norm_summary_path.read_text())
    if norm_summary.get("artifact_identity", {}).get("action_indices_identity") != v4_index["action_indices_identity"]:
        raise ValueError("V4 norm stats action indices differ from the V5/V4 action indices")
    norm_sha = sha256_file(v4_norm_stats_path)
    if norm_summary.get("norm_stats_sha256") != norm_sha:
        raise ValueError("V4 norm stats SHA is invalid")

    action_identity = v4_index["action_indices_identity"]
    if action_identity["all"] != {
        "count": len(all_v4_indices),
        "sha256": sha256_json(all_v4_indices),
    }:
        raise ValueError("V5 execution indices are not byte-for-byte identical to V4")
    h30_target_identity = {
        "action_horizon": horizon,
        "execution_indices": action_identity,
        "lerobot_identity": v4_index["lerobot_identity"],
        "lerobot_parquet_sha256": {
            name: value
            for name, value in sorted(v4_index["source_files"]["lerobot_parquet"].items())
        },
    }
    h30_target_identity["sha256"] = sha256_json(h30_target_identity)
    index: dict[str, Any] = {
        "schema_version": V5_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V5,
        "prompt_profile": PHASE_PROMPT_PROFILE,
        "experiment_kind": PHASE_EXPERIMENT_KIND,
        "data_config_hash": sha256_json(
            {
                "selection_hash": selection["selection_hash"],
                "v4_profile_config_hash": profile["profile_config_hash"],
                "prompt_profile": PHASE_PROMPT_PROFILE,
                "experiment_kind": PHASE_EXPERIMENT_KIND,
            }
        ),
        "selection_hash": selection["selection_hash"],
        "v4_profile_config_hash": profile["profile_config_hash"],
        "dataset_dir": str(dataset_dir.resolve()),
        "action_horizon": horizon,
        "expected_episode_count": len(episode_ids),
        "expected_attempt_count": len(profile_by_attempt),
        "selected_tasks": sorted(selected_tasks),
        "split_episode_ids": split_ids,
        "splits": split_entries,
        "action_indices_identity": action_identity,
        "v4_action_indices_identity": v4_index["action_indices_identity"],
        "h30_target_identity": h30_target_identity,
        "v4_h30_target_identity": h30_target_identity,
        "v4_lerobot_identity": v4_index["lerobot_identity"],
        "v4_training_data_hash": v4_index["training_data_hash"],
        "v4_norm_stats_sha256": norm_sha,
        "source_files": {
            "selection": _file_identity(selection_path),
            "profile": _file_identity(profile_path),
            "splits": _file_identity(split_path),
            "v4_training_index": _file_identity(v4_index_file),
            "phase_boundary_overrides": _file_identity(overrides_file),
            "v4_norm_summary": _file_identity(v4_norm_summary_path),
            "v4_norm_stats": _file_identity(v4_norm_stats_path),
            "raw_label_sources": raw_identities,
        },
        "phase_boundaries_identity": {
            "count": len(boundary_rows),
            "content_sha256": sha256_json(boundary_rows),
        },
        "action_phase_manifest_identity": {
            "count": len(action_rows),
            "content_sha256": sha256_json(action_rows),
        },
    }
    summary = {
        "schema_version": V5_SUMMARY_SCHEMA,
        "data_profile": ROTATION_PHASE_V5,
        "prompt_profile": PHASE_PROMPT_PROFILE,
        "experiment_kind": PHASE_EXPERIMENT_KIND,
        "selection_hash": selection["selection_hash"],
        "selected_episodes": len(episode_ids),
        "selected_attempts": len(profile_by_attempt),
        "selected_tasks": dict(Counter(str(row["task"]) for row in selection_attempts)),
        "attempt_ids": dict(Counter(str(row["attempt_id"]) for row in selection_attempts)),
        "no_grasp": 0,
        "small_grasp": 0,
        "phase_frame_counts": dict(phase_counts),
        "boundary_label_sources": dict(label_sources),
        "action_indices_identity": action_identity,
        "v4_norm_stats_sha256": norm_sha,
        "h30_target_identity_sha256": h30_target_identity["sha256"],
        "splits": {split: split_entries[split]["summary"] for split in SPLITS},
        "override_version": override_payload["version"],
        "override_count": len(overrides),
    }
    return boundary_rows, action_rows, index, summary


def _validate_file_identity(identity: Mapping[str, Any], *, context: str) -> Path:
    path = Path(str(identity.get("path", "")))
    expected = str(identity.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if len(expected) != 64 or actual != expected:
        raise ValueError(f"V5 {context} hash mismatch: expected={expected}, actual={actual}")
    return path


def validate_v5_training_index(
    payload: Mapping[str, Any],
    *,
    index_path: Path | None = None,
    dataset_dir: Path | None = None,
    revalidate_raw_streams: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Strictly validate a persisted prompt-only V5 training index."""

    expected_header = {
        "schema_version": V5_TRAINING_INDEX_SCHEMA,
        "data_profile": ROTATION_PHASE_V5,
        "prompt_profile": PHASE_PROMPT_PROFILE,
        "experiment_kind": PHASE_EXPERIMENT_KIND,
    }
    for key, expected in expected_header.items():
        if payload.get(key) != expected:
            raise ValueError(f"V5 index {key}={payload.get(key)!r}, expected={expected!r}")
    stored_hash = str(payload.get("training_data_hash", ""))
    actual_hash = sha256_json({key: value for key, value in payload.items() if key != "training_data_hash"})
    if not stored_hash or stored_hash != actual_hash:
        raise ValueError("V5 training_data_hash does not match the index payload")
    if index_path is not None and not Path(index_path).is_file():
        raise FileNotFoundError(index_path)
    sources = payload.get("source_files")
    if not isinstance(sources, Mapping):
        raise ValueError("V5 index lacks source_files")
    required = {
        "selection", "profile", "splits", "v4_training_index",
        "phase_boundary_overrides", "phase_boundaries", "action_phase_manifest",
        "v4_norm_summary", "v4_norm_stats", "raw_label_sources",
    }
    missing = sorted(required - set(sources))
    if missing:
        raise ValueError(f"V5 index lacks source identities: {missing}")
    paths = {
        name: _validate_file_identity(sources[name], context=name)
        for name in required - {"raw_label_sources"}
    }
    v4_index = json.loads(paths["v4_training_index"].read_text())
    effective_dataset_dir = dataset_dir or Path(str(payload["dataset_dir"]))
    _, v4_global_lookup = validate_v4_index_dataset(v4_index, effective_dataset_dir)
    if payload.get("selection_hash") != v4_index.get("selection_hash"):
        raise ValueError("V5/V4 selection hashes differ")
    if payload.get("v4_training_data_hash") != v4_index.get("training_data_hash"):
        raise ValueError("V5 references a different V4 unified index")
    if payload.get("action_indices_identity") != v4_index.get("action_indices_identity"):
        raise ValueError("V5/V4 action index identities differ")
    h30_identity = payload.get("h30_target_identity")
    if not isinstance(h30_identity, Mapping):
        raise ValueError("V5 index lacks the immutable V4 H30 target identity")
    stored_h30_hash = str(h30_identity.get("sha256", ""))
    actual_h30_hash = sha256_json(
        {key: value for key, value in h30_identity.items() if key != "sha256"}
    )
    if stored_h30_hash != actual_h30_hash or payload.get("v4_h30_target_identity") != h30_identity:
        raise ValueError("V5/V4 H30 target identities differ")
    for split in SPLITS:
        if payload["splits"][split]["execution_indices"] != v4_index["splits"][split]["execution_indices"]:
            raise ValueError(f"V5/V4 {split} execution indices differ")

    selection = json.loads(paths["selection"].read_text())
    profile = json.loads(paths["profile"].read_text())
    split_payload = json.loads(paths["splits"].read_text())
    if selection.get("selection_hash") != payload.get("selection_hash"):
        raise ValueError("Persisted V5 selection differs from V4")
    if {str(row["task"]) for row in selection.get("attempts", [])} != ALLOWED_TASKS:
        raise ValueError("Persisted V5 selection contains tasks outside the phase-prompt experiment")
    if len(selection.get("selected_episode_ids", [])) != int(payload["expected_episode_count"]):
        raise ValueError("V5 selected episode count changed")
    if len(profile.get("attempts", [])) != int(payload["expected_attempt_count"]):
        raise ValueError("V5 selected attempt count changed")
    actual_split_ids = {
        split: [int(value) for value in split_payload["original_episode_ids"][split]]
        for split in SPLITS
    }
    if actual_split_ids != payload.get("split_episode_ids"):
        raise ValueError("V5 split is not exactly the V4 split")

    boundary_rows = load_jsonl(paths["phase_boundaries"])
    action_rows = load_jsonl(paths["action_phase_manifest"])
    if len(boundary_rows) != int(payload["phase_boundaries_identity"]["count"]):
        raise ValueError("V5 phase boundary row count mismatch")
    if sha256_json(boundary_rows) != payload["phase_boundaries_identity"]["content_sha256"]:
        raise ValueError("V5 phase boundary content identity mismatch")
    if len(action_rows) != int(payload["action_phase_manifest_identity"]["count"]):
        raise ValueError("V5 action phase row count mismatch")
    if sha256_json(action_rows) != payload["action_phase_manifest_identity"]["content_sha256"]:
        raise ValueError("V5 action phase content identity mismatch")
    action_lookup: dict[int, dict[str, Any]] = {}
    expected_actions: list[int] = []
    for split in SPLITS:
        row_indices = payload["splits"][split]["action_phase_manifest_row_indices"]
        indices = payload["splits"][split]["execution_indices"]
        if len(row_indices) != len(indices):
            raise ValueError(f"V5 {split} phase rows/action indices lengths differ")
        for row_index, global_index in zip(row_indices, indices, strict=True):
            row = action_rows[int(row_index)]
            if int(row["global_index"]) != int(global_index) or row["split"] != split:
                raise ValueError("V5 action phase manifest identity/order mismatch")
            if row["phase"] not in ACTION_PHASES or not isinstance(row.get("chunk_phase_pure"), bool):
                raise ValueError("V5 action phase row has an invalid phase/purity flag")
            v4_frame = v4_global_lookup[int(global_index)]
            identity = (int(row["episode_id"]), int(row["attempt_id"]), int(row["frame_index"]))
            if identity != v4_frame.key:
                raise ValueError("V5 action phase row differs from V4 frame identity")
            if int(global_index) in action_lookup:
                raise ValueError(f"Duplicate V5 action phase global index {global_index}")
            action_lookup[int(global_index)] = row
            expected_actions.append(int(global_index))
    if payload["action_indices_identity"]["all"] != {
        "count": len(expected_actions),
        "sha256": sha256_json(expected_actions),
    }:
        raise ValueError("V5 action indices identity is invalid")

    norm_summary = json.loads(paths["v4_norm_summary"].read_text())
    norm_sha = sha256_file(paths["v4_norm_stats"])
    if (
        norm_sha != payload.get("v4_norm_stats_sha256")
        or norm_summary.get("norm_stats_sha256") != norm_sha
        or norm_summary.get("artifact_identity", {}).get("action_indices_identity")
        != payload.get("action_indices_identity")
    ):
        raise ValueError("V5 did not preserve V4 norm stats/action identity")

    raw_sources = sources["raw_label_sources"]
    if not isinstance(raw_sources, Mapping) or len(raw_sources) != int(payload["expected_attempt_count"]):
        raise ValueError("V5 raw label source identities do not cover every attempt")
    for key, identity in raw_sources.items():
        if not isinstance(identity, Mapping) or len(str(identity.get("sha256", ""))) != 64:
            raise ValueError(f"V5 raw label source {key} lacks a complete SHA256 identity")
        raw_path = Path(str(identity.get("path", "")))
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if revalidate_raw_streams:
            actual = load_raw_attempt_streams(raw_path).source_sha256
            if actual != identity["sha256"]:
                raise ValueError(f"V5 raw label source hash mismatch for {key}")
    return action_rows, action_lookup
