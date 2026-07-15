#!/usr/bin/env python3
"""Copy raw episodes into a dataset while assigning new episode IDs.

The source episode directories are sorted by their numeric suffix and mapped
contiguously to ``episode<start_episode_id>`` and above.  Both ``meta.json``
and ``data.hdf5/meta/episode_id`` are updated in the copied attempt folders.

The importer validates every source attempt before writing anything.  Each
episode is copied through a temporary directory and renamed atomically after
validation, so ``--resume`` can safely skip completed episodes after an
interrupted import.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import h5py
import numpy as np


DEFAULT_SOURCE_DIR = Path("/home/test/qxh/workspace/tac_ws/data")
DEFAULT_TARGET_DIR = Path("/data1/tac_data/raw_data")
DEFAULT_START_EPISODE_ID = 121

EPISODE_PATTERN = re.compile(r"episode(\d+)")
ATTEMPT_PATTERN = re.compile(r"attempt(\d+)")

REQUIRED_FRAME_DATASETS = (
    "timestamp",
    "arm/jointStatePosition/masterRight",
    "arm/jointStatePosition/puppetRight",
    "camera/color/front",
    "camera/color/left",
    "label/rotation_state_id",
    "label/need_recovery",
    "label/input_recovery_plan",
    "label/failure_recovery_memory",
    "label/rotation_state_name",
    "label/tactile_caption",
    "label/failure_reason",
    "label/recovery_plan",
)
REQUIRED_SCALAR_DATASETS = (
    "size",
    "meta/episode_id",
    "meta/attempt_id",
    "meta/case_id",
    "meta/instruction",
    "meta/valid",
    "reasoning/has_sample",
)
CAMERA_DATASETS = ("camera/color/front", "camera/color/left")


@dataclass(frozen=True)
class TreeInventory:
    file_count: int
    total_bytes: int
    structural_digest: str
    content_digest: str | None


@dataclass(frozen=True)
class EpisodeMapping:
    source_episode_id: int
    target_episode_id: int
    source_dir: Path
    target_dir: Path
    attempt_count: int
    frame_count: int
    valid_attempt_count: int
    inventory: TreeInventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--start-episode-id", type=int, default=DEFAULT_START_EPISODE_ID)
    parser.add_argument(
        "--verify-images",
        choices=("none", "first-last", "all"),
        default="all",
        help="Check that image paths stored in HDF5 exist inside each attempt directory.",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help=(
            "Hash unchanged auxiliary file contents too; rewritten meta.json/HDF5 files are checked semantically. "
            "This reads most of the source and destination and is slow for this dataset."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and skip already completed target episodes; continue an unfinished temporary episode copy.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only validate and print the import plan.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path. By default it is stored directly under target-dir.",
    )
    return parser.parse_args()


def _numeric_dirs(root: Path, pattern: re.Pattern[str], kind: str) -> list[tuple[int, Path]]:
    matches: list[tuple[int, Path]] = []
    unexpected: list[Path] = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            unexpected.append(path)
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinked {kind} directory is not allowed: {path}")
        matches.append((int(match.group(1)), path))
    if unexpected:
        names = ", ".join(str(path) for path in sorted(unexpected))
        raise ValueError(f"Unexpected directories under {root}: {names}")
    matches.sort(key=lambda item: item[0])
    return matches


def discover_source_episodes(source_dir: Path) -> list[tuple[int, Path]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    episodes = _numeric_dirs(source_dir, EPISODE_PATTERN, "episode")
    if not episodes:
        raise ValueError(f"No episode<N> directories found under {source_dir}")
    episode_ids = [episode_id for episode_id, _ in episodes]
    expected = list(range(episode_ids[0], episode_ids[-1] + 1))
    if episode_ids != expected:
        missing = sorted(set(expected) - set(episode_ids))
        raise ValueError(f"Source episode IDs must be contiguous; missing IDs: {missing}")
    return episodes


def _decode_path(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _safe_relative_path(attempt_dir: Path, value: Any, dataset_name: str) -> Path:
    text = _decode_path(value)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image path in {attempt_dir / 'data.hdf5'}:{dataset_name}: {text!r}")
    return attempt_dir / relative


def _image_indices(size: int, mode: str) -> range | tuple[int, ...]:
    if mode == "none" or size == 0:
        return ()
    if mode == "first-last":
        return (0,) if size == 1 else (0, size - 1)
    return range(size)


def validate_attempt(
    attempt_dir: Path,
    *,
    expected_episode_id: int,
    expected_attempt_id: int,
    verify_images: str,
) -> tuple[int, bool]:
    meta_path = attempt_dir / "meta.json"
    hdf5_path = attempt_dir / "data.hdf5"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Missing HDF5 file: {hdf5_path}")

    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {meta_path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"Expected a JSON object in {meta_path}")
    if int(meta.get("episode_id", -1)) != expected_episode_id:
        raise ValueError(
            f"episode_id mismatch in {meta_path}: {meta.get('episode_id')!r} != {expected_episode_id}"
        )
    if int(meta.get("attempt_id", -1)) != expected_attempt_id:
        raise ValueError(
            f"attempt_id mismatch in {meta_path}: {meta.get('attempt_id')!r} != {expected_attempt_id}"
        )

    try:
        with h5py.File(hdf5_path, "r") as episode:
            missing = [name for name in REQUIRED_SCALAR_DATASETS + REQUIRED_FRAME_DATASETS if name not in episode]
            if missing:
                raise KeyError(f"Missing HDF5 datasets: {missing}")
            size = int(episode["size"][()])
            if size <= 0:
                raise ValueError(f"Invalid frame count in {hdf5_path}: {size}")
            hdf5_episode_id = int(episode["meta/episode_id"][()])
            hdf5_attempt_id = int(episode["meta/attempt_id"][()])
            if hdf5_episode_id != expected_episode_id:
                raise ValueError(
                    f"episode_id mismatch in {hdf5_path}: {hdf5_episode_id} != {expected_episode_id}"
                )
            if hdf5_attempt_id != expected_attempt_id:
                raise ValueError(
                    f"attempt_id mismatch in {hdf5_path}: {hdf5_attempt_id} != {expected_attempt_id}"
                )
            for dataset_name in REQUIRED_FRAME_DATASETS:
                dataset = episode[dataset_name]
                if not dataset.shape or int(dataset.shape[0]) != size:
                    raise ValueError(
                        f"Frame length mismatch in {hdf5_path}:{dataset_name}: {dataset.shape} vs size={size}"
                    )
            for dataset_name in (
                "arm/jointStatePosition/masterRight",
                "arm/jointStatePosition/puppetRight",
            ):
                if episode[dataset_name].shape != (size, 7):
                    raise ValueError(
                        f"Expected shape {(size, 7)} in {hdf5_path}:{dataset_name}, "
                        f"got {episode[dataset_name].shape}"
                    )

            for dataset_name in CAMERA_DATASETS:
                dataset = episode[dataset_name]
                for index in _image_indices(size, verify_images):
                    image_path = _safe_relative_path(attempt_dir, dataset[index], dataset_name)
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"Missing image referenced by {hdf5_path}:{dataset_name}[{index}]: {image_path}"
                        )

            reasoning_has_sample = bool(episode["reasoning/has_sample"][()])
            if reasoning_has_sample:
                if "reasoning/failed_frame_index" not in episode:
                    raise KeyError(f"Missing reasoning/failed_frame_index in {hdf5_path}")
                failed_index = int(episode["reasoning/failed_frame_index"][()])
                if not 0 <= failed_index < size:
                    raise ValueError(f"Invalid reasoning frame {failed_index} for size={size} in {hdf5_path}")
            valid = bool(episode["meta/valid"][()])
    except OSError as exc:
        raise ValueError(f"Cannot open HDF5 file {hdf5_path}: {exc}") from exc

    if bool(meta.get("valid", valid)) != valid:
        raise ValueError(f"valid flag differs between {meta_path} and {hdf5_path}")
    return size, valid


def inventory_tree(root: Path, *, checksum: bool) -> TreeInventory:
    structural_hasher = hashlib.sha256()
    content_hasher = hashlib.sha256() if checksum else None
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in raw episode data: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_count += 1
        total_bytes += size
        # meta.json changes length when its episode ID is rewritten. It is
        # validated semantically instead of through the structural digest.
        if path.name != "meta.json":
            structural_hasher.update(relative.encode("utf-8"))
            structural_hasher.update(b"\0")
            structural_hasher.update(str(size).encode("ascii"))
            structural_hasher.update(b"\n")
        if content_hasher is not None and path.name not in {"meta.json", "data.hdf5"}:
            content_hasher.update(relative.encode("utf-8"))
            content_hasher.update(b"\0")
            with path.open("rb") as file:
                while chunk := file.read(8 * 1024 * 1024):
                    content_hasher.update(chunk)
    return TreeInventory(
        file_count=file_count,
        total_bytes=total_bytes,
        structural_digest=structural_hasher.hexdigest(),
        content_digest=content_hasher.hexdigest() if content_hasher is not None else None,
    )


def validate_episode(
    episode_dir: Path,
    *,
    expected_episode_id: int,
    verify_images: str,
    checksum: bool,
) -> tuple[int, int, int, TreeInventory]:
    attempts = _numeric_dirs(episode_dir, ATTEMPT_PATTERN, "attempt")
    if not attempts:
        raise ValueError(f"No attempt<N> directories found under {episode_dir}")
    attempt_ids = [attempt_id for attempt_id, _ in attempts]
    if attempt_ids != list(range(1, attempt_ids[-1] + 1)):
        raise ValueError(f"Attempt IDs must start at 1 and be contiguous in {episode_dir}: {attempt_ids}")

    frame_count = 0
    valid_attempt_count = 0
    for attempt_id, attempt_dir in attempts:
        size, valid = validate_attempt(
            attempt_dir,
            expected_episode_id=expected_episode_id,
            expected_attempt_id=attempt_id,
            verify_images=verify_images,
        )
        frame_count += size
        valid_attempt_count += int(valid)
    inventory = inventory_tree(episode_dir, checksum=checksum)
    return len(attempts), frame_count, valid_attempt_count, inventory


def build_mappings(args: argparse.Namespace) -> list[EpisodeMapping]:
    episodes = discover_source_episodes(args.source_dir)
    mappings: list[EpisodeMapping] = []
    for offset, (source_episode_id, source_dir) in enumerate(episodes):
        target_episode_id = args.start_episode_id + offset
        print(f"preflight source={source_dir} target=episode{target_episode_id}", flush=True)
        attempt_count, frame_count, valid_attempt_count, inventory = validate_episode(
            source_dir,
            expected_episode_id=source_episode_id,
            verify_images=args.verify_images,
            checksum=args.checksum,
        )
        mappings.append(
            EpisodeMapping(
                source_episode_id=source_episode_id,
                target_episode_id=target_episode_id,
                source_dir=source_dir,
                target_dir=args.target_dir / f"episode{target_episode_id}",
                attempt_count=attempt_count,
                frame_count=frame_count,
                valid_attempt_count=valid_attempt_count,
                inventory=inventory,
            )
        )
    return mappings


def update_episode_id(episode_dir: Path, target_episode_id: int) -> None:
    for attempt_id, attempt_dir in _numeric_dirs(episode_dir, ATTEMPT_PATTERN, "attempt"):
        meta_path = attempt_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["episode_id"] = target_episode_id
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        hdf5_path = attempt_dir / "data.hdf5"
        with h5py.File(hdf5_path, "r+") as episode:
            episode["meta/episode_id"][()] = target_episode_id
            if int(episode["meta/attempt_id"][()]) != attempt_id:
                raise ValueError(f"attempt_id changed unexpectedly while updating {hdf5_path}")


def inventories_match(source: TreeInventory, target: TreeInventory, *, checksum: bool) -> bool:
    if source.file_count != target.file_count or source.structural_digest != target.structural_digest:
        return False
    return not checksum or source.content_digest == target.content_digest


def validate_copied_episode(mapping: EpisodeMapping, args: argparse.Namespace, episode_dir: Path) -> None:
    attempt_count, frame_count, valid_attempt_count, inventory = validate_episode(
        episode_dir,
        expected_episode_id=mapping.target_episode_id,
        verify_images=args.verify_images,
        checksum=args.checksum,
    )
    if (attempt_count, frame_count, valid_attempt_count) != (
        mapping.attempt_count,
        mapping.frame_count,
        mapping.valid_attempt_count,
    ):
        raise ValueError(
            f"Copied episode summary differs for {episode_dir}: "
            f"{(attempt_count, frame_count, valid_attempt_count)} != "
            f"{(mapping.attempt_count, mapping.frame_count, mapping.valid_attempt_count)}"
        )
    if not inventories_match(mapping.inventory, inventory, checksum=args.checksum):
        raise ValueError(f"Copied file inventory differs from source for {episode_dir}")


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"No existing parent found for {path}")
        candidate = candidate.parent
    return candidate


def preflight_targets(mappings: list[EpisodeMapping], args: argparse.Namespace) -> tuple[list[int], int]:
    completed: list[int] = []
    missing_bytes = 0
    for mapping in mappings:
        temporary_dir = args.target_dir / f".{mapping.target_dir.name}.importing"
        if mapping.target_dir.exists():
            if not args.resume:
                raise FileExistsError(
                    f"Target already exists: {mapping.target_dir}. Refusing to overwrite; use --resume only if it "
                    "was created by an earlier run of this importer."
                )
            if temporary_dir.exists():
                raise FileExistsError(
                    f"Both completed and temporary directories exist: {mapping.target_dir}, {temporary_dir}"
                )
            print(f"validating completed target={mapping.target_dir}", flush=True)
            validate_copied_episode(mapping, args, mapping.target_dir)
            completed.append(mapping.target_episode_id)
        else:
            if temporary_dir.exists() and not args.resume:
                raise FileExistsError(f"Unfinished import exists: {temporary_dir}; rerun with --resume")
            missing_bytes += mapping.inventory.total_bytes

    disk_root = _nearest_existing_parent(args.target_dir)
    free_bytes = shutil.disk_usage(disk_root).free
    reserve = max(1024**3, missing_bytes // 20)
    if free_bytes < missing_bytes + reserve:
        raise OSError(
            f"Not enough free space under {disk_root}: need about {(missing_bytes + reserve) / 1024**3:.1f} GiB "
            f"including reserve, have {free_bytes / 1024**3:.1f} GiB"
        )
    return completed, missing_bytes


def default_manifest_path(args: argparse.Namespace, mappings: list[EpisodeMapping]) -> Path:
    if args.manifest is not None:
        return args.manifest
    return args.target_dir / (
        f"import_manifest_episode{mappings[0].target_episode_id}_{mappings[-1].target_episode_id}.json"
    )


def manifest_payload(
    args: argparse.Namespace,
    mappings: list[EpisodeMapping],
    statuses: dict[int, str],
    *,
    status: str,
    created_at: str,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir.resolve()),
        "target_dir": str(args.target_dir.resolve()),
        "start_episode_id": args.start_episode_id,
        "verify_images": args.verify_images,
        "checksum": args.checksum,
        "summary": {
            "episodes": len(mappings),
            "attempts": sum(mapping.attempt_count for mapping in mappings),
            "valid_attempts": sum(mapping.valid_attempt_count for mapping in mappings),
            "frames": sum(mapping.frame_count for mapping in mappings),
            "source_bytes": sum(mapping.inventory.total_bytes for mapping in mappings),
        },
        "mappings": [],
    }
    for mapping in mappings:
        item = asdict(mapping)
        item["source_dir"] = str(mapping.source_dir)
        item["target_dir"] = str(mapping.target_dir)
        item["status"] = statuses[mapping.target_episode_id]
        payload["mappings"].append(item)
    if error is not None:
        payload["error"] = error
    return payload


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def print_plan(mappings: list[EpisodeMapping], missing_bytes: int, completed: list[int]) -> None:
    summary = {
        "episodes": len(mappings),
        "attempts": sum(mapping.attempt_count for mapping in mappings),
        "valid_attempts": sum(mapping.valid_attempt_count for mapping in mappings),
        "frames": sum(mapping.frame_count for mapping in mappings),
        "source_id_range": [mappings[0].source_episode_id, mappings[-1].source_episode_id],
        "target_id_range": [mappings[0].target_episode_id, mappings[-1].target_episode_id],
        "already_completed": len(completed),
        "bytes_left_to_copy": missing_bytes,
        "gib_left_to_copy": round(missing_bytes / 1024**3, 2),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run(args: argparse.Namespace) -> Path | None:
    if args.start_episode_id < 0:
        raise ValueError(f"start-episode-id must be non-negative: {args.start_episode_id}")
    source_resolved = args.source_dir.resolve()
    target_resolved = args.target_dir.resolve()
    if source_resolved == target_resolved or source_resolved in target_resolved.parents:
        raise ValueError("target-dir must not be the source directory or a child of it")

    mappings = build_mappings(args)
    completed, missing_bytes = preflight_targets(mappings, args)
    print_plan(mappings, missing_bytes, completed)
    if args.dry_run:
        print("dry-run complete; no files were copied")
        return None

    args.target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = default_manifest_path(args, mappings)
    statuses = {
        mapping.target_episode_id: (
            "validated_existing" if mapping.target_episode_id in completed else "pending"
        )
        for mapping in mappings
    }
    created_at = datetime.now(timezone.utc).isoformat()
    write_manifest(
        manifest_path,
        manifest_payload(args, mappings, statuses, status="in_progress", created_at=created_at),
    )

    try:
        for index, mapping in enumerate(mappings, start=1):
            if mapping.target_episode_id in completed:
                continue
            temporary_dir = args.target_dir / f".{mapping.target_dir.name}.importing"
            statuses[mapping.target_episode_id] = "copying"
            write_manifest(
                manifest_path,
                manifest_payload(args, mappings, statuses, status="in_progress", created_at=created_at),
            )
            print(
                f"[{index}/{len(mappings)}] copying {mapping.source_dir} -> {mapping.target_dir}",
                flush=True,
            )
            shutil.copytree(
                mapping.source_dir,
                temporary_dir,
                copy_function=shutil.copy2,
                dirs_exist_ok=args.resume,
            )
            update_episode_id(temporary_dir, mapping.target_episode_id)
            validate_copied_episode(mapping, args, temporary_dir)
            temporary_dir.rename(mapping.target_dir)
            statuses[mapping.target_episode_id] = "copied_and_validated"
            write_manifest(
                manifest_path,
                manifest_payload(args, mappings, statuses, status="in_progress", created_at=created_at),
            )
    except Exception as exc:
        write_manifest(
            manifest_path,
            manifest_payload(
                args,
                mappings,
                statuses,
                status="failed",
                created_at=created_at,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise

    write_manifest(
        manifest_path,
        manifest_payload(args, mappings, statuses, status="completed", created_at=created_at),
    )
    print(f"import complete; manifest={manifest_path}")
    return manifest_path


def main() -> None:
    try:
        run(parse_args())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
