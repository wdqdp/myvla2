#!/usr/bin/env python3
"""Download and validate the OpenPI pi05 base checkpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import sys
import time

import filelock
import fsspec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
DEFAULT_CACHE_DIR = Path("/data1/outputs/openpi_cache")
CHECKPOINT_URL = "gs://openpi-assets/checkpoints/pi05_base/params"
CHECKPOINT_RELATIVE_PATH = Path("openpi-assets/checkpoints/pi05_base/params")

sys.path.insert(0, str(OPENPI_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=20, help="Number of checkpoint objects to download in parallel.")
    parser.add_argument("--force", action="store_true", help="Redownload even when a valid checkpoint exists.")
    return parser.parse_args()


def _download_file(fs, remote_path: str, local_path: Path, expected_size: int) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_name(f"{local_path.name}.download")
    if temporary_path.exists():
        temporary_path.unlink()
    fs.get_file(remote_path, str(temporary_path))
    actual_size = temporary_path.stat().st_size
    if actual_size != expected_size:
        temporary_path.unlink(missing_ok=True)
        raise IOError(f"Downloaded size mismatch for {remote_path}: {actual_size} != {expected_size}")
    temporary_path.replace(local_path)
    return actual_size


def _parallel_download(url: str, target_path: Path, *, workers: int, force: bool) -> Path:
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    scratch_path = target_path.with_suffix(".partial")
    if force:
        shutil.rmtree(target_path, ignore_errors=True)
        shutil.rmtree(scratch_path, ignore_errors=True)
    scratch_path.mkdir(parents=True, exist_ok=True)

    fs, remote_root = fsspec.core.url_to_fs(url)
    remote_entries = fs.find(remote_root, detail=True)
    remote_files = {
        remote_path: int(info["size"])
        for remote_path, info in remote_entries.items()
        if info.get("type") != "directory"
    }
    total_bytes = sum(remote_files.values())
    completed_bytes = 0
    pending: list[tuple[str, Path, int]] = []
    for remote_path, expected_size in remote_files.items():
        relative_path = Path(remote_path).relative_to(remote_root)
        local_path = scratch_path / relative_path
        if local_path.is_file() and local_path.stat().st_size == expected_size:
            completed_bytes += expected_size
        else:
            pending.append((remote_path, local_path, expected_size))

    print(
        f"checkpoint_objects={len(remote_files)} total_bytes={total_bytes} "
        f"already_complete_bytes={completed_bytes} workers={workers}",
        flush=True,
    )
    pending.sort(key=lambda item: item[2], reverse=True)
    started_at = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_file, fs, remote_path, local_path, expected_size): remote_path
            for remote_path, local_path, expected_size in pending
        }
        completed_objects = len(remote_files) - len(pending)
        remaining = set(futures)
        while remaining:
            done, remaining = concurrent.futures.wait(
                remaining,
                timeout=10,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                completed_bytes += future.result()
                completed_objects += 1

            in_progress_bytes = 0
            for future in remaining:
                remote_path = futures[future]
                expected_size = remote_files[remote_path]
                relative_path = Path(remote_path).relative_to(remote_root)
                local_path = scratch_path / relative_path
                temporary_path = local_path.with_name(f"{local_path.name}.download")
                if temporary_path.is_file():
                    in_progress_bytes += min(temporary_path.stat().st_size, expected_size)

            observed_bytes = completed_bytes + in_progress_bytes
            elapsed = max(time.monotonic() - started_at, 1e-6)
            print(
                f"download_progress={completed_objects}/{len(remote_files)} "
                f"bytes={observed_bytes}/{total_bytes} "
                f"percent={100.0 * observed_bytes / total_bytes:.2f}% "
                f"average_speed={observed_bytes / elapsed / 1024**2:.2f}MiB/s",
                flush=True,
            )

    for remote_path, expected_size in remote_files.items():
        relative_path = Path(remote_path).relative_to(remote_root)
        local_path = scratch_path / relative_path
        if not local_path.is_file() or local_path.stat().st_size != expected_size:
            raise IOError(f"Checkpoint object is incomplete: {local_path}")

    shutil.rmtree(target_path, ignore_errors=True)
    shutil.move(str(scratch_path), str(target_path))
    return target_path


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENPI_DATA_HOME"] = str(cache_dir)

    expected_path = cache_dir / CHECKPOINT_RELATIVE_PATH
    metadata_path = expected_path / "_METADATA"
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = expected_path.with_suffix(".lock")
    print(f"checkpoint_path={expected_path}", flush=True)
    print(f"acquiring_download_lock={lock_path}", flush=True)
    with filelock.FileLock(lock_path):
        print("download_lock_acquired", flush=True)
        if metadata_path.is_file() and not args.force:
            downloaded_path = expected_path
        else:
            print("Downloading with parallel gcsfs; system gsutil is not used", flush=True)
            downloaded_path = _parallel_download(
                CHECKPOINT_URL,
                expected_path,
                workers=args.workers,
                force=args.force,
            )

    if downloaded_path != expected_path:
        raise RuntimeError(f"Unexpected checkpoint path: {downloaded_path}; expected {expected_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint download is incomplete; missing {metadata_path}")

    print(f"checkpoint_path={downloaded_path}")
    print(f"metadata_path={metadata_path}")
    print("pi05 base checkpoint is complete")


if __name__ == "__main__":
    main()
