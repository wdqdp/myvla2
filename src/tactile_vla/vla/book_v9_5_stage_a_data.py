"""Book V9.5 Stage A: current data only, never the history9_4_5 archive."""

from __future__ import annotations

import json
from pathlib import Path

from tactile_vla.vla import book_stage_a_data as common
from tactile_vla.vla.artifacts import sha256_json

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book")
VERSION_TAG = "book_v9_5"
RUN_NAME = "pi05_delta_tac_book_stage_a_v9_5_no_history"
ARCHIVE_NAME = "history9_4_5"
DATA_PROFILE = common.DATA_PROFILE
EXPERIMENT_KIND = common.EXPERIMENT_KIND
SCOPE_POLICY = {
    "schema_version": "book_v9_5_stage_a_source_scope_v1",
    "selection": "explicit_current_lerobot_and_v4_sources_not_recursive_book_scan",
    "excluded_directory": ARCHIVE_NAME,
    "archive_reference": "reject_including_symlinks",
    "phase_and_targets": "unchanged_book_stage_a_v1_native_R_raw_contiguous_H30",
}


def reject_archive(path: Path) -> Path:
    path = path.expanduser()
    resolved = path.resolve()
    if ARCHIVE_NAME in path.absolute().parts or ARCHIVE_NAME in resolved.parts:
        raise ValueError(f"V9.5 must not use {ARCHIVE_NAME}, including symlinks: {path}")
    return resolved


def current_path(path: Path, book_root: Path) -> Path:
    resolved = reject_archive(path)
    if not resolved.is_relative_to(book_root):
        raise ValueError(f"V9.5 source is outside the current book root: {path}")
    return resolved


def validate_source_scope(*, book_root: Path, dataset_dir: Path, v4_index_file: Path, norm_stats_dir: Path) -> dict:
    book_root = reject_archive(book_root)
    dataset_dir = current_path(dataset_dir, book_root)
    v4_index_file = current_path(v4_index_file, book_root)
    norm_stats_dir = current_path(norm_stats_dir, book_root)
    if not dataset_dir.is_relative_to(book_root / "lerobot_data"):
        raise ValueError("V9.5 requires the current book/lerobot_data dataset")
    if not v4_index_file.is_relative_to(book_root / "outputs") or not norm_stats_dir.is_relative_to(
        book_root / "outputs"
    ):
        raise ValueError("V9.5 requires current book/outputs V4 index and norm stats")
    v4_index = json.loads(v4_index_file.read_text())
    if current_path(Path(v4_index["dataset_dir"]), book_root) != dataset_dir:
        raise ValueError("V9.5 V4 index declares a different dataset directory")
    for name, source in v4_index["source_files"].items():
        if name == "lerobot_parquet":
            for relative in source:
                current_path(dataset_dir / relative, book_root)
        else:
            current_path(Path(source["path"]), book_root)
    # Inspect only the explicitly selected dataset, including image/video files.
    # Do not recurse through book or merge archives that reuse episode IDs.
    for path in dataset_dir.rglob("*"):
        current_path(path, book_root)
    profile_path = Path(v4_index["source_files"]["profile"]["path"])
    profile = json.loads(profile_path.read_text())
    for name in ("hdf5_data_dir", "raw_data_dir"):
        if profile.get(name):
            current_path(Path(profile[name]), book_root)
    hdf5_root = current_path(Path(profile["hdf5_data_dir"]), book_root)
    for attempt in profile["attempts"]:
        current_path(hdf5_root / attempt["hdf5_path"], book_root)
    return {
        "policy": SCOPE_POLICY,
        "book_root": str(book_root),
        "excluded_directory": str(book_root / ARCHIVE_NAME),
        "dataset_dir": str(dataset_dir),
        "v4_index_file": str(v4_index_file),
        "norm_stats_dir": str(norm_stats_dir),
    }


def build_artifacts(*, book_root: Path, dataset_dir: Path, v4_index_file: Path, norm_stats_dir: Path):
    scope = validate_source_scope(
        book_root=book_root,
        dataset_dir=dataset_dir,
        v4_index_file=v4_index_file,
        norm_stats_dir=norm_stats_dir,
    )
    rows, index, summary = common.build_artifacts(
        dataset_dir=dataset_dir,
        v4_index_file=v4_index_file,
        norm_stats_dir=norm_stats_dir,
    )
    # Preserve the common schema/summary contract for existing Stage A consumers.
    # The extra version/scope fields participate in the final training_data_hash.
    index.update(experiment_version=VERSION_TAG, source_scope=scope)
    summary = summary | {"experiment_version": VERSION_TAG, "source_scope": scope}
    return rows, index, summary


def validate_training_index(payload, *, index_path: Path, dataset_dir: Path, book_root: Path):
    if payload.get("experiment_version") != VERSION_TAG:
        raise ValueError("V9.5 needs an independently rebuilt V9.5 Stage A index")
    reject_archive(index_path)
    for source in payload["source_files"].values():
        reject_archive(Path(source["path"]))
    sources = payload["source_files"]
    expected_scope = validate_source_scope(
        book_root=book_root,
        dataset_dir=dataset_dir,
        v4_index_file=Path(sources["v4_training_index"]["path"]),
        norm_stats_dir=Path(sources["norm_stats"]["path"]).parent,
    )
    if payload.get("source_scope") != expected_scope:
        raise ValueError("V9.5 source scope differs from the current book inputs")
    if payload.get("training_data_hash") != sha256_json(
        {k: v for k, v in payload.items() if k != "training_data_hash"}
    ):
        raise ValueError("V9.5 training_data_hash mismatch")
    return common.validate_training_index(payload, index_path=index_path, dataset_dir=dataset_dir)
