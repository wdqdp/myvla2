from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import import_raw_episodes


def _write_attempt(root: Path, episode_id: int, attempt_id: int) -> None:
    attempt = root / f"episode{episode_id}" / f"attempt{attempt_id}"
    (attempt / "camera/color/front").mkdir(parents=True)
    (attempt / "camera/color/left").mkdir(parents=True)
    for camera in ("front", "left"):
        for frame in range(2):
            (attempt / f"camera/color/{camera}/{frame}.jpg").write_bytes(f"{camera}-{frame}".encode())

    (attempt / "meta.json").write_text(
        json.dumps({"episode_id": episode_id, "attempt_id": attempt_id, "valid": True}) + "\n"
    )
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(attempt / "data.hdf5", "w") as episode:
        episode.create_dataset("size", data=2)
        episode.create_dataset("timestamp", data=np.asarray([1.0, 2.0]))
        episode.create_dataset("meta/episode_id", data=episode_id)
        episode.create_dataset("meta/attempt_id", data=attempt_id)
        episode.create_dataset("meta/case_id", data="case_001", dtype=string_dtype)
        episode.create_dataset("meta/instruction", data="test", dtype=string_dtype)
        episode.create_dataset("meta/valid", data=True)
        episode.create_dataset("reasoning/has_sample", data=False)
        episode.create_dataset("arm/jointStatePosition/masterRight", data=np.zeros((2, 7)))
        episode.create_dataset("arm/jointStatePosition/puppetRight", data=np.zeros((2, 7)))
        for camera in ("front", "left"):
            episode.create_dataset(
                f"camera/color/{camera}",
                data=[f"camera/color/{camera}/0.jpg", f"camera/color/{camera}/1.jpg"],
                dtype=string_dtype,
            )
        episode.create_dataset("label/rotation_state_id", data=np.zeros(2, dtype=np.int64))
        episode.create_dataset("label/need_recovery", data=np.zeros(2, dtype=np.bool_))
        for name in (
            "input_recovery_plan",
            "failure_recovery_memory",
            "rotation_state_name",
            "tactile_caption",
            "failure_reason",
            "recovery_plan",
        ):
            episode.create_dataset(f"label/{name}", data=["", ""], dtype=string_dtype)


def _args(source: Path, target: Path, *, dry_run: bool = False, resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        source_dir=source,
        target_dir=target,
        start_episode_id=121,
        verify_images="all",
        checksum=True,
        resume=resume,
        dry_run=dry_run,
        manifest=None,
    )


def test_import_renumbers_validates_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_attempt(source, episode_id=1, attempt_id=1)
    _write_attempt(source, episode_id=2, attempt_id=1)

    assert import_raw_episodes.run(_args(source, target, dry_run=True)) is None
    assert not target.exists()

    manifest_path = import_raw_episodes.run(_args(source, target))
    assert manifest_path == target / "import_manifest_episode121_122.json"
    assert json.loads((target / "episode121/attempt1/meta.json").read_text())["episode_id"] == 121
    assert json.loads((target / "episode122/attempt1/meta.json").read_text())["episode_id"] == 122
    with h5py.File(target / "episode121/attempt1/data.hdf5", "r") as episode:
        assert int(episode["meta/episode_id"][()]) == 121
    assert json.loads(manifest_path.read_text())["status"] == "completed"

    with pytest.raises(FileExistsError):
        import_raw_episodes.run(_args(source, target))

    resumed_manifest = import_raw_episodes.run(_args(source, target, resume=True))
    payload = json.loads(resumed_manifest.read_text())
    assert payload["status"] == "completed"
    assert {item["status"] for item in payload["mappings"]} == {"validated_existing"}
