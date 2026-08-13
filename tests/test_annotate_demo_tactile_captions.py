from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts import annotate_demo_tactile_captions as annotation
from tactile_vla.common.labels import DEFAULT_TACTILE_CAPTION
from tactile_vla.common.labels import LABEL_FIELDS
from tactile_vla.common.labels import labels_to_caption


class FakeBatchPredictor:
    window_size = 3

    def __init__(self) -> None:
        self.batches: list[list[list[int]]] = []

    def predict_batch(self, mesh_motion: np.ndarray, force: np.ndarray) -> list[SimpleNamespace]:
        del force
        batch_windows = [
            [int(value) for value in window[:, 0, 0, 0]]
            for window in mesh_motion
        ]
        self.batches.append(batch_windows)
        predictions = []
        for window in batch_windows:
            rotation = "clockwise" if window[-1] % 2 else "counterclockwise"
            labels = {
                "area": "medium",
                "fx_state": "near_zero",
                "fy_state": "near_zero",
                "fz_state": "negative",
                "rotation": rotation,
            }
            predictions.append(
                SimpleNamespace(
                    caption=labels_to_caption(labels),
                    label_names=labels,
                )
            )
        return predictions


def make_tactile_data(frame_count: int = 6) -> annotation.AttemptTactileData:
    mesh = np.empty((frame_count, 35, 20, 12), dtype=np.float32)
    force = np.empty((frame_count, 35, 20, 6), dtype=np.float32)
    for index in range(frame_count):
        mesh[index].fill(index)
        force[index].fill(index)
    return annotation.AttemptTactileData(
        timestamps=np.arange(frame_count, dtype=np.float64),
        mesh_motion=mesh,
        force=force,
    )


def write_attempt_hdf5(path: Path, *, frame_count: int = 4) -> None:
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as root:
        root.create_dataset("timestamp", data=np.arange(frame_count, dtype=np.float64))
        root.create_dataset(
            "tactile/mesh_motion",
            data=np.zeros((frame_count, 35, 20, 12), dtype=np.float32),
        )
        root.create_dataset(
            "tactile/force_concat",
            data=np.zeros((frame_count, 35, 20, 6), dtype=np.float32),
        )


def test_predict_frame_captions_uses_trailing_windows_and_batches() -> None:
    predictor = FakeBatchPredictor()
    captions, counts = annotation.predict_frame_captions(
        make_tactile_data(),
        predictor,
        batch_size=2,
    )

    assert captions[:2] == [DEFAULT_TACTILE_CAPTION, DEFAULT_TACTILE_CAPTION]
    assert predictor.batches == [[[0, 1, 2], [1, 2, 3]], [[2, 3, 4], [3, 4, 5]]]
    assert len(captions) == 6
    assert counts["area"] == {"none": 2, "medium": 4}
    assert counts["rotation"] == {"none": 2, "clockwise": 2, "counterclockwise": 2}


def test_discover_and_load_attempts(tmp_path: Path) -> None:
    write_attempt_hdf5(tmp_path / "episode2/attempt2/data.hdf5")
    write_attempt_hdf5(tmp_path / "episode2/attempt1/data.hdf5")
    write_attempt_hdf5(tmp_path / "episode10/attempt1/data.hdf5")

    attempts = annotation.discover_attempts(
        tmp_path,
        episode_start=2,
        episode_end=2,
        attempt_ids={2},
    )

    assert [(item.episode_id, item.attempt_id) for item in attempts] == [(2, 2)]
    data = annotation.load_attempt_tactile_data(attempts[0])
    assert data.mesh_motion.shape == (4, 35, 20, 12)
    assert data.force.shape == (4, 35, 20, 6)


def test_label_payload_preserves_unrelated_overrides(tmp_path: Path) -> None:
    data = make_tactile_data(frame_count=3)
    captions = [DEFAULT_TACTILE_CAPTION] * 3
    counts = {field: annotation.Counter() for field in LABEL_FIELDS}
    payload = annotation.make_label_payload(
        existing={"recovery_plan": ["keep-me"] * 3},
        captions=captions,
        data=data,
        checkpoint=tmp_path / "best.pt",
        window_size=3,
        field_counts=counts,
    )

    assert payload["recovery_plan"] == ["keep-me"] * 3
    assert payload["tactile_caption"] == captions
    assert payload["_tactile_caption_annotation"]["num_frames"] == 3

    destination = tmp_path / "labels.json"
    annotation._atomic_write_json(destination, payload)
    assert json.loads(destination.read_text(encoding="utf-8"))["tactile_caption"] == captions


def test_preflight_requires_explicit_existing_file_policy(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "episode1/attempt1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "labels.json").write_text("{}", encoding="utf-8")
    attempts = [annotation.AttemptRef(episode_id=1, attempt_id=1, attempt_dir=attempt_dir)]

    with pytest.raises(FileExistsError, match="--skip-existing"):
        annotation._preflight_destinations(
            attempts,
            label_file_name="labels.json",
            overwrite=False,
            skip_existing=False,
        )
    annotation._preflight_destinations(
        attempts,
        label_file_name="labels.json",
        overwrite=True,
        skip_existing=False,
    )
