from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts import inspect_tactile_captioner_attempt as inspection
from tactile_vla.common.labels import LABEL_FIELDS
from tactile_vla.common.labels import LABEL_MAPS
from tactile_vla.common.labels import label_id_to_name
from tactile_vla.common.labels import labels_to_caption


class FakePredictor:
    window_size = 3
    device = "cpu"

    def __init__(self) -> None:
        self.windows: list[list[int]] = []

    def predict(self, mesh_motion: np.ndarray, force: np.ndarray) -> SimpleNamespace:
        del force
        frame_values = [int(value) for value in mesh_motion[:, 0, 0, 0]]
        self.windows.append(frame_values)
        end_frame = frame_values[-1]
        label_ids = {
            field: end_frame % len(LABEL_MAPS[field])
            for field in LABEL_FIELDS
        }
        label_names = {
            field: label_id_to_name(field, label_id)
            for field, label_id in label_ids.items()
        }
        probabilities = {}
        for field, label_id in label_ids.items():
            num_classes = len(LABEL_MAPS[field])
            remaining = 0.1 / (num_classes - 1)
            values = [remaining] * num_classes
            values[label_id] = 0.9
            probabilities[field] = values
        return SimpleNamespace(
            caption=labels_to_caption(label_names),
            label_ids=label_ids,
            label_names=label_names,
            probabilities=probabilities,
        )


def make_attempt_data() -> inspection.AttemptData:
    frames = 5
    mesh_motion = np.empty((frames, 35, 20, 12), dtype=np.float32)
    force = np.empty((frames, 35, 20, 6), dtype=np.float32)
    for index in range(frames):
        mesh_motion[index].fill(index)
        force[index].fill(index)
    return inspection.AttemptData(
        hdf5_path=Path("/dataset/episode7/attempt2/data.hdf5"),
        episode_id=7,
        attempt_id=2,
        result="failure",
        timestamps=np.asarray([10.0, 11.0, 12.0, 13.0, 14.0]),
        mesh_motion=mesh_motion,
        force=force,
        shift_timestamp=12.6,
    )


def write_hdf5(path: Path, *, bad_force_shape: bool = False) -> None:
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as root:
        root.create_dataset("timestamp", data=np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
        root.create_dataset("tactile/mesh_motion", data=np.zeros((3, 35, 20, 12), dtype=np.float32))
        force_shape = (2, 35, 20, 6) if bad_force_shape else (3, 35, 20, 6)
        root.create_dataset("tactile/force_concat", data=np.zeros(force_shape, dtype=np.float32))
        root.create_dataset("meta/episode_id", data=7)
        root.create_dataset("meta/attempt_id", data=2)
        root.create_dataset("meta/result", data="failure", dtype=h5py.string_dtype("utf-8"))
        root.create_dataset("meta/shift_timestamp", data=2.4)


def test_report_uses_trailing_windows_and_locates_shift() -> None:
    predictor = FakePredictor()
    report = inspection.build_report(
        make_attempt_data(),
        predictor,
        low_confidence_threshold=0.8,
    )

    timeline = inspection.predict_timeline(make_attempt_data(), FakePredictor())
    assert [entry["warmup"] for entry in timeline] == [True, True, False, False, False]
    assert predictor.windows == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert timeline[0]["probabilities"] == {field: None for field in LABEL_FIELDS}
    assert timeline[2]["relative_time_sec"] == pytest.approx(2.0)
    assert [entry["shift_relation"] for entry in timeline] == ["before"] * 3 + ["after"] * 2

    shift = report["shift_position"]
    assert shift["nearest_frame_index"] == 3
    assert shift["nearest_frame_timestamp"] == pytest.approx(13.0)
    assert shift["nearest_offset_sec"] == pytest.approx(0.4)
    assert shift["first_at_or_after_frame_index"] == 3

    statistics = report["statistics"]
    assert statistics["total_frames"] == 5
    assert statistics["warmup_frames"] == 2
    assert statistics["model_prediction_frames"] == 3
    assert statistics["all_predictions"]["num_model_predictions"] == 3
    assert statistics["before_shift"]["num_model_predictions"] == 1
    assert statistics["after_shift"]["num_model_predictions"] == 2
    assert statistics["all_predictions"]["confidence"]["area"]["low_confidence_count"] == 0
    assert set(report) == {"statistics", "segments", "shift_position"}


def test_load_attempt_data_validates_hdf5(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "episode7/attempt2/data.hdf5"
    write_hdf5(hdf5_path)

    data = inspection.load_attempt_data(tmp_path, 7, 2)

    assert data.episode_id == 7
    assert data.attempt_id == 2
    assert data.result == "failure"
    assert data.shift_timestamp == pytest.approx(2.4)
    assert data.mesh_motion.shape == (3, 35, 20, 12)
    assert data.force.shape == (3, 35, 20, 6)


def test_load_attempt_data_rejects_incomplete_hdf5(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "episode7/attempt2/data.hdf5"
    write_hdf5(hdf5_path, bad_force_shape=True)

    with pytest.raises(ValueError, match="force_concat must have shape"):
        inspection.load_attempt_data(tmp_path, 7, 2)


def test_write_report_requires_explicit_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    inspection.write_report({"value": 1}, destination, overwrite=False)
    assert json.loads(destination.read_text()) == {"value": 1}

    with pytest.raises(FileExistsError, match="--overwrite"):
        inspection.write_report({"value": 2}, destination, overwrite=False)

    inspection.write_report({"value": 3}, destination, overwrite=True)
    assert json.loads(destination.read_text()) == {"value": 3}
