from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tactile_vla.captioner.model import TactileCaptioner
from tactile_vla.captioner.predictor import TactileCaptionerPredictor
from tactile_vla.captioner.training import compute_class_weights
from tactile_vla.common.labels import DEFAULT_TACTILE_CAPTION
from tactile_vla.common.labels import LABEL_FIELDS
from tactile_vla.common.labels import LABEL_MAPS
from tactile_vla.common.labels import LABEL_SCHEMA_VERSION
from tactile_vla.common.labels import labels_to_caption
from tactile_vla.data.tactile_captioner_dataset import TactileCaptionerDataset
from tactile_vla.data.tactile_captioner_dataset import label_counts


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_dataset(root: Path, *, window_size: int = 3) -> None:
    (root / "indices").mkdir(parents=True)
    (root / "shards").mkdir()
    meta = {
        "dataset_format": "tactile_captioner_shards_v2",
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "window_size": window_size,
        "label_maps": LABEL_MAPS,
        "mesh_motion_shape": [6, 5, 12],
        "force_shape": [6, 5, 6],
        "sequences": [
            {
                "sequence_index": 0,
                "episode_id": 7,
                "target_class": "rotation:clockwise",
                "shard": "shard_000.hdf5",
                "group": "/sequences/000000",
            }
        ],
    }
    write_json(root / "meta.json", meta)
    write_json(
        root / "norm_stats.json",
        {
            "mesh_motion": {"mean": [0.0] * 12, "std": [2.0] * 12},
            "force": {"mean": [0.0] * 6, "std": [4.0] * 6},
        },
    )
    with h5py.File(root / "shards" / "shard_000.hdf5", "w") as shard:
        group = shard.create_group("sequences/000000")
        group.create_dataset("mesh_motion", data=np.full((5, 6, 5, 12), 2.0, dtype=np.float32))
        group.create_dataset("force", data=np.full((5, 6, 5, 6), 4.0, dtype=np.float32))
        group.create_dataset("timestamp", data=np.arange(5, dtype=np.float64))
    index = {
        "sequence_index": np.asarray([0, 0], dtype=np.int64),
        "end_frame_index": np.asarray([2, 4], dtype=np.int64),
        "area_label": np.asarray([0, 2], dtype=np.int64),
        "fx_state_label": np.asarray([1, 0], dtype=np.int64),
        "fy_state_label": np.asarray([1, 2], dtype=np.int64),
        "fz_state_label": np.asarray([1, 0], dtype=np.int64),
        "rotation_label": np.asarray([0, 1], dtype=np.int64),
    }
    for split in ("train", "train_balanced", "val", "test"):
        np.savez_compressed(root / "indices" / f"{split}.npz", **index)


def test_dataset_reads_v3_multilabel_shards(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    dataset = TactileCaptionerDataset(tmp_path, split="train", balanced=True)

    item = dataset[1]
    assert item["mesh_motion"].shape == (3, 12, 6, 5)
    assert item["force"].shape == (3, 6, 6, 5)
    assert torch.allclose(item["mesh_motion"], torch.ones_like(item["mesh_motion"]))
    assert torch.allclose(item["force"], torch.ones_like(item["force"]))
    assert {field: int(item["labels"][field]) for field in LABEL_FIELDS} == {
        "area": 2,
        "fx_state": 0,
        "fy_state": 2,
        "fz_state": 0,
        "rotation": 1,
    }
    assert label_counts(dataset)["rotation"] == {0: 1, 1: 1, 2: 0}
    dataset.close()


def test_multihead_model_output_shapes_and_backward() -> None:
    model = TactileCaptioner(frame_feature_dim=8, temporal_hidden_dim=12, temporal_dilations=(1,), dropout=0.0)
    logits = model(
        torch.randn(2, 4, 12, 8, 8),
        torch.randn(2, 4, 6, 8, 8),
    )

    assert tuple(logits) == LABEL_FIELDS
    for field in LABEL_FIELDS:
        assert logits[field].shape == (2, len(LABEL_MAPS[field]))
    sum(value.mean() for value in logits.values()).backward()
    assert model.classifiers["rotation"].weight.grad is not None


def test_sqrt_inverse_weights_are_normalized() -> None:
    weights = compute_class_weights({0: 100, 1: 25, 2: 4}, num_classes=3)
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] < weights[1] < weights[2]
    assert compute_class_weights({0: 1, 1: 2}, num_classes=2, mode="none").tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="needs samples"):
        compute_class_weights({0: 3, 1: 0}, num_classes=2)


def test_structured_caption_format() -> None:
    assert DEFAULT_TACTILE_CAPTION == (
        "Touch[area=none; Fx=near_zero; Fy=near_zero; Fz=near_zero; rotation=none]"
    )
    assert labels_to_caption(
        {
            "area": "small",
            "fx_state": "negative",
            "fy_state": "near_zero",
            "fz_state": "negative",
            "rotation": "clockwise",
        }
    ) == "Touch[area=small; Fx=negative; Fy=near_zero; Fz=negative; rotation=clockwise]"


def test_predictor_outputs_all_heads_and_checks_window_size(tmp_path: Path) -> None:
    model = TactileCaptioner(frame_feature_dim=8, temporal_hidden_dim=12, temporal_dilations=(1,), dropout=0.0)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config_dict(),
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_maps": LABEL_MAPS,
        "dataset_meta": {"window_size": 3},
        "normalization": {
            "mesh_motion_mean": [0.0] * 12,
            "mesh_motion_std": [1.0] * 12,
            "force_mean": [0.0] * 6,
            "force_std": [1.0] * 6,
        },
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    predictor = TactileCaptionerPredictor(checkpoint_path, device="cpu")

    prediction = predictor.predict(
        np.zeros((3, 8, 8, 12), dtype=np.float32),
        np.zeros((3, 8, 8, 6), dtype=np.float32),
    )
    assert tuple(prediction.label_names) == LABEL_FIELDS
    assert tuple(prediction.probabilities) == LABEL_FIELDS
    assert prediction.caption.startswith("Touch[area=")
    batch_predictions = predictor.predict_batch(
        np.zeros((2, 3, 8, 8, 12), dtype=np.float32),
        np.zeros((2, 3, 8, 8, 6), dtype=np.float32),
    )
    assert len(batch_predictions) == 2
    for batch_prediction in batch_predictions:
        assert batch_prediction.label_ids == prediction.label_ids
        assert batch_prediction.label_names == prediction.label_names
        assert batch_prediction.caption == prediction.caption
        for field in LABEL_FIELDS:
            assert batch_prediction.probabilities[field] == pytest.approx(
                prediction.probabilities[field],
                abs=1e-6,
            )
    with pytest.raises(ValueError, match="window_size=3"):
        predictor.predict(
            np.zeros((4, 8, 8, 12), dtype=np.float32),
            np.zeros((4, 8, 8, 6), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="window_size=3"):
        predictor.predict_batch(
            np.zeros((2, 4, 8, 8, 12), dtype=np.float32),
            np.zeros((2, 4, 8, 8, 6), dtype=np.float32),
        )
