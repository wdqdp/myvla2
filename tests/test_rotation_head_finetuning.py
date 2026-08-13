from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tactile_vla.captioner.model import TactileCaptioner
from tactile_vla.captioner.rotation_finetuning import BalancedRotationSampler
from tactile_vla.captioner.rotation_finetuning import DemoRotationDataset
from tactile_vla.captioner.rotation_finetuning import DemoSequence
from tactile_vla.captioner.rotation_finetuning import assert_only_rotation_head_changed
from tactile_vla.captioner.rotation_finetuning import build_demo_windows
from tactile_vla.captioner.rotation_finetuning import freeze_for_rotation_head
from tactile_vla.captioner.rotation_finetuning import rotation_counts
from tactile_vla.captioner.rotation_finetuning import run_rotation_epoch
from tactile_vla.captioner.rotation_finetuning import stratified_episode_split
from tactile_vla.common.labels import LABEL_MAPS


def write_demo_hdf5(path: Path, timestamps: np.ndarray) -> None:
    path.parent.mkdir(parents=True)
    frames = len(timestamps)
    with h5py.File(path, "w") as root:
        root.create_dataset("timestamp", data=timestamps)
        root.create_dataset("tactile/mesh_motion", data=np.ones((frames, 35, 20, 12), dtype=np.float32))
        root.create_dataset("tactile/force_concat", data=np.full((frames, 35, 20, 6), 2.0, dtype=np.float32))


def sequence(
    path: Path,
    *,
    episode_id: int,
    direction: str,
    shift_timestamp: float = 3.0,
    num_frames: int = 6,
) -> DemoSequence:
    rotation_name = "clockwise" if direction in {"right", "front"} else "counterclockwise"
    return DemoSequence(
        hdf5_path=path,
        episode_id=episode_id,
        physical_direction=direction,
        rotation_label=LABEL_MAPS["rotation"][rotation_name],
        shift_timestamp=shift_timestamp,
        num_frames=num_frames,
    )


def test_window_labels_use_end_timestamp_and_keep_boundary_windows(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "episode41/attempt1/data.hdf5"
    write_demo_hdf5(hdf5_path, np.arange(6, dtype=np.float64))
    record = sequence(hdf5_path, episode_id=41, direction="right")

    windows = build_demo_windows([record], window_size=3)

    assert [window.end_frame_index for window in windows] == [2, 3, 4, 5]
    assert [window.rotation_label for window in windows] == [0, 1, 1, 1]
    assert rotation_counts(windows) == {"none": 1, "clockwise": 3, "counterclockwise": 0}

    dataset = DemoRotationDataset([record], windows, window_size=3)
    item = dataset[1]
    assert item["mesh_motion"].shape == (3, 12, 35, 20)
    assert item["force"].shape == (3, 6, 35, 20)
    assert int(item["rotation_label"]) == 1
    dataset.close()


def test_stratified_split_is_16_2_2_per_direction_and_holds_out_episode75() -> None:
    records = []
    for direction, start in (("right", 41), ("left", 61), ("front", 81), ("back", 101)):
        records.extend(
            sequence(Path(f"/episode{episode_id}.hdf5"), episode_id=episode_id, direction=direction)
            for episode_id in range(start, start + 20)
        )

    splits = stratified_episode_split(records, seed=42, forced_test_episodes=(75,))

    assert {split: len(values) for split, values in splits.items()} == {"train": 64, "val": 8, "test": 8}
    for direction in ("right", "left", "front", "back"):
        assert sum(record.physical_direction == direction for record in splits["train"]) == 16
        assert sum(record.physical_direction == direction for record in splits["val"]) == 2
        assert sum(record.physical_direction == direction for record in splits["test"]) == 2
    assert 75 in {record.episode_id for record in splits["test"]}
    episode_sets = [{record.episode_id for record in splits[split]} for split in ("train", "val", "test")]
    assert not episode_sets[0] & episode_sets[1]
    assert not episode_sets[0] & episode_sets[2]
    assert not episode_sets[1] & episode_sets[2]


def test_balanced_sampler_has_exact_counts_and_changes_across_epochs() -> None:
    labels = [0] * 8 + [1] * 5 + [2] * 3
    sampler = BalancedRotationSampler(labels, seed=7)

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert len(first) == 9
    assert {label: sum(labels[index] == label for index in first) for label in range(3)} == {0: 3, 1: 3, 2: 3}
    assert first != second
    with pytest.raises(ValueError, match="classes have no samples"):
        BalancedRotationSampler([0, 0, 1])


class TinyRotationDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "mesh_motion": torch.full((3, 12, 8, 8), float(index)),
            "force": torch.full((3, 6, 8, 8), float(index)),
            "rotation_label": torch.tensor(index, dtype=torch.long),
        }


def test_training_step_changes_only_rotation_head_and_preserves_batchnorm() -> None:
    model = TactileCaptioner(
        frame_feature_dim=8,
        temporal_hidden_dim=12,
        temporal_dilations=(1,),
        dropout=0.0,
    )
    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trainable = freeze_for_rotation_head(model)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.1)
    normalization = {
        "mesh_motion_mean": torch.zeros(12),
        "mesh_motion_std": torch.ones(12),
        "force_mean": torch.zeros(6),
        "force_std": torch.ones(6),
    }

    metrics = run_rotation_epoch(
        model,
        DataLoader(TinyRotationDataset(), batch_size=3),
        device=torch.device("cpu"),
        normalization=normalization,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        desc="test",
    )

    assert set(trainable) == {"classifiers.rotation.weight", "classifiers.rotation.bias"}
    assert metrics["num_samples"] == 3
    assert_only_rotation_head_changed(base_state, model.state_dict())
    assert not torch.equal(base_state["classifiers.rotation.weight"], model.state_dict()["classifiers.rotation.weight"])
