from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from tactile_vla.captioner.model import TactileCaptioner
from tactile_vla.captioner.predictor import TactileCaptionerPredictor
from tactile_vla.captioner.training import _make_criteria, _masked_head_losses, _run_epoch
from tactile_vla.common.labels_v4 import DATASET_FORMAT, LABEL_FIELDS, LABEL_MAPS, LABEL_SCHEMA_VERSION
from tactile_vla.common.labels_v4 import labels_to_caption
from tactile_vla.data.tactile_captioner_dataset import TactileCaptionerDataset, label_counts


def make_masked_dataset(root: Path) -> None:
    (root / 'indices').mkdir()
    (root / 'shards').mkdir()
    targets = [(field, name, label_id) for field, mapping in LABEL_MAPS.items() for name, label_id in mapping.items()]
    records = []
    with h5py.File(root / 'shards/shard_000.hdf5', 'w') as shard:
        for idx, (field, name, _) in enumerate(targets):
            record = {
                'sequence_index': idx, 'episode_id': idx + 1, 'label_task': field, 'target_label': name,
                'target_class': f'{field}:{name}', 'num_frames': 3,
                'shard': 'shard_000.hdf5', 'group': f'/sequences/{idx:06d}',
            }
            records.append(record)
            group = shard.create_group(record['group'])
            group.create_dataset('mesh_motion', data=np.ones((3, 8, 8, 12), dtype=np.float32))
            group.create_dataset('force', data=np.ones((3, 8, 8, 6), dtype=np.float32))
            group.create_dataset('timestamp', data=np.arange(3) / 30)
    (root / 'meta.json').write_text(json.dumps({
        'dataset_format': DATASET_FORMAT, 'label_schema_version': LABEL_SCHEMA_VERSION,
        'label_policy': 'episode_constant', 'window_size': 3, 'label_maps': LABEL_MAPS, 'sequences': records,
    }))
    (root / 'norm_stats.json').write_text(json.dumps({
        'mesh_motion': {'mean': [0.] * 12, 'std': [2.] * 12},
        'force': {'mean': [0.] * 6, 'std': [4.] * 6},
    }))
    index = {'sequence_index': np.arange(len(targets)), 'end_frame_index': np.full(len(targets), 2)}
    for field in LABEL_FIELDS:
        index[f'{field}_mask'] = np.asarray([task == field for task, _, _ in targets], dtype=np.bool_)
        index[f'{field}_label'] = np.asarray([value if task == field else -1 for task, _, value in targets])
    for split in ('train', 'train_balanced', 'val', 'test'):
        np.savez(root / 'indices' / f'{split}.npz', **index)


def test_masked_loader_counts_and_batches_all_18_classes(tmp_path: Path) -> None:
    make_masked_dataset(tmp_path)
    dataset = TactileCaptionerDataset(tmp_path, balanced=True, include_metadata=True)
    assert dataset.head_num_classes == {'area': 4, 'fx_state': 3, 'fy_state': 3, 'fz_state': 2, 'fz_bias': 3, 'rotation': 3}
    assert label_counts(dataset) == {field: {idx: 1 for idx in mapping.values()} for field, mapping in LABEL_MAPS.items()}
    batch = next(iter(DataLoader(dataset, batch_size=18)))
    assert batch['mesh_motion'].shape == (18, 3, 12, 8, 8)
    assert torch.all(batch['mesh_motion'] == .5)
    assert torch.all(batch['force'] == .25)
    assert torch.stack(list(batch['masks'].values())).sum(dim=0).tolist() == [1] * 18
    bias = dataset[12]
    assert bool(bias['masks']['fz_bias']) and int(bias['labels']['fz_bias']) == 0
    assert bias['target_class'] == 'fz_bias:left'
    dataset.close()


@pytest.mark.parametrize('corruption', ['missing_mask', 'no_active', 'two_active', 'inactive_id', 'active_id', 'changed_target'])
def test_loader_rejects_incorrect_supervision(tmp_path: Path, corruption: str) -> None:
    make_masked_dataset(tmp_path)
    path = tmp_path / 'indices/train.npz'
    with np.load(path) as source:
        index = {key: source[key] for key in source.files}
    if corruption == 'missing_mask':
        del index['area_mask']
    elif corruption == 'no_active':
        index['area_mask'][0] = False
        index['area_label'][0] = -1
    elif corruption == 'two_active':
        index['fx_state_mask'][0] = True
        index['fx_state_label'][0] = 1
    elif corruption == 'inactive_id':
        index['fx_state_label'][0] = 1
    elif corruption == 'active_id':
        index['area_label'][0] = 99
    else:
        index['area_label'][0] = 1
    np.savez(path, **index)
    with pytest.raises(ValueError):
        TactileCaptionerDataset(tmp_path)


def test_masked_loss_excludes_inactive_targets_and_gradients() -> None:
    logits = {field: torch.randn(2, len(mapping), requires_grad=True) for field, mapping in LABEL_MAPS.items()}
    labels = {field: torch.full((2,), -1) for field in LABEL_FIELDS}
    masks = {field: torch.zeros(2, dtype=torch.bool) for field in LABEL_FIELDS}
    masks['area'][0] = True
    masks['rotation'][1] = True
    labels['area'][0] = 3
    labels['rotation'][1] = 2
    criteria = _make_criteria({field: torch.ones(len(mapping)) for field, mapping in LABEL_MAPS.items()}, label_smoothing=0)
    losses = _masked_head_losses(logits, labels, masks, criteria)
    actual = torch.cat(list(losses.values())).mean()
    expected = (nn.functional.cross_entropy(logits['area'][:1], labels['area'][:1]) +
                nn.functional.cross_entropy(logits['rotation'][1:], labels['rotation'][1:])) / 2
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits['fz_bias'].grad is None
    assert torch.count_nonzero(logits['area'].grad[1]) == 0
    assert torch.count_nonzero(logits['rotation'].grad[0]) == 0
    assert torch.count_nonzero(logits['area'].grad[0]) > 0


def test_epoch_reports_only_supervised_samples_and_keeps_inactive_heads() -> None:
    class FixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head_num_classes = {field: len(mapping) for field, mapping in LABEL_MAPS.items()}
            self.values = nn.ParameterDict({field: nn.Parameter(torch.tensor([3.] + [0.] * (size - 1)))
                                            for field, size in self.head_num_classes.items()})

        def forward(self, mesh: torch.Tensor, force: torch.Tensor) -> dict[str, torch.Tensor]:
            return {field: value.unsqueeze(0).expand(len(mesh), -1) for field, value in self.values.items()}

    model = FixedModel()
    untouched = model.values['fz_bias'].detach().clone()
    item = {'mesh_motion': torch.zeros(3, 12, 8, 8), 'force': torch.zeros(3, 6, 8, 8),
            'labels': {field: torch.tensor(0 if field == 'area' else -1) for field in LABEL_FIELDS},
            'masks': {field: torch.tensor(field == 'area') for field in LABEL_FIELDS}}
    loader = DataLoader([item, item], batch_size=2)
    criteria = _make_criteria({field: torch.ones(size) for field, size in model.head_num_classes.items()}, label_smoothing=0)
    report = _run_epoch(model, loader, device=torch.device('cpu'), criteria=criteria,
                        optimizer=torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1), desc='test')
    assert report['num_samples'] == report['num_supervised_labels'] == 2
    assert report['supervised_heads'] == ['area']
    assert report['heads']['area']['num_samples'] == 2
    assert report['heads']['fz_bias']['num_samples'] == 0
    assert report['mean_macro_f1'] == report['heads']['area']['macro_f1']
    assert report['mean_macro_f1'] > 0
    assert torch.equal(model.values['fz_bias'], untouched)
    assert model.values['fz_bias'].grad is None


def test_six_head_checkpoint_predictor_includes_fz_bias(tmp_path: Path) -> None:
    model = TactileCaptioner(head_num_classes={field: len(mapping) for field, mapping in LABEL_MAPS.items()},
                            frame_feature_dim=8, temporal_hidden_dim=12, temporal_dilations=(1,), dropout=0)
    checkpoint = {'model_state_dict': model.state_dict(), 'model_config': model.config_dict(),
                  'label_schema_version': LABEL_SCHEMA_VERSION, 'label_maps': LABEL_MAPS,
                  'dataset_meta': {'window_size': 3},
                  'normalization': {'mesh_motion_mean': [0.] * 12, 'mesh_motion_std': [1.] * 12,
                                    'force_mean': [0.] * 6, 'force_std': [1.] * 6}}
    path = tmp_path / 'six_head.pt'
    torch.save(checkpoint, path)
    prediction = TactileCaptionerPredictor(path, device='cpu').predict(np.zeros((3, 8, 8, 12)), np.zeros((3, 8, 8, 6)))
    assert tuple(prediction.label_names) == LABEL_FIELDS
    assert len(prediction.probabilities['fz_bias']) == 3
    assert '; Fz_bias=' in prediction.caption
    assert labels_to_caption({'area': 'none', 'fx_state': 'near_zero', 'fy_state': 'near_zero',
                              'fz_state': 'near_zero', 'fz_bias': 'balanced', 'rotation': 'none'}) == (
        'Touch[area=none; Fx=near_zero; Fy=near_zero; Fz=near_zero; Fz_bias=balanced; rotation=none]')


def test_window_random_loader_allows_one_episode_in_multiple_splits(tmp_path: Path) -> None:
    make_masked_dataset(tmp_path)
    meta_path = tmp_path / 'meta.json'
    meta = json.loads(meta_path.read_text())
    meta['split_policy'] = 'window_random'
    for record in meta['sequences']:
        record['split'] = 'mixed'
    meta_path.write_text(json.dumps(meta))
    for split in ('train', 'val', 'test'):
        dataset = TactileCaptionerDataset(tmp_path, split=split)
        assert len(dataset) == 18
        assert dataset[0]['masks']['area']
        dataset.close()
    # Existing episode-level data still checks sequence ownership.
    meta.pop('split_policy')
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='different split'):
        TactileCaptionerDataset(tmp_path, split='val')
