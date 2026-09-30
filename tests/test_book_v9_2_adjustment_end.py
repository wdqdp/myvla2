from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vla.vla.book_v9_2_adjustment_end_data import adjustment_end_label, selected_train_frames
from tactile_vla.vla.v7_5_runtime_history import RUNTIME_SAMPLE_OFFSETS
from tactile_vla.vla.v7_6_adjustment_end_data import DeterministicNaturalBatchSampler


def test_book_train_selection_preserves_boundary_and_uses_one_to_two_ratio():
    r = 150
    selected = selected_train_frames(r, episode_id=261)
    assert selected == selected_train_frames(r, episode_id=261)
    assert len(selected) == 33
    assert set(range(r - 10, r + 1)) <= selected
    assert set(range(r - 20, r - 10)) <= selected
    assert len(selected & set(range(r - 40, r - 20))) == 8
    assert len(selected & set(range(0, r - 40))) == 4
    assert len([frame for frame in selected if frame >= r - 10]) == 11
    with pytest.raises(ValueError, match="three negative buckets"):
        selected_train_frames(43, episode_id=261)


def test_book_label_window_is_inclusive_at_reexecution_frame():
    assert not adjustment_end_label(139, 150)
    assert adjustment_end_label(140, 150)
    assert adjustment_end_label(150, 150)
    with pytest.raises(ValueError, match="inclusive"):
        adjustment_end_label(151, 150)


def test_raw_h100_offsets_include_current_and_match_runtime_protocol():
    assert len(RUNTIME_SAMPLE_OFFSETS) == 11
    assert RUNTIME_SAMPLE_OFFSETS[0] == 0
    assert RUNTIME_SAMPLE_OFFSETS[-1] == 99
    assert len(set(RUNTIME_SAMPLE_OFFSETS)) == 11
    attempt1 = np.arange(200)
    attempt2 = np.arange(200, 350)
    timeline = np.concatenate((attempt1, attempt2))
    p = 0
    sample = timeline[200 + p - 99 + np.asarray(RUNTIME_SAMPLE_OFFSETS)]
    assert sample[0] == 101
    assert sample[-1] == 200
    assert np.count_nonzero(sample < 200) == 10
    p = 99
    sample = timeline[200 + p - 99 + np.asarray(RUNTIME_SAMPLE_OFFSETS)]
    assert sample[0] == 200
    assert sample[-1] == 299


def test_natural_sampler_accepts_reduced_manifest_without_forced_batch_ratio():
    labels = [True] * 264 + [False] * 528
    sampler = DeterministicNaturalBatchSampler(labels=labels, num_batches=2000, seed=42)
    first = next(iter(sampler))
    assert len(first) == 8
    assert len(set(first)) == 8
    assert first == next(iter(DeterministicNaturalBatchSampler(labels=labels, num_batches=2000, seed=42)))


def test_training_entry_imports_the_natural_sampler():
    from scripts import train_vla_adjustment_end_multitask_book_v9_2 as trainer

    assert trainer.DeterministicNaturalBatchSampler is DeterministicNaturalBatchSampler
