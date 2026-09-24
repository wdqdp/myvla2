from __future__ import annotations

import pytest

from tactile_vla.vla.artifacts import sha256_json
from tactile_vla.vla.v7_7_2_multitask_data import (
    DATA_PROFILE, INDEX_SCHEMA, factual_adjustment_end, validate_index,
)
from tactile_vla.vla.v7_7_multitask_data import TASK_CYCLE
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE


def test_factual_stop_window_includes_ten_frames_on_both_sides():
    stop = 100
    assert not factual_adjustment_end(89, stop)
    assert factual_adjustment_end(90, stop)
    assert factual_adjustment_end(100, stop)
    assert factual_adjustment_end(110, stop)
    assert not factual_adjustment_end(111, stop)


def test_v772_index_rejects_modified_content_and_source_hash(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    from tactile_vla.vla.artifacts import sha256_file

    index = {
        "schema_version": INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "task_cycle": list(TASK_CYCLE),
        "adjustment_sampling_policy": "ratio_1_to_2",
        "adjustment_label_policy": "factual_true_inclusive_[S-10,S+10]",
        "history_policy": {"idle_perturbation": "none_raw_contiguous"},
        "need_successful_recovery_start": "arm_adjustment_stop_plus_1",
        "source_hashes": {str(source): sha256_file(source)},
        "splits": {split: {task: {"manifest_row_indices": [], "global_indices": []}
                           for task in ("adjustment", "need", "failure", "plan")}
                   for split in ("train", "val", "test")},
    }
    index["training_data_hash"] = sha256_json(index)
    validate_index(index)
    changed = index | {"seed": 43}
    with pytest.raises(ValueError, match="training data hash"):
        validate_index(changed)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="source hash changed"):
        validate_index(index)
