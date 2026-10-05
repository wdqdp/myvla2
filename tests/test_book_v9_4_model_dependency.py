from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]

from tactile_vla.vla import book_v9_2_adjustment_end_data as data  # noqa: E402
from tactile_vla.vla.artifacts import sha256_file  # noqa: E402
from tactile_vla.vla.book_v9_4_multitask_data import validate_stage_a_for_training  # noqa: E402
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_adjustment_labels_and_sampling_identical_without_model(monkeypatch, tmp_path):
    frames, events, indices = [], {}, {}
    for episode, split in enumerate(("train", "val", "test"), 1):
        indices[split] = {"execution_indices": []}
        events[episode, 2] = 44
        for attempt, length in ((1, 100), (2, 45)):
            for frame in range(length):
                global_index = len(frames)
                frames.append(
                    SimpleNamespace(
                        episode_id=episode,
                        attempt_id=attempt,
                        frame_index=frame,
                        global_index=global_index,
                        attempt_key=(episode, attempt),
                        instruction="Book.",
                        tactile_caption="Touch[area=none; Fx=near_zero; Fy=near_zero; Fz=near_zero; Fz_bias=balanced; rotation=none]",
                        input_recovery_plan="recovery_plan=move horizontally right moderately, move vertically none moderately.",
                    )
                )
                indices[split]["execution_indices"].append(global_index)
    v4_path, action_path, norm_path = (tmp_path / name for name in ("v4.json", "action.json", "norm.json"))
    write_json(norm_path, {})
    write_json(v4_path, {"selection_hash": "selection", "splits": indices})
    write_json(
        action_path,
        {
            "selection_hash": "selection",
            "training_data_hash": "datahash",
            "v4_norm_stats_sha256": sha256_file(norm_path),
            "native_reexecution_timing_identity": "timing",
        },
    )
    monkeypatch.setattr(data, "validate_v4_index_dataset", lambda *args: (frames, {}))
    validation_calls = []
    monkeypatch.setattr(data, "validate_training_index", lambda *args, **kwargs: validation_calls.append(kwargs))
    monkeypatch.setattr(data, "native_reexecution_events", lambda **kwargs: (events, "timing"))
    monkeypatch.setattr(data, "scan_selected_qpos", lambda **kwargs: {f.global_index: np.zeros(7) for f in frames})
    monkeypatch.setattr(data, "load_state_quantiles", lambda *args: StateQuantileStats(np.zeros(7), np.ones(7)))
    tokenizer = SimpleNamespace(encode_text=lambda *args, **kwargs: [1] * 10)
    arguments = dict(
        dataset_dir=tmp_path,
        v4_index_file=v4_path,
        action_index_file=action_path,
        norm_stats_file=norm_path,
        tokenizer=tokenizer,
        expected_counts={"train": (1, 11, 22), "val": (1, 11, 34), "test": (1, 11, 34)},
    )
    unbound_rows, unbound_index, _ = data.build_artifacts(**arguments, stage_a_checkpoint=None)
    assert len(unbound_rows) == 135 and len(validation_calls) == 1
    assert "stage_a_checkpoint" not in unbound_index
    assert set(unbound_index["source_files"]) == {"v4_training_index", "book_stage_a_index", "v4_norm_stats"}
    missing = tmp_path / "missing_model/15000"
    with pytest.raises(ValueError, match="checkpoint"):
        data.build_artifacts(**arguments, stage_a_checkpoint=missing)
    checkpoint = tmp_path / "model/15000"
    write_json(checkpoint / "params/_METADATA", {})
    write_json(
        checkpoint.parent / "config.json",
        {"data_profile": "book_stage_a_v1", "num_steps": 15000, "use_state_history": False},
    )
    bound_rows, bound_index, _ = data.build_artifacts(**arguments, stage_a_checkpoint=checkpoint)
    assert bound_rows == unbound_rows
    assert bound_index["splits"] == unbound_index["splits"]
    assert "backbone_config" in bound_index["source_files"]
    assert bound_index["stage_a_checkpoint"]["step"] == 15000


def test_v94_data_clis_have_no_checkpoint_argument():
    from scripts import build_book_v9_4_adjustment_end_data as adjustment
    from scripts import build_book_v9_4_multitask_data as multitask

    for entry in (adjustment, multitask):
        assert not hasattr(entry.parse_args([]), "stage_a_checkpoint")


@pytest.mark.parametrize("corruption", [None, "missing", "step", "profile", "history", "data"])
def test_stage_a_is_validated_at_training_time(tmp_path, corruption):
    checkpoint = tmp_path / "model/15000"
    config = {
        "data_profile": "book_stage_a_v1",
        "num_steps": 15000,
        "use_state_history": False,
        "artifact_identity": {"training_data_hash": "datahash"},
    }
    if corruption == "step":
        config["num_steps"] = 10000
    elif corruption == "profile":
        config["data_profile"] = "other"
    elif corruption == "history":
        config["use_state_history"] = True
    elif corruption == "data":
        config["artifact_identity"]["training_data_hash"] = "other"
    write_json(checkpoint.parent / "config.json", config)
    if corruption != "missing":
        write_json(checkpoint / "params/_METADATA", {})
    index = {"action_training_data_hash": "datahash"}
    if corruption:
        with pytest.raises(ValueError):
            validate_stage_a_for_training(checkpoint, index)
    else:
        identity = validate_stage_a_for_training(checkpoint, index)
        assert identity["step"] == 15000
        assert identity["config_sha256"] == sha256_file(checkpoint.parent / "config.json")
        assert identity["params_metadata_sha256"] == sha256_file(checkpoint / "params/_METADATA")


def test_training_entry_checks_model_and_records_identity(monkeypatch, tmp_path):
    from scripts import train_vla_multitask_book_v9_4 as trainer

    dataset, norms = tmp_path / "dataset", tmp_path / "norms"
    write_json(norms / "norm_stats.json", {})
    index = {
        "dataset_dir": str(dataset),
        "source_hashes": {str(norms / "norm_stats.json"): sha256_file(norms / "norm_stats.json")},
        "splits": {"val": {"need": {"sample_count": 20}, "adjustment": {"sample_count": 30}}},
        "training_target_coverage": {},
        "captioner_identity": {},
        "plan_token_validation": {},
    }
    index_path = tmp_path / "index.json"
    write_json(index_path, index)
    calls = []
    monkeypatch.setattr(trainer, "validate_index", lambda value: calls.append("data"))

    def check_model(path, value):
        assert value == index
        calls.append("model")
        return {"path": str(path), "step": 15000}

    monkeypatch.setattr(trainer, "validate_stage_a_for_training", check_model)
    monkeypatch.setattr(trainer, "scan_v4_lerobot_frames", lambda path: ["frames"])
    monkeypatch.setattr(trainer.training_base, "EXTRA_CONFIG", copy.deepcopy(trainer.training_base.EXTRA_CONFIG))
    args = SimpleNamespace(
        index_file=index_path,
        dataset_dir=dataset,
        norm_stats_dir=norms,
        stage_a_checkpoint=tmp_path / "model/15000",
        eval_max_need_samples=1000,
    )
    assert trainer.ensure_index(args) == (index, ["frames"])
    assert calls == ["data", "model"] and args.eval_max_need_samples == 30
    assert trainer.training_base.EXTRA_CONFIG["stage_a_initialization_identity"]["step"] == 15000
