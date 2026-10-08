from __future__ import annotations

# ruff: noqa: E402
import copy
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from tactile_vla.vla import book_v9_5_stage_a_data as data
from tactile_vla.vla.artifacts import sha256_json


def source_fixture(tmp_path):
    root = tmp_path / "book"
    dataset = root / "lerobot_data/tactile_vla_rotation_v4"
    v4 = root / "outputs/rotation_v4"
    norm = v4 / "norm_stats"
    dataset.mkdir(parents=True)
    norm.mkdir(parents=True)
    hdf5 = root / "hdf5/episode1/attempt2"
    hdf5.mkdir(parents=True)
    (hdf5 / "data.hdf5").touch()
    profile = {
        "hdf5_data_dir": str(root / "hdf5"),
        "raw_data_dir": str(root / "raw_data"),
        "attempts": [{"hdf5_path": "episode1/attempt2/data.hdf5"}],
    }
    (v4 / "profile.json").write_text(json.dumps(profile))
    payload = {
        "dataset_dir": str(dataset),
        "source_files": {
            "profile": {"path": str(v4 / "profile.json")},
            "lerobot_parquet": {},
        },
    }
    index = v4 / "v4_training_index.json"
    index.write_text(json.dumps(payload))
    return {"book_root": root, "dataset_dir": dataset, "v4_index_file": index, "norm_stats_dir": norm}


def test_current_sources_without_archive_are_accepted(tmp_path):
    inputs = source_fixture(tmp_path)
    archive = inputs["book_root"] / data.ARCHIVE_NAME
    archive.mkdir()
    # Its existence is harmless; the builder must not traverse/read it.
    (archive / "profile.json").write_text("not valid JSON")
    scope = data.validate_source_scope(**inputs)
    assert scope["excluded_directory"] == str(archive)
    assert scope["policy"] == data.SCOPE_POLICY


@pytest.mark.parametrize("location", ["dataset_dir", "v4_index_file", "norm_stats_dir", "book_root"])
def test_archive_input_rejected(tmp_path, location):
    inputs = source_fixture(tmp_path)
    inputs[location] = inputs["book_root"] / data.ARCHIVE_NAME / location
    with pytest.raises(ValueError, match="history9_4_5"):
        data.validate_source_scope(**inputs)


@pytest.mark.parametrize("location", ["dataset", "video", "hdf5", "source"])
def test_symlink_and_embedded_source_paths_into_archive_rejected(tmp_path, location):
    inputs = source_fixture(tmp_path)
    archive = inputs["book_root"] / data.ARCHIVE_NAME
    archive.mkdir()
    if location == "dataset":
        target = archive / "dataset"
        target.mkdir()
        link = inputs["book_root"] / "lerobot_data/linked_dataset"
        link.symlink_to(target, target_is_directory=True)
        inputs["dataset_dir"] = link
    elif location == "video":
        target = archive / "old.mp4"
        target.touch()
        (inputs["dataset_dir"] / "video.mp4").symlink_to(target)
    elif location == "hdf5":
        target = archive / "old.hdf5"
        target.touch()
        (inputs["book_root"] / "hdf5/episode1/attempt2/old.hdf5").symlink_to(target)
        p = inputs["v4_index_file"].parent / "profile.json"
        d = json.loads(p.read_text())
        d["attempts"][0]["hdf5_path"] = "episode1/attempt2/old.hdf5"
        p.write_text(json.dumps(d))
    else:
        p = inputs["v4_index_file"]
        d = json.loads(p.read_text())
        d["source_files"]["selection"] = {"path": str(archive / "inventory.json")}
        p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="history9_4_5"):
        data.validate_source_scope(**inputs)


def test_outside_root_and_different_dataset_rejected(tmp_path):
    inputs = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        data.validate_source_scope(**(inputs | {"dataset_dir": tmp_path / "other"}))
    p = inputs["v4_index_file"]
    d = json.loads(p.read_text())
    d["dataset_dir"] = str(inputs["book_root"] / "lerobot_data/other")
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="different dataset"):
        data.validate_source_scope(**inputs)


def test_version_fields_do_not_mutate_common_phase_contract(tmp_path, monkeypatch):
    inputs = source_fixture(tmp_path)
    original_index = {"summary": {"splits": {}}, "action_horizon": 30}
    original_summary = copy.deepcopy(original_index["summary"])
    monkeypatch.setattr(
        data.common, "build_artifacts", lambda **_: ([], copy.deepcopy(original_index), copy.deepcopy(original_summary))
    )
    rows, index, summary = data.build_artifacts(**inputs)
    assert rows == []
    assert index["summary"] == original_summary
    assert index["experiment_version"] == summary["experiment_version"] == "book_v9_5"
    assert index["source_scope"] == summary["source_scope"]
    assert "experiment_version" not in original_index
    payload = index | {
        "source_files": {
            "v4_training_index": {"path": str(inputs["v4_index_file"])},
            "norm_stats": {"path": str(inputs["norm_stats_dir"] / "norm_stats.json")},
        }
    }
    payload["training_data_hash"] = sha256_json(payload)
    monkeypatch.setattr(data.common, "validate_training_index", lambda *a, **kw: ("rows", "lookup"))
    assert data.validate_training_index(
        payload, index_path=tmp_path / "index.json", dataset_dir=inputs["dataset_dir"], book_root=inputs["book_root"]
    ) == ("rows", "lookup")
    with pytest.raises(ValueError, match="rebuilt"):
        data.validate_training_index(
            payload | {"experiment_version": "book_v9_4"},
            index_path=tmp_path / "index.json",
            dataset_dir=inputs["dataset_dir"],
            book_root=inputs["book_root"],
        )
    changed = copy.deepcopy(payload)
    changed["source_scope"]["policy"]["excluded_directory"] = "other"
    with pytest.raises(ValueError, match="source scope"):
        data.validate_training_index(
            changed,
            index_path=tmp_path / "index.json",
            dataset_dir=inputs["dataset_dir"],
            book_root=inputs["book_root"],
        )


def test_builder_defaults_and_overwrite_protection(tmp_path):
    from scripts.prepare_book_stage_a_v9_5_index import main, parse_args

    args = parse_args(["--book-root", str(tmp_path / "book")])
    assert args.output_dir == tmp_path / "book/outputs/book_stage_a_v9_5"
    assert args.dataset_dir == tmp_path / "book/lerobot_data/tactile_vla_rotation_v4"
    assert not hasattr(args, "checkpoint")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "book_stage_a_training_index.json").touch()
    with pytest.raises(FileExistsError):
        main(["--book-root", str(tmp_path / "book")])


def test_training_defaults_and_equals_form(monkeypatch, tmp_path):
    from scripts import train_vla_stage_a_book_v9_5 as entry

    args = entry.parse_args(["--book-root", str(tmp_path / "book")])
    assert args.run_name == data.RUN_NAME
    assert args.index_file == tmp_path / "book/outputs/book_stage_a_v9_5/book_stage_a_training_index.json"
    assert args.data_profile == "book_stage_a_v1" and args.experiment_version == "book_v9_5"
    assert args.prompt_profile == "phase_v2" and args.experiment_kind == data.EXPERIMENT_KIND
    assert args.num_steps == 15000 and args.batch_size == 8 and args.fsdp_devices == 2
    assert not args.use_state_history and args.state_history_len == args.history_hidden_dim == 0
    entry.trainer.validate_v4_training_protocol(args)
    alternate = tmp_path / "book/outputs/alternate/index.json"
    args2 = entry.parse_args(["--book-root", str(tmp_path / "book"), f"--index-file={alternate}", "--run-name=custom"])
    assert args2.index_file == alternate and args2.run_name == "custom"
    with pytest.raises(ValueError, match="history9_4_5"):
        entry.parse_args(["--dataset-dir", str(data.ROOT / "history9_4_5/lerobot_data")])
    with pytest.raises(ValueError, match="requires"):
        entry.parse_args(["--data-profile=rotation_v4"])
    with pytest.raises(ValueError, match="protocol mismatch"):
        entry.trainer.validate_v4_training_protocol(entry.parse_args(["--use-state-history"]))


def test_training_index_and_config_resume_wiring(monkeypatch, tmp_path):
    from scripts import train_vla_stage_a_book_v9_5 as entry

    args = entry.parse_args(["--book-root", str(tmp_path / "book")])
    args.index_file.parent.mkdir(parents=True)
    index = {"source_scope": {"policy": data.SCOPE_POLICY}}
    args.index_file.write_text(json.dumps(index))
    calls = []

    def validate(payload, **kwargs):
        calls.append(kwargs)
        return [], {42: {"phase": "adjustment"}}

    monkeypatch.setattr(entry, "validate_training_index", validate)
    assert entry.ensure_index(args) == index
    assert calls[0]["book_root"] == tmp_path / "book"
    assert args._v5_action_phase_lookup == {42: {"phase": "adjustment"}}
    identity = {
        "data_config_hash": "a",
        "action_frame_manifest_hash": "b",
        "action_indices_identity": {},
        "index_sha256": "c",
    }
    config = entry.checkpoint_config_payload(args, identity)
    assert config["experiment_version"] == "book_v9_5"
    assert config["source_scope"] == index["source_scope"]
    assert "_v5_action_phase_lookup" not in config and "_v9_5_source_scope" not in config
    entry.validate_resume_config(config, args)
    with pytest.raises(ValueError, match="different version"):
        entry.validate_resume_config(config | {"experiment_version": "book_v9_4"}, args)
    with pytest.raises(ValueError, match="different version"):
        entry.validate_resume_config(config | {"source_scope": {}}, args)
    for name in ("parse_args", "ensure_index", "checkpoint_config_payload", "validate_v4_resume_config"):
        monkeypatch.setattr(entry.trainer, name, getattr(entry.trainer, name))
    entry.configure_version()
    assert entry.trainer.ensure_index is entry.ensure_index
    assert entry.trainer.parse_args is entry.parse_args
