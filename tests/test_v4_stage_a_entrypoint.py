from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi" / "src"))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "train_vla_stage_a_v4_test_module",
        PROJECT_ROOT / "scripts" / "train_vla_stage_a_openpi.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_a_defaults_to_single_process_data_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    monkeypatch.setattr(sys, "argv", ["train_vla_stage_a_openpi.py"])
    assert module.parse_args().num_workers == 0


def test_checkpoint_config_excludes_private_runtime_phase_lookup() -> None:
    module = _load_script()
    args = SimpleNamespace(
        run_name="phase-v5.2",
        num_workers=0,
        _v5_action_phase_lookup={1: {"phase": "adjustment"}},
    )
    identity = {
        "data_config_hash": "a" * 64,
        "action_frame_manifest_hash": "b" * 64,
        "action_indices_identity": {"all": {"count": 1}},
        "index_sha256": "c" * 64,
    }

    payload = module.checkpoint_config_payload(args, identity)

    assert payload["run_name"] == "phase-v5.2"
    assert payload["num_workers"] == 0
    assert payload["artifact_identity"] is identity
    assert "_v5_action_phase_lookup" not in payload


def test_stage_a_v4_args_require_minimal_prompt_and_norm() -> None:
    module = _load_script()
    module.validate_v4_args(
        SimpleNamespace(data_profile="rotation_v4", prompt_profile="minimal_v1", no_norm=False)
    )
    with pytest.raises(ValueError, match="minimal_v1"):
        module.validate_v4_args(
            SimpleNamespace(data_profile="rotation_v4", prompt_profile="legacy", no_norm=False)
        )
    with pytest.raises(ValueError, match="norm stats"):
        module.validate_v4_args(
            SimpleNamespace(data_profile="rotation_v4", prompt_profile="minimal_v1", no_norm=True)
        )


def _v4_stage_a_protocol_args(**overrides):
    values = dict(
        data_profile="rotation_v4",
        prompt_profile="minimal_v1",
        split="train",
        batch_size=8,
        num_steps=15_000,
        lr=5e-5,
        lr_final=5e-7,
        lr_transition_steps=7_000,
        save_interval=1_000,
        keep_period=5_000,
        action_horizon=30,
        action_dim=32,
        use_state_history=True,
        state_history_len=60,
        state_history_dim=7,
        history_hidden_dim=256,
        max_token_len=200,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        train_lora_only=True,
        allow_random_init=False,
        weight_decay=1e-10,
        grad_clip=1.0,
        log_interval=20,
        seed=42,
        precision="auto",
        ema_decay=None,
        fsdp_devices=1,
        video_backend="pyav",
        checkpoint="/models/pi05_base/params",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stage_a_v4_protocol_and_resume_are_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "DEFAULT_BASE_CHECKPOINT", Path("/models/pi05_base/params"))
    args = _v4_stage_a_protocol_args()
    module.validate_v4_training_protocol(args)
    saved = vars(args).copy()
    module.validate_v4_resume_config(saved, args)

    with pytest.raises(ValueError, match="protocol mismatch"):
        module.validate_v4_training_protocol(_v4_stage_a_protocol_args(action_dim=7))
    with pytest.raises(ValueError, match="checkpoint"):
        module.validate_v4_training_protocol(
            _v4_stage_a_protocol_args(checkpoint="/models/old_stage_a/params")
        )
    with pytest.raises(ValueError, match="resume config mismatch"):
        module.validate_v4_resume_config(saved | {"lr": 1e-4}, args)


def test_stage_a_v4_requires_existing_dedicated_index_and_validates_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    args = SimpleNamespace(
        index_file=tmp_path / "index.json",
        split_file=tmp_path / "splits.json",
        dataset_dir=tmp_path / "dataset",
        data_profile="rotation_v4",
        seed=42,
        action_horizon=30,
    )
    with pytest.raises(FileNotFoundError, match="matching versioned"):
        module.ensure_index(args)

    payload = {
        "schema_version": "tactile_vla_v4_training_index_v1",
        "data_profile": "rotation_v4",
        "action_horizon": 30,
    }
    args.index_file.write_text(json.dumps(payload))
    calls = []
    monkeypatch.setattr(
        module,
        "validate_v4_index_dataset",
        lambda actual, dataset_dir: calls.append((actual, dataset_dir)),
    )
    assert module.ensure_index(args) == payload
    assert calls == [(payload, args.dataset_dir)]


def test_stage_a_v4_loader_uses_index_execution_indices_and_v4_repo_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    captured = {}

    class RawDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    monkeypatch.setattr(module, "TactileVLAFrameDataset", RawDataset)
    monkeypatch.setattr(module, "TransformedTactileVLADataset", lambda dataset, transform: dataset)
    monkeypatch.setattr(module, "build_transform", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "DataLoader", lambda dataset, **kwargs: {"dataset": dataset, **kwargs})
    args = SimpleNamespace(
        split="train",
        max_frames=None,
        no_norm=True,
        norm_stats_dir=tmp_path,
        dataset_dir=tmp_path,
        action_horizon=30,
        use_state_history=True,
        state_history_len=60,
        video_backend="pyav",
        prompt_profile="minimal_v1",
        data_profile="rotation_v4",
        num_workers=0,
        batch_size=8,
    )
    payload = {"splits": {"train": {"execution_indices": [7, 3]}}}
    loader = module.build_loader(args, object(), payload)
    assert captured["indices"] == [7, 3]
    assert captured["dataset_repo_id"] == "tactile_vla_rotation_v4"
    assert captured["state_history_len"] == 60
    assert loader["shuffle"] is True

    args.use_state_history = False
    args.state_history_len = 0
    module.build_loader(args, object(), payload)
    assert captured["state_history_len"] == 0
