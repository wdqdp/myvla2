from __future__ import annotations

# ruff: noqa: E402
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "openpi/src"), str(ROOT / "openpi/inference/agilex/inference")]

from scripts import serve_tactile_vla_book_v9_2 as server
import agilex_inference_book_v9_2_asyn_direction_keys as client
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v7_5_phase_change import helper_identity


@pytest.fixture
def deployment(tmp_path):
    stage = tmp_path / "stage_a" / "15000"
    stage.mkdir(parents=True)
    norm = tmp_path / "norm"
    norm.mkdir()
    (norm / "norm_stats.json").write_text("{}")
    (stage.parent / "config.json").write_text(
        json.dumps({"artifact_identity": {"v4_norm_stats_sha256": sha256_file(norm / "norm_stats.json")}})
    )
    config = {
        "data_profile": server.DATA_PROFILE,
        "prompt_profile": server.PHASE_CHANGE_PROMPT_PROFILE,
        "experiment_kind": server.EXPERIMENT_KIND,
        "checkpoint_format": server.MULTITASK_CHECKPOINT_FORMAT,
        "num_steps": 4000,
        "phase_change_max_token_len": 512,
        "use_state_history": False,
        "state_history_len": 0,
        "history_hidden_dim": 0,
        "label_policy": server.LABEL_POLICY,
        "classification_sampling_ratio": {"positive": 1, "negative": 2},
        "classification_sampling_policy": {
            "strategy": "deterministic_natural_manifest_stream",
            **server.TRAIN_SAMPLING_POLICY,
        },
        "stage_a_checkpoint": str(stage),
        "caption_source": {"field": "tactile_caption", "source": "book_v4_lerobot"},
        "prompt_helper": helper_identity(),
    }
    metadata = {
        "checkpoint_format": server.MULTITASK_CHECKPOINT_FORMAT,
        "official_step": 4000,
        "label_policy": server.LABEL_POLICY,
        "history_policy": server.HISTORY_POLICY,
        "train_sampling_policy": server.TRAIN_SAMPLING_POLICY,
        "adjustment_end_threshold": 0.9999775886535645,
        "prompt_helper": helper_identity(),
    }
    args = SimpleNamespace(checkpoint_load_mode="full", norm_stats_dir=norm, captioner_checkpoint_sha256="a" * 64)
    return args, config, metadata


def test_factual_book_config_and_saved_threshold(deployment):
    args, config, metadata = deployment
    threshold, sha = server._validate_config(args, config, metadata)
    assert threshold == metadata["adjustment_end_threshold"]
    assert sha == "a" * 64
    bad = copy.deepcopy(metadata)
    bad["history_policy"]["counterfactual"] = True
    with pytest.raises(ValueError, match="history policy"):
        server._validate_config(args, config, bad)
    (args.norm_stats_dir / "norm_stats.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="norm stats"):
        server._validate_config(args, config, metadata)


def test_default_and_override_threshold_change_classifier_decision():
    policy = object.__new__(server.BookV92Policy)
    policy._phase_input = lambda x: x
    policy._observation = lambda x: x
    policy._adjustment_end_logits = lambda _: np.log(np.asarray([[0.2, 0.8]], dtype=np.float32))
    request = {"prompt": "Mode: adjustment.\nqpos_h100_11:[[0]]"}
    policy._threshold, overridden = server.v7.v53._resolve_runtime_threshold(0.9999775886535645, None)
    assert not overridden
    assert not policy._infer_adjustment_end(request)["adjustment_end"]
    policy._threshold, overridden = server.v7.v53._resolve_runtime_threshold(0.9999775886535645, 0.7)
    assert overridden
    assert policy._infer_adjustment_end(request)["adjustment_end"]
    with pytest.raises(ValueError):
        server.v7.v53._resolve_runtime_threshold(0.9, 1.1)


def test_book_disallows_counterfactual_gripper_probe():
    args = SimpleNamespace(classification_gripper_open_threshold=0.04, classification_gripper_open_value=0.06)
    with pytest.raises(SystemExit):
        client.validate_raw_classification_qpos(args, argparse.ArgumentParser())


def test_book_metadata_rejects_rotation_model_before_requests():
    with pytest.raises(ValueError, match="metadata mismatch"):
        client.validate_server_metadata(SimpleNamespace(), {"data_profile": "rotation_phase_v7_4_adjustment"})
