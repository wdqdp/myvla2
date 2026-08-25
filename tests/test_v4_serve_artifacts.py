from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("USE_TF", "0")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi" / "src"))

from tactile_vla.vla.artifacts import selected_best_metrics_summary  # noqa: E402
from tactile_vla.vla.artifacts import sha256_file  # noqa: E402


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity() -> dict:
    return {
        "data_profile": "rotation_v4",
        "prompt_profile": "minimal_v1",
        "norm_stats_sha256": "a" * 64,
    }


def _best_metrics(**overrides) -> dict:
    metrics = {
        "step": 1_000,
        "val_score": 0.8,
        "action_loss": 1.05,
        "action_loss_baseline": 1.0,
        "action_loss_degradation": 0.05,
        "action_gate_passed": True,
        "need_recovery": {"macro_f1": 0.8},
        "failure_reason": {"exact_match": 0.75},
        "recovery_plan": {"exact_match": 0.85},
    }
    metrics.update(overrides)
    return metrics


def test_v4_merge_uses_stage_a_configured_final_step_and_requires_best(tmp_path: Path) -> None:
    merge = _load_script("merge_v4_validation_test_module", "scripts/merge_v3_stage_b_delta.py")
    config = {
        "data_profile": "rotation_v4",
        "artifact_identity": _identity(),
        "action_loss_degradation_limit": 0.10,
        "stage_a_checkpoint_step": 15_000,
    }
    delta_dir = tmp_path / "stage_b" / "best" / "delta_params"
    delta_dir.mkdir(parents=True)
    metrics_path = delta_dir.parent / "metrics.json"
    metrics_path.write_text(json.dumps(_best_metrics()))
    artifact = merge.validate_versioned_merge_layout(
        config=config,
        delta_dir=delta_dir,
        stage_a_checkpoint=Path("/stage_a/15000"),
        output=Path("/deploy/merged_best"),
    )
    assert artifact["summary"]["step"] == 1_000
    assert artifact["sha256"] == sha256_file(metrics_path)
    with pytest.raises(ValueError, match="selected best"):
        merge.validate_versioned_merge_layout(
            config=config,
            delta_dir=Path("/stage_b/4000/delta_params"),
            stage_a_checkpoint=Path("/stage_a/15000"),
            output=Path("/deploy/merged_best"),
        )
    with pytest.raises(ValueError, match="checkpoint step mismatch"):
        merge.validate_versioned_merge_layout(
            config=config,
            delta_dir=delta_dir,
            stage_a_checkpoint=Path("/stage_a/10000"),
            output=Path("/deploy/merged_best"),
        )
    with pytest.raises(ValueError, match="norm_stats_sha256"):
        merge.validate_versioned_merge_layout(
            config=config | {"artifact_identity": _identity() | {"norm_stats_sha256": ""}},
            delta_dir=delta_dir,
            stage_a_checkpoint=Path("/stage_a/15000"),
            output=Path("/deploy/merged_best"),
        )

    metrics_path.unlink()
    with pytest.raises(FileNotFoundError, match="metrics.json"):
        merge.validate_versioned_merge_layout(
            config=config,
            delta_dir=delta_dir,
            stage_a_checkpoint=Path("/stage_a/15000"),
            output=Path("/deploy/merged_best"),
        )
    for metrics, message in (
        (_best_metrics(action_gate_passed=False), "action_gate_passed"),
        (_best_metrics(action_loss_degradation=0.11), "exceeds action degradation"),
        (_best_metrics(step=750), "500-step evaluation"),
    ):
        metrics_path.write_text(json.dumps(metrics))
        with pytest.raises(ValueError, match=message):
            merge.validate_versioned_merge_layout(
                config=config,
                delta_dir=delta_dir,
                stage_a_checkpoint=Path("/stage_a/15000"),
                output=Path("/deploy/merged_best"),
            )


def test_v4_full_server_validates_portable_merge_manifest_and_norm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve = _load_script("serve_v4_validation_test_module", "scripts/serve_tactile_vla_policy_v3.py")
    root = tmp_path / "merged_best"
    root.mkdir()
    identity = _identity()
    stage_a = Path("/source/stage_a/15000").resolve()
    stage_b_delta = str(Path("/source/stage_b/best/delta_params").resolve())
    metrics = _best_metrics()
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    metrics_summary = selected_best_metrics_summary(
        metrics,
        action_loss_degradation_limit=0.10,
        context="test",
    )
    (root / "merge_manifest.json").write_text(
        json.dumps(
            {
                "stage_a_checkpoint": str(stage_a),
                "stage_a_checkpoint_step": 15_000,
                "stage_b_delta": stage_b_delta,
                "artifact_identity": identity,
                "norm_stats_sha256": identity["norm_stats_sha256"],
                "selected_best_metrics": metrics_summary,
                "selected_best_metrics_sha256": sha256_file(metrics_path),
            }
        )
    )
    config = {
        "data_profile": "rotation_v4",
        "prompt_profile": "minimal_v1",
        "checkpoint_format": "stage_b_v3_merged_full_v1",
        "stage_a_checkpoint": str(stage_a),
        "stage_a_checkpoint_step": 15_000,
        "stage_b_delta": stage_b_delta,
        "action_loss_degradation_limit": 0.10,
        "artifact_identity": identity,
    }
    monkeypatch.setattr(
        serve,
        "validate_norm_stats_identity",
        lambda *_args, **_kwargs: {"norm_stats_sha256": identity["norm_stats_sha256"]},
    )
    args = argparse.Namespace(checkpoint=root, norm_stats_dir=tmp_path / "norm", no_norm=False)
    summary = serve.validate_v4_serve_artifacts(args, config)
    assert summary == {"norm_stats_sha256": "a" * 64}

    with pytest.raises(ValueError, match="config Stage A checkpoint step mismatch"):
        serve.validate_v4_serve_artifacts(
            args,
            config | {"stage_a_checkpoint_step": 10_000},
        )

    bad_manifest = json.loads((root / "merge_manifest.json").read_text())
    bad_manifest["norm_stats_sha256"] = "b" * 64
    (root / "merge_manifest.json").write_text(json.dumps(bad_manifest))
    with pytest.raises(ValueError, match="merge manifest norm_stats"):
        serve.validate_v4_serve_artifacts(args, config)

    bad_manifest["norm_stats_sha256"] = identity["norm_stats_sha256"]
    bad_manifest["selected_best_metrics"] = dict(metrics_summary) | {
        "action_loss_degradation": 0.09,
    }
    (root / "merge_manifest.json").write_text(json.dumps(bad_manifest))
    with pytest.raises(ValueError, match="metrics summary mismatch"):
        serve.validate_v4_serve_artifacts(args, config)


def test_v4_stage_a_action_server_validates_actual_norm_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve = _load_script("serve_v4_action_validation_test_module", "scripts/serve_tactile_vla_action_ablation.py")
    identity = _identity()
    config = {
        "data_profile": "rotation_v4",
        "prompt_profile": "minimal_v1",
        "artifact_identity": identity,
    }
    monkeypatch.setattr(
        serve,
        "validate_norm_stats_identity",
        lambda *_args, **_kwargs: {"norm_stats_sha256": identity["norm_stats_sha256"]},
    )
    args = argparse.Namespace(
        checkpoint=tmp_path / "stage_a" / "15000",
        checkpoint_kind="stage-a",
        norm_stats_dir=tmp_path / "norm",
    )
    assert serve.validate_v4_norm_artifacts(args, config) == {
        "norm_stats_sha256": identity["norm_stats_sha256"]
    }
    with pytest.raises(ValueError, match="numeric checkpoint step"):
        serve.validate_v4_norm_artifacts(
            argparse.Namespace(
                checkpoint=tmp_path / "stage_a" / "latest",
                checkpoint_kind="stage-a",
                norm_stats_dir=tmp_path / "norm",
            ),
            config,
        )
    monkeypatch.setattr(
        serve,
        "validate_norm_stats_identity",
        lambda *_args, **_kwargs: {"norm_stats_sha256": "b" * 64},
    )
    with pytest.raises(ValueError, match="does not match checkpoint identity"):
        serve.validate_v4_norm_artifacts(args, config)


def test_action_server_metadata_excludes_v5_per_frame_training_lookup() -> None:
    serve = _load_script(
        "serve_v5_metadata_summary_test_module",
        "scripts/serve_tactile_vla_action_ablation.py",
    )
    config = {
        "run_name": "phase-v5",
        "data_profile": "rotation_phase_v5",
        "prompt_profile": "phase_v1",
        "experiment_kind": "phase_prompt_only",
        "num_steps": 15_000,
        "artifact_identity": {
            "selection_hash": "a" * 64,
            "v4_norm_stats_sha256": "b" * 64,
            "source_file_hashes": {"large": ["must", "not", "be", "served"]},
        },
        "_v5_action_phase_lookup": {
            str(index): {"phase": "execution"} for index in range(100)
        },
    }

    summary = serve.metadata_config_summary(config)

    assert summary["data_profile"] == "rotation_phase_v5"
    assert summary["prompt_profile"] == "phase_v1"
    assert summary["artifact_identity"] == {
        "selection_hash": "a" * 64,
        "v4_norm_stats_sha256": "b" * 64,
    }
    assert "_v5_action_phase_lookup" not in summary
    assert "source_file_hashes" not in summary["artifact_identity"]
    assert len(json.dumps(summary)) < 2_000


def test_v4_merged_action_server_uses_recorded_stage_a_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve = _load_script(
        "serve_v4_merged_action_validation_test_module",
        "scripts/serve_tactile_vla_action_ablation.py",
    )
    root = tmp_path / "merged_best"
    root.mkdir()
    identity = _identity()
    stage_a = Path("/source/stage_a/15000").resolve()
    stage_b_delta = str(Path("/source/stage_b/best/delta_params").resolve())
    metrics = _best_metrics()
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    metrics_summary = selected_best_metrics_summary(
        metrics,
        action_loss_degradation_limit=0.10,
        context="test",
    )
    manifest = {
        "stage_a_checkpoint": str(stage_a),
        "stage_a_checkpoint_step": 15_000,
        "stage_b_delta": stage_b_delta,
        "artifact_identity": identity,
        "norm_stats_sha256": identity["norm_stats_sha256"],
        "selected_best_metrics": metrics_summary,
        "selected_best_metrics_sha256": sha256_file(metrics_path),
    }
    (root / "merge_manifest.json").write_text(json.dumps(manifest))
    config = {
        "data_profile": "rotation_v4",
        "prompt_profile": "minimal_v1",
        "stage_a_checkpoint": str(stage_a),
        "stage_a_checkpoint_step": 15_000,
        "stage_b_delta": stage_b_delta,
        "action_loss_degradation_limit": 0.10,
        "artifact_identity": identity,
    }
    monkeypatch.setattr(
        serve,
        "validate_norm_stats_identity",
        lambda *_args, **_kwargs: {"norm_stats_sha256": identity["norm_stats_sha256"]},
    )
    args = argparse.Namespace(
        checkpoint=root,
        checkpoint_kind="stage-b",
        norm_stats_dir=tmp_path / "norm",
    )
    assert serve.validate_v4_norm_artifacts(args, config) == {
        "norm_stats_sha256": identity["norm_stats_sha256"]
    }

    with pytest.raises(ValueError, match="config Stage A checkpoint step mismatch"):
        serve.validate_v4_norm_artifacts(
            args,
            config | {"stage_a_checkpoint_step": 10_000},
        )

    manifest["stage_a_checkpoint_step"] = 10_000
    (root / "merge_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest Stage A checkpoint step mismatch"):
        serve.validate_v4_norm_artifacts(args, config)
