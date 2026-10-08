from __future__ import annotations

# ruff: noqa: E402
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from tactile_vla.vla.book_v9_4_5_action_replay import (
    PHASES,
    PhaseBalancedActionBatchSampler,
    phase_eval_positions,
    phase_positions,
)


def example():
    indices, lookup = [], {}
    for phase, count in (("execution", 40), ("adjustment", 12)):
        for i in range(count):
            index = len(indices)
            indices.append(index)
            lookup[index] = {
                "phase": phase,
                "trainable": True,
                "chunk_phase_pure": True,
                "episode_id": i // 10,
                "attempt_id": 1 if phase == "execution" else 2,
                "frame_index": i % 10,
            }
    return indices, lookup


def test_every_batch_is_exactly_four_plus_four_and_reproducible():
    indices, lookup = example()
    pools = phase_positions(indices, lookup)
    sampler = PhaseBalancedActionBatchSampler(pools)
    first = list(sampler)
    assert first == list(PhaseBalancedActionBatchSampler(pools))
    assert first != list(sampler)
    assert len(first) == 10
    for batch in first:
        assert len(batch) == len(set(batch)) == 8
        assert sum(lookup[indices[p]]["phase"] == "adjustment" for p in batch) == 4
    execution = [p for batch in first for p in batch if lookup[indices[p]]["phase"] == "execution"]
    assert len(execution) == len(set(execution)) == 40
    assert len([p for b in first for p in b if lookup[indices[p]]["phase"] == "adjustment"]) == 40


@pytest.mark.parametrize("bad", ["empty", "overlap", "duplicate", "odd"])
def test_invalid_phase_pools_fail(bad):
    pools = {"adjustment": list(range(4)), "execution": list(range(4, 12))}
    batch_size = 8
    if bad == "empty":
        pools["adjustment"] = []
    elif bad == "overlap":
        pools["execution"][0] = 0
    elif bad == "duplicate":
        pools["execution"][1] = 4
    else:
        batch_size = 7
    with pytest.raises(ValueError):
        PhaseBalancedActionBatchSampler(pools, batch_size=batch_size)


def test_phase_eval_covers_trajectories_and_not_prefix():
    indices, lookup = example()
    result = phase_eval_positions(indices, lookup, samples_per_phase=8)
    assert result == phase_eval_positions(indices, lookup, samples_per_phase=8)
    for phase in PHASES:
        selected = result[phase]
        assert len(selected) == len(set(selected)) == 8
        assert all(lookup[indices[p]]["phase"] == phase for p in selected)
        expected_groups = {(r["episode_id"], r["attempt_id"]) for r in lookup.values() if r["phase"] == phase}
        groups = {(lookup[indices[p]]["episode_id"], lookup[indices[p]]["attempt_id"]) for p in selected}
        assert groups == expected_groups
    assert any(p >= 10 for p in result["execution"])
    assert phase_eval_positions(indices, lookup, samples_per_phase=100)["adjustment"]
    with pytest.raises(ValueError, match="cover all"):
        phase_eval_positions(indices, lookup, samples_per_phase=1)
    with pytest.raises(ValueError):
        phase_eval_positions(indices, lookup, samples_per_phase=0)


def test_ineligible_action_frame_rejected():
    indices, lookup = example()
    lookup[0]["chunk_phase_pure"] = False
    with pytest.raises(ValueError, match="phase-pure"):
        phase_positions(indices, lookup)
    with pytest.raises(ValueError, match="Duplicate"):
        phase_positions(indices + indices[:1], lookup)


def test_phase_gate_rejects_adjustment_degradation_hidden_by_execution_improvement():
    from scripts.train_vla_stage_b_v3 import action_retention_metrics

    baseline = {
        "loss": 2.5,
        "by_phase": {"adjustment": 1.0, "execution": 4.0},
        "support_by_phase": {"adjustment": 256, "execution": 256},
    }
    report = {
        "loss": 2.15,
        "by_phase": {"adjustment": 1.3, "execution": 3.0},
        "support_by_phase": baseline["support_by_phase"],
    }
    metrics = action_retention_metrics(report, baseline, limit=0.1)
    assert metrics["action_loss_degradation"] < 0
    assert metrics["action_loss_degradation_by_phase"]["adjustment"] == pytest.approx(0.3)
    assert not metrics["action_gate_passed"]
    good = report | {"loss": 2.0, "by_phase": {"adjustment": 0.9, "execution": 3.1}}
    assert action_retention_metrics(good, baseline, limit=0.1)["action_gate_passed"]
    old = action_retention_metrics({"loss": 1.05}, {"loss": 1.0}, limit=0.1)
    assert old["action_gate_passed"] and "action_loss_by_phase" not in old


@pytest.mark.parametrize("bad", ["coverage", "support", "nan", "zero"])
def test_bad_baseline_or_phase_support_rejected(bad):
    from scripts.train_vla_stage_b_v3 import action_retention_metrics

    baseline = {
        "loss": 1.0,
        "by_phase": {"adjustment": 1.0, "execution": 1.0},
        "support_by_phase": {"adjustment": 8, "execution": 8},
    }
    current = baseline.copy()
    if bad == "coverage":
        current["by_phase"] = {"adjustment": 1.0}
    elif bad == "support":
        current["support_by_phase"] = {"adjustment": 0, "execution": 8}
    elif bad == "nan":
        current["loss"] = float("nan")
    else:
        baseline["loss"] = 0.0
    with pytest.raises(ValueError):
        action_retention_metrics(current, baseline, limit=0.1)


def test_hook_evaluates_all_selected_samples_with_sample_weighted_means(monkeypatch):
    from scripts import train_vla_multitask_book_v9_4_5 as entry

    calls = []
    loaders = {"action_" + phase: SimpleNamespace(dataset=list(range(257))) for phase in PHASES}

    def evaluate(state, loader, sharding, **kwargs):
        calls.append(kwargs)
        return 1.0 if loader is loaders["action_adjustment"] else 3.0

    monkeypatch.setattr(entry.training_base, "evaluate_action_loss", evaluate)
    result = entry.evaluate_action_by_phase(None, loaders, None, seed=142)
    assert result["loss"] == 2.0
    assert result["support_by_phase"] == {phase: 257 for phase in PHASES}
    assert all(c == {"seed": 142, "max_batches": None, "sample_weighted": True} for c in calls)


def test_partial_eval_batch_is_weighted_by_valid_samples(monkeypatch):
    from scripts import train_vla_stage_b_v3 as base
    import numpy as np

    model = SimpleNamespace(
        eval=lambda: None,
        backbone=SimpleNamespace(
            compute_loss=lambda rng, observation, actions: np.asarray(observation["losses"], dtype=np.float32)
        ),
    )
    monkeypatch.setattr(base.nnx, "merge", lambda *args: model)
    monkeypatch.setattr(base.nnx_utils, "module_jit", lambda fn: fn)
    monkeypatch.setattr(base, "pad_eval_batch", lambda batch, **kwargs: (batch, len(batch["losses"])))
    monkeypatch.setattr(base, "batch_to_jax", lambda batch, *args: (batch, None))
    state = SimpleNamespace(model_def=None, params=None)
    sharding = SimpleNamespace(device_set=[0])
    loader = [{"losses": [1.0] * 8}, {"losses": [3.0]}]
    assert base.evaluate_action_loss(
        state, loader, sharding, seed=42, max_batches=None, sample_weighted=True
    ) == pytest.approx(11 / 9)
    assert base.evaluate_action_loss(state, loader, sharding, seed=42) == 2.0
    assert base.evaluate_action_loss(state, loader, sharding, seed=42, max_batches=1) == 1.0


def test_old_evaluation_path_remains_the_default(monkeypatch):
    from scripts import train_vla_stage_b_v3 as base

    monkeypatch.setattr(base, "ACTION_EVALUATION_HOOK", None)
    monkeypatch.setattr(base, "evaluate_action_loss", lambda *args, **kwargs: 3.0)
    assert base.evaluate_action_report(None, {"action": object()}, None, seed=42) == {"loss": 3.0}


def test_loader_wiring_keeps_other_tasks_and_uses_same_eval_indices(monkeypatch, tmp_path):
    from scripts import train_vla_multitask_book_v9_4_5 as entry

    indices, lookup = example()
    path = tmp_path / "action_index.json"
    path.write_text("{}")
    dataset = [{"global_index": i} for i in indices]
    original = {split: {"action": SimpleNamespace(dataset=dataset), "need": object()} for split in ("train", "val")}
    monkeypatch.setattr(entry, "_BASE_BUILD_LOADERS", lambda *args: original)
    monkeypatch.setattr(entry.trainer, "validate_v7_4_adjustment_training_index", lambda *args, **kwargs: ({}, lookup))
    monkeypatch.setattr(entry.training_base, "EXTRA_CONFIG", {})
    args = SimpleNamespace(
        dataset_dir=tmp_path, batch_size=8, num_workers=0, seed=42, action_eval_samples_per_phase=8, dry_run=True
    )
    index = {
        "action_index_file": str(path),
        "splits": {split: {"action": {"indices": indices}} for split in ("train", "val")},
    }
    need = original["train"]["need"]
    loaders = entry.build_loaders(args, None, index, None, None, None, None)
    assert loaders["train"]["need"] is need
    batch = next(iter(loaders["train"]["action"]))["global_index"]
    assert sum(lookup[int(i)]["phase"] == "adjustment" for i in batch) == 4
    selection = entry.training_base.EXTRA_CONFIG["action_eval_selection"]
    for phase in PHASES:
        actual = [int(i) for b in loaders["val"]["action_" + phase] for i in b["global_index"]]
        assert actual == selection[phase]["global_indices"]


@pytest.mark.parametrize("steps", [2000, 4000])
def test_cli_defaults_and_export_identify_model_separately_from_data(monkeypatch, tmp_path, steps):
    from scripts import train_vla_multitask_book_v9_4_5 as entry

    trainer = entry.trainer
    for name in (
        "DATA_PROFILE",
        "DEFAULT_INDEX",
        "DEFAULT_STAGE_A",
        "DEFAULT_OUTPUT",
        "DEFAULT_DATASET",
        "DEFAULT_NORM_STATS",
        "RUN_NAME",
        "VERSION_TAG",
        "validate_index",
        "validate_v7_4_adjustment_training_index",
        "parse_args",
        "configure",
        "ensure_index",
        "ROOT",
        "_ARGS",
        "build_loaders",
    ):
        monkeypatch.setattr(trainer, name, getattr(trainer, name))
    monkeypatch.setattr(entry.training_base, "evaluate_text", entry.training_base.evaluate_text)
    argv = ["train_vla_multitask_book_v9_4_5.py", "--action-eval-samples-per-phase=257"]
    if steps != 2000:
        argv += ["--num-steps", str(steps)]
    monkeypatch.setattr(sys, "argv", argv)
    entry.configure_version()
    args = trainer.parse_args()
    assert args.num_steps == steps and args.action_eval_samples_per_phase == 257
    assert args.run_name == entry.RUN_NAME and args.output_dir == entry.DEFAULT_OUTPUT
    assert args.data_profile == entry.DATA_PROFILE and args.index_file == entry.DEFAULT_INDEX
    assert "v9_4_5" in str(args.index_file) and "v9_4_2" in str(args.stage_a_checkpoint)
    monkeypatch.setattr(entry.training_base, "EXTRA_CONFIG", {"action_eval_policy": {}, "action_eval_selection": {}})

    def export(run_dir, state, step, filter_):
        path = run_dir / str(step) / f"{entry.VERSION_TAG}_export.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"step": step}))

    monkeypatch.setattr(trainer, "export_checkpoint", export)
    entry.export_checkpoint(tmp_path, None, steps, None)
    metadata = json.loads((tmp_path / f"{steps}/{entry.VERSION_TAG}_export.json").read_text())
    assert metadata["experiment_version"] == "book_v9_4_5"
    assert metadata["data_experiment_version"] == "book_v9_4_5"
    assert metadata["need_label_policy"]["direction_offsets_frames"] == {"left": 15, "right": 8}
    assert metadata["need_label_policy"]["rotation_none_splits"] == ["train"]
    assert metadata["reasoning_window_policy"]["direction_offsets_frames"] == {"left": 15, "right": 8}
    assert metadata["action_sampling_policy"]["ratio"] == "1:1_per_batch"
