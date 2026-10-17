"""Joint task identity, filtering, shared normalization, histories and sampling."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "openpi/src")]

from tactile_vla.vla import v12_1_data as data
from tactile_vla.vla.v12_1_runtime import BalancedGroupsBatchSampler, check_args


def touch(direction):
    return f"Touch[area=full; Fx=near_zero; Fy=near_zero; Fz=negative; rotation={direction}]"


@pytest.mark.parametrize(
    "domain,direction,observed,expected",
    [
        ("blackbox", "right", "clockwise", True),
        ("blackbox", "left", "counterclockwise", True),
        ("blackbox", "right", "counterclockwise", False),
        ("blackbox", "left", "clockwise", False),
        ("book", "left", "clockwise", True),
        ("book", "right", "counterclockwise", True),
        ("book", "left", "counterclockwise", False),
        ("book", "right", "clockwise", False),
        ("blackbox", "left", "none", False),
        ("book", "right", "none", False),
    ],
)
def test_direction_mapping_is_task_conditioned(domain, direction, observed, expected):
    row = {"episode_id": 261, "attempt_id": 1, "prompt": touch(observed)}
    assert data.correct_rotation(row, domain, {(261, 1): {"rotation_direction": direction}}) is expected


def test_canonical_touch_marks_unmeasured_bias_unknown_and_preserves_measured_bias():
    caption = touch("clockwise")
    canonical = data.canonical_touch(caption)
    assert "Fz_bias=unknown" in canonical and data.rotation(canonical) == "clockwise"
    assert data.canonical_touch(canonical) == canonical
    measured = caption.replace("; rotation=", "; Fz_bias=left; rotation=")
    assert data.canonical_touch(measured) == measured


@pytest.mark.parametrize("bad", ["", touch("none") + touch("clockwise"), "Touch[rotation=wrong]"])
def test_invalid_caption_is_not_silently_treated_as_none(bad):
    with pytest.raises(ValueError):
        data.rotation(bad)


def test_exact_shared_statistics_are_equal_task_mass_not_equal_frame_mass():
    a = np.tile(np.linspace(0, 2, 10)[:, None], (1, 7))
    b = np.tile(np.linspace(10, 12, 100)[:, None], (1, 7))
    stats = data.weighted_statistics([a, b])
    assert np.allclose(stats["mean"], 6)
    assert not np.allclose(stats["mean"], np.concatenate([a, b]).mean(axis=0))
    assert np.allclose(data.weighted_statistics([a, np.repeat(b, 3, axis=0)])["mean"], stats["mean"])
    assert np.all(np.asarray(stats["q99"]) > np.asarray(stats["q01"]))


@pytest.mark.parametrize("parts", [[np.zeros((0, 7)), np.ones((4, 7))], [np.zeros((3, 7)), np.zeros((3, 7))]])
def test_shared_stats_reject_empty_or_degenerate_domains(parts):
    with pytest.raises(ValueError):
        data.weighted_statistics(parts)


def test_action_sampler_has_two_samples_in_each_domain_phase_and_is_deterministic():
    pools = {
        "blackbox/adjustment": list(range(12)),
        "blackbox/execution": list(range(12, 48)),
        "book/adjustment": list(range(48, 56)),
        "book/execution": list(range(56, 120)),
    }
    first = list(BalancedGroupsBatchSampler(pools, batch_size=8, seed=42))
    assert first == list(BalancedGroupsBatchSampler(pools, batch_size=8, seed=42))
    for batch in first:
        assert len(batch) == len(set(batch)) == 8
        assert all(sum(i in p for i in batch) == 2 for p in pools.values())
    small_pass = [i for batch in first[:4] for i in batch if i in pools["book/adjustment"]]
    assert len(set(small_pass)) == 8


def test_classification_sampler_is_one_to_one_between_tasks():
    pools = {"blackbox": list(range(16)), "book": list(range(16, 40))}
    assert all(sum(i < 16 for i in batch) == 4 for batch in BalancedGroupsBatchSampler(pools))


@pytest.mark.parametrize(
    "pools,batch", [({"a": [0, 1], "b": [1, 2]}, 4), ({"a": [0], "b": [1]}, 8), ({"a": [0], "b": [1]}, 3)]
)
def test_sampler_rejects_overlaps_undersized_pools_and_invalid_batch(pools, batch):
    with pytest.raises(ValueError):
        BalancedGroupsBatchSampler(pools, batch_size=batch)


def test_history_is_rediscretized_not_resampled_and_cannot_cross_episode():
    stats = data.StateQuantileStats(np.full(7, -1), np.full(7, 1))
    frame = {"episode_id": 261, "attempt_id": 1, "frame_index": 4, "ros_timestamp": 2.0, "observation.state": [0] * 7}
    row = {
        "episode_id": 261,
        "attempt_id": 1,
        "frame_index": 4,
        "global_index": 8,
        "timestamp": 2.0,
        "history_sources": [{"global_index": 8, "attempt_id": 1, "frame_index": 4}] * 11,
        "prompt": f"Mode: phase\nTouch: {touch('clockwise')}\nState history H100 sampled to 11 points: []",
        "need_recovery": True,
    }
    result = data.rediscretize(row, {"frames": {8: frame}}, stats)
    assert result["history_sources"] == row["history_sources"]
    assert result["need_recovery"] is True
    assert np.asarray(result["qpos_h100_11_discrete"]).shape == (11, 7)
    # The production quantile transform includes +1e-6 in its denominator.
    assert np.all(np.asarray(result["qpos_h100_11_discrete"]) == 127)
    assert row["prompt"].endswith("[]")  # Source artifact was not mutated.
    bad = copy.deepcopy(frame)
    bad["episode_id"] = 262
    with pytest.raises(ValueError, match="History crosses episode"):
        data.rediscretize(row, {"frames": {8: bad}}, stats)


def need_context():
    frames, metas, lookup = {}, {}, {}
    for episode, attempt, count, task, result in [
        (261, 1, 120, "moderate_lift", "failure"),
        (262, 1, 400, "one_success", "success"),
        (261, 2, 400, "moderate_lift", "success"),
    ]:
        metas[episode, attempt] = {
            "split": "train",
            "task": task,
            "result": result,
            "shift_frame_index": 50 if result == "failure" else None,
            "rotation_direction": "right" if result == "failure" else "none",
        }
        for local in range(count):
            g = len(frames)
            observed = "clockwise" if result == "failure" and local >= 52 else "none"
            if result == "failure" and local == 51:
                observed = "counterclockwise"
            frames[g] = {
                "index": g,
                "episode_id": episode,
                "attempt_id": attempt,
                "frame_index": local,
                "ros_timestamp": float(g),
                "observation.state": [0] * 7,
                "instruction": "transfer",
                "tactile_caption": touch(observed),
                "input_recovery_plan": "none",
            }
            lookup[g] = {
                "episode_id": episode,
                "attempt_id": attempt,
                "frame_index": local,
                "phase": "adjustment" if attempt == 2 and local <= 20 else "execution",
            }
    rows = [
        {
            "episode_id": 261,
            "attempt_id": 1,
            "frame_index": g,
            "global_index": g,
            "need_recovery": True,
            "prompt": touch("none" if g == 50 else "counterclockwise" if g == 51 else "clockwise"),
        }
        for g in range(50, 120)
    ]
    return {
        "frames": frames,
        "metas": metas,
        "lookup": lookup,
        "manifests": {"need": rows},
        "index": {"splits": {"train": {"need": {"manifest_row_indices": list(range(len(rows)))}}}},
    }


def test_need_drops_none_and_opposite_without_relabel_and_reserves_preF30():
    ctx = need_context()
    selected, dropped = data.blackbox_need(ctx, "train", 42)
    positive = [r for r in selected if r["need_recovery"]]
    negative = [r for r in selected if not r["need_recovery"]]
    assert len(positive) == 68 and len(negative) == 204
    assert {r["global_index"] for r in dropped} == {50, 51}
    assert {50, 51}.isdisjoint(r["global_index"] for r in selected)
    assert set(range(20, 50)).issubset(r["global_index"] for r in negative)
    assert all(r["history_sources"][-1]["global_index"] == r["global_index"] for r in negative)
    assert selected == data.blackbox_need(ctx, "train", 42)[0]


def test_phase_history_joins_attempt1_tail_without_episode_leak():
    ctx = need_context()
    timeline = [r for r in ctx["frames"].values() if r["episode_id"] == 261]
    current = next(r for r in timeline if r["attempt_id"] == 2 and r["frame_index"] == 1)
    timelines = {261: (timeline, {r["index"]: i for i, r in enumerate(timeline)})}
    row = data.phase_row(current, ctx, timelines)
    assert row["history_sources"][0]["attempt_id"] == 1
    assert row["history_sources"][-1]["attempt_id"] == 2


def test_cli_allows_manual_steps_batch_fsdp_and_transition(monkeypatch):
    from scripts import train_vla_stage_a_v12_1 as stage

    monkeypatch.setattr(
        sys, "argv", ["train", "--num-steps=7500", "--batch-size=16", "--fsdp-devices=2", "--lr-transition-steps=3500"]
    )
    args = stage.parse_args()
    assert (args.num_steps, args.batch_size, args.fsdp_devices, args.lr_transition_steps) == (7500, 16, 2, 3500)
    assert args.use_state_history is False
    assert "pi05_base" in args.checkpoint


def test_multitask_cli_free_steps_and_five_task_cycle(monkeypatch):
    from scripts import train_vla_multitask_v12_1 as multi

    monkeypatch.setattr(sys, "argv", ["train", "--num-steps=2000", "--batch-size=16", "--fsdp-devices=2"])
    args = multi.parse_args()
    assert (args.num_steps, args.batch_size, args.fsdp_devices) == (2000, 16, 2)
    assert args.state_history_len == 0 and args.adjustment_loss_weight == 1
    monkeypatch.setattr(sys, "argv", ["train", "--num-steps=2001"])
    with pytest.raises(ValueError, match="five-task"):
        multi.parse_args()


def test_bad_joint_batch_is_rejected():
    args = SimpleNamespace(batch_size=6)
    with pytest.raises(ValueError, match="divisible by 4"):
        check_args(args)


def test_data_builder_refuses_overwrite_and_does_not_require_checkpoint(tmp_path):
    with pytest.raises(FileExistsError):
        data.build(tmp_path)


def test_filtered_need_rows_cannot_change_source_labels_by_mutation():
    ctx = need_context()
    before = json.dumps(ctx["manifests"])
    data.blackbox_need(ctx, "train", 42)
    assert json.dumps(ctx["manifests"]) == before


def test_old_blackbox_dataset_is_rejected_before_reading_frames(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(data.seal({"dataset_dir": str(data.ROOT / "black_box/lerobot_data/tactile_vla_rotation_v4")}))
    )
    with pytest.raises(ValueError, match="old blackbox data are forbidden"):
        data.load_sources({"blackbox": source, "book": source})


def test_stage_a_rejects_old_task_initialization(monkeypatch):
    from scripts import train_vla_stage_a_v12_1 as stage

    monkeypatch.setattr(sys, "argv", ["train", "--checkpoint", "/tmp/v7_4/15000/params"])
    with pytest.raises(ValueError, match="raw pi05_base"):
        stage.parse_args()


def test_action_gate_cannot_hide_one_tasks_regression_in_the_overall_mean():
    from scripts.train_vla_stage_b_v3 import action_retention_metrics

    names = [f"{d}/{p}" for d in data.DOMAINS for p in ("adjustment", "execution")]
    baseline = {"loss": 1, "by_phase": dict.fromkeys(names, 1), "support_by_phase": dict.fromkeys(names, 256)}
    report = copy.deepcopy(baseline)
    report["by_phase"]["blackbox/adjustment"] = 1.2
    report["loss"] = 1.05
    metrics = action_retention_metrics(report, baseline, limit=0.1)
    assert metrics["action_gate_passed"] is False


def test_classification_best_score_is_equal_task_weight_not_equal_sample_weight():
    from scripts.train_vla_multitask_v12_1 import grouped_classification

    loader = SimpleNamespace(dataset=list(range(110)))
    loader.domain_loaders = {
        "blackbox": SimpleNamespace(dataset=list(range(100))),
        "book": SimpleNamespace(dataset=list(range(10))),
    }

    def evaluate(_state, selected, _sharding, *, max_samples):
        assert max_samples == len(selected.dataset)
        return {"macro_f1": 0.2 if len(selected.dataset) == 100 else 1.0, "confusion_matrix": [[4, 1], [1, 4]]}

    report = grouped_classification(evaluate, None, loader, None, max_samples=1)
    assert report["macro_f1"] == pytest.approx(0.6)
    assert report["by_task"]["blackbox"]["false_positive_rate"] == pytest.approx(0.2)
