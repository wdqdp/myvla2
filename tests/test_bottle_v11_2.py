# ruff: noqa: E402
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src"), str(PROJECT / "openpi/src")]

from tactile_vla.vla.bottle_v11_2_stage_a_data import (
    compressed_h30_offsets,
    detect_idle_gap,
    target_policy,
)
from tactile_vla.vla.bottle_v11_2_multitask_data import (
    build_training_grammars,
    sample_compressed_adjustment_history,
    validate_stage_a_for_training,
)
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_7_phase_prompt import TimelinePoint


def signals(partial=False):
    actions = np.zeros((220, 7))
    if partial:
        actions[10:20, 6] = np.linspace(0, 0.04, 10)
        actions[20:30, 6] = 0.04
        actions[30:40, 6] = np.linspace(0.04, 0.1, 10)
        actions[40:, 6] = 0.1
    else:
        actions[10:20, 6] = np.linspace(0, 0.1, 10)
        actions[20:, 6] = 0.1
    actions[70:150, 5] = np.linspace(0, 1.2, 80)
    actions[150:, 5] = 1.2
    states = np.concatenate([np.repeat(actions[:1], 3, axis=0), actions[:-3]])
    return actions, states, np.arange(220) / 30


def test_detector_keeps_actual_gripper_arrival_and_settling():
    gap = detect_idle_gap(*signals(), reexecution_frame=200)
    assert gap["master_full_open"] == 19
    assert gap["puppet_full_open"] == 22
    assert gap["rotation_start"] == 71
    assert gap["removed_start"] == 28
    assert gap["removed_stop"] == 71
    assert gap["removed_frame_count"] == 43


def test_partial_opening_is_not_mistaken_for_full_open():
    gap = detect_idle_gap(*signals(partial=True), reexecution_frame=200)
    assert gap["master_full_open"] == 39
    assert gap["puppet_full_open"] == 42
    assert gap["removed_start"] == 48


def test_detector_refuses_meaningful_motion_in_removed_gap():
    actions, states, ts = signals()
    actions[45:48, 0] = 0.1
    with pytest.raises(ValueError, match="non-idle"):
        detect_idle_gap(actions, states, ts, reexecution_frame=200)


@pytest.mark.parametrize("settle", [-1, True, 1.5])
def test_settle_frames_must_be_nonnegative_integer(settle):
    with pytest.raises(ValueError, match="nonnegative integer"):
        target_policy(settle)


def test_offsets_skip_idle_and_fill_from_real_rotation_without_crossing_R():
    gap = {"removed_start": 30, "removed_stop": 60, "rexecution_frame": 100}
    assert compressed_h30_offsets(20, gap) == [*range(10), *range(40, 60)]
    assert compressed_h30_offsets(0, gap) == list(range(30))
    assert compressed_h30_offsets(70, gap) == list(range(30))
    assert compressed_h30_offsets(71, gap) is None
    for start in range(30, 60):
        assert compressed_h30_offsets(start, gap) is None
    for start in range(100):
        offsets = compressed_h30_offsets(start, gap)
        if offsets is not None:
            assert len(offsets) == 30 and offsets[0] == 0
            assert all(a < b for a, b in zip(offsets, offsets[1:]))
            frames = [start + x for x in offsets]
            assert max(frames) < 100
            assert not any(30 <= x < 60 for x in frames)


def timeline():
    points = []
    for attempt, count in ((1, 120), (2, 200)):
        for frame in range(count):
            index = len(points)
            points.append(TimelinePoint(1, attempt, frame, index, index / 30, np.full(7, index)))
    return points


def test_adjustment_history_compresses_before_sampling_and_connects_attempt1():
    gaps = {(1, 2): {"removed_start": 30, "removed_stop": 60}}
    raw = timeline()
    points, length = sample_compressed_adjustment_history(raw, 120 + 75, gaps=gaps)
    assert length == 100 and len(points) == 11
    assert points[0].global_index == 66
    assert points[-1].frame_index == 75
    assert any(p.attempt_id == 1 for p in points)
    assert all(p.attempt_id != 2 or not 30 <= p.frame_index < 60 for p in points)
    assert all(p.global_index <= 195 for p in points)


def test_current_inside_gap_is_preserved_but_previous_idle_frames_are_not():
    gaps = {(1, 2): {"removed_start": 30, "removed_stop": 60}}
    points, _ = sample_compressed_adjustment_history(timeline(), 165, gaps=gaps)
    assert points[-1].frame_index == 45
    assert all(p.attempt_id != 2 or not 30 <= p.frame_index < 60 for p in points[:-1])


def test_history_has_no_future_access_and_padding_remains_episode_local():
    raw = timeline()
    gaps = {(1, 2): {"removed_start": 30, "removed_stop": 60}}
    one, length = sample_compressed_adjustment_history(raw, 5, gaps=gaps)
    assert length == 6 and one[0].global_index == 0 and one[-1].global_index == 5
    past, _ = sample_compressed_adjustment_history(raw, 160, gaps=gaps)
    prefix, _ = sample_compressed_adjustment_history(raw[:161], 160, gaps=gaps)
    assert [p.global_index for p in past] == [p.global_index for p in prefix]


def test_checkpoint_identity_checks_do_not_pin_saved_or_planned_step(tmp_path):
    import json

    checkpoint = tmp_path / "10000"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params/_METADATA").write_text("{}")
    action = {
        "source_scope": {"data": "bottle"},
        "target_policy": {"failure_action_filter": {}},
        "idle_gap_audit": {"attempts": []},
        "v4_norm_stats_sha256": "norm",
    }
    action_path = tmp_path / "action.json"
    action_path.write_text(json.dumps(action))
    index = {"action_index_file": str(action_path), "action_training_data_hash": "data"}
    config = {
        "experiment_version": "bottle_v11_2",
        "num_steps": 99999,
        "use_state_history": False,
        "data_profile": "book_stage_a_v1",
        "source_scope": action["source_scope"],
        "failure_action_policy": {},
        "action_target_policy": action["target_policy"],
        "idle_gap_audit_sha256": sha256_json(action["idle_gap_audit"]),
        "artifact_identity": {
            "training_data_hash": "data",
            "index_sha256": sha256_file(action_path),
            "v4_norm_stats_sha256": "norm",
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert validate_stage_a_for_training(checkpoint, index)["step"] == 10000
    config["experiment_version"] = "bottle_v11_1"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_stage_a_for_training(checkpoint, index)


def test_text_grammar_keeps_full_output_space_but_requires_v11_2_identity():
    from types import SimpleNamespace
    from tactile_vla.vla.bottle_v11_1_grammar import TEXT_SCHEMA_POLICY

    class Tokenizer:
        def encode_text(self, text, *, add_eos):
            return [ord(c) for c in text] + ([2] if add_eos else [])

    args = SimpleNamespace(data_profile="bottle_v11_2_five_task_h100")
    index = {"experiment_version": "bottle_v11_2", "text_schema_policy": TEXT_SCHEMA_POLICY}
    failure, plan = build_training_grammars(Tokenizer(), args=args, index=index)
    assert len(failure.texts) == 179 and len(plan.texts) == 636
    index["experiment_version"] = "bottle_v11_1"
    with pytest.raises(ValueError, match="versioned data/schema"):
        build_training_grammars(Tokenizer(), args=args, index=index)


def test_stage_a_configurable_fields_do_not_relax_legacy_or_h30_protocol(monkeypatch):
    from scripts import train_vla_stage_a_bottle_v11_2 as entry

    trainer = entry.previous.trainer
    args = entry.parse_args(
        [
            "--num-steps",
            "7500",
            "--batch-size",
            "16",
            "--fsdp-devices",
            "2",
            "--lr-transition-steps",
            "3500",
        ]
    )
    assert (args.num_steps, args.batch_size, args.fsdp_devices, args.lr_transition_steps) == (7500, 16, 2, 3500)
    monkeypatch.setattr(trainer, "selected_stage_a_protocol", entry.selected_stage_a_protocol)
    trainer.validate_v4_training_protocol(args)
    _, protocol = entry.selected_stage_a_protocol(args)
    assert not entry.CONFIGURABLE_FIELDS & protocol.keys()
    assert protocol["action_horizon"] == 30 and protocol["use_state_history"] is False
    schedule = trainer.make_lr_schedule(args)
    assert float(schedule(3499)) == pytest.approx(5e-5)
    assert float(schedule(3500)) == pytest.approx(5e-7)
    args.action_horizon = 20
    with pytest.raises(ValueError, match="protocol mismatch"):
        trainer.validate_v4_training_protocol(args)
    args.action_horizon = 30
    args.experiment_version = "bottle_v11_1"
    with pytest.raises(ValueError, match="protocol mismatch"):
        trainer.validate_v4_training_protocol(args)


@pytest.mark.parametrize(
    "flag,value",
    [("--num-steps", "0"), ("--batch-size", "0"), ("--fsdp-devices", "0"), ("--lr-transition-steps", "-1")],
)
def test_stage_a_configurable_fields_still_require_legal_values(flag, value):
    from scripts import train_vla_stage_a_bottle_v11_2 as entry

    with pytest.raises(ValueError):
        entry.parse_args([flag, value])
