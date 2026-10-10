from __future__ import annotations

# ruff: noqa: E402
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from tactile_vla.vla import sponge_v10_1_stage_a_data as stage
from tactile_vla.vla import sponge_v10_1_multitask_data as multi
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.structured_text import legal_failure_reasons
from tactile_vla.vla.v7_7_multitask_data import NEGATIVE_SOURCES


def failed_meta(**changes):
    return {
        "episode_id": 21,
        "attempt_id": 1,
        "result": "failure",
        "frame_count": 797,
        "failure_boundary_source": "captioner_fz_bias",
        "failure_boundary_status": "resolved",
        "fz_bias_direction": "left",
        "rotation_direction": "none",
        "grasp_position": "appropriate",
        "captioner_left_start_frame": 711,
        "failure_boundary_frame_index": 714,
        "failure_window_start": 714,
    } | changes


@pytest.mark.parametrize("s,keep", [(0, True), (684, True), (685, False), (713, False), (714, False), (767, False)])
def test_H30_failure_boundary_off_by_one(s, keep):
    assert stage.failure_safe_chunk(s, 714) is keep


def test_native_C_is_not_the_book_shift_or_R():
    assert stage.failure_boundary(failed_meta(shift_frame_index=None, rexecution_frame_index=None)) == (711, 714)


@pytest.mark.parametrize(
    "change",
    [
        {"failure_boundary_status": "pending"},
        {"failure_boundary_source": "shift"},
        {"fz_bias_direction": "right"},
        {"rotation_direction": "right"},
        {"failure_boundary_frame_index": 713},
        {"failure_window_start": 715},
        {"frame_count": 728},
    ],
)
def test_invalid_or_short_native_boundary_rejected(change):
    with pytest.raises(ValueError):
        stage.failure_boundary(failed_meta(**change))


@pytest.mark.parametrize(
    "frame,label,source",
    [
        (680, False, NEGATIVE_SOURCES[0]),
        (710, False, NEGATIVE_SOURCES[0]),
        (711, None, None),
        (712, None, None),
        (713, None, None),
        (714, True, "failure_active"),
        (796, True, "failure_active"),
    ],
)
def test_need_B_C_boundary(frame, label, source):
    f = SimpleNamespace(frame_index=frame, attempt_id=1)
    assert multi.need_role(f, failed_meta()) == (label, source)


@pytest.mark.parametrize("frame,label", [(0, None), (214, None), (215, False), (216, False)])
def test_need_excludes_attempt2_adjustment(frame, label):
    result, _ = multi.need_role(
        SimpleNamespace(frame_index=frame, attempt_id=2), {"result": "success", "rexecution_frame_index": 215}
    )
    assert result is label


def need_row(g, source, label=False):
    return {
        "global_index": g,
        "episode_id": 21,
        "attempt_id": 1,
        "frame_index": g,
        "source": source,
        "need_recovery": label,
    }


def test_need_reserves_boundary_negatives_and_balances_sources():
    positives = [need_row(i, "failure_active", True) for i in range(100)]
    negatives = {s: [need_row(1000 + k * 1000 + i, s) for i in range(500)] for k, s in enumerate(NEGATIVE_SOURCES)}
    reserved = negatives[NEGATIVE_SOURCES[0]][:30]
    first, audit = multi.select_need(positives, negatives, reserved, split="train")
    second, _ = multi.select_need(positives, negatives, reserved, split="train")
    assert first == second and len(first) == 400
    assert audit["selected_negative"] == dict.fromkeys(NEGATIVE_SOURCES, 100)
    assert {r["global_index"] for r in reserved} <= {r["global_index"] for r in first}
    assert sum(r["need_recovery"] for r in first) == 100


def test_need_reservation_exceeding_one_source_quota_is_not_dropped():
    positives = [need_row(i, "failure_active", True) for i in range(10)]
    negatives = {s: [need_row(100 + k * 100 + i, s) for i in range(50)] for k, s in enumerate(NEGATIVE_SOURCES)}
    rows, audit = multi.select_need(positives, negatives, negatives[NEGATIVE_SOURCES[0]][:20], split="train")
    assert len(rows) == 40 and audit["selected_negative"][NEGATIVE_SOURCES[0]] == 20
    with pytest.raises(ValueError):
        multi.select_need(positives, negatives, negatives[NEGATIVE_SOURCES[0]][:31], split="train")


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_fz_bias_history_preserves_initial_real_terminal_and_no_rotation(length):
    memory = multi.make_memory(length, split="train", global_index=714)
    assert len(memory) == length and memory[0]["recovery_plan"] == "initial plan"
    assert memory[-1]["source_type"] == "real_current_failure"
    assert all(m["failure_reason"] == multi.FAILURE_TARGET for m in memory)
    assert all(m["recovery_plan"] == multi.PLAN_TARGET for m in memory[1:])
    assert all(m["source_type"] == "synthetic" for m in memory[:-1])
    assert not any("donor_episode_id" in m or "slightly" in str(m) for m in memory)
    assert memory == multi.make_memory(length, split="train", global_index=714)


def test_extended_grammar_required_and_rotation_none_is_not_relabelled():
    assert multi.FAILURE_TARGET == "failure_reason=rotate none,grasp appropriate,fz_bias left."
    assert multi.FAILURE_TARGET not in legal_failure_reasons()
    assert multi.FAILURE_TARGET in legal_failure_reasons(include_fz_bias=True)
    assert multi.PLAN_TARGET == "recovery_plan=move horizontally none moderately, move vertically up moderately."
    assert multi.NEED_POLICY["counterfactual"] is False


def test_stage_a_filter_build_is_source_derived(tmp_path, monkeypatch):
    v4_file = tmp_path / "v4.json"
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps({"attempts": [failed_meta()]}))
    v4_file.write_text(json.dumps({"source_files": {"profile": {"path": str(profile_file)}}}))
    rows = [
        {
            "global_index": s,
            "episode_id": 21,
            "attempt_id": 1,
            "frame_index": s,
            "split": "train",
            "phase": "execution",
            "trainable": True,
            "chunk_phase_pure": True,
            "raw_chunk_phase_pure": True,
            "exclusion_reason": None,
        }
        for s in (684, 685, 714)
    ]
    common_index = {"source_files": {}, "data_config_hash": "original"}
    monkeypatch.setattr(
        stage.common, "build_artifacts", lambda **_: (copy.deepcopy(rows), copy.deepcopy(common_index), {})
    )
    args = {"dataset_dir": tmp_path / "dataset", "v4_index_file": v4_file, "norm_stats_dir": tmp_path}
    result, index, summary = stage.build_artifacts(**args)
    assert index["splits"]["train"]["execution_indices"] == [684]
    assert result[1]["exclusion_reason"] == "crosses_failure_C"
    assert result[2]["exclusion_reason"] == "failure_at_or_after_C"
    assert summary["splits"]["train"]["trainable_count"] == 1
    assert rows[1]["trainable"]  # Original common rows were not mutated.


def test_builders_are_checkpoint_free_and_refuse_overwrite(tmp_path):
    from scripts.prepare_sponge_stage_a_v10_1_index import parse_args as stage_args, main as stage_main
    from scripts.build_sponge_v10_1_multitask_data import parse_args as multi_args, main as multi_main

    s = stage_args(["--sponge-root", str(tmp_path)])
    m = multi_args(["--sponge-root", str(tmp_path)])
    assert not hasattr(s, "checkpoint") and not hasattr(m, "stage_a_checkpoint")
    assert s.output_dir == tmp_path / "outputs/sponge_stage_a_v10_1"
    assert m.output_dir == tmp_path / "outputs/sponge_v10_1_multitask"
    for entry, args in [(stage_main, s), (multi_main, m)]:
        args.output_dir.mkdir(parents=True)
        with pytest.raises(FileExistsError):
            entry(["--sponge-root", str(tmp_path)])


def test_stage_a_training_defaults_and_wrong_initialization_rejected(tmp_path):
    from scripts import train_vla_stage_a_sponge_v10_1 as entry

    args = entry.parse_args(["--sponge-root", str(tmp_path)])
    assert args.experiment_version == "sponge_v10_1" and args.num_steps == 15000
    assert args.run_name == stage.RUN_NAME and args.batch_size == 8 and args.fsdp_devices == 2
    assert not args.use_state_history and args.state_history_len == args.history_hidden_dim == 0
    entry.trainer.validate_v4_training_protocol(args)
    with pytest.raises(ValueError, match="pi05_base"):
        entry.parse_args(["--checkpoint", str(tmp_path / "old_multitask/params")])


def test_multitask_training_defaults_equals_and_data_only_guard(tmp_path):
    from scripts import train_vla_multitask_sponge_v10_1 as entry

    args = entry.parse_args(["--sponge-root", str(tmp_path), "--data-only-dry-run", "--num-steps=2000"])
    assert args.dry_run and args.data_only_dry_run
    assert args.num_steps == 2000 and args.eval_interval == args.save_interval == args.keep_period == 1000
    assert args.stage_a_checkpoint == tmp_path / "outputs/stage_a_action" / stage.RUN_NAME / "15000"
    assert args.data_profile == multi.DATA_PROFILE and args.run_name == multi.RUN_NAME
    assert not args.use_state_history and args.max_token_len == 512 and args.reasoning_max_token_len == 320
    with pytest.raises(ValueError, match="protocol mismatch"):
        entry.parse_args(["--batch-size", "16"])
    with pytest.raises(ValueError, match="five-task|multiple of 5"):
        entry.parse_args(["--num-steps", "2001"])


def stage_checkpoint_fixture(tmp_path):
    run = tmp_path / "stage_a"
    checkpoint = run / "15000"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params/_METADATA").write_text("metadata")
    action_path = tmp_path / "action_index.json"
    action = {"source_scope": {"dataset_dir": "sponge"}, "v4_norm_stats_sha256": "norm"}
    action_path.write_text(json.dumps(action))
    config = {
        "experiment_version": stage.VERSION_TAG,
        "use_state_history": False,
        "data_profile": "book_stage_a_v1",
        "num_steps": 15000,
        "failure_action_policy": stage.FAILURE_ACTION_POLICY,
        "source_scope": action["source_scope"],
        "artifact_identity": {
            "training_data_hash": "action_hash",
            "index_sha256": sha256_file(action_path),
            "v4_norm_stats_sha256": "norm",
        },
    }
    (run / "config.json").write_text(json.dumps(config))
    return checkpoint, {"action_index_file": str(action_path), "action_training_data_hash": "action_hash"}, config


def test_stage_a_checkpoint_identity_is_checked_only_at_training(tmp_path):
    checkpoint, index, config = stage_checkpoint_fixture(tmp_path)
    assert multi.validate_stage_a_for_training(checkpoint, index)["step"] == 15000
    config["experiment_version"] = "book_v9_5"
    (checkpoint.parent / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mismatch"):
        multi.validate_stage_a_for_training(checkpoint, index)
    with pytest.raises(ValueError, match="completed"):
        multi.validate_stage_a_for_training(checkpoint.parent / "10000", index)


def test_rehashing_multitask_index_cannot_change_label_policy(monkeypatch, tmp_path):
    expected = {
        "schema_version": multi.INDEX_SCHEMA,
        "experiment_version": stage.VERSION_TAG,
        "data_profile": multi.DATA_PROFILE,
        "source_hashes": {},
        "dataset_dir": str(tmp_path),
        "v4_dir": str(tmp_path),
        "action_index_file": str(tmp_path / "action.json"),
        "incremental_state_dir": str(tmp_path),
        "need_policy": multi.NEED_POLICY,
    }
    monkeypatch.setattr(multi, "build_artifacts", lambda **_: (expected, {}, {}))
    changed = copy.deepcopy(expected)
    changed["need_policy"]["C"] = "B+27"
    changed["training_data_hash"] = sha256_json(changed)
    with pytest.raises(ValueError, match="need_policy"):
        multi.validate_index(changed)


def test_need_group_eval_uses_integer_support_not_None(monkeypatch):
    from scripts import train_vla_multitask_sponge_v10_1 as entry

    calls = []

    def evaluate(*args, max_samples):
        assert isinstance(max_samples, int)
        calls.append(max_samples)
        return {"confusion_matrix": [[2, 1], [0, 0]]}

    monkeypatch.setattr(entry, "_BASE_NEED", evaluate)
    group = SimpleNamespace(dataset=[1, 2, 3])
    loader = SimpleNamespace(v10_1_groups={"pre_failure_hard_negative": group})
    result = entry.evaluate_need(None, loader, None, max_samples=632)
    assert calls == [632, 3]
    assert result["by_source"]["pre_failure_hard_negative"]["false_positive_rate"] == 1 / 3


def test_plan_eval_visits_all_memory_lengths(monkeypatch):
    from scripts import train_vla_multitask_sponge_v10_1 as entry

    raw = SimpleNamespace(rows=[{"memory_length": k} for k in range(1, 5)], row_indices=list(range(4)))

    class Dataset:
        dataset = raw

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return i

    loader = SimpleNamespace(dataset=Dataset(), batch_size=8)
    monkeypatch.setattr(entry.base, "_loader", lambda dataset, **_: SimpleNamespace(dataset=dataset))
    calls = []

    def evaluate(*args, max_samples):
        calls.append(max_samples)
        return {"num_samples": 1, "exact_match": 1.0}

    monkeypatch.setattr(entry, "_BASE_TEXT", evaluate)
    result = entry.evaluate_text(None, loader, "plan", None, None, max_samples=1)
    assert calls == [None] * 4
    assert result["num_samples"] == 4 and set(result["by_memory_length"]) == {"1", "2", "3", "4"}


def test_fz_bias_grammar_opt_in_is_scoped_and_exported(monkeypatch):
    from scripts import train_vla_multitask_sponge_v10_1 as entry

    monkeypatch.setattr(entry, "_BASE_CONFIGURE", lambda: None)
    monkeypatch.setattr(entry.base, "EXTRA_CONFIG", {})
    for name in (
        "CHECKPOINT_EXPORT_HOOK",
        "ACTION_EVALUATION_HOOK",
        "evaluate_text",
        "evaluate_need",
        "validate_v4_resume_config",
    ):
        monkeypatch.setattr(entry.base, name, getattr(entry.base, name))
    entry.configure_training()
    assert entry.base.EXTRA_CONFIG["include_fz_bias_failure_grammar"] is True
    assert entry.base.EXTRA_CONFIG["experiment_version"] == stage.VERSION_TAG
    assert entry.base.CHECKPOINT_EXPORT_HOOK is entry.export_checkpoint
    assert entry.base.ACTION_EVALUATION_HOOK is entry.evaluate_action_by_phase
