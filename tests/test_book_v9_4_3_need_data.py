from __future__ import annotations

# ruff: noqa: E402
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from tactile_vla.vla.book_v9_4_3_multitask_data import (
    DEFAULT_ACTION_INDEX,
    DEFAULT_ADJUSTMENT_DIR,
    DEFAULT_MULTITASK_DIR,
    DEFAULT_STAGE_A,
    expected_counts,
    validate_index,
)
from tactile_vla.vla.book_v9_4_3_need_data import (
    LABEL_POLICY,
    SAMPLING_POLICY,
    boundary_annotation,
    build_need_rows,
    failure_boundary,
    need_role,
    positive_counts,
    select_need_rows,
    validate_need_rows,
)
from tactile_vla.vla.book_v9_4_multitask_data import expected_counts as old_counts
from tactile_vla.vla.v7_7_multitask_data import NEGATIVE_SOURCES


def example_profile():
    attempts = []
    for i, split in enumerate(("train", "val", "test")):
        for episode, attempt, direction, result, task, f, n, r in (
            (10 + i * 10, 1, "left", "failure", "moderate_lift", 50, 100, None),
            (11 + i * 10, 1, "right", "failure", "moderate_lift", 60, 120, None),
            (12 + i * 10, 1, "none", "success", "one_success", None, 180, None),
            (10 + i * 10, 2, "none", "success", "moderate_lift", None, 180, 50),
        ):
            attempts.append(
                {
                    "episode_id": episode,
                    "attempt_id": attempt,
                    "rotation_direction": direction,
                    "result": result,
                    "task": task,
                    "split": split,
                    "shift_frame_index": f,
                    "frame_count": n,
                    "rexecution_frame_index": r,
                }
            )
    return {"attempts": attempts}


def base_row(frame, split, source):
    return {
        "episode_id": frame.attempt_key[0],
        "attempt_id": frame.attempt_key[1],
        "frame_index": frame.frame_index,
        "global_index": frame.global_index,
        "split": split,
        "source": source,
    }


def build_example():
    profile = example_profile()
    frames, global_index = [], 0
    for meta in profile["attempts"]:
        for f in range(meta["frame_count"]):
            frames.append(
                SimpleNamespace(
                    attempt_key=(meta["episode_id"], meta["attempt_id"]), frame_index=f, global_index=global_index
                )
            )
            global_index += 1
    rows, summaries = [], {}
    for split in ("train", "val", "test"):
        sampled, summaries[split] = build_need_rows(
            frames=frames, profile=profile, split=split, base_row=base_row, seed=42
        )
        rows.extend(sampled)
    return profile, frames, rows, summaries


@pytest.mark.parametrize("direction,offset", [("left", 12), ("right", 5)])
def test_need_delayed_positive_inclusive_and_ignore_not_false(direction, offset):
    meta = example_profile()["attempts"][0] | {"rotation_direction": direction}
    f, c = failure_boundary(meta)
    assert c == f + offset
    assert need_role(f - 1, meta) == (False, NEGATIVE_SOURCES[0])
    assert all(need_role(frame, meta) == (None, None) for frame in range(f, c))
    assert need_role(c, meta) == (True, "failure_active")
    assert need_role(meta["frame_count"] - 1, meta) == (True, "failure_active")
    assert boundary_annotation(f - 30, meta)["pre_failure_boundary"]
    assert not boundary_annotation(f - 31, meta)["pre_failure_boundary"]


def test_success_domains_and_invalid_failure_are_rejected():
    profile = example_profile()
    recovery = profile["attempts"][3]
    assert need_role(49, recovery) == (None, None)
    assert need_role(50, recovery) == (False, NEGATIVE_SOURCES[2])
    assert need_role(0, profile["attempts"][2]) == (False, NEGATIVE_SOURCES[1])
    for change in ({"rotation_direction": "front"}, {"frame_count": 62}, {"shift_frame_index": -1}):
        with pytest.raises(ValueError):
            failure_boundary(profile["attempts"][0] | change)
    with pytest.raises(ValueError, match="native R"):
        need_role(55, recovery | {"rexecution_frame_index": None})


def test_boundary_first_sampling_is_complete_balanced_and_reproducible():
    profile, frames, rows, summaries = build_example()
    audit = validate_need_rows(profile, rows)
    assert positive_counts(profile) == {"train": 93, "val": 93, "test": 93}
    for split in ("train", "val", "test"):
        assert summaries[split]["ignored_count"] == 17
        assert summaries[split]["selected_counts"] == dict.fromkeys(NEGATIVE_SOURCES, 93)
        assert summaries[split]["boundary_selected_count"] == 60
        assert audit["splits"][split]["reserved_boundary_count"] == 60
        repeated, _ = build_need_rows(
            frames=list(reversed(frames)), profile=profile, split=split, base_row=base_row, seed=42
        )
        assert repeated == [row for row in rows if row["split"] == split]
    assert len(audit["attempts"]) == 6
    for attempt in audit["attempts"]:
        f = attempt["failure_frame"]
        assert attempt["reserved_negative_frames"] == list(range(f - 30, f))


@pytest.mark.parametrize(
    "corruption", ["ignored", "missing_boundary", "missing_positive", "annotation", "split", "duplicate"]
)
def test_bad_labels_missing_boundary_or_duplicate_frames_are_rejected(corruption):
    profile, _, rows, _ = build_example()
    rows = copy.deepcopy(rows)
    meta = profile["attempts"][0]
    if corruption == "ignored":
        row = next(row for row in rows if row["episode_id"] == 10 and row["attempt_id"] == 1 and row["need_recovery"])
        row["frame_index"] = 50
        row["need_boundary"] = boundary_annotation(50, meta)
    elif corruption in ("missing_boundary", "missing_positive"):
        if corruption == "missing_boundary":
            position = next(i for i, row in enumerate(rows) if row.get("need_boundary", {}).get("pre_failure_boundary"))
        else:
            position = next(i for i, row in enumerate(rows) if row["need_recovery"])
        rows.pop(position)
    elif corruption == "annotation":
        next(row for row in rows if "need_boundary" in row)["need_boundary"]["positive_start_frame"] += 1
    elif corruption == "split":
        rows[0]["split"] = "test"
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        validate_need_rows(profile, rows)


def test_boundary_quota_overflow_is_not_silently_downsampled():
    positive = {
        "episode_id": 1,
        "attempt_id": 1,
        "frame_index": 80,
        "global_index": 80,
        "source": "failure_active",
        "need_recovery": True,
    }
    hard = [
        positive
        | {
            "frame_index": i,
            "global_index": i,
            "need_recovery": False,
            "source": NEGATIVE_SOURCES[0],
            "need_boundary": {"pre_failure_boundary": True},
        }
        for i in range(3)
    ]
    easy = {
        source: [positive | {"episode_id": j + 2, "global_index": 100 + j, "source": source, "need_recovery": False}]
        for j, source in enumerate(NEGATIVE_SOURCES[1:])
    }
    with pytest.raises(ValueError, match="quota"):
        select_need_rows([positive], {NEGATIVE_SOURCES[0]: hard} | easy)


def test_absent_easy_source_redistributes_capacity_without_losing_boundary():
    profile = example_profile()
    profile["attempts"] = [meta for meta in profile["attempts"] if meta["task"] != "one_success"]
    frames = []
    for meta in profile["attempts"]:
        if meta["attempt_id"] == 2:
            meta["frame_count"] = 300
        for frame in range(meta["frame_count"]):
            frames.append(
                SimpleNamespace(
                    attempt_key=(meta["episode_id"], meta["attempt_id"]), frame_index=frame, global_index=len(frames)
                )
            )
    rows = []
    for split in ("train", "val", "test"):
        sampled, _ = build_need_rows(frames=frames, profile=profile, split=split, base_row=base_row, seed=42)
        rows.extend(sampled)
    audit = validate_need_rows(profile, rows)
    assert audit["splits"]["train"]["negative_source_counts"] == dict(zip(NEGATIVE_SOURCES, (110, 0, 169), strict=True))
    assert audit["splits"]["train"]["reserved_boundary_count"] == 60


def test_new_counts_do_not_reuse_old_positive_manifest_or_change_old_version(tmp_path):
    (tmp_path / "profile.json").write_text(json.dumps(example_profile()))
    for split in ("train", "val", "test"):
        for folder in ("need", "reasoning_manifests/failure_reason", "reasoning_manifests/reasoning"):
            path = tmp_path / folder / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            values = [{"need_recovery": True}] if folder == "need" else [{"frame_offset": 0}, {"frame_offset": 14}]
            path.write_text("".join(json.dumps(row) + "\n" for row in values))
    assert expected_counts(tmp_path)["train"]["need"] == 372
    assert old_counts(tmp_path)["train"]["need"] == 4
    assert expected_counts(tmp_path)["train"]["plan"] == 8
    assert expected_counts(tmp_path, expand_plan=False)["train"]["plan"] == 2


def test_old_index_and_modified_policies_are_rejected():
    with pytest.raises(ValueError, match="rebuild old data"):
        validate_index({"experiment_version": "book_v9_4"})
    with pytest.raises(ValueError, match="rebuild old data"):
        validate_index(
            {
                "experiment_version": "book_v9_4_3",
                "need_label_policy": LABEL_POLICY,
                "need_sampling_policy": SAMPLING_POLICY | {"pre_failure_window_frames": 10},
            }
        )


def test_new_builder_defaults_reuse_v942_data_without_model_dependency():
    from scripts.build_book_v9_4_3_multitask_data import parse_args

    args = parse_args([])
    assert args.action_index == DEFAULT_ACTION_INDEX
    assert args.adjustment_dir == DEFAULT_ADJUSTMENT_DIR
    assert args.output_dir == DEFAULT_MULTITASK_DIR
    assert not hasattr(args, "stage_a_checkpoint")
    assert "v9_4_2" in str(DEFAULT_STAGE_A) and DEFAULT_STAGE_A.name == "15000"


def test_training_cli_uses_new_profile_outputs_and_old_stage_a(monkeypatch):
    from scripts import train_vla_multitask_book_v9_4_3 as entry
    from tactile_vla.vla.book_v9_4_3_multitask_data import DATA_PROFILE, DEFAULT_INDEX, RUN_NAME, VERSION_TAG

    trainer = entry.previous.trainer
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
    ):
        monkeypatch.setattr(trainer, name, getattr(trainer, name))
    monkeypatch.setattr(entry.previous.training_base, "evaluate_text", entry.previous.training_base.evaluate_text)
    monkeypatch.setattr(sys, "argv", ["train_vla_multitask_book_v9_4_3.py"])
    entry.configure_version()
    args = trainer.parse_args()
    assert args.data_profile == DATA_PROFILE
    assert args.index_file == DEFAULT_INDEX
    assert args.stage_a_checkpoint == DEFAULT_STAGE_A
    assert args.run_name == RUN_NAME
    assert args.num_steps == 4000 and args.eval_interval == 2000
    assert trainer.VERSION_TAG == VERSION_TAG


def test_export_metadata_records_new_need_contract(monkeypatch, tmp_path):
    from scripts import train_vla_multitask_book_v9_4_3 as entry

    def fake_export(run_dir, state, step, filter_):
        path = run_dir / str(step) / "book_v9_4_3_export.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"step": step, "exports": {"full_params": {}, "delta_params": {}}}))

    monkeypatch.setattr(entry.previous.trainer, "export_checkpoint", fake_export)
    entry.export_checkpoint(tmp_path, None, 800, None)
    metadata = json.loads((tmp_path / "800/book_v9_4_3_export.json").read_text())
    assert metadata["experiment_version"] == "book_v9_4_3"
    assert metadata["need_label_policy"] == LABEL_POLICY
    assert metadata["need_sampling_policy"] == SAMPLING_POLICY
