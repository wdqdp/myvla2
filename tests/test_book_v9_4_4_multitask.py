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

from tactile_vla.vla.book_v9_4_4_multitask_data import (
    DATA_PROFILE,
    DEFAULT_ACTION_INDEX,
    DEFAULT_ADJUSTMENT_DIR,
    DEFAULT_INDEX,
    DEFAULT_MULTITASK_DIR,
    DEFAULT_STAGE_A,
    LABEL_POLICY,
    REASONING_WINDOW_POLICY,
    RUN_NAME,
    VERSION_TAG,
    build_reasoning_rows,
    reasoning_window,
    validate_index,
    validate_reasoning_rows,
)
from tactile_vla.vla.book_v9_4_memory import validate_balanced_variants, validate_plan_row


@pytest.mark.parametrize("direction,offset", [("left", 12), ("right", 5)])
def test_C_window_is_inclusive_and_does_not_change_F(direction, offset):
    m = {"episode_id": 1, "shift_frame_index": 50, "rotation_direction": direction, "frame_count": 100}
    fields = reasoning_window(m)
    assert fields["failure_reference_frame"] == 50
    assert fields["need_positive_start_frame"] == 50 + offset
    assert fields["reasoning_window_end"] == 50 + offset + 14
    assert fields["source_anchor_frame_index"] == 50
    reasoning_window(m | {"frame_count": 50 + offset + 15})
    with pytest.raises(ValueError, match="cannot support"):
        reasoning_window(m | {"frame_count": 50 + offset + 14})


def fixture(tmp_path):
    metas, anchors, frames, need = [], {}, {}, []
    for split in ("train", "val", "test"):
        sources = []
        for direction in ("left", "right"):
            ep = len(metas) + 1
            meta = {
                "episode_id": ep,
                "attempt_id": 1,
                "split": split,
                "result": "failure",
                "shift_frame_index": 50,
                "rotation_direction": direction,
                "grasp_position": "appropriate",
                "frame_count": 100,
            }
            recovery = {
                "episode_id": ep,
                "attempt_id": 2,
                "split": split,
                "result": "success",
                "horizontal_direction": direction,
                "horizontal_magnitude": "moderately",
                "vertical_direction": "none",
                "vertical_magnitude": "moderately",
            }
            metas.extend((meta, recovery))
            target = f"failure_reason=rotate {direction},grasp appropriate."
            source = {
                "frame_offset": 0,
                "frame_index": 50,
                "variant_id": f"source-{ep}",
                "current_observation": {"episode_id": ep, "attempt_id": 1, "failure_reason": target},
                "target_recovery_plan": f"recovery_plan=move horizontally {direction} moderately, move vertically none moderately.",
                "target_source": {
                    "source_type": "real",
                    "episode_id": ep,
                    "failed_attempt_id": 1,
                    "plan_attempt_id": 2,
                },
            }
            anchors[split, ep, 1] = source
            sources.append(source)
            fields = reasoning_window(meta)
            for frame in range(fields["reasoning_window_start"], fields["reasoning_window_end"] + 1):
                global_index = ep * 100 + frame
                caption = f"Touch[rotation={'clockwise' if direction == 'left' else 'counterclockwise'}]"
                frames[ep, 1, frame] = SimpleNamespace(
                    global_index=global_index,
                    episode_id=ep,
                    attempt_id=1,
                    frame_index=frame,
                    ros_timestamp=float(global_index),
                    tactile_caption=caption,
                    instruction="book",
                )
                need.append(
                    {
                        "split": split,
                        "global_index": global_index,
                        "episode_id": ep,
                        "frame_index": frame,
                        "need_recovery": True,
                        "prompt": f"phase {global_index} {caption}",
                    }
                )
        path = tmp_path / f"reasoning_manifests/reasoning/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in sources))

    def base(frame, split, source):
        return {
            "split": split,
            "global_index": frame.global_index,
            "episode_id": frame.episode_id,
            "attempt_id": 1,
            "frame_index": frame.frame_index,
            "timestamp": frame.ros_timestamp,
            "source": source,
        }

    failure, plan = build_reasoning_rows(
        profile={"attempts": metas},
        frame_by_key=frames,
        anchors=anchors,
        phase_fields=lambda f: {"prompt": f"phase {f.global_index} {f.tactile_caption}"},
        base_row=base,
    )
    manifests = {"failure": failure, "plan": plan, "need": need}
    index = {"v4_dir": str(tmp_path), "splits": {}}
    for split in ("train", "val", "test"):
        index["splits"][split] = {
            task: {
                "manifest_row_indices": [
                    i
                    for i, r in enumerate(manifests[task])
                    if r["split"] == split and (split == "train" or r["frame_offset"] == 14)
                ]
            }
            for task in ("failure", "plan")
        }
    return index, manifests, {"attempts": metas}, frames, anchors


def test_rebuild_uses_actual_new_observations_and_all_memory_lengths(tmp_path):
    index, manifests, profile, _, _ = fixture(tmp_path)
    audit = validate_reasoning_rows(index, manifests, profile)
    assert len(audit["attempts"]) == 6
    assert len(manifests["failure"]) == 90 and len(manifests["plan"]) == 360
    validate_balanced_variants(manifests["plan"])
    failures = {r["global_index"]: r["target_failure_reason"] for r in manifests["failure"]}
    for row in manifests["plan"]:
        validate_plan_row(row, failures[row["global_index"]])
        assert (
            f"rotation={'clockwise' if 'horizontally left' in row['target_recovery_plan'] else 'counterclockwise'}"
            in row["prompt"]
        )
    for row in manifests["failure"]:
        assert row["frame_index"] >= row["need_positive_start_frame"]
        assert row["frame_offset"] == row["frame_index"] - row["need_positive_start_frame"]
    assert len(index["splits"]["train"]["failure"]["manifest_row_indices"]) == 30
    assert len(index["splits"]["val"]["failure"]["manifest_row_indices"]) == 2


@pytest.mark.parametrize("corruption", ["old_F", "offset", "target", "missing", "memory_missing", "eval", "anchor"])
def test_rejects_old_window_wrong_target_and_incomplete_stream(tmp_path, corruption):
    index, manifests, profile, _, _ = fixture(tmp_path)
    manifests = copy.deepcopy(manifests)
    row = manifests["failure"][0]
    if corruption == "old_F":
        row["frame_index"] = 50
    elif corruption == "offset":
        row["frame_offset"] += 12
    elif corruption == "target":
        row["target_failure_reason"] = "failure_reason=rotate right,grasp appropriate."
    elif corruption == "missing":
        manifests["failure"].pop()
    elif corruption == "memory_missing":
        manifests["plan"].pop()
    elif corruption == "eval":
        index["splits"]["val"]["failure"]["manifest_row_indices"] = []
    else:
        manifests["plan"][0]["source_variant_id"] = "wrong"
    with pytest.raises(ValueError):
        validate_reasoning_rows(index, manifests, profile)


def test_missing_real_C_frame_is_rejected(tmp_path):
    _, _, profile, frames, anchors = fixture(tmp_path)
    frames.pop(next(iter(frames)))
    with pytest.raises(ValueError, match="missing real frame"):
        build_reasoning_rows(
            profile=profile, frame_by_key=frames, anchors=anchors, phase_fields=lambda f: {}, base_row=lambda *args: {}
        )


def test_version_defaults_are_independent_and_builder_has_no_checkpoint():
    from scripts.build_book_v9_4_4_multitask_data import parse_args

    args = parse_args([])
    assert args.output_dir == DEFAULT_MULTITASK_DIR
    assert args.action_index == DEFAULT_ACTION_INDEX and args.adjustment_dir == DEFAULT_ADJUSTMENT_DIR
    assert not hasattr(args, "stage_a_checkpoint")
    with pytest.raises(ValueError, match="rebuild old data"):
        validate_index({"experiment_version": "book_v9_4_3"})


def test_training_cli_and_export_record_new_window(monkeypatch, tmp_path):
    from scripts import train_vla_multitask_book_v9_4_4 as entry

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
    monkeypatch.setattr(sys, "argv", ["train_vla_multitask_book_v9_4_4.py"])
    entry.configure_version()
    args = trainer.parse_args()
    assert (args.data_profile, args.index_file, args.run_name) == (DATA_PROFILE, DEFAULT_INDEX, RUN_NAME)
    assert args.stage_a_checkpoint == DEFAULT_STAGE_A
    assert args.num_steps == 4000 and args.eval_interval == 2000
    assert trainer.VERSION_TAG == VERSION_TAG

    def export(run_dir, state, step, filter_):
        path = run_dir / str(step) / f"{VERSION_TAG}_export.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"step": step}))

    monkeypatch.setattr(trainer, "export_checkpoint", export)
    entry.export_checkpoint(tmp_path, None, 2000, None)
    metadata = json.loads((tmp_path / f"2000/{VERSION_TAG}_export.json").read_text())
    assert metadata["reasoning_window_policy"] == REASONING_WINDOW_POLICY
    assert metadata["need_label_policy"] == LABEL_POLICY
