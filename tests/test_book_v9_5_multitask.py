from __future__ import annotations

# ruff: noqa: E402
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tactile_vla.vla import book_v9_5_need_data as data
from tactile_vla.vla.book_v9_5_multitask_data import reasoning_window
from tactile_vla.vla.v7_7_multitask_data import NEGATIVE_SOURCES, V77ManifestDataset


def caption(rotation):
    return f"Touch[area=small; Fx=negative; Fy=positive; Fz=negative; Fz_bias=left; rotation={rotation}]"


def example():
    metas, frames = [], []
    for i, split in enumerate(("train", "val", "test")):
        for ep, attempt, direction, result, task, n, f, r in (
            (10 * i + 1, 1, "left", "failure", "moderate_lift", 190, 100, None),
            (10 * i + 2, 1, "right", "failure", "moderate_lift", 190, 100, None),
            (10 * i + 3, 1, "none", "success", "one_success", 600, None, None),
            (10 * i + 1, 2, "none", "success", "moderate_lift", 600, None, 60),
        ):
            metas.append(
                {
                    "episode_id": ep,
                    "attempt_id": attempt,
                    "split": split,
                    "rotation_direction": direction,
                    "result": result,
                    "task": task,
                    "frame_count": n,
                    "shift_frame_index": f,
                    "rexecution_frame_index": r,
                }
            )
            for frame in range(n):
                observed = "none"
                if result == "failure" and frame >= f + 1:
                    observed = data.EXPECTED_ROTATION[direction]
                if direction == "left" and frame == f + 1:
                    observed = "counterclockwise"
                frames.append(
                    SimpleNamespace(
                        attempt_key=(ep, attempt),
                        frame_index=frame,
                        global_index=len(frames),
                        tactile_caption=caption(observed),
                    )
                )
    return {"attempts": metas}, frames


def base_row(frame, split, source):
    return {
        "episode_id": frame.attempt_key[0],
        "attempt_id": frame.attempt_key[1],
        "frame_index": frame.frame_index,
        "global_index": frame.global_index,
        "split": split,
        "source": source,
    }


def fields(frame):
    return {
        "prompt": f"Mode: phase\nTask: book\nTouch: {frame.tactile_caption}\nRecovery plan: none\nState history: [[1,2,3]]",
        "qpos_h100_11_discrete": [[1, 2, 3]],
        "history_sources": [{"global_index": frame.global_index}],
        "effective_episode_history_length": 100,
    }


def build_example():
    profile, frames = example()
    rows, summary = [], {}
    for split in ("train", "val", "test"):
        selected, summary[split] = data.build_need_rows(
            frames=frames,
            profile=profile,
            split=split,
            base_row=base_row,
            seed=42,
            phase_fields=fields,
        )
        rows.extend(selected)
    return profile, frames, rows, summary


@pytest.mark.parametrize("direction,offset", [("left", 27), ("right", 8)])
def test_boundaries_and_correct_rotation_before_C(direction, offset):
    profile, frames = example()
    meta = next(m for m in profile["attempts"] if m["rotation_direction"] == direction)
    f, c = data.failure_boundary(meta)
    assert c == f + offset
    frame = next(x for x in frames if x.attempt_key == (meta["episode_id"], 1) and x.frame_index == f + 2)
    assert data.need_role(frame, meta) == (True, "failure_active")
    assert reasoning_window(meta)["reasoning_window_start"] == c
    assert reasoning_window(meta)["reasoning_window_end"] == c + 14
    assert "need_positive_start_frame" not in reasoning_window(meta)


def test_train_pairs_real_validation_and_reserved_sampling():
    profile, frames, rows, summaries = build_example()
    audit = data.validate_need_rows(profile, rows, frames)
    assert data.positive_counts(profile, frames) == dict.fromkeys(("train", "val", "test"), 177)
    for split in ("train", "val", "test"):
        assert summaries[split]["selected_counts"] == dict.fromkeys(NEGATIVE_SOURCES, 177)
        assert summaries[split]["negative_count"] == 531
        assert summaries[split]["pair_count"] == (32 if split == "train" else 0)
        assert len(summaries[split]["excluded_wrong_rotation"]) == 1
        repeated, _ = data.build_need_rows(
            frames=list(reversed(frames)), profile=profile, split=split, base_row=base_row, seed=42, phase_fields=fields
        )
        assert repeated == [r for r in rows if r["split"] == split]
    assert audit["splits"]["train"]["pairs_by_direction"] == {"left": 25, "right": 7}
    real = {(r["split"], r["global_index"]): r for r in rows if r["need_variant"] == "real"}
    for row in rows:
        if row["need_variant"] != "rotation_none":
            continue
        paired = real[row["split"], row["global_index"]]
        assert paired["need_recovery"] is True and row["need_recovery"] is False
        assert row["need_pair_id"] == paired["need_pair_id"]
        assert row["prompt"] == paired["prompt"].replace(
            "rotation=" + row["need_boundary"]["observed_rotation"], "rotation=none"
        )


@pytest.mark.parametrize("mutation", ["drop_pair", "label", "extra_variant", "history", "prompt", "caption", "leak"])
def test_strict_pair_audit_rejects_corruption(mutation):
    profile, frames, rows, _ = build_example()
    rows = copy.deepcopy(rows)
    target = next(r for r in rows if r["need_variant"] == "rotation_none")
    if mutation == "drop_pair":
        rows.remove(target)
    elif mutation == "label":
        target["need_recovery"] = True
    elif mutation == "extra_variant":
        rows.append(copy.deepcopy(target))
    elif mutation == "history":
        target["history_sources"] = []
    elif mutation == "prompt":
        target["prompt"] += " changed"
    elif mutation == "caption":
        target["need_counterfactual"]["field"] = "Fz"
    else:
        target["split"] = "val"
    with pytest.raises(ValueError):
        data.validate_need_rows(profile, rows, frames)


def test_variant_manifest_loader_preserves_same_images_and_state():
    import numpy as np

    _, _, rows, _ = build_example()
    cf = next(r for r in rows if r["need_variant"] == "rotation_none")
    real = next(r for r in rows if r.get("need_pair_id") == cf["need_pair_id"] and r["need_variant"] == "real")
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)
    item = {
        "index": real["global_index"],
        "episode_id": real["episode_id"],
        "attempt_id": 1,
        "frame_index": real["frame_index"],
        "observation.images.front": image,
        "observation.images.left": image,
        "observation.state": state,
    }
    ds = V77ManifestDataset(
        rows=[real, cf],
        row_indices=[0, 1],
        global_indices=[real["global_index"]] * 2,
        task="need",
        lerobot_dataset={real["global_index"]: item},
    )
    assert ds[0]["need_recovery_label"] == 1 and ds[1]["need_recovery_label"] == 0
    assert ds[0]["observation/image"] is ds[1]["observation/image"]
    np.testing.assert_array_equal(ds[0]["observation/state"], ds[1]["observation/state"])


def test_need_eval_groups_include_real_transition_and_reject_synthetic():
    from scripts.train_vla_multitask_book_v9_5 import need_eval_groups

    _, _, rows, _ = build_example()
    selected = [i for i, r in enumerate(rows) if r["split"] == "val"]
    groups = need_eval_groups(rows, selected)
    assert len(groups["transition_left"]) == 26  # One wrong-direction frame excluded, none retained.
    assert len(groups["transition_right"]) == 8
    with pytest.raises(ValueError):
        need_eval_groups(rows, list(range(len(rows))))


def test_build_cli_has_no_model_dependency():
    from scripts import build_book_v9_5_multitask_data as entry
    from scripts import build_book_v9_5_adjustment_end_data as adjustment

    args = entry.parse_args([])
    assert "book_stage_a_v9_5" in str(args.action_index)
    assert args.adjustment_dir.name == "book_adjustment_end_v9_5"
    assert not hasattr(args, "stage_a_checkpoint")
    assert not hasattr(adjustment.parse_args([]), "stage_a_checkpoint")
    with pytest.raises(ValueError):
        entry.parse_args(["--dataset-dir", "/tmp/history9_4_5/dataset"])


def test_reasoning_short_attempt_is_not_padded():
    meta = {"episode_id": 1, "shift_frame_index": 50, "rotation_direction": "left", "frame_count": 91}
    with pytest.raises(ValueError):
        reasoning_window(meta)


def save_training_globals(monkeypatch, entry):
    for module in (entry.trainer, entry.training_base):
        for key, value in list(vars(module).items()):
            if not key.startswith("__"):
                monkeypatch.setattr(module, key, value)


@pytest.mark.parametrize("data_only", [False, True])
def test_training_defaults_and_data_only_cannot_train(monkeypatch, data_only):
    from scripts import train_vla_multitask_book_v9_5 as entry

    save_training_globals(monkeypatch, entry)
    monkeypatch.setattr(sys, "argv", ["train9_5"] + (["--data-only-dry-run"] if data_only else []))
    entry.configure_version()
    args = entry.trainer.parse_args()
    assert (args.num_steps, args.eval_interval, args.save_interval, args.keep_period) == (2000, 1000, 1000, 1000)
    assert args.run_name == "pi05_book_v9_5_five_task_h100_no_history"
    assert "book_stage_a_v9_5" in str(args.stage_a_checkpoint)
    assert args.stage_a_checkpoint.name == "15000"
    assert args.data_only_dry_run is data_only and args.dry_run is data_only
    assert args.batch_size == 8 and args.lr == 1e-4
    assert args.action_eval_samples_per_phase == 256
    assert not args.use_state_history and args.state_history_len == 0


def test_cli_equals_and_archive_checkpoint_rejected(monkeypatch):
    from scripts import train_vla_multitask_book_v9_5 as entry

    save_training_globals(monkeypatch, entry)
    monkeypatch.setattr(sys, "argv", ["train9_5", "--num-steps=4000", "--eval-interval=800"])
    entry.configure_version()
    args = entry.trainer.parse_args()
    assert args.num_steps == 4000 and args.eval_interval == 800
    monkeypatch.setattr(sys, "argv", ["train9_5", "--stage-a-checkpoint=/tmp/history9_4_5/15000"])
    with pytest.raises(ValueError):
        entry.trainer.parse_args()


def test_need_direction_and_transition_metrics_are_recorded(monkeypatch):
    from scripts import train_vla_multitask_book_v9_5 as entry

    loader = SimpleNamespace(book_v9_5_groups={"transition_left": SimpleNamespace(dataset=[0, 1, 2])})
    calls = []

    def evaluate(state, loader, sharding, **kwargs):
        calls.append(kwargs)
        return {"confusion_matrix": [[1, 1], [0, 1]], "macro_f1": 0.5}

    monkeypatch.setattr(entry, "_BASE_EVALUATE_NEED", evaluate)
    result = entry.evaluate_need(None, loader, None, max_samples=100)
    assert result["by_direction_and_transition"]["transition_left"]["false_positive_rate"] == 0.5
    assert calls == [{"max_samples": 100}, {"max_samples": 3}]


def test_stage_a_binding_rejects_missing_or_wrong_version(tmp_path, monkeypatch):
    import json
    from tactile_vla.vla import book_v9_5_multitask_data as contract

    checkpoint = tmp_path / "model/15000"
    checkpoint.mkdir(parents=True)
    with pytest.raises(ValueError):
        contract.validate_stage_a_for_training(checkpoint, {})
    monkeypatch.setattr(contract.common, "validate_stage_a_for_training", lambda *args: {"step": 15000})
    config = checkpoint.parent / "config.json"
    config.write_text(json.dumps({"experiment_version": "book_v9_4", "source_scope": {}}))
    with pytest.raises(ValueError):
        contract.validate_stage_a_for_training(checkpoint, {"source_scope": {}})
    config.write_text(json.dumps({"experiment_version": "book_v9_5", "source_scope": {"dataset": "current"}}))
    assert contract.validate_stage_a_for_training(checkpoint, {"source_scope": {"dataset": "current"}}) == {
        "step": 15000
    }


def test_none_after_C_is_false_wrong_direction_excluded():
    profile, frames = example()
    meta = profile["attempts"][0]
    _, c = data.failure_boundary(meta)
    frame = next(f for f in frames if f.attempt_key == (meta["episode_id"], 1) and f.frame_index == c)
    frame.tactile_caption = caption("none")
    assert data.need_role(frame, meta) == (False, NEGATIVE_SOURCES[0])
    frame.tactile_caption = caption("counterclockwise")
    assert data.need_role(frame, meta) == (None, None)


def test_export_records_labels_pairs_source_and_initialization(monkeypatch, tmp_path):
    import json
    from scripts import train_vla_multitask_book_v9_5 as entry

    def export(run_dir, state, step, filter_):
        path = run_dir / str(step) / "book_v9_5_export.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"step": step}))

    monkeypatch.setattr(entry.trainer, "export_checkpoint", export)
    extra = {
        "action_eval_policy": {},
        "action_eval_selection": {},
        "training_data_hash": "datahash",
        "source_scope": {"book_root": "current"},
        "stage_a_initialization_identity": {"step": 15000},
        "need_boundary_audit_sha256": "needhash",
        "reasoning_boundary_audit_sha256": "reasonhash",
    }
    monkeypatch.setattr(entry.training_base, "EXTRA_CONFIG", extra)
    entry.export_checkpoint(tmp_path, None, 1000, None)
    metadata = json.loads((tmp_path / "1000/book_v9_5_export.json").read_text())
    assert metadata["need_label_policy"]["direction_offsets_frames"] == {"left": 27, "right": 8}
    assert metadata["training_data_hash"] == "datahash"
    assert metadata["stage_a_initialization_identity"]["step"] == 15000
    assert metadata["need_sampling_policy"]["pair_selection"].startswith("keep_both_rows")
