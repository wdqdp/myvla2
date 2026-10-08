from __future__ import annotations

# ruff: noqa: E402
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from tactile_vla.vla.book_v9_4_5_need_data import (
    boundary_annotation,
    build_need_rows,
    failure_boundary,
    need_role,
    positive_counts,
    replace_current_caption,
    rotation_none_caption,
    select_need_rows,
    validate_need_rows,
)
from tactile_vla.vla.book_v9_4_3_need_data import OFFSETS as OLD_OFFSETS
from tactile_vla.vla.v7_7_multitask_data import NEGATIVE_SOURCES

CAPTION = "Touch[area=small; Fx=negative; Fy=positive; Fz=negative; Fz_bias=left; rotation=clockwise]"


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


def phase_fields(frame):
    return {
        "prompt": f"Mode: phase\nTask: book\nTouch: {frame.tactile_caption}\nRecovery plan: none\nState history: [[1,2,3]]",
        "qpos_h100_11_discrete": [[1, 2, 3]],
        "history_sources": [{"global_index": frame.global_index}],
        "effective_episode_history_length": 100,
    }


def build_example():
    profile = example_profile()
    frames = []
    for meta in profile["attempts"]:
        for f in range(meta["frame_count"]):
            frames.append(
                SimpleNamespace(
                    attempt_key=(meta["episode_id"], meta["attempt_id"]),
                    frame_index=f,
                    global_index=len(frames),
                    tactile_caption=CAPTION,
                )
            )
    rows, summaries = [], {}
    for split in ("train", "val", "test"):
        selected, summaries[split] = build_need_rows(
            frames=frames,
            profile=profile,
            split=split,
            base_row=base_row,
            seed=42,
            phase_fields=phase_fields,
        )
        rows.extend(selected)
    return profile, frames, rows, summaries


@pytest.mark.parametrize("direction,offset", [("left", 15), ("right", 8)])
def test_C_half_open_boundary_and_real_eval(direction, offset):
    meta = example_profile()["attempts"][0] | {"rotation_direction": direction}
    f, c = failure_boundary(meta)
    assert c == f + offset
    assert need_role(f - 1, meta) == (False, NEGATIVE_SOURCES[0])
    for frame in range(f, c):
        assert need_role(frame, meta) == (False, NEGATIVE_SOURCES[0])
        assert boundary_annotation(frame, meta)["rotation_none_counterfactual"]
        assert need_role(frame, meta | {"split": "val"}) == (None, None)
    assert need_role(c, meta) == (True, "failure_active")
    assert not boundary_annotation(c, meta)["rotation_none_counterfactual"]
    assert OLD_OFFSETS == {"left": 12, "right": 5}


def test_only_rotation_changes():
    modified = rotation_none_caption(CAPTION)
    assert modified == CAPTION.replace("rotation=clockwise", "rotation=none")
    assert rotation_none_caption(modified) == modified
    original = phase_fields(SimpleNamespace(tactile_caption=CAPTION, global_index=10))["prompt"]
    assert replace_current_caption(original, CAPTION, modified) == original.replace(CAPTION, modified)
    with pytest.raises(ValueError):
        rotation_none_caption("Touch[Fz_bias=left]")
    with pytest.raises(ValueError):
        replace_current_caption(original + original, CAPTION, modified)


def test_manifest_loader_keeps_images_and_state_but_uses_synthetic_prompt():
    import numpy as np
    from tactile_vla.vla.v7_7_multitask_data import V77ManifestDataset

    _, _, rows, _ = build_example()
    row = next(r for r in rows if "need_counterfactual" in r)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    wrist = np.ones((2, 2, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)
    item = {
        "index": row["global_index"],
        "episode_id": row["episode_id"],
        "attempt_id": row["attempt_id"],
        "frame_index": row["frame_index"],
        "observation.images.front": image,
        "observation.images.left": wrist,
        "observation.state": state,
    }
    dataset = V77ManifestDataset(
        rows=[row],
        row_indices=[0],
        global_indices=[row["global_index"]],
        task="need",
        lerobot_dataset={row["global_index"]: item},
    )
    sample = dataset[0]
    assert sample["prompt"] == row["prompt"] and "rotation=none" in sample["prompt"]
    assert sample["need_recovery_label"] == 0
    assert sample["observation/image"] is image and sample["observation/wrist_image"] is wrist
    np.testing.assert_array_equal(sample["observation/state"], state)


def test_reserved_union_quota_overflow_rejected():
    prototype = {
        "episode_id": 1,
        "attempt_id": 1,
        "frame_index": 80,
        "global_index": 80,
        "source": "failure_active",
        "need_recovery": True,
    }
    hard = [
        prototype
        | {
            "frame_index": i,
            "global_index": i,
            "need_recovery": False,
            "source": NEGATIVE_SOURCES[0],
            "need_boundary": {"pre_failure_boundary": i < 2, "rotation_none_counterfactual": i == 2},
        }
        for i in range(3)
    ]
    negatives = {NEGATIVE_SOURCES[0]: hard}
    for i, source in enumerate(NEGATIVE_SOURCES[1:]):
        negatives[source] = [
            prototype | {"episode_id": i + 2, "global_index": i + 100, "source": source, "need_recovery": False}
        ]
    with pytest.raises(ValueError, match="quota"):
        select_need_rows([prototype], negatives)


def test_all_synthetic_and_preF_negatives_retained_with_1_to_3():
    profile, frames, rows, summaries = build_example()
    audit = validate_need_rows(profile, rows)
    assert positive_counts(profile) == dict.fromkeys(("train", "val", "test"), 87)
    for split in ("train", "val", "test"):
        summary = summaries[split]
        assert summary["selected_counts"] == dict.fromkeys(NEGATIVE_SOURCES, 87)
        assert summary["boundary_selected_count"] == 60
        assert summary["counterfactual_selected_count"] == (23 if split == "train" else 0)
        assert summary["ignored_count"] == (0 if split == "train" else 23)
        assert audit["splits"][split]["rotation_none_counterfactual_count"] == (23 if split == "train" else 0)
        repeated, _ = build_need_rows(
            frames=list(reversed(frames)),
            profile=profile,
            split=split,
            base_row=base_row,
            seed=42,
            phase_fields=phase_fields,
        )
        assert repeated == [row for row in rows if row["split"] == split]
    lookup = {frame.global_index: frame for frame in frames}
    for row in rows:
        original = phase_fields(lookup[row["global_index"]])
        for field in ("qpos_h100_11_discrete", "history_sources", "effective_episode_history_length"):
            assert row[field] == original[field]
        if "need_counterfactual" in row:
            assert row["split"] == "train" and row["need_recovery"] is False
            assert row["need_counterfactual"]["original_prompt"] == original["prompt"]
            assert row["prompt"] == original["prompt"].replace("rotation=clockwise", "rotation=none")
        else:
            assert row["prompt"] == original["prompt"]


@pytest.mark.parametrize(
    "corruption",
    ["missing_synthetic", "true_synthetic", "wrong_rotation", "other_touch", "history", "C_false", "outside_interval"],
)
def test_invalid_counterfactual_labels_are_rejected(corruption):
    profile, _, rows, _ = build_example()
    rows = copy.deepcopy(rows)
    row = next(r for r in rows if "need_counterfactual" in r)
    if corruption == "missing_synthetic":
        rows.remove(row)
    elif corruption == "true_synthetic":
        row["need_recovery"] = True
    elif corruption == "wrong_rotation":
        row["prompt"] = row["prompt"].replace("rotation=none", "rotation=counterclockwise")
    elif corruption == "other_touch":
        row["prompt"] = row["prompt"].replace("Fz_bias=left", "Fz_bias=right")
    elif corruption == "history":
        row["prompt"] = row["prompt"].replace("[[1,2,3]]", "[[9,9,9]]")
    elif corruption == "C_false":
        positive = next(
            r for r in rows if r["need_recovery"] and r["frame_index"] == r["need_boundary"]["positive_start_frame"]
        )
        positive["need_recovery"] = False
    else:
        factual = next(r for r in rows if r["need_recovery"])
        factual["need_counterfactual"] = row["need_counterfactual"]
    with pytest.raises(ValueError):
        validate_need_rows(profile, rows)
