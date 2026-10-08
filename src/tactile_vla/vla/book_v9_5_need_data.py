"""Book V9.5 real rotation labels and train-only matched rotation-none negatives."""

from __future__ import annotations

from collections import Counter
import re

from tactile_vla.vla.book_v9_4_5_need_data import replace_current_caption, rotation_none_caption
from tactile_vla.vla.v4_data import SPLITS
from tactile_vla.vla.v7_7_multitask_data import (
    NEGATIVE_SOURCES,
    allocate_balanced_without_replacement,
    deterministic_uniform_select,
    stable_priority,
)

OFFSETS = {"left": 27, "right": 8}
EXPECTED_ROTATION = {"left": "clockwise", "right": "counterclockwise"}
HARD_WINDOW = 30
LABEL_POLICY = {
    "schema_version": "book_v9_5_need_label_policy_v1",
    "direction_offsets_frames": OFFSETS,
    "C": "stable_reasoning_window_start_not_first_need_true",
    "failed_attempt_real": "frame<F:false;frame>=F:correct_rotation_true_none_false_wrong_excluded",
    "counterfactual_interval": "[F,C)_half_open_correct_rotation_only",
    "counterfactual_fields": "only_current_tactile_rotation_none_all_other_inputs_unchanged",
    "counterfactual_splits": ["train"],
    "successful_execution": "unchanged_one_success_and_attempt2_frame>=native_R_false",
    "direction_mapping": EXPECTED_ROTATION,
}
SAMPLING_POLICY = {
    "schema_version": "book_v9_5_need_sampling_policy_v1",
    "positive_selection": "all_real_correct_rotation_frames_from_F",
    "negative_to_positive_ratio": "3:1",
    "negative_source_target": "as_close_as_possible_to_1:1:1",
    "hard_negative_selection": "reserve_all_preF30_postF_real_none_and_train_paired_none_then_uniform_remaining",
    "pair_selection": "keep_both_rows_no_same_batch_requirement_no_left_downsampling",
    "boundary_quota_overflow": "error_never_drop_reserved_frames",
    "seed": 42,
}


def rotation(caption):
    matches = re.findall(r"(?<![A-Za-z_])rotation=(clockwise|counterclockwise|none)(?=;|\])", caption)
    if len(matches) != 1:
        raise ValueError("Expected one valid tactile rotation field")
    return matches[0]


def failure_boundary(meta):
    direction = meta["rotation_direction"]
    if direction not in OFFSETS:
        raise ValueError("V9.5 supports left/right failed attempts")
    f = int(meta["shift_frame_index"])
    c = f + OFFSETS[direction]
    if f < 0 or c >= int(meta["frame_count"]):
        raise ValueError("Invalid V9.5 F/C boundary")
    return f, c


def need_role(frame, meta):
    if meta["result"] == "failure":
        f, _ = failure_boundary(meta)
        if frame.frame_index < f:
            return False, NEGATIVE_SOURCES[0]
        observed = rotation(frame.tactile_caption)
        if observed == EXPECTED_ROTATION[meta["rotation_direction"]]:
            return True, "failure_active"
        if observed == "none":
            return False, NEGATIVE_SOURCES[0]
        return None, None
    if meta["task"] == "one_success":
        return False, NEGATIVE_SOURCES[1]
    if meta["result"] == "success" and int(meta["attempt_id"]) == 2:
        r = meta.get("rexecution_frame_index")
        if r is None:
            raise ValueError("V9.5 recovery attempt requires native R")
        if frame.frame_index >= int(r):
            return False, NEGATIVE_SOURCES[2]
    return None, None


def boundary_annotation(frame, meta):
    f, c = failure_boundary(meta)
    return {
        "rotation_direction": meta["rotation_direction"],
        "failure_frame": f,
        "stable_window_start_frame": c,
        "shift_frames": c - f,
        "pre_failure_boundary": max(0, f - HARD_WINDOW) <= frame.frame_index < f,
        "transition_interval": f <= frame.frame_index < c,
        "observed_rotation": rotation(frame.tactile_caption),
    }


def pair_id(frame):
    return f"{frame.attempt_key[0]}:{frame.attempt_key[1]}:{frame.frame_index}"


def candidate_rows(*, frames, profile, split, base_row, phase_fields):
    metadata = {(int(m["episode_id"]), int(m["attempt_id"])): m for m in profile["attempts"]}
    positives, negatives, excluded = [], {s: [] for s in NEGATIVE_SOURCES}, []
    for frame in sorted(frames, key=lambda f: f.global_index):
        meta = metadata[frame.attempt_key]
        if meta["split"] != split:
            continue
        label, source = need_role(frame, meta)
        if label is None:
            if meta["result"] == "failure":
                excluded.append(
                    {
                        "episode_id": frame.attempt_key[0],
                        "attempt_id": frame.attempt_key[1],
                        "frame_index": frame.frame_index,
                        "observed_rotation": rotation(frame.tactile_caption),
                    }
                )
            continue
        row = (
            base_row(frame, split, source)
            | phase_fields(frame)
            | {
                "need_recovery": label,
                "need_variant": "real",
            }
        )
        if meta["result"] == "failure":
            row["need_boundary"] = boundary_annotation(frame, meta)
        paired = label and row.get("need_boundary", {}).get("transition_interval", False) and split == "train"
        if paired:
            row["need_pair_id"] = pair_id(frame)
        if label:
            positives.append(row)
        else:
            negatives[source].append(row)
        if paired:
            original = frame.tactile_caption
            modified = rotation_none_caption(original)
            negatives[NEGATIVE_SOURCES[0]].append(
                row
                | {
                    "source": NEGATIVE_SOURCES[0],
                    "need_recovery": False,
                    "need_variant": "rotation_none",
                    "prompt": replace_current_caption(row["prompt"], original, modified),
                    "need_counterfactual": {
                        "original_tactile_caption": original,
                        "input_tactile_caption": modified,
                        "original_prompt": row["prompt"],
                        "field": "rotation",
                        "value": "none",
                    },
                }
            )
    return positives, negatives, excluded


def reserved_negative(row):
    b = row.get("need_boundary", {})
    return (
        row["need_variant"] == "rotation_none"
        or b.get("pre_failure_boundary", False)
        or (row.get("need_boundary") is not None and row["frame_index"] >= b["failure_frame"])
    )


def select_need_rows(positives, negatives, *, seed=42):
    if seed != 42 or not positives:
        raise ValueError("V9.5 requires seed42 and nonempty real positives")
    capacities = {s: len(negatives[s]) for s in NEGATIVE_SOURCES}
    allocation = allocate_balanced_without_replacement(capacities, 3 * len(positives), seed=seed)
    hard = NEGATIVE_SOURCES[0]
    reserved = [r for r in negatives[hard] if reserved_negative(r)]
    if len(reserved) > allocation[hard]:
        raise ValueError("V9.5 hard-negative quota cannot retain all reserved/pair rows")
    selected = reserved + deterministic_uniform_select(
        [r for r in negatives[hard] if not reserved_negative(r)],
        allocation[hard] - len(reserved),
        seed=seed,
        source=hard,
    )
    for source in NEGATIVE_SOURCES[1:]:
        selected.extend(deterministic_uniform_select(negatives[source], allocation[source], seed=seed, source=source))
    rows = positives + selected
    rows.sort(
        key=lambda r: stable_priority(seed, "v9_5-need-shuffle", r["global_index"], r["source"] + r["need_variant"])
    )
    return rows, {
        "positive_count": len(positives),
        "negative_count": len(selected),
        "candidate_counts": capacities,
        "selected_counts": allocation,
        "reserved_negative_count": len(reserved),
        "pair_count": sum(r["need_variant"] == "rotation_none" for r in selected),
        "positive_by_direction": dict(Counter(r["need_boundary"]["rotation_direction"] for r in positives)),
        "pairs_by_direction": dict(
            Counter(r["need_boundary"]["rotation_direction"] for r in selected if r["need_variant"] == "rotation_none")
        ),
    }


def build_need_rows(*, frames, profile, split, base_row, seed, phase_fields):
    positive, negative, excluded = candidate_rows(
        frames=frames,
        profile=profile,
        split=split,
        base_row=base_row,
        phase_fields=lambda frame: {"prompt": f"\nTouch: {frame.tactile_caption}\n"},
    )
    rows, summary = select_need_rows(positive, negative, seed=seed)
    lookup = {frame.global_index: frame for frame in frames}
    for row in rows:
        canonical = phase_fields(lookup[row["global_index"]])
        row.update(canonical)
        if row["need_variant"] == "rotation_none":
            cf = row["need_counterfactual"]
            cf["original_prompt"] = canonical["prompt"]
            row["prompt"] = replace_current_caption(
                canonical["prompt"], cf["original_tactile_caption"], cf["input_tactile_caption"]
            )
    summary["excluded_wrong_rotation"] = excluded
    return rows, summary


def positive_counts(profile, frames):
    metadata = {(m["episode_id"], m["attempt_id"]): m for m in profile["attempts"]}
    counts = dict.fromkeys(SPLITS, 0)
    for frame in frames:
        meta = metadata[frame.attempt_key]
        if need_role(frame, meta)[0] is True:
            counts[meta["split"]] += 1
    return counts


def validate_need_rows(profile, rows, frames):
    """Re-derive labels, exact selected identities, reserved rows and both pair members."""
    from types import SimpleNamespace

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
        return {"prompt": f"\nTouch: {frame.tactile_caption}\n"}

    expected = {}
    summaries = {}
    for split in SPLITS:
        selected, summaries[split] = build_need_rows(
            frames=frames,
            profile=profile,
            split=split,
            base_row=base_row,
            seed=42,
            phase_fields=fields,
        )
        for r in selected:
            expected[r["split"], r["global_index"], r["need_variant"]] = r
    seen, pairs = set(), {}
    for row in rows:
        key = row["split"], row["global_index"], row["need_variant"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected V9.5 need variant")
        seen.add(key)
        e = expected[key]
        for name in (
            *base_row(SimpleNamespace(attempt_key=(0, 0), frame_index=0, global_index=0), "", ""),
            "need_recovery",
            "need_variant",
            "need_boundary",
            "need_pair_id",
        ):
            if row.get(name) != e.get(name):
                raise ValueError(f"V9.5 need label/identity mismatch: {name}")
        if row.get("need_pair_id"):
            pairs.setdefault(row["need_pair_id"], {})[row["need_variant"]] = row
        if row["need_variant"] == "rotation_none" and row["split"] != "train":
            raise ValueError("V9.5 synthetic validation leakage")
        if row["need_variant"] == "real" and "need_counterfactual" in row:
            raise ValueError("Real need row must not claim counterfactual metadata")
    if seen != set(expected):
        raise ValueError("V9.5 dropped real positives, reserved negatives or selected pairs")
    for pair in pairs.values():
        if set(pair) != {"real", "rotation_none"}:
            raise ValueError("Incomplete V9.5 need pair")
        real, cf = pair["real"], pair["rotation_none"]
        metadata = cf.get("need_counterfactual", {})
        original = metadata.get("original_tactile_caption", "")
        modified = rotation_none_caption(original)
        if metadata != {
            "original_tactile_caption": original,
            "input_tactile_caption": modified,
            "original_prompt": real["prompt"],
            "field": "rotation",
            "value": "none",
        }:
            raise ValueError("Invalid V9.5 pair caption provenance")
        ignore = {"prompt", "source", "need_recovery", "need_variant", "need_counterfactual"}
        if {k: v for k, v in real.items() if k not in ignore} != {k: v for k, v in cf.items() if k not in ignore}:
            raise ValueError("V9.5 pair changed non-rotation inputs")
        if cf["prompt"] != replace_current_caption(real["prompt"], original, modified):
            raise ValueError("V9.5 pair changed non-rotation prompt fields")
    attempts = []
    for meta in profile["attempts"]:
        if meta["result"] != "failure":
            continue
        f, c = failure_boundary(meta)
        key = meta["episode_id"], meta["attempt_id"]
        actual = [frame for frame in frames if frame.attempt_key == key]
        transition = Counter(rotation(frame.tactile_caption) for frame in actual if f <= frame.frame_index < c)
        selected = [r for r in rows if (r["episode_id"], r["attempt_id"]) == key]
        attempts.append(
            {
                "episode_id": key[0],
                "attempt_id": key[1],
                "split": meta["split"],
                "rotation_direction": meta["rotation_direction"],
                "failure_frame": f,
                "stable_window_start_frame": c,
                "transition_range_half_open": [f, c],
                "transition_rotation_counts": dict(transition),
                "real_positive_count": sum(r["need_recovery"] for r in selected),
                "pair_count": sum(r["need_variant"] == "rotation_none" for r in selected),
            }
        )
    return {
        "schema_version": "book_v9_5_need_boundary_audit_v1",
        "label_policy": LABEL_POLICY,
        "sampling_policy": SAMPLING_POLICY,
        "splits": summaries,
        "attempts": attempts,
    }
