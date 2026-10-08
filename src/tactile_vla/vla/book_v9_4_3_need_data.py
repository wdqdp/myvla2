"""Need-only delayed labels and boundary-first factual sampling for Book V9.4.3."""

from __future__ import annotations

from collections import Counter

from tactile_vla.vla.v4_data import SPLITS
from tactile_vla.vla.v7_7_multitask_data import (
    NEGATIVE_SOURCES,
    allocate_balanced_without_replacement,
    deterministic_uniform_select,
    stable_priority,
)

OFFSETS = {"left": 12, "right": 5}
HARD_WINDOW = 30
LABEL_POLICY = {
    "schema_version": "book_v9_4_3_need_label_policy_v1",
    "failure_reference": "original_shift_frame_index_F",
    "direction_offsets_frames": OFFSETS,
    "negative": "frame<F",
    "ignored": "F<=frame<F+c",
    "positive": "frame>=F+c",
    "scope": "need_only_original_shift_caption_action_failure_plan_unchanged",
}
SAMPLING_POLICY = {
    "schema_version": "book_v9_4_3_need_sampling_policy_v1",
    "positive_selection": "all",
    "negative_to_positive_ratio": "3:1",
    "negative_source_target": "as_close_as_possible_to_1:1:1",
    "pre_failure_window_frames": HARD_WINDOW,
    "hard_negative_selection": "reserve_all_[max(0,F-30),F)_then_uniform_remaining",
    "easy_negative_selection": "deterministic_uniform_without_replacement",
    "boundary_quota_overflow": "error_never_silently_drop_reserved_frames",
    "splits": list(SPLITS),
    "seed": 42,
}


def failure_boundary(meta: dict) -> tuple[int, int]:
    direction = meta["rotation_direction"]
    if direction not in OFFSETS:
        raise ValueError(f"V9.4.3 need supports only left/right failure directions: {direction}")
    f = int(meta["shift_frame_index"])
    c = f + OFFSETS[direction]
    if f < 0 or c >= int(meta["frame_count"]):
        raise ValueError("V9.4.3 failed attempt has an invalid/empty delayed positive range")
    return f, c


def need_role(frame: int, meta: dict) -> tuple[bool | None, str | None]:
    """None is excluded, never silently converted to a false label."""
    if meta["result"] == "failure":
        f, c = failure_boundary(meta)
        if frame < f:
            return False, NEGATIVE_SOURCES[0]
        if frame < c:
            return None, None
        return True, "failure_active"
    if meta["task"] == "one_success":
        return False, NEGATIVE_SOURCES[1]
    if meta["result"] == "success" and int(meta["attempt_id"]) == 2:
        r = meta.get("rexecution_frame_index")
        if r is None:
            raise ValueError("V9.4.3 successful recovery lacks native R")
        if frame >= int(r):
            return False, NEGATIVE_SOURCES[2]
    return None, None


def boundary_annotation(frame: int, meta: dict) -> dict:
    f, c = failure_boundary(meta)
    return {
        "rotation_direction": meta["rotation_direction"],
        "failure_frame": f,
        "positive_start_frame": c,
        "shift_frames": c - f,
        "pre_failure_boundary": max(0, f - HARD_WINDOW) <= frame < f,
    }


def positive_counts(profile: dict) -> dict[str, int]:
    counts = {split: 0 for split in SPLITS}
    for meta in profile["attempts"]:
        if meta["result"] == "failure":
            _, c = failure_boundary(meta)
            counts[meta["split"]] += int(meta["frame_count"]) - c
    if min(counts.values()) <= 0:
        raise ValueError("V9.4.3 requires delayed need positives in every split")
    return counts


def select_need_rows(positives, negatives, *, seed=42):
    if seed != 42:
        raise ValueError("V9.4.3 fixes the need sampling seed to 42")
    capacities = {source: len(negatives[source]) for source in NEGATIVE_SOURCES}
    allocation = allocate_balanced_without_replacement(capacities, 3 * len(positives), seed=seed)
    hard_source = NEGATIVE_SOURCES[0]
    reserved = [row for row in negatives[hard_source] if row["need_boundary"]["pre_failure_boundary"]]
    remaining = [row for row in negatives[hard_source] if not row["need_boundary"]["pre_failure_boundary"]]
    if len(reserved) > allocation[hard_source]:
        raise ValueError("V9.4.3 hard-negative quota cannot retain every F-before-30 boundary frame")
    selected = list(reserved) + deterministic_uniform_select(
        remaining,
        allocation[hard_source] - len(reserved),
        seed=seed,
        source=hard_source,
    )
    for source in NEGATIVE_SOURCES[1:]:
        selected.extend(deterministic_uniform_select(negatives[source], allocation[source], seed=seed, source=source))
    rows = list(positives) + selected
    identities = [(row["episode_id"], row["attempt_id"], row["frame_index"]) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("V9.4.3 need candidates/sampling contain duplicate factual frames")
    rows.sort(key=lambda row: stable_priority(seed, "need-shuffle", row["global_index"], row["source"]))
    return rows, {
        "positive_count": len(positives),
        "negative_target": 3 * len(positives),
        "candidate_counts": capacities,
        "selected_counts": allocation,
        "negative_to_positive_ratio": "3:1",
        "negative_source_target": SAMPLING_POLICY["negative_source_target"],
        "boundary_candidate_count": len(reserved),
        "boundary_selected_count": len(reserved),
        "other_hard_negative_selected_count": allocation[hard_source] - len(reserved),
    }


def build_need_rows(*, frames, profile, split, base_row, seed):
    metadata = {(int(row["episode_id"]), int(row["attempt_id"])): row for row in profile["attempts"]}
    positives, negatives = [], {source: [] for source in NEGATIVE_SOURCES}
    ignored = 0
    for frame in frames:
        meta = metadata[frame.attempt_key]
        if meta["split"] != split:
            continue
        label, source = need_role(frame.frame_index, meta)
        if label is None:
            if meta["result"] == "failure":
                ignored += 1
            continue
        row = base_row(frame, split, source) | {"need_recovery": label}
        if meta["result"] == "failure":
            row["need_boundary"] = boundary_annotation(frame.frame_index, meta)
        if label:
            positives.append(row)
        else:
            negatives[source].append(row)
    rows, summary = select_need_rows(positives, negatives, seed=seed)
    if len(positives) != positive_counts(profile)[split]:
        raise ValueError("V9.4.3 source frames omit delayed need positives")
    summary["ignored_count"] = ignored
    return rows, summary


def validate_need_rows(profile: dict, rows: list[dict]) -> dict:
    """Check labels, complete positives, split quotas and all reserved negatives; return an audit."""
    metadata = {(int(row["episode_id"]), int(row["attempt_id"])): row for row in profile["attempts"]}
    expected_positive, expected_boundary = set(), set()
    capacities = {split: {source: 0 for source in NEGATIVE_SOURCES} for split in SPLITS}
    for key, meta in metadata.items():
        split, n = meta["split"], int(meta["frame_count"])
        if meta["result"] == "failure":
            f, c = failure_boundary(meta)
            expected_positive.update((*key, frame) for frame in range(c, n))
            expected_boundary.update((*key, frame) for frame in range(max(0, f - HARD_WINDOW), f))
            capacities[split][NEGATIVE_SOURCES[0]] += f
        elif meta["task"] == "one_success":
            capacities[split][NEGATIVE_SOURCES[1]] += n
        elif meta["result"] == "success" and key[1] == 2:
            need_role(n - 1, meta)  # Validate native R even if no selected rows use this attempt.
            capacities[split][NEGATIVE_SOURCES[2]] += max(0, n - int(meta["rexecution_frame_index"]))
    seen, positives, boundary = set(), set(), set()
    negative_counts = {split: Counter() for split in SPLITS}
    by_attempt = {key: [] for key in metadata}
    for row in rows:
        key = (int(row["episode_id"]), int(row["attempt_id"]))
        frame = int(row["frame_index"])
        identity = (*key, frame)
        if identity in seen:
            raise ValueError("Duplicate V9.4.3 need frame")
        seen.add(identity)
        meta = metadata[key]
        if row["split"] != meta["split"] or not 0 <= frame < int(meta["frame_count"]):
            raise ValueError("V9.4.3 need frame/split identity mismatch")
        label, source = need_role(frame, meta)
        if label is None or row["need_recovery"] is not label or row["source"] != source:
            raise ValueError("V9.4.3 need ignored frame, label or source mismatch")
        if meta["result"] == "failure":
            if row.get("need_boundary") != boundary_annotation(frame, meta):
                raise ValueError("V9.4.3 need boundary annotation mismatch")
            by_attempt[key].append(row)
        elif "need_boundary" in row:
            raise ValueError("V9.4.3 successful need row has a failure boundary")
        if label:
            positives.add(identity)
        else:
            negative_counts[row["split"]][source] += 1
            if identity in expected_boundary:
                boundary.add(identity)
    if positives != expected_positive:
        raise ValueError("V9.4.3 need must retain every delayed positive frame")
    if boundary != expected_boundary:
        raise ValueError("V9.4.3 need dropped a reserved F-before-30 hard negative")
    counts = positive_counts(profile)
    audit = {
        "schema_version": "book_v9_4_3_need_boundary_audit_v1",
        "label_policy": LABEL_POLICY,
        "sampling_policy": SAMPLING_POLICY,
        "splits": {},
        "attempts": [],
    }
    for split in SPLITS:
        allocation = allocate_balanced_without_replacement(capacities[split], 3 * counts[split], seed=42)
        if {source: negative_counts[split][source] for source in NEGATIVE_SOURCES} != allocation:
            raise ValueError("V9.4.3 need negative source quota mismatch")
        audit["splits"][split] = {
            "positive_count": counts[split],
            "negative_count": 3 * counts[split],
            "ignored_count": 0,
            "reserved_boundary_count": 0,
            "negative_source_counts": allocation,
        }
    for key, meta in sorted(metadata.items()):
        if meta["result"] != "failure":
            continue
        f, c = failure_boundary(meta)
        selected_frames = sorted(row["frame_index"] for row in by_attempt[key] if not row["need_recovery"])
        reserved_frames = [frame for frame in selected_frames if frame >= max(0, f - HARD_WINDOW)]
        audit["attempts"].append(
            {
                "episode_id": key[0],
                "attempt_id": key[1],
                "split": meta["split"],
                "rotation_direction": meta["rotation_direction"],
                "failure_frame": f,
                "positive_start_frame": c,
                "shift_frames": c - f,
                "ignored_frame_range_half_open": [f, c],
                "ignored_count": c - f,
                "positive_count": int(meta["frame_count"]) - c,
                "reserved_negative_frame_range_half_open": [max(0, f - HARD_WINDOW), f],
                "reserved_negative_frames": reserved_frames,
                "selected_pre_failure_negative_count": len(selected_frames),
            }
        )
        audit["splits"][meta["split"]]["ignored_count"] += c - f
        audit["splits"][meta["split"]]["reserved_boundary_count"] += len(reserved_frames)
    return audit
