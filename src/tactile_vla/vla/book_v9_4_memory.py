"""V9.4 text-only recovery histories; no changes to real observations or targets."""

from __future__ import annotations

from collections import Counter, defaultdict
import re

from tactile_vla.vla.artifacts import sha256_json
from tactile_vla.vla.prompts import format_memory
from tactile_vla.vla.v7_7_multitask_data import stable_priority

MEMORY_LENGTHS = (1, 2, 3, 4)
RULE_VERSION = "book_v9_4_left_right_moderately_memory_v1"
PLAN_MAX_TOKEN_LEN = 320
MEMORY_POLICY = {
    "rule_version": RULE_VERSION,
    "lengths": list(MEMORY_LENGTHS),
    "variants_per_observation": 4,
    "seed": 42,
    "first_plan": "initial plan",
    "directions": ["left", "right"],
    "horizontal_magnitude": "moderately",
    "vertical": "none moderately",
    "terminal_failure": "current_real_failure",
    "target": "unchanged_adjacent_real_attempt2",
    "history_source": "synthetic_text_only",
    "runtime_retention": "initial_pair_plus_latest_three",
    "max_recovery_attempts": None,
}


def failure_direction(text: str) -> str:
    match = re.fullmatch(r"failure_reason=rotate (left|right),grasp appropriate\.", text)
    if not match:
        raise ValueError(f"V9.4 history requires a real left/right rotation failure: {text}")
    return match.group(1)


def plan_direction(text: str) -> str:
    match = re.fullmatch(
        r"recovery_plan=move horizontally (left|right) moderately, move vertically none moderately\.", text
    )
    if not match:
        raise ValueError(f"V9.4 plans require left/right moderately, vertical none moderately: {text}")
    return match.group(1)


def plan_for_failure(text: str) -> str:
    return f"recovery_plan=move horizontally {failure_direction(text)} moderately, move vertically none moderately."


def expand_plan_rows(rows: list[dict], failure_rows: list[dict], *, sources: dict | None = None) -> list[dict]:
    failures = {(row["split"], row["global_index"]): row["target_failure_reason"] for row in failure_rows}
    expanded = []
    for row in rows:
        real_failure = failures[row["split"], row["global_index"]]
        failure_direction(real_failure)
        plan_direction(row["target_recovery_plan"])
        if row["memory_length"] != 1 or row["failure_recovery_memory"][-1]["failure_reason"] != real_failure:
            raise ValueError("V9.4 expects unchanged one-pair source histories ending in the real failure")
        head, separator, _ = row["prompt"].partition("Failure-recovery memory: ")
        if not separator:
            raise ValueError("V9.4 source plan prompt has no memory heading")
        source = None if sources is None else sources[row["split"], row["global_index"]]
        if source is not None and (
            source["current_observation"]["failure_reason"] != real_failure
            or source["target_recovery_plan"] != row["target_recovery_plan"]
            or source["target_source"]["source_type"] != "real"
            or source["target_source"]["episode_id"] != row["episode_id"]
            or source["target_source"]["failed_attempt_id"] != row["attempt_id"]
            or source["target_source"]["plan_attempt_id"] != row["attempt_id"] + 1
        ):
            raise ValueError("V9.4 plan must retain the adjacent real attempt2 target")
        for length in MEMORY_LENGTHS:
            key = (42, row["split"], row["global_index"], length, RULE_VERSION)
            variant_id = "book-v9_4-" + sha256_json(key)[:24]
            bits = stable_priority(*key)
            reasons = [
                f"failure_reason=rotate {'right' if (bits >> pair) & 1 else 'left'},grasp appropriate."
                for pair in range(length - 1)
            ] + [real_failure]
            memory = [
                {
                    "recovery_plan": "initial plan" if pair == 0 else plan_for_failure(reasons[pair - 1]),
                    "failure_reason": reason,
                    "source_type": "real_current_failure" if pair == length - 1 else "synthetic",
                    "plan_source_type": "initial" if pair == 0 else "synthetic",
                    "rule_version": RULE_VERSION,
                    "seed": 42,
                    "variant_id": variant_id,
                    "pair_index": pair,
                }
                for pair, reason in enumerate(reasons)
            ]
            text = format_memory(memory)
            expanded.append(
                row
                | {
                    "source": "book_v9_4_synthetic_text_history",
                    "memory_length": length,
                    "failure_recovery_memory": memory,
                    "current_real_failure": real_failure,
                    "variant_id": variant_id,
                    "rule_version": RULE_VERSION,
                    "seed": 42,
                    "prompt": head + "Failure-recovery memory: " + text,
                    **(
                        {"target_source": dict(source["target_source"]), "source_variant_id": source["variant_id"]}
                        if source is not None
                        else {}
                    ),
                }
            )
    return expanded


def validate_plan_row(row: dict, real_failure: str) -> None:
    length = row["memory_length"]
    memory = row["failure_recovery_memory"]
    if length not in MEMORY_LENGTHS or len(memory) != length:
        raise ValueError("V9.4 memory length mismatch")
    if row.get("rule_version") != RULE_VERSION or row.get("seed") != 42 or not row.get("variant_id"):
        raise ValueError("V9.4 memory provenance mismatch")
    expected_id = "book-v9_4-" + sha256_json((42, row["split"], row["global_index"], length, RULE_VERSION))[:24]
    if row["variant_id"] != expected_id:
        raise ValueError("V9.4 memory variant ID mismatch")
    if row.get("current_real_failure") != real_failure or memory[-1]["failure_reason"] != real_failure:
        raise ValueError("V9.4 terminal memory failure is not the current real failure")
    failure_direction(real_failure)
    plan_direction(row["target_recovery_plan"])
    for pair, entry in enumerate(memory):
        failure_direction(entry["failure_reason"])
        expected = {
            "recovery_plan": "initial plan" if pair == 0 else plan_for_failure(memory[pair - 1]["failure_reason"]),
            "source_type": "real_current_failure" if pair == length - 1 else "synthetic",
            "plan_source_type": "initial" if pair == 0 else "synthetic",
            "pair_index": pair,
            "rule_version": RULE_VERSION,
            "seed": 42,
            "variant_id": row["variant_id"],
        }
        if any(entry.get(k) != v for k, v in expected.items()) or set(entry) != set(expected) | {"failure_reason"}:
            raise ValueError("V9.4 memory prefix/provenance mismatch; synthetic pairs cannot claim donor episodes")
    _, separator, text = row["prompt"].partition("Failure-recovery memory: ")
    if not separator or text != format_memory(memory):
        raise ValueError("V9.4 prompt does not contain the complete memory")


def validate_balanced_variants(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["split"], row["global_index"]].append(row)
    for variants in grouped.values():
        if Counter(row["memory_length"] for row in variants) != Counter(MEMORY_LENGTHS):
            raise ValueError("V9.4 requires one variant per memory length per real observation")
        invariant = {
            (
                row["target_recovery_plan"],
                row["current_real_failure"],
                row["prompt"].partition("Failure-recovery memory: ")[0],
                tuple(row.get(key) for key in ("episode_id", "attempt_id", "frame_index", "timestamp", "frame_offset")),
            )
            for row in variants
        }
        if len(invariant) != 1:
            raise ValueError("V9.4 memory variants changed the real observation/target")


def validate_plan_tokens(rows: list[dict], *, tokenizer, normalized_states: dict) -> dict:
    """Use the exact training tokenizer, including current state, Answer and target EOS."""
    lengths = defaultdict(list)
    for row in rows:
        target = tokenizer.encode_text(row["target_recovery_plan"], add_eos=True)
        _, mask, _, _, prefix = tokenizer.tokenize_structured_response(
            row["prompt"], normalized_states[row["global_index"]], target, max_len=PLAN_MAX_TOKEN_LEN
        )
        total = int(mask.sum())
        row["plan_token_lengths"] = {"prefix": int(prefix), "target": len(target), "total": total}
        lengths[str(row["memory_length"])].append(total)
    return {
        "max_len": PLAN_MAX_TOKEN_LEN,
        "includes": "current_qpos_answer_target_eos",
        "truncation": "error",
        "by_memory_length": {
            key: {"count": len(values), "min": min(values), "max": max(values)}
            for key, values in sorted(lengths.items())
        },
    }


def validate_dataset_plan_tokens(rows: list[dict], *, dataset_dir, norm_stats_dir) -> dict:
    """Read actual qpos and mirror the training normalization without loading images."""
    import numpy as np
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.transforms import Normalize
    from tactile_vla.vla.v5_3_adjustment_end_data import scan_selected_qpos

    qpos = scan_selected_qpos(dataset_dir=dataset_dir, selected_episode_ids={row["episode_id"] for row in rows})
    normalizer = Normalize(normalize.load(norm_stats_dir), use_quantiles=True)
    states = {
        global_index: np.asarray(
            normalizer({"state": np.asarray(qpos[global_index], dtype=np.float32)})["state"], dtype=np.float32
        )
        for global_index in {row["global_index"] for row in rows}
    }
    return validate_plan_tokens(rows, tokenizer=PaligemmaTokenizer(PLAN_MAX_TOKEN_LEN), normalized_states=states)


def append_runtime_memory(memory: list[dict], entry: dict) -> list[dict]:
    """Unlimited recoveries, at most four pairs: initial plus latest three."""
    if len(memory) > 4:
        raise ValueError("V9.4 runtime memory already exceeds four pairs")
    updated = [dict(pair) for pair in memory] + [dict(entry)]
    if updated[0].get("recovery_plan") not in {"initial plan", "recovery_plan=initial plan"}:
        raise ValueError("V9.4 runtime must preserve the initial-plan pair")
    return updated if len(updated) <= 4 else [updated[0], *updated[-3:]]


def plan_eval_groups(rows: list[dict], row_indices: list[int]) -> dict[tuple[int, str], list[int]]:
    groups = defaultdict(list)
    for position, row_index in enumerate(row_indices):
        row = rows[row_index]
        groups[row["memory_length"], plan_direction(row["target_recovery_plan"])].append(position)
    return dict(groups)


def merge_plan_group_metrics(metrics: dict[tuple[int, str], dict]) -> dict:
    def merge(values):
        support = sum(value["num_samples"] for value in values)
        return {
            "exact_match": sum(v["exact_match"] * v["num_samples"] for v in values) / support if support else 0.0,
            "support": support,
        }

    overall = merge(list(metrics.values()))
    return {
        "exact_match": overall["exact_match"],
        "num_samples": overall["support"],
        "by_memory_length": {
            str(length): merge([v for (group_length, _), v in metrics.items() if group_length == length])
            for length in MEMORY_LENGTHS
        },
        "by_direction": {
            d: merge([v for (_, direction), v in metrics.items() if direction == d]) for d in ("left", "right")
        },
        "by_direction_magnitude": {
            d + "/moderately": merge([v for (_, direction), v in metrics.items() if direction == d])
            for d in ("left", "right")
        },
        "by_memory_length_direction": {
            f"{length}/{direction}": merge([value]) for (length, direction), value in sorted(metrics.items())
        },
    }
