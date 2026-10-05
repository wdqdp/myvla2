from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

from tactile_vla.vla.book_v9_4_memory import (  # noqa: E402
    MEMORY_LENGTHS,
    append_runtime_memory,
    expand_plan_rows,
    merge_plan_group_metrics,
    plan_eval_groups,
    validate_balanced_variants,
    validate_plan_row,
    validate_plan_tokens,
)
from tactile_vla.vla.prompts import format_memory  # noqa: E402


def source_rows(n=2):
    rows, failures, sources = [], [], {}
    for global_index in range(n):
        direction = "left" if global_index % 2 else "right"
        reason = f"failure_reason=rotate {direction},grasp appropriate."
        target = f"recovery_plan=move horizontally {direction} moderately, move vertically none moderately."
        row = {
            "split": "train",
            "global_index": global_index,
            "episode_id": global_index + 1,
            "attempt_id": 1,
            "frame_index": 50,
            "timestamp": 100,
            "frame_offset": 14,
            "memory_length": 1,
            "failure_recovery_memory": [{"recovery_plan": "initial plan", "failure_reason": reason}],
            "target_recovery_plan": target,
            "prompt": "Mode: reasoning. Task: book. Touch[same actual observation] Failure-recovery memory: old",
        }
        rows.append(row)
        failures.append(row | {"target_failure_reason": reason})
        sources["train", global_index] = {
            "current_observation": {"failure_reason": reason},
            "target_recovery_plan": target,
            "variant_id": "original-source-" + str(global_index),
            "target_source": {
                "source_type": "real",
                "episode_id": global_index + 1,
                "failed_attempt_id": 1,
                "plan_attempt_id": 2,
            },
        }
    return rows, failures, sources


def test_four_deterministic_text_variants_keep_real_terminal_and_target():
    rows, failures, sources = source_rows(40)
    before = copy.deepcopy((rows, failures, sources))
    expanded = expand_plan_rows(rows, failures, sources=sources)
    assert expanded == expand_plan_rows(rows, failures, sources=sources)
    assert (rows, failures, sources) == before
    assert len(expanded) == 160
    assert len({row["variant_id"] for row in expanded}) == 160
    validate_balanced_variants(expanded)
    transitions = set()
    for row in expanded:
        original = rows[row["global_index"]]
        validate_plan_row(row, failures[row["global_index"]]["target_failure_reason"])
        assert row["target_recovery_plan"] == original["target_recovery_plan"]
        assert row["target_source"] == sources["train", row["global_index"]]["target_source"]
        for key in ("episode_id", "attempt_id", "frame_index", "timestamp"):
            assert row[key] == original[key]
        assert (
            row["prompt"].split("Failure-recovery memory:")[0]
            == original["prompt"].split("Failure-recovery memory:")[0]
        )
        assert "slightly" not in row["prompt"] and "front" not in row["prompt"] and "back" not in row["prompt"]
        memory = row["failure_recovery_memory"]
        transitions.update((a["failure_reason"], b["failure_reason"]) for a, b in zip(memory, memory[1:]))
    # Both repetitions and alternating directions are reachable, not forced alternation.
    assert any(a == b for a, b in transitions)
    assert any(a != b for a, b in transitions)


@pytest.mark.parametrize("change", ["terminal", "prefix", "slightly", "donor", "pair_index", "prompt", "seed"])
def test_corrupt_memory_is_rejected(change):
    rows, failures, _ = source_rows()
    row = expand_plan_rows(rows, failures)[3]
    if change == "terminal":
        row["failure_recovery_memory"][-1]["failure_reason"] = "failure_reason=rotate left,grasp appropriate."
    elif change == "prefix":
        row["failure_recovery_memory"][0]["recovery_plan"] = row["target_recovery_plan"]
    elif change == "slightly":
        row["failure_recovery_memory"][1]["recovery_plan"] = row["failure_recovery_memory"][1]["recovery_plan"].replace(
            "moderately", "slightly"
        )
    elif change == "donor":
        row["failure_recovery_memory"][0]["episode_id"] = 123
    elif change == "pair_index":
        row["failure_recovery_memory"][1]["pair_index"] = 3
    elif change == "prompt":
        row["prompt"] = row["prompt"][:50]
    else:
        row["seed"] = 99
    with pytest.raises(ValueError):
        validate_plan_row(row, failures[0]["target_failure_reason"])


def test_missing_length_or_changed_real_target_rejected():
    rows, failures, sources = source_rows()
    expanded = expand_plan_rows(rows, failures)
    with pytest.raises(ValueError, match="one variant"):
        validate_balanced_variants(expanded[1:])
    expanded[0]["target_recovery_plan"] = rows[1]["target_recovery_plan"]
    with pytest.raises(ValueError, match="changed"):
        validate_balanced_variants(expanded)
    sources["train", 0]["target_source"]["plan_attempt_id"] = 3
    with pytest.raises(ValueError, match="adjacent real"):
        expand_plan_rows(rows, failures, sources=sources)


def test_runtime_unlimited_recoveries_preserve_initial_and_latest_three():
    memory = []
    for number in range(20):
        entry = {
            "recovery_plan": "initial plan" if number == 0 else f"plan{number}",
            "failure_reason": f"reason{number}",
        }
        old = copy.deepcopy(memory)
        updated = append_runtime_memory(memory, entry)
        assert memory == old
        memory = updated
        assert len(memory) == min(4, number + 1)
        assert memory[0]["failure_reason"] == "reason0"
        if number >= 4:
            assert [pair["failure_reason"] for pair in memory[1:]] == [
                f"reason{i}" for i in range(number - 2, number + 1)
            ]
        assert "recovery_plan=initial plan" in format_memory(memory)
    with pytest.raises(ValueError):
        append_runtime_memory([], {"recovery_plan": "not initial"})


def test_token_check_includes_real_state_target_eos_and_rejects_overflow():
    rows, failures, _ = source_rows(1)
    expanded = expand_plan_rows(rows, failures)
    calls = []

    class Tokenizer:
        def encode_text(self, text, *, add_eos):
            assert add_eos and text == rows[0]["target_recovery_plan"]
            return [7, 8, 1]

        def tokenize_structured_response(self, prompt, state, target, *, max_len):
            assert max_len == 320 and target == [7, 8, 1]
            assert np.array_equal(state, np.arange(7))
            calls.append(prompt)
            return None, np.ones(123, dtype=bool), None, None, 120

    summary = validate_plan_tokens(expanded, tokenizer=Tokenizer(), normalized_states={0: np.arange(7)})
    assert len(calls) == 4
    assert summary["by_memory_length"]["4"] == {"count": 1, "min": 123, "max": 123}
    assert expanded[0]["plan_token_lengths"] == {"prefix": 120, "target": 3, "total": 123}

    class Overflow(Tokenizer):
        def tokenize_structured_response(self, *args, **kwargs):
            raise ValueError("exceeds max token length")

    with pytest.raises(ValueError, match="exceeds"):
        validate_plan_tokens(expanded, tokenizer=Overflow(), normalized_states={0: np.arange(7)})


def test_eval_group_positions_and_weighted_metrics():
    rows, failures, _ = source_rows()
    expanded = expand_plan_rows(rows, failures)
    positions = [7, 0, 5, 2, 4, 1, 3, 6]
    groups = plan_eval_groups(expanded, positions)
    assert len(groups) == 8
    for (length, direction), indices in groups.items():
        for index in indices:
            row = expanded[positions[index]]
            assert row["memory_length"] == length
            assert f"horizontally {direction} moderately" in row["target_recovery_plan"]
    metrics = {
        (length, direction): {
            "exact_match": 0.5 if direction == "right" else 1.0,
            "num_samples": 2 if direction == "right" else 1,
        }
        for length in MEMORY_LENGTHS
        for direction in ("left", "right")
    }
    result = merge_plan_group_metrics(metrics)
    assert result["num_samples"] == 12 and result["exact_match"] == pytest.approx(2 / 3)
    assert result["by_memory_length"]["4"]["support"] == 3
    assert result["by_memory_length_direction"]["4/right"] == {"exact_match": 0.5, "support": 2}


def test_training_eval_uses_all_variant_groups_without_loading_images(monkeypatch):
    from types import SimpleNamespace
    from scripts import train_vla_multitask_book_v9_4 as entry

    rows, failures, _ = source_rows()
    expanded = expand_plan_rows(rows, failures)
    raw = SimpleNamespace(rows=expanded, row_indices=list(range(8)))
    transformed = SimpleNamespace(dataset=raw)
    loader = SimpleNamespace(dataset=transformed, batch_size=8)
    calls = []
    monkeypatch.setattr(entry.training_base, "_loader", lambda dataset, **kwargs: SimpleNamespace(dataset=dataset))

    def evaluate(state, group_loader, task, grammar, sharding, *, max_samples):
        assert max_samples is None
        calls.extend(group_loader.dataset.indices)
        return {"exact_match": 1.0, "num_samples": len(group_loader.dataset.indices)}

    monkeypatch.setattr(entry, "_BASE_EVALUATE_TEXT", evaluate)
    result = entry.evaluate_text(None, loader, "plan", None, None, max_samples=1)
    assert sorted(calls) == list(range(8))
    assert result["num_samples"] == 8
