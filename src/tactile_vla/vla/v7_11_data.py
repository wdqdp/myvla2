"""V7.11 direct tactile recovery inputs derived from the V7.7.2 samples."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


TASK_CYCLE = ("action", "adjustment", "need")
DATA_PROFILE = "rotation_phase_v7_11_direct_tactile_h100"
PROMPT_PROFILE = "v7_11_direct_tactile_h100"


def phase_prompt_without_plan(prompt: str) -> str:
    """Keep V7.7.2 task, Touch, and H100 exactly; remove its plan line."""
    lines = prompt.splitlines()
    plan_lines = [i for i, line in enumerate(lines) if line.startswith("Recovery plan: ")]
    if len(plan_lines) != 1:
        raise ValueError("Expected exactly one V7.7.2 recovery plan line")
    del lines[plan_lines[0]]
    if not any(line.startswith("Touch: ") and line[7:].strip() for line in lines):
        raise ValueError("Missing tactile caption in phase prompt")
    if not any(line.startswith("State history H100 sampled to 11 points: ") for line in lines):
        raise ValueError("Missing H100 in phase prompt")
    return "\n".join(lines)


def direct_recovery_prompt(instruction: str, tactile_caption: str) -> str:
    instruction, tactile_caption = instruction.strip(), tactile_caption.strip()
    if not instruction or not tactile_caption:
        raise ValueError("Recovery action requires instruction and current tactile caption")
    if not instruction.endswith("."):
        instruction += "."
    return f"Mode: adjustment. Task: {instruction}\nTouch: {tactile_caption}"


def factual_selection(rows: Sequence[Mapping[str, Any]], stream: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """Filter the *selected* V7.7.2 stream; do not restore excluded source frames."""
    row_indices, global_indices = [], []
    for ri, gi in zip(stream["manifest_row_indices"], stream["global_indices"], strict=True):
        row = rows[int(ri)]
        if int(row["global_index"]) != int(gi):
            raise ValueError("Source manifest and stream global indices disagree")
        if row.get("sample_variant") == "factual":
            row_indices.append(int(ri))
            global_indices.append(int(gi))
    if not row_indices:
        raise ValueError("No factual adjustment samples selected")
    return row_indices, global_indices


def selected_label_counts(rows, row_indices, label):
    return dict(Counter(str(bool(rows[i][label])).lower() for i in row_indices))
