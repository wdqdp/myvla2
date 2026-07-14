"""Prompt builders used by VLA training and runtime inference."""

from __future__ import annotations

import json
from typing import Any


def plan_text(plan: str | None) -> str:
    plan = (plan or "").strip()
    return plan if plan else "none"


def parse_memory(memory: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if memory is None:
        return []
    if isinstance(memory, list):
        return memory
    memory = memory.strip()
    if not memory:
        return []
    parsed = json.loads(memory)
    if not isinstance(parsed, list):
        raise ValueError(f"failure_recovery_memory must be a JSON list, got: {type(parsed).__name__}")
    return parsed


def format_memory(memory: str | list[dict[str, Any]] | None) -> str:
    entries = parse_memory(memory)
    if not entries:
        return "none"
    lines = []
    for entry in entries:
        attempt_id = entry.get("attempt_id", "?")
        recovery_plan = plan_text(str(entry.get("recovery_plan", "")))
        failure_reason = str(entry.get("failure_reason", "")).strip() or "unknown"
        lines.append(f"attempt {attempt_id}: recovery_plan={recovery_plan}; failure_reason={failure_reason}")
    return " | ".join(lines)


def build_execution_prompt(
    *,
    instruction: str,
    tactile_caption: str,
    input_recovery_plan: str | None = "",
    case_id: str | None = None,
    attempt_id: int | None = None,
) -> str:
    parts = [
        "Mode: execution.",
        f"Task: {instruction.strip()}",
        tactile_caption.strip(),
        f"Recovery plan: {plan_text(input_recovery_plan)}.",
    ]
    if case_id:
        parts.append(f"Case: {case_id}.")
    if attempt_id is not None:
        parts.append(f"Attempt: {attempt_id}.")
    parts.append("Output the next robot action chunk and monitor whether recovery is needed.")
    return " ".join(parts)


def build_reasoning_prompt(
    *,
    instruction: str,
    failed_tactile_caption: str,
    failure_recovery_memory: str | list[dict[str, Any]] | None,
    case_id: str | None = None,
    failed_attempt_id: int | None = None,
) -> str:
    parts = [
        "Mode: reasoning.",
        f"Task: {instruction.strip()}",
        failed_tactile_caption.strip(),
        f"Failure-recovery memory: {format_memory(failure_recovery_memory)}.",
    ]
    if case_id:
        parts.append(f"Case: {case_id}.")
    if failed_attempt_id is not None and failed_attempt_id >= 0:
        parts.append(f"Failed attempt: {failed_attempt_id}.")
    parts.append("Choose the next recovery plan.")
    return " ".join(parts)
