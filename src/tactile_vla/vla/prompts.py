"""Prompt builders used by VLA training and runtime inference."""

from __future__ import annotations

import json
from typing import Any


LEGACY_PROMPT_PROFILE = "legacy"
MINIMAL_PROMPT_PROFILE = "minimal_v1"
PROMPT_PROFILES = (LEGACY_PROMPT_PROFILE, MINIMAL_PROMPT_PROFILE)
MAX_MEMORY_PAIRS = 4
MAX_SUPPORTED_ATTEMPTS = MAX_MEMORY_PAIRS + 1


def resolve_prompt_profile(profile: str | None) -> str:
    """Resolve an omitted profile to the pre-profile checkpoint behavior."""

    resolved = (profile or LEGACY_PROMPT_PROFILE).strip()
    if resolved not in PROMPT_PROFILES:
        raise ValueError(
            f"Unknown prompt profile {resolved!r}; expected one of {PROMPT_PROFILES}"
        )
    return resolved


def plan_text(plan: str | None) -> str:
    plan = (plan or "").strip()
    return plan if plan else "none"


def _with_period(text: str) -> str:
    return text if text.endswith(".") else f"{text}."


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
        recovery_plan = plan_text(str(entry.get("recovery_plan", "")))
        failure_reason = str(entry.get("failure_reason", "")).strip() or "unknown"
        plan_field = (
            recovery_plan
            if recovery_plan.startswith("recovery_plan=")
            else f"recovery_plan={recovery_plan}"
        )
        reason_field = (
            failure_reason
            if failure_reason.startswith("failure_reason=")
            else f"failure_reason={failure_reason}"
        )
        lines.append(f"{plan_field}; {reason_field}")
    return " | ".join(lines)


def update_failure_recovery_memory(
    memory: list[dict[str, Any]],
    entry: dict[str, Any],
    *,
    prompt_profile: str | None,
) -> list[dict[str, Any]]:
    """Apply the checkpoint-specific runtime memory retention policy."""

    updated_entry = dict(entry)
    if resolve_prompt_profile(prompt_profile) == MINIMAL_PROMPT_PROFILE:
        if len(memory) >= MAX_MEMORY_PAIRS:
            raise ValueError(
                f"minimal_v1 recovery memory already contains the maximum "
                f"{MAX_MEMORY_PAIRS} pairs; refusing to discard the initial pair"
            )
        return [*memory, updated_entry]
    return [*memory, updated_entry]


def build_execution_prompt(
    *,
    instruction: str,
    tactile_caption: str,
    input_recovery_plan: str | None = "",
    case_id: str | None = None,
    attempt_id: int | None = None,
    prompt_profile: str | None = None,
) -> str:
    profile = resolve_prompt_profile(prompt_profile)
    if profile == MINIMAL_PROMPT_PROFILE:
        return " ".join(
            [
                "Mode: execution.",
                f"Task: {instruction.strip()}",
                f"Recovery plan: {_with_period(plan_text(input_recovery_plan))}",
            ]
        )

    parts = [
        "Mode: execution.",
        f"Task: {instruction.strip()}",
        tactile_caption.strip(),
        f"Recovery plan: {plan_text(input_recovery_plan)}.",
    ]
    # case_id and attempt_id remain API arguments for data organization and
    # logging compatibility, but identifiers are intentionally excluded from
    # the model prompt.
    parts.append("Output the next robot action chunk and monitor whether recovery is needed.")
    return " ".join(parts)


def build_assessment_prompt(
    *,
    instruction: str,
    tactile_caption: str,
    input_recovery_plan: str | None = "",
    prompt_profile: str | None = None,
) -> str:
    """Build the shared V3 need-recovery/failure-diagnosis prefix."""

    parts = [
        "Mode: tactile assessment.",
        f"Task: {instruction.strip()}",
        tactile_caption.strip(),
        (
            f"Recovery plan: {_with_period(plan_text(input_recovery_plan))}"
            if resolve_prompt_profile(prompt_profile) == MINIMAL_PROMPT_PROFILE
            else f"Recovery plan: {plan_text(input_recovery_plan)}."
        ),
    ]
    if resolve_prompt_profile(prompt_profile) == LEGACY_PROMPT_PROFILE:
        parts.append(
            "Determine whether recovery is needed and, if needed, "
            "output only the structured failure_reason."
        )
    return " ".join(parts)


def build_monitor_prompt(
    *,
    instruction: str,
    tactile_caption: str,
    input_recovery_plan: str | None = "",
    prompt_profile: str | None = None,
) -> str:
    """Backward-compatible alias for the shared V3 assessment prompt."""

    return build_assessment_prompt(
        instruction=instruction,
        tactile_caption=tactile_caption,
        input_recovery_plan=input_recovery_plan,
        prompt_profile=prompt_profile,
    )


def build_failure_prompt(
    *,
    instruction: str,
    tactile_caption: str,
    input_recovery_plan: str | None = "",
    prompt_profile: str | None = None,
) -> str:
    """Backward-compatible alias for the shared V3 assessment prompt."""

    return build_assessment_prompt(
        instruction=instruction,
        tactile_caption=tactile_caption,
        input_recovery_plan=input_recovery_plan,
        prompt_profile=prompt_profile,
    )


def build_reasoning_prompt(
    *,
    instruction: str,
    failed_tactile_caption: str,
    failure_recovery_memory: str | list[dict[str, Any]] | None,
    case_id: str | None = None,
    failed_attempt_id: int | None = None,
    prompt_profile: str | None = None,
) -> str:
    profile = resolve_prompt_profile(prompt_profile)
    memory_text = format_memory(failure_recovery_memory)
    parts = [
        "Mode: reasoning.",
        f"Task: {instruction.strip()}",
        failed_tactile_caption.strip(),
        (
            f"Failure-recovery memory: {_with_period(memory_text)}"
            if profile == MINIMAL_PROMPT_PROFILE
            else f"Failure-recovery memory: {memory_text}."
        ),
    ]
    # Identifiers remain available to callers for logging, never as model
    # inputs.
    if profile == LEGACY_PROMPT_PROFILE:
        parts.append("Choose the next recovery plan.")
    return " ".join(parts)


def build_recovery_prompt(
    *,
    instruction: str,
    failed_tactile_caption: str,
    failure_recovery_memory: str | list[dict[str, Any]] | None,
    prompt_profile: str | None = None,
) -> str:
    """Identifier-free V3 recovery prompt.

    Keep ``build_reasoning_prompt`` as the backward-compatible public entry
    point while giving the V3 task an explicit name.
    """

    return build_reasoning_prompt(
        instruction=instruction,
        failed_tactile_caption=failed_tactile_caption,
        failure_recovery_memory=failure_recovery_memory,
        prompt_profile=prompt_profile,
    )
