"""Shared runtime state names for closed-loop inference."""

from enum import StrEnum


class RuntimeMode(StrEnum):
    NORMAL_EXECUTION = "normal_execution"
    REASONING = "reasoning"
    RECOVERY_ATTEMPT = "recovery_attempt"
    SUCCESS_OR_FAILURE = "success_or_failure"
