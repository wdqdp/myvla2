"""Safety-critical generation/hold state for V7.7 asynchronous deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AsyncPhaseState:
    phase: str = "execution"
    attempt_id: int = 1
    action_generation: int = 0
    phase_generation: int = 0
    stop_latched: bool = False
    pending_actions: list[Any] = field(default_factory=list)
    accepted_phase_request_id: str | None = None

    def begin_action_request(self) -> int:
        return self.action_generation

    def accept_action_result(self, generation: int, actions) -> bool:
        if self.stop_latched or int(generation) != self.action_generation:
            return False
        self.pending_actions = list(actions)
        return True

    def begin_phase_request(self, request_id: str) -> tuple[int, str]:
        self.accepted_phase_request_id = str(request_id)
        return self.phase_generation, self.accepted_phase_request_id

    def apply_phase_decision(self, event: dict[str, Any]) -> bool:
        if event.get("event") != "phase_decision":
            raise ValueError("expected phase_decision")
        if event.get("request_id") != self.accepted_phase_request_id or event.get("phase") != self.phase:
            return False
        triggered = (
            bool(event.get("need_recovery")) if self.phase == "execution"
            else bool(event.get("adjustment_end"))
        )
        if triggered:
            self.stop_latched = True
            self.pending_actions.clear()
            self.action_generation += 1
        return triggered

    def accept_failure_event(self, event: dict[str, Any]) -> bool:
        return bool(
            self.stop_latched and self.phase == "execution"
            and event.get("event") == "failure_reason"
            and event.get("request_id") == self.accepted_phase_request_id
        )

    def switch_phase(self, phase: str, *, increment_attempt: bool = False) -> None:
        if phase not in {"execution", "adjustment"} or phase == self.phase:
            raise ValueError(f"invalid phase transition {self.phase!r}->{phase!r}")
        self.phase = phase
        if increment_attempt:
            self.attempt_id += 1
        self.phase_generation += 1
        self.action_generation += 1
        self.pending_actions.clear()
        self.accepted_phase_request_id = None
        # Hold remains latched until a fresh chunk for the new generation is ready.

    def release_hold_with_fresh_actions(self, generation: int, actions) -> bool:
        if int(generation) != self.action_generation or actions is None or len(actions) == 0:
            return False
        self.pending_actions = list(actions)
        self.stop_latched = False
        return True


__all__ = ["AsyncPhaseState"]
