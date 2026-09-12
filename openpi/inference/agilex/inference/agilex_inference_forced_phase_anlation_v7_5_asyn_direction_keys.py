#!/usr/bin/env python3
"""V7.5 async inference with keyboard-selected horizontal recovery plans."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Literal

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

import agilex_inference_forced_phase_anlation as v52
import agilex_inference_forced_phase_anlation_v7_5_asyn as implementation

Phase = Literal["execution", "adjustment"]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/forced_phase_ablation_v7_5_async_direction_keys"
DIRECTION_KEYS = {
    "a": ("left", "moderately"),
    "s": ("left", "slightly"),
    "d": ("right", "slightly"),
    "f": ("right", "moderately"),
}
FIXED_SELECTION_OPTIONS = (
    "--rotation-direction",
    "--rotation-magnitude",
    "--forced-failure-reason",
    "--forced-recovery-plan",
)


def _select_adjustment(args: argparse.Namespace, key: str) -> None:
    direction, magnitude = DIRECTION_KEYS[key]
    failure_reason, recovery_plan = v52.rotation_targets(direction, magnitude)
    args.rotation_direction = direction
    args.rotation_magnitude = magnitude
    args.forced_failure_reason = failure_reason
    args.forced_recovery_plan = recovery_plan
    args.adjustment_selection_key = key
    print(f"[ADJUSTMENT SELECT] key={key} direction={direction} magnitude={magnitude} plan={recovery_plan}")


def _poll_key(args: argparse.Namespace, keyboard: Any, *, phase: Phase) -> str | None:
    """Handle keys between chunks; direction keys replace the old SPACE trigger."""

    key = keyboard.get_key()
    if key is None:
        return None
    if key == args.quit_key:
        return "quit"
    if phase == "execution" and key in DIRECTION_KEYS:
        _select_adjustment(args, key)
        return "trigger"
    return None


def _poll_control_key(
    args: argparse.Namespace,
    keyboard: Any,
    *,
    phase: Phase,
    chunk_paused: bool = False,
) -> v52.ControlSignal | None:
    """Handle keys while actions are publishing; this client never pauses chunks."""

    del chunk_paused
    key = keyboard.get_key()
    if key is None:
        return None
    if key == args.quit_key:
        return v52.ControlSignal("quit", key)
    if phase == "execution" and key in DIRECTION_KEYS:
        _select_adjustment(args, key)
        return v52.ControlSignal("trigger", key)
    return None


def _keyboard_selection_validate_args(
    original_validate,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Run the shared validator with a temporary legal plan, then require a key."""

    if any(
        value == option or value.startswith(f"{option}=")
        for value in sys.argv[1:]
        for option in FIXED_SELECTION_OPTIONS
    ):
        parser.error("This client selects recovery with a/s/d/f; remove fixed rotation and recovery-plan options")
    placeholder_failure, placeholder_plan = v52.rotation_targets("right", "moderately")
    args.rotation_direction = "right"
    args.rotation_magnitude = "moderately"
    args.forced_failure_reason = placeholder_failure
    args.forced_recovery_plan = placeholder_plan
    original_validate(args, parser)
    args.rotation_direction = None
    args.rotation_magnitude = None
    args.forced_failure_reason = None
    args.forced_recovery_plan = None
    args.adjustment_selection_key = None


def main() -> None:
    original_validate = implementation.base.validate_args

    def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        _keyboard_selection_validate_args(original_validate, args, parser)

    implementation.DEFAULT_LOG_ROOT = DEFAULT_LOG_ROOT
    implementation.base._poll_key = _poll_key
    implementation.v52._poll_control_key = _poll_control_key
    implementation.base.validate_args = validate_args
    print(
        "V7.5 async direction controls: "
        "a=left moderately, s=left slightly, d=right slightly, "
        "f=right moderately, q=quit; SPACE is disabled."
    )
    implementation.main()


if __name__ == "__main__":
    main()
