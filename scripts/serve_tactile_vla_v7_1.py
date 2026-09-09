#!/usr/bin/env python3
"""Serve a V7.1 Stage A action checkpoint with strict config identity checks."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.serve_tactile_vla_action_ablation import main


def _has_option(name: str) -> bool:
    return name in sys.argv or any(value.startswith(name + "=") for value in sys.argv[1:])


if __name__ == "__main__":
    if not _has_option("--expected-data-profile"):
        sys.argv.extend(["--expected-data-profile", "rotation_phase_v7_1_adjustment"])
    main()
