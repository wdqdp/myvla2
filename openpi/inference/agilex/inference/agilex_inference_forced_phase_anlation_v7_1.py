#!/usr/bin/env python3
"""Synchronous single-arm V7.1 forced-phase action inference (no H60)."""

# ruff: noqa: E402, I001

from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import agilex_inference_forced_phase_anlation as implementation


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
implementation.DEFAULT_NORM_STATS = ROOT / "outputs/rotation_v4/norm_stats/norm_stats.json"
implementation.DEFAULT_LOG_ROOT = implementation.PROJECT_ROOT / "outputs/runtime/forced_phase_ablation_v7_1"


def _has_option(name: str) -> bool:
    return name in sys.argv or any(value.startswith(name + "=") for value in sys.argv[1:])


if __name__ == "__main__":
    if not _has_option("--expected-data-profile"):
        sys.argv.extend(["--expected-data-profile", "rotation_phase_v7_1_adjustment"])
    implementation.main()
