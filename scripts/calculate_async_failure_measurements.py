#!/usr/bin/env python3
"""Offline asynchronous need=true lift height relative to b, plus adjustment distances."""

# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calculate_sync_failure_measurements import main


if __name__ == "__main__":
    main(expected_mode="asynchronous", description=__doc__)
