#!/usr/bin/env python3
"""Serve V7.9 using the unchanged tactile V7.7 streamed phase/output protocol."""
# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / 'src'), str(PROJECT_ROOT / 'openpi/src')]

from scripts import serve_tactile_vla_v7_7 as server
from tactile_vla.vla.v7_9_multitask_data import DATA_PROFILE


def main():
    server.DATA_PROFILE = DATA_PROFILE
    server.main()


if __name__ == '__main__':
    main()
