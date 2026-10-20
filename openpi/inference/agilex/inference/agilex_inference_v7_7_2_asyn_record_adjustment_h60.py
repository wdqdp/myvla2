#!/usr/bin/env python3
"""V7.7.2 async recording with 11 phase qpos points from the latest 60 frames."""

# ruff: noqa: E402
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import agilex_inference_v7_7_2_asyn_record_adjustment as recording


def main():
    recording.main(
        default_log_root=PROJECT_ROOT / "outputs/runtime/v7_7_2_h60_adjustment_measurements",
        deployment_label="V7.7.2 H60 uniform history",
        phase_history_window_frames=60,
    )


if __name__ == "__main__":
    main()
