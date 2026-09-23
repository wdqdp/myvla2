#!/usr/bin/env python3
"""Run V8.3-compatible horizontal prompts plus down-moderately on GPUs 0-3.

The checkpoint data profile is read by ``infer_vla_action_from_dataset.py``.
V8.3 uses the same Phase-V2 recovery-plan prompt format as V8.2; its only
training-time difference is the short-descending H30 target construction.
"""

from __future__ import annotations

import infer_vla_four_adjustments as base


InferenceJob = base.InferenceJob
build_command = base.build_command
run_cli = base.run_cli


JOBS = (
    InferenceJob(gpu=0, direction="left", degree="slightly"),
    InferenceJob(gpu=1, direction="left", degree="moderately"),
    InferenceJob(gpu=2, direction="right", degree="slightly"),
    InferenceJob(gpu=3, direction="right", degree="moderately"),
    InferenceJob(
        gpu=0,
        vertical_direction="down",
        vertical_degree="moderately",
    ),
)


if __name__ == "__main__":
    run_cli(JOBS, temporary_prefix="vla_five_adjustments_", description=__doc__)
