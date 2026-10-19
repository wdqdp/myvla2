#!/usr/bin/env python3
"""V7.9 tactile async inference with checkpoint and norm identity validation."""
# ruff: noqa: E402
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

import agilex_inference_v7_7_asyn as runtime
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v7_9_multitask_data import ADJUSTMENT_SAMPLING_POLICY
from tactile_vla.vla.v7_9_multitask_data import COUNTERFACTUAL_POLICY
from tactile_vla.vla.v7_9_multitask_data import DATA_PROFILE
from tactile_vla.vla.v7_9_multitask_data import ROOT

_shared_validate_metadata = runtime.validate_server_metadata


def get_arguments(argv=None):
    args, parser = runtime.get_arguments(argv)

    def validate_metadata(metadata):
        _shared_validate_metadata(metadata)
        expected = {
            "experiment_version": "v7_9",
            "counterfactual_policy": COUNTERFACTUAL_POLICY,
            "adjustment_sampling_policy": ADJUSTMENT_SAMPLING_POLICY,
            "requires_captioner": True,
            "requires_tactile_topics": True,
            "norm_stats_sha256": sha256_file(args.norm_stats_file),
        }
        mismatch = {key: (metadata.get(key), value) for key, value in expected.items()
                    if metadata.get(key) != value}
        if mismatch:
            raise ValueError(f"V7.9 client/server deployment mismatch: {mismatch}")

    runtime.validate_server_metadata = validate_metadata
    return args, parser


def main():
    runtime.DATA_PROFILE = DATA_PROFILE
    runtime.DEFAULT_NORM_STATS = ROOT / "outputs/rotation_v4_1_new_env/norm_stats/norm_stats.json"
    runtime.DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs/runtime/v7_9_async"
    runtime.main(argument_parser=get_arguments, deployment_label="V7.9")


if __name__ == "__main__":
    main()
