#!/usr/bin/env python3
"""Run joint V12.1 Blackbox/Book inference using the streamed asynchronous control loop."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT / "src")]

import agilex_inference_book_v9_5_asyn as shared
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v12_1_deployment import DEFAULT_NORM_DIR
from tactile_vla.vla.v12_1_deployment import TASKS
from tactile_vla.vla.v12_1_deployment import append_memory
from tactile_vla.vla.v12_1_deployment import runtime_caption
from tactile_vla.vla.v12_1_deployment import validate_server_metadata
from tactile_vla.vla.v12_1_deployment import validate_tactile_caption


def configure_task(task):
    """Configure the imported control loop before opening either WebSocket."""
    spec = TASKS[task]
    shared.BOOK_INSTRUCTION = spec["instruction"]
    shared.LABEL_SCHEMA_VERSION = spec["schema_version"]
    shared.LABEL_FIELDS = tuple(spec["label_fields"])
    shared.validate_book_server_metadata = lambda metadata: validate_server_metadata(metadata, task)
    shared.validate_tactile_caption = lambda caption: validate_tactile_caption(caption, task)
    shared.append_runtime_memory = lambda memory, entry: append_memory(memory, entry, task)


def get_arguments(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--task", choices=tuple(TASKS))
    selected, _ = selector.parse_known_args(argv)
    # Help remains available without selecting a task; actual launches require --task.
    task = selected.task or "blackbox"
    configure_task(task)
    parser = shared.build_argument_parser()
    parser.description = __doc__
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, default=1000)
    parser.set_defaults(
        instruction=TASKS[task]["instruction"],
        captioner_checkpoint=TASKS[task]["captioner"],
        norm_stats_file=DEFAULT_NORM_DIR / "norm_stats.json",
        log_dir=PROJECT_ROOT / f"outputs/runtime/v12_1_{task}_async",
    )
    return parser.parse_args(argv), parser


def validate_args(args, parser):
    configure_task(args.task)
    shared.validate_args(args, parser)
    if args.expected_checkpoint_step <= 0:
        parser.error("--expected-checkpoint-step must be positive")
    actual_sha = sha256_file(args.captioner_checkpoint)
    if actual_sha != TASKS[args.task]["captioner_sha256"]:
        parser.error(f"V12.1 {args.task} captioner SHA256 differs from the versioned deployment identity")


class CanonicalCaptioner:
    """Preserve predictor metadata and probabilities, adding only the missing source field."""

    def __init__(self, predictor, task):
        self.predictor, self.task = predictor, task

    def __getattr__(self, name):
        return getattr(self.predictor, name)

    def predict(self, mesh_motion, force):
        result = self.predictor.predict(mesh_motion, force)
        return replace(result, caption=runtime_caption(result.caption, self.task))


def run_v12_1_async(args, operator, action_policy, phase_client, captioner, keyboard, logger):
    configure_task(args.task)
    metadata = action_policy.get_server_metadata()
    validate_server_metadata(metadata, args.task)
    if metadata.get("checkpoint_step") != args.expected_checkpoint_step:
        raise ValueError(f"V12.1 client requires checkpoint step {args.expected_checkpoint_step}")
    shared.run_book_v9_5_async(
        args,
        operator,
        action_policy,
        phase_client,
        CanonicalCaptioner(captioner, args.task),
        keyboard,
        logger,
    )


def main():
    shared.main(
        argument_parser=get_arguments,
        argument_validator=validate_args,
        run=run_v12_1_async,
        deployment_label="V12.1 Blackbox/Book",
    )


if __name__ == "__main__":
    main()
