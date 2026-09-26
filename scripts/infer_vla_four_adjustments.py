#!/usr/bin/env python3
"""Run the four horizontal-adjustment prompts concurrently on GPUs 0-3."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SCRIPT = PROJECT_ROOT / "scripts" / "infer_vla_action_from_dataset.py"
MODEL_ROOT = Path(
    "/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/"
    "stage_a_action"
)
V7_7_2_RUN = Path(
    "/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/"
    "multitask_v7_7_2/pi05_rotation_v7_7_2_five_task_h100_no_history"
)


@dataclass(frozen=True)
class InferenceJob:
    gpu: int
    direction: str
    degree: str

    @property
    def name(self) -> str:
        return f"{self.direction}_{self.degree}"


JOBS = (
    InferenceJob(gpu=0, direction="left", degree="slightly"),
    InferenceJob(gpu=1, direction="left", degree="moderately"),
    InferenceJob(gpu=2, direction="right", degree="slightly"),
    InferenceJob(gpu=3, direction="right", degree="moderately"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--episode", type=int, required=True, help="Original raw-data episode id.")
    parser.add_argument("--attempt", type=int, required=True, help="Attempt id within the episode.")
    parser.add_argument("--timestamp", type=float, required=True, help="ROS timestamp in seconds.")
    parser.add_argument(
        "--model",
        "--model-folder",
        dest="model",
        required=True,
        help=(
            f"Model folder name under {MODEL_ROOT}, 'v7_7_2', or a checkpoint/run directory path."
        ),
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Compact combined JSON output path."
    )
    parser.add_argument(
        "--gpus", type=int, nargs=4, metavar=("G0", "G1", "G2", "G3"),
        default=[0, 1, 2, 3], help="GPU indices for the four adjustment jobs in output order.",
    )
    return parser.parse_args()


def resolve_model_checkpoint(model: str) -> Path:
    """Resolve a Stage-A model, V7.7.2 alias, or explicit checkpoint path."""

    if not model:
        raise ValueError("--model must not be empty")
    requested = Path(model).expanduser()
    if model in {"v7_7_2", "multitask_v7_7_2", V7_7_2_RUN.name}:
        checkpoint = V7_7_2_RUN.resolve()
    elif requested.is_absolute() or len(requested.parts) > 1:
        checkpoint = requested.resolve()
        if checkpoint == V7_7_2_RUN.parent.resolve():
            checkpoint = V7_7_2_RUN.resolve()
    else:
        if model in {".", ".."}:
            raise ValueError("--model must be a model folder name or checkpoint path")
        checkpoint = (MODEL_ROOT / requested).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Model/checkpoint directory not found: {checkpoint}")
    return checkpoint


def build_command(
    job: InferenceJob,
    *,
    checkpoint: Path,
    episode: int,
    attempt: int,
    timestamp: float,
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--episode",
        str(episode),
        "--attempt",
        str(attempt),
        "--timestamp",
        repr(timestamp),
        "--mode",
        "adjustment",
        "--direction",
        job.direction,
        "--degree",
        job.degree,
        "--num-inference-steps",
        "10",
        "--noise-seed",
        "0",
        "--output-json",
        str(result_path),
    ]


def run_job(
    job: InferenceJob,
    *,
    checkpoint: Path,
    episode: int,
    attempt: int,
    timestamp: float,
    result_path: Path,
) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    env.setdefault("OPENPI_DATA_HOME", "/home/qxh/.cache/openpi")
    env["USE_TF"] = "0"
    command = build_command(
        job,
        checkpoint=checkpoint,
        episode=episode,
        attempt=attempt,
        timestamp=timestamp,
        result_path=result_path,
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if len(stderr) > 4000:
            stderr = "..." + stderr[-4000:]
        raise RuntimeError(
            f"{job.name} failed on GPU {job.gpu} with exit code "
            f"{completed.returncode}:\n{stderr}"
        )


def endpoint_change(result_path: Path) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    try:
        endpoint = payload["h30_cartesian_changes"]["endpoint_change_from_current"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing endpoint change in {result_path}") from error
    if not isinstance(endpoint, dict):
        raise ValueError(f"Endpoint change is not an object in {result_path}")
    required = {"translation_mm", "rotation_zyx_deg", "gripper"}
    missing = required.difference(endpoint)
    if missing:
        raise ValueError(f"Endpoint change in {result_path} is missing {sorted(missing)}")
    return endpoint


def main() -> None:
    args = parse_args()
    if any(gpu < 0 for gpu in args.gpus) or len(set(args.gpus)) != len(JOBS):
        raise ValueError("--gpus requires four distinct non-negative GPU indices")
    checkpoint = resolve_model_checkpoint(args.model)
    jobs = tuple(replace(job, gpu=gpu) for job, gpu in zip(JOBS, args.gpus, strict=True))
    with tempfile.TemporaryDirectory(prefix="vla_four_adjustments_") as temporary_dir:
        temp_root = Path(temporary_dir)
        result_paths = {job: temp_root / f"{job.name}.json" for job in jobs}
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {
                executor.submit(
                    run_job,
                    job,
                    checkpoint=checkpoint,
                    episode=args.episode,
                    attempt=args.attempt,
                    timestamp=args.timestamp,
                    result_path=result_paths[job],
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except Exception as error:
                    failures.append(f"{job.name}: {error}")
        if failures:
            raise RuntimeError("One or more parallel inferences failed:\n" + "\n".join(failures))

        # Keep only H30's final target relative to the selected input frame.
        compact_output = {
            job.name: endpoint_change(result_paths[job])
            for job in jobs
        }

    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(compact_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
