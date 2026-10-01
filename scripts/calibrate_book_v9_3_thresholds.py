#!/usr/bin/env python3
"""Score complete Book V9.3 val/test manifests; select both thresholds on val only."""

# ruff: noqa: E402, SLF001
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]
os.environ.setdefault("USE_TF", "0")

import jax
import jax.numpy as jnp
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.models.model import Observation
from torch.utils.data import DataLoader
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.book_v9_3_multitask_data import DATA_PROFILE, validate_index
from tactile_vla.vla.book_v9_3_runtime import DEFAULT_INDEX, DEFAULT_NORM_DIR, DEFAULT_RUN, THRESHOLD_SCHEMA
from tactile_vla.vla.openpi_bridge import collate_numpy
from tactile_vla.vla.v5_3_adjustment_end_evaluation import (
    select_max_recall_under_early_fpr, threshold_metrics,
)
from tactile_vla.vla.v7_7_multitask_data import V77ManifestDataset, TransformedV77Dataset, load_jsonl
from tactile_vla.vla.v7_7_multitask_evaluation import detection_delays
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE
from scripts.serve_tactile_vla_book_v9_3 import inspect_artifacts, load_policy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "4000")
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--norm-stats-dir", type=Path, default=DEFAULT_NORM_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/calibration/book_v9_3/4000")
    parser.add_argument("--batch-size", type=int, default=1, help="1 matches online single-observation inference")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--maximum-negative-fpr", type=float, default=0.01,
                        help="Maximize val recall subject to this false-positive-rate ceiling for each head")
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit very slow CPU weight loading/scoring")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or not 0 <= args.maximum_negative_fpr <= 1:
        parser.error("batch-size must be positive; num-workers nonnegative; maximum-negative-fpr in [0,1]")
    # These values exist only to construct the scorer. They do not select any operating point.
    args.output_action_dim, args.reasoning_max_token_len, args.no_norm = 7, None, False
    args.num_inference_steps = 10
    args.need_recovery_threshold, args.adjustment_end_threshold = 0.5, 0.5
    args.captioner_checkpoint_sha256 = None  # Offline scores use the manifest's cached captions.
    return args


def collect_predictions(policy, index, task, split, manifests, dataset, args):
    stream = index["splits"][split][task]
    raw = V77ManifestDataset(rows=manifests[task], row_indices=stream["manifest_row_indices"],
                             global_indices=stream["global_indices"], task=task, lerobot_dataset=dataset)
    transformed = TransformedV77Dataset(raw, policy._assessment_transform)
    kwargs = {"multiprocessing_context": "spawn"} if args.num_workers else {}
    loader = DataLoader(transformed, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, collate_fn=collate_numpy, **kwargs)
    compact = jnp.asarray(policy._failure_grammar.compact_token_ids, dtype=jnp.int32)
    label_key = "need_recovery_label" if task == "need" else "adjustment_end_label"
    rows = []
    for batch in loader:
        observation = Observation.from_dict(jax.tree.map(jnp.asarray, batch))
        # Use the same need prefill route as online, including its shared failure KV cache.
        if task == "need":
            logits, *_ = policy._assessment_prefill(observation, compact)
        else:
            logits = policy._adjustment_logits(observation)
        probabilities = np.asarray(jax.device_get(jax.nn.softmax(logits, axis=-1)))[:, 1]
        for position, probability in enumerate(probabilities):
            source_row = manifests[task][int(batch["manifest_row_index"][position])]
            rows.append({
                "label": int(batch[label_key][position]), "probability": float(probability),
                "global_index": int(batch["global_index"][position]),
                "episode_id": int(batch["episode_id"][position]),
                "attempt_id": int(batch["attempt_id"][position]),
                "frame_index": int(batch["frame_index"][position]), "source": source_row["source"],
            })
        if len(rows) % 100 < args.batch_size:
            logging.info("%s/%s scored %d/%d", task, split, len(rows), len(raw))
    if len(rows) != stream["sample_count"]:
        raise ValueError("Calibration must score the full manifest without subsampling")
    return rows


def select_from_val(val_rows, test_rows, *, maximum_negative_fpr):
    for rows in (val_rows, test_rows):
        if not rows or any(row["label"] not in (0, 1) or not np.isfinite(row["probability"])
                           or not 0 <= row["probability"] <= 1 for row in rows):
            raise ValueError("Calibration needs finite binary probability rows")
    selection = select_max_recall_under_early_fpr(
        val_rows, maximum_early_false_positive_rate=maximum_negative_fpr,
    )
    threshold = selection["selected"]["threshold"]
    # Test is only reported at the val-selected operating point; never used for selection.
    return {
        "threshold": threshold, "selection": selection,
        "val": threshold_metrics(val_rows, threshold), "test": threshold_metrics(test_rows, threshold),
        "val_negative_source_fpr": negative_source_rates(val_rows, threshold),
        "test_negative_source_fpr": negative_source_rates(test_rows, threshold),
        "positive_window_detection": {
            split: detection_delays(rows, threshold)
            for split, rows in (("val", val_rows), ("test", test_rows))
            if all("episode_id" in row and "attempt_id" in row and "frame_index" in row for row in rows)
        },
    }


def negative_source_rates(rows, threshold):
    result = {}
    for source in sorted({row.get("source", "unknown") for row in rows if not row["label"]}):
        values = [row for row in rows if not row["label"] and row.get("source", "unknown") == source]
        count = sum(row["probability"] >= threshold for row in values)
        result[source] = {"support": len(values), "false_positive_count": count,
                          "false_positive_rate": count / len(values)}
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    warnings.filterwarnings("once", message="The video decoding and encoding capabilities of torchvision.*")
    args = parse_args()
    config, identity = inspect_artifacts(args)
    index = json.loads(args.index_file.read_text())
    validate_index(index)
    if index["training_data_hash"] != identity["training_data_hash"]:
        raise ValueError("Calibration index differs from checkpoint training data")
    if sha256_file(args.index_file) != config["artifact_identity"]["book_v9_3_index_sha256"]:
        raise ValueError("Calibration index SHA256 differs from checkpoint")
    manifests = {}
    for task in ("need", "adjustment"):
        path = Path(index[f"{task}_manifest_file"])
        if sha256_file(path) != index[f"{task}_manifest_sha256"]:
            raise ValueError(f"Changed {task} manifest")
        manifests[task] = load_jsonl(path)
    if args.validate_only:
        print(json.dumps(identity | {"scoring_counts": {
            split: {task: index["splits"][split][task]["sample_count"] for task in manifests}
            for split in ("val", "test")
        }}, indent=2))
        return
    output = args.output_dir / "book_v9_3_thresholds.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty calibration output directory: {args.output_dir}")
    if not args.allow_cpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("No JAX GPU available; calibrate on the model server (or explicitly --allow-cpu)")
    policy = load_policy(args, config, identity, {"thresholds_status": "calibration_in_progress"})
    dataset_dir = Path(index["dataset_dir"])
    dataset = LeRobotDataset(dataset_dir.name, root=dataset_dir, download_videos=False,
                             video_backend=args.video_backend)
    tasks, scored = {}, {}
    for task, name in (("need", "need_recovery"), ("adjustment", "adjustment_end")):
        scored[name] = {
            split: collect_predictions(policy, index, task, split, manifests, dataset, args)
            for split in ("val", "test")
        }
        tasks[name] = select_from_val(scored[name]["val"], scored[name]["test"],
                                      maximum_negative_fpr=args.maximum_negative_fpr)
    report = identity | {
        "schema_version": THRESHOLD_SCHEMA, "data_profile": DATA_PROFILE, "prompt_profile": PROMPT_PROFILE,
        "selection_split": "val", "thresholds_status": "calibrated_on_book_val",
        "selection_policy": "maximum_val_recall_under_negative_fpr_ceiling_ties_higher_threshold",
        "thresholds": {name: value["threshold"] for name, value in tasks.items()},
        "tasks": tasks, "accepted_for_robot": False, "batch_size": args.batch_size,
        "maximum_negative_fpr": args.maximum_negative_fpr,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, splits in scored.items():
        for split, rows in splits.items():
            (args.output_dir / f"{name}_{split}_predictions.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Thresholds saved to {output}")


if __name__ == "__main__":
    main()
