#!/usr/bin/env python3
"""Build Book V9.4 five-task data from an incremental V4 dataset and new Stage A."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import build_book_v9_3_multitask_data as base
from scripts.build_v7_7_multitask_data import _write_json, _write_jsonl
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_4_memory import MEMORY_POLICY, expand_plan_rows, validate_dataset_plan_tokens
from tactile_vla.vla.v4_data import load_jsonl
from tactile_vla.vla.book_v9_4_multitask_data import (
    ROOT,
    DEFAULT_ACTION_INDEX,
    DEFAULT_ADJUSTMENT_DIR,
    DEFAULT_MULTITASK_DIR,
    DATA_PROFILE,
    INDEX_SCHEMA,
    MANIFEST_SCHEMA,
    expected_counts,
    load_captioner_provenance,
    target_coverage,
    validate_index,
    validate_manifest_rows,
    validate_upload_metadata,
)


def parse_args(argv=None, *, default_action_index=DEFAULT_ACTION_INDEX,
               default_adjustment_dir=DEFAULT_ADJUSTMENT_DIR, default_output_dir=DEFAULT_MULTITASK_DIR):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "lerobot_data/tactile_vla_rotation_v4")
    parser.add_argument("--v4-dir", type=Path, default=ROOT / "outputs/rotation_v4")
    parser.add_argument("--action-index", type=Path, default=default_action_index)
    parser.add_argument("--adjustment-dir", type=Path, default=default_adjustment_dir)
    parser.add_argument("--incremental-state-dir", type=Path, default=ROOT / "outputs/incremental_state")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build(args, *, data_profile=DATA_PROFILE, index_schema=INDEX_SCHEMA,
          manifest_schema=MANIFEST_SCHEMA, counts_fn=expected_counts,
          need_builder=None, validate_rows=validate_manifest_rows,
          adjustment_experiment_version="book_v9_4", need_rows_complete=False):
    profile = json.loads((args.v4_dir / "profile.json").read_text())
    validate_upload_metadata(args.dataset_dir, profile)
    identity, provenance_hashes = load_captioner_provenance(args.incremental_state_dir, profile=profile)
    adjustment_index = json.loads((args.adjustment_dir / "adjustment_end_training_index.json").read_text())
    if (
        adjustment_index.get("captioner_identity") != identity
        or adjustment_index.get("experiment_version") != adjustment_experiment_version
        or adjustment_index.get("model_dependency") != "none_data_only"
    ):
        raise ValueError("Rebuild the model-independent V9.4 adjustment subset with the current captioner")
    index, summary, manifests = base.build(
        args,
        expected_counts=counts_fn(args.v4_dir, expand_plan=False),
        data_profile=data_profile,
        index_schema=index_schema,
        manifest_schema=manifest_schema,
        validate_stage_a_model=False,
        need_builder=need_builder,
        need_rows_complete=need_rows_complete,
    )
    # Expand only the derived plan stream, never the V4 source used by Stage A.
    sources = {}
    for split in ("train", "val", "test"):
        by_frame = {
            (row["episode_id"], row["attempt_id"], row["frame_index"]): row["global_index"]
            for row in manifests["plan"]
            if row["split"] == split
        }
        for row in load_jsonl(args.v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl"):
            obs = row["current_observation"]
            key = (obs["episode_id"], obs["attempt_id"], row["frame_index"])
            identity_key = (split, by_frame[key])
            if identity_key in sources:
                raise ValueError("V9.4 source contains duplicate plan observations")
            sources[identity_key] = row
    manifests["plan"] = expand_plan_rows(manifests["plan"], manifests["failure"], sources=sources)
    token_summary = validate_dataset_plan_tokens(
        manifests["plan"], dataset_dir=args.dataset_dir, norm_stats_dir=args.v4_dir / "norm_stats"
    )
    for split in ("train", "val", "test"):
        chosen = [
            i
            for i, row in enumerate(manifests["plan"])
            if row["split"] == split and (split == "train" or row["frame_offset"] == 14)
        ]
        index["splits"][split]["plan"] = {
            "manifest_row_indices": chosen,
            "global_indices": [manifests["plan"][i]["global_index"] for i in chosen],
            "sample_count": len(chosen),
        }
        if len(chosen) != counts_fn(args.v4_dir)[split]["plan"]:
            raise ValueError("V9.4 expanded plan stream count mismatch")
        summary["selected_counts"][split]["plan"] = len(chosen)
    index["manifest_content_hashes"]["plan"] = sha256_json(manifests["plan"])
    summary["counts"]["plan"] = {
        split: sum(row["split"] == split for row in manifests["plan"]) for split in ("train", "val", "test")
    }
    summary["plan_memory_lengths"] = dict(Counter(str(row["memory_length"]) for row in manifests["plan"]))
    index.update(
        {
            "v4_dir": str(args.v4_dir.resolve()),
            "incremental_state_dir": str(args.incremental_state_dir.resolve()),
            "captioner_identity": identity,
            "training_target_coverage": target_coverage(manifests),
            "count_policy": "derived_from_source_profile_and_manifests_not_fixed_dataset_size",
            "model_dependency": "none_data_only",
            "plan_memory_policy": MEMORY_POLICY,
            "plan_token_validation": token_summary,
        }
    )
    index["source_hashes"].update(provenance_hashes)
    validate_rows(index, manifests)
    summary.update(
        {
            "schema_version": "tactile_vla_book_v9_4_summary_v1",
            "data_profile": data_profile,
            "model_dependency": "none_data_only",
            "captioner_identity": identity,
            "training_target_coverage": index["training_target_coverage"],
            "plan_memory_policy": MEMORY_POLICY,
            "plan_token_validation": token_summary,
        }
    )
    return index, summary, manifests


def main(argv=None):
    args = parse_args(argv)
    filenames = {
        "adjustment": "adjustment_end_manifest.jsonl",
        "need": "need_recovery_manifest.jsonl",
        "failure": "failure_reason_manifest.jsonl",
        "plan": "recovery_plan_manifest.jsonl",
        "index": "book_v9_4_multitask_training_index.json",
        "summary": "summary.json",
        "hashes": "artifact_hashes.json",
    }
    paths = {key: args.output_dir / value for key, value in filenames.items()}
    if not args.dry_run and not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("V9.4 multitask output exists; use a new directory or explicit --overwrite")
    index, summary, manifests = build(args)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    for task, rows in manifests.items():
        _write_jsonl(paths[task], rows)
        index[f"{task}_manifest_file"] = str(paths[task].resolve())
        index[f"{task}_manifest_sha256"] = sha256_file(paths[task])
    index["training_data_hash"] = sha256_json(index)
    validate_index(index)
    summary["training_data_hash"] = index["training_data_hash"]
    _write_json(paths["index"], index)
    _write_json(paths["summary"], summary)
    _write_json(
        paths["hashes"],
        {
            "artifacts": {
                key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for key, path in paths.items()
                if key != "hashes"
            }
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
