#!/usr/bin/env python3
"""Build model-independent V9.4.3 multitask data; reuse V9.4.2 action/adjustment artifacts."""

# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import build_book_v9_4_multitask_data as previous
from scripts.build_v7_7_multitask_data import _write_json, _write_jsonl
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_4_3_multitask_data import (
    DATA_PROFILE,
    DEFAULT_ACTION_INDEX,
    DEFAULT_ADJUSTMENT_DIR,
    DEFAULT_MULTITASK_DIR,
    INDEX_SCHEMA,
    MANIFEST_SCHEMA,
    VERSION_TAG,
    expected_counts,
    validate_index,
    validate_manifest_rows,
)
from tactile_vla.vla.book_v9_4_3_need_data import LABEL_POLICY, SAMPLING_POLICY, build_need_rows
from tactile_vla.vla.book_v9_4_multitask_data import validate_manifest_rows as validate_common_rows


def parse_args(argv=None):
    return previous.parse_args(
        argv,
        default_action_index=DEFAULT_ACTION_INDEX,
        default_adjustment_dir=DEFAULT_ADJUSTMENT_DIR,
        default_output_dir=DEFAULT_MULTITASK_DIR,
    )


def build(args):
    index, summary, manifests = previous.build(
        args,
        data_profile=DATA_PROFILE,
        index_schema=INDEX_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
        counts_fn=expected_counts,
        need_builder=build_need_rows,
        validate_rows=lambda index, rows: validate_common_rows(
            index,
            rows,
            data_profile=DATA_PROFILE,
            manifest_schema=MANIFEST_SCHEMA,
        ),
    )
    index.update(
        {"experiment_version": VERSION_TAG, "need_label_policy": LABEL_POLICY, "need_sampling_policy": SAMPLING_POLICY}
    )
    audit = validate_manifest_rows(index, manifests)
    for split, counts in audit["splits"].items():
        observed = summary["need"][split]
        if (
            observed["ignored_count"] != counts["ignored_count"]
            or observed["boundary_selected_count"] != counts["reserved_boundary_count"]
        ):
            raise ValueError("V9.4.3 build summary and actual need frames disagree")
    summary.update(
        {
            "schema_version": "tactile_vla_book_v9_4_3_summary_v1",
            "experiment_version": VERSION_TAG,
            "need_label_policy": LABEL_POLICY,
            "need_sampling_policy": SAMPLING_POLICY,
            "need_boundary_audit": audit["splits"],
        }
    )
    return index, summary, manifests, audit


def main(argv=None):
    args = parse_args(argv)
    names = {
        "adjustment": "adjustment_end_manifest.jsonl",
        "need": "need_recovery_manifest.jsonl",
        "failure": "failure_reason_manifest.jsonl",
        "plan": "recovery_plan_manifest.jsonl",
        "index": "book_v9_4_3_multitask_training_index.json",
        "summary": "summary.json",
        "audit": "need_boundary_audit.json",
        "hashes": "artifact_hashes.json",
    }
    paths = {key: args.output_dir / name for key, name in names.items()}
    if not args.dry_run and not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("V9.4.3 output exists; use a new directory or explicit --overwrite")
    index, summary, manifests, audit = build(args)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    for task, rows in manifests.items():
        _write_jsonl(paths[task], rows)
        index[f"{task}_manifest_file"] = str(paths[task].resolve())
        index[f"{task}_manifest_sha256"] = sha256_file(paths[task])
    _write_json(paths["audit"], audit)
    index["need_boundary_audit_file"] = str(paths["audit"].resolve())
    index["need_boundary_audit_sha256"] = sha256_file(paths["audit"])
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
