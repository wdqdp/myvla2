#!/usr/bin/env python3
"""Build Book V9.5 with real failure/plan observations from [C,C+14]."""

# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts import build_book_v9_4_multitask_data as previous
from scripts.build_book_v9_3_multitask_data import _base_row, _phase_fields, _timeline
from scripts.build_v7_7_multitask_data import _write_json, _write_jsonl
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_5_multitask_data import (
    DATA_PROFILE,
    DEFAULT_ACTION_INDEX,
    DEFAULT_ADJUSTMENT_DIR,
    DEFAULT_MULTITASK_DIR,
    INDEX_SCHEMA,
    LABEL_POLICY,
    MANIFEST_SCHEMA,
    REASONING_WINDOW_POLICY,
    SAMPLING_POLICY,
    VERSION_TAG,
    build_reasoning_rows,
    expected_counts,
    source_plan_anchors,
    validate_index,
    validate_manifest_rows,
)
from tactile_vla.vla.book_v9_5_need_data import build_need_rows
from tactile_vla.vla.book_v9_4_multitask_data import validate_manifest_rows as validate_common_rows
from tactile_vla.vla.book_v9_5_stage_a_data import reject_archive, validate_training_index as validate_action_index
from tactile_vla.vla.book_v9_4_memory import validate_dataset_plan_tokens
from tactile_vla.vla.v4_data import SPLITS, scan_v4_lerobot_frames
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles, scan_selected_qpos
from tactile_vla.vla.v7_7_phase_prompt import sample_episode_history


def parse_args(argv=None):
    args = previous.parse_args(
        argv,
        default_action_index=DEFAULT_ACTION_INDEX,
        default_adjustment_dir=DEFAULT_ADJUSTMENT_DIR,
        default_output_dir=DEFAULT_MULTITASK_DIR,
    )
    for name in ("dataset_dir", "v4_dir", "action_index", "adjustment_dir", "incremental_state_dir", "output_dir"):
        reject_archive(getattr(args, name))
    return args


def build(args):
    # The existing builder validates/reuses action, adjustment, need and real target anchors.
    # Its derived F-window reasoning rows are replaced below, never written to disk.
    expected_counts(args.v4_dir)  # Fail before expensive work if C+14 exceeds an attempt.
    action = json.loads(args.action_index.read_text())
    scope = action["source_scope"]
    validate_action_index(
        action, index_path=args.action_index, dataset_dir=args.dataset_dir, book_root=Path(scope["book_root"])
    )
    index, summary, manifests = previous.build(
        args,
        data_profile=DATA_PROFILE,
        index_schema=INDEX_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
        counts_fn=expected_counts,
        need_builder=build_need_rows,
        need_rows_complete=True,
        adjustment_experiment_version=VERSION_TAG,
        validate_rows=lambda i, rows: validate_common_rows(
            i,
            rows,
            data_profile=DATA_PROFILE,
            manifest_schema=MANIFEST_SCHEMA,
            need_variant_key="need_variant",
        ),
    )
    profile = json.loads((args.v4_dir / "profile.json").read_text())
    frames = scan_v4_lerobot_frames(args.dataset_dir)
    qpos = scan_selected_qpos(dataset_dir=args.dataset_dir, selected_episode_ids={f.episode_id for f in frames})
    _, positions = _timeline(frames, qpos)
    stats = load_state_quantiles(args.v4_dir / "norm_stats/norm_stats.json")

    def phase_fields(frame):
        timeline, position = positions[frame.global_index]
        points, length = sample_episode_history(timeline, position)
        return _phase_fields(frame, qpos_points=points, effective_length=length, stats=stats)

    def base_row(frame, split, source):
        return _base_row(frame, split, source, data_profile=DATA_PROFILE, manifest_schema=MANIFEST_SCHEMA)

    need_summaries = summary["need"]
    manifests["failure"], manifests["plan"] = build_reasoning_rows(
        profile=profile,
        frame_by_key={f.key: f for f in frames},
        anchors=source_plan_anchors(args.v4_dir),
        phase_fields=phase_fields,
        base_row=base_row,
    )
    for rows in manifests.values():
        for row in rows:
            row.update(schema_version=MANIFEST_SCHEMA, data_profile=DATA_PROFILE)
    for task in ("need", "failure", "plan"):
        rows = manifests[task]
        for split in SPLITS:
            chosen = [
                i
                for i, r in enumerate(rows)
                if r["split"] == split and (task == "need" or split == "train" or r["frame_offset"] == 14)
            ]
            index["splits"][split][task] = {
                "manifest_row_indices": chosen,
                "global_indices": [rows[i]["global_index"] for i in chosen],
                "sample_count": len(chosen),
            }
    token_summary = validate_dataset_plan_tokens(
        manifests["plan"],
        dataset_dir=args.dataset_dir,
        norm_stats_dir=args.v4_dir / "norm_stats",
    )
    index.update(
        schema_version=INDEX_SCHEMA,
        data_profile=DATA_PROFILE,
        experiment_version=VERSION_TAG,
        need_label_policy=LABEL_POLICY,
        need_sampling_policy=SAMPLING_POLICY,
        reasoning_window_policy=REASONING_WINDOW_POLICY,
        plan_token_validation=token_summary,
        manifest_content_hashes={task: sha256_json(rows) for task, rows in manifests.items()},
        source_scope=scope,
    )
    need_audit, reasoning_audit = validate_manifest_rows(index, manifests)
    summary.update(
        schema_version="tactile_vla_book_v9_5_summary_v1",
        data_profile=DATA_PROFILE,
        experiment_version=VERSION_TAG,
        need_label_policy=LABEL_POLICY,
        reasoning_window_policy=REASONING_WINDOW_POLICY,
        plan_token_validation=token_summary,
        reasoning_boundary_audit=reasoning_audit,
        need_boundary_audit=need_audit,
        need=need_summaries,
        need_sampling_policy=SAMPLING_POLICY,
        counts={
            task: {split: sum(r["split"] == split for r in rows) for split in SPLITS}
            for task, rows in manifests.items()
        },
        selected_counts={
            split: {task: index["splits"][split][task]["sample_count"] for task in manifests} for split in SPLITS
        },
        splits={
            split: {
                task: index["splits"][split][task]["sample_count"] for task in ("need", "adjustment", "failure", "plan")
            }
            for split in SPLITS
        },
    )
    return index, summary, manifests, need_audit, reasoning_audit


def main(argv=None):
    args = parse_args(argv)
    names = {
        "adjustment": "adjustment_end_manifest.jsonl",
        "need": "need_recovery_manifest.jsonl",
        "failure": "failure_reason_manifest.jsonl",
        "plan": "recovery_plan_manifest.jsonl",
        "index": "book_v9_5_multitask_training_index.json",
        "summary": "summary.json",
        "need_boundary_audit": "need_boundary_audit.json",
        "reasoning_boundary_audit": "reasoning_boundary_audit.json",
        "hashes": "artifact_hashes.json",
    }
    paths = {key: args.output_dir / name for key, name in names.items()}
    if not args.dry_run and not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("V9.5 output exists; use a new directory or explicit --overwrite")
    index, summary, manifests, need_audit, reasoning_audit = build(args)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    for task, rows in manifests.items():
        _write_jsonl(paths[task], rows)
        index[f"{task}_manifest_file"] = str(paths[task].resolve())
        index[f"{task}_manifest_sha256"] = sha256_file(paths[task])
    for name, audit in (("need_boundary_audit", need_audit), ("reasoning_boundary_audit", reasoning_audit)):
        _write_json(paths[name], audit)
        index[f"{name}_file"] = str(paths[name].resolve())
        index[f"{name}_sha256"] = sha256_file(paths[name])
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
