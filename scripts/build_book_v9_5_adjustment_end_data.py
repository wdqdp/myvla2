#!/usr/bin/env python3
"""Build the V9.5 adjustment subset using unchanged V9.2 label/sampling rules."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_5_multitask_data import ROOT, DEFAULT_ACTION_INDEX, DEFAULT_ADJUSTMENT_DIR
from tactile_vla.vla.book_v9_4_multitask_data import (
    adjustment_counts,
    load_captioner_provenance,
    validate_upload_metadata,
)
from scripts.prepare_book_stage_a_index import write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "lerobot_data/tactile_vla_rotation_v4")
    parser.add_argument("--v4-dir", type=Path, default=ROOT / "outputs/rotation_v4")
    parser.add_argument("--action-index-file", type=Path, default=DEFAULT_ACTION_INDEX)
    parser.add_argument("--norm-stats-file", type=Path, default=ROOT / "outputs/rotation_v4/norm_stats/norm_stats.json")
    parser.add_argument("--incremental-state-dir", type=Path, default=ROOT / "outputs/incremental_state")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ADJUSTMENT_DIR)
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build(args):
    from tactile_vla.vla.book_v9_2_adjustment_end_data import build_artifacts
    from tactile_vla.vla.v7_5_phase_change import PHASE_CHANGE_MAX_TOKEN_LEN
    from tactile_vla.vla.book_v9_5_stage_a_data import reject_archive, validate_training_index

    for name in (
        "dataset_dir",
        "v4_dir",
        "action_index_file",
        "norm_stats_file",
        "incremental_state_dir",
        "output_dir",
    ):
        reject_archive(getattr(args, name))
    action = json.loads(args.action_index_file.read_text())
    validate_training_index(
        action,
        index_path=args.action_index_file,
        dataset_dir=args.dataset_dir,
        book_root=Path(action["source_scope"]["book_root"]),
    )

    profile = json.loads((args.v4_dir / "profile.json").read_text())
    validate_upload_metadata(args.dataset_dir, profile)
    identity, hashes = load_captioner_provenance(args.incremental_state_dir, profile=profile)
    from openpi.models.tokenizer import PaligemmaTokenizer

    manifest, index, summary = build_artifacts(
        dataset_dir=args.dataset_dir,
        v4_index_file=args.v4_dir / "v4_training_index.json",
        action_index_file=args.action_index_file,
        norm_stats_file=args.norm_stats_file,
        stage_a_checkpoint=None,
        tokenizer=PaligemmaTokenizer(max_len=PHASE_CHANGE_MAX_TOKEN_LEN),
        seed=args.seed,
        expected_counts=adjustment_counts(profile),
    )
    # Keep the V9.2 subset schema so the mature selector remains reusable. The final
    # phase prompts/manifests/index have independent V9.5 identities.
    index["experiment_version"] = "book_v9_5"
    index["model_dependency"] = "none_data_only"
    index["captioner_identity"] = identity
    index["source_scope"] = action["source_scope"]
    index["source_files"].update(
        {
            f"caption_provenance_{number}": {"path": path, "sha256": digest}
            for number, (path, digest) in enumerate(hashes.items())
        }
    )
    summary["experiment_version"] = "book_v9_5"
    summary["model_dependency"] = "none_data_only"
    summary["captioner_identity"] = identity
    return manifest, index, summary


def main(argv=None):
    args = parse_args(argv)
    paths = {
        name: args.output_dir / filename
        for name, filename in {
            "manifest": "adjustment_end_manifest.jsonl",
            "index": "adjustment_end_training_index.json",
            "summary": "adjustment_end_summary.json",
            "hashes": "artifact_hashes.json",
        }.items()
    }
    if not args.dry_run and not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("V9.5 adjustment output exists; use a new directory or explicit --overwrite")
    manifest, index, summary = build(args)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    from scripts.build_v7_7_multitask_data import _write_jsonl
    from tactile_vla.vla.book_v9_2_adjustment_end_data import load_indexed_manifest_rows

    _write_jsonl(paths["manifest"], manifest)
    index["manifest_file"] = str(paths["manifest"].resolve())
    index["manifest_identity"]["file_sha256"] = sha256_file(paths["manifest"])
    index["training_data_hash"] = sha256_json(index)
    load_indexed_manifest_rows(index=index, manifest_path=paths["manifest"])
    summary["training_data_hash"] = index["training_data_hash"]
    write_json(paths["index"], index)
    write_json(paths["summary"], summary)
    write_json(
        paths["hashes"],
        {
            "artifacts": {
                key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for key, path in paths.items()
                if key != "hashes"
            }
        },
    )
    print(f"wrote V9.5 adjustment data to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
