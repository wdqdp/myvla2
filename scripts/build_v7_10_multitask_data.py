#!/usr/bin/env python3
"""Build V7.10 0s/1s/2s qpos histories from frozen V7.7.2 samples."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts.build_v7_7_multitask_data import _write_json, _write_jsonl
from tactile_vla.vla import v7_10_multitask_data as data
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_10_phase_prompt import WINDOW_FRAMES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=data.DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=data.DEFAULT_OUTPUT,
                        help="Output root; each window uses its own 0s/1s/2s subdirectory.")
    parser.add_argument("--history-window", choices=(*WINDOW_FRAMES, "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Audit all inputs and token lengths without writing files.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.source_index, args.output_dir = args.source_index.resolve(), args.output_dir.resolve()
    windows = list(WINDOW_FRAMES) if args.history_window == "all" else [args.history_window]
    for window in windows:
        output = args.output_dir / window
        if not args.dry_run and output.exists():
            raise FileExistsError(f"V7.10 output exists: {output}; choose a new root, no implicit overwrite")
    original, original_rows = data.load_source(args.source_index)
    positions = data.load_timelines(original)
    reports = {}
    for window in windows:
        print(f"Building/auditing V7.10 window={window}", file=sys.stderr, flush=True)
        index, rows = data.derive(original, original_rows, args.source_index, window, positions=positions)
        report = data.summary(index, rows)
        report["token_validation"] = data.audit_tokens(index, rows)
        if not args.dry_run:
            output = args.output_dir / window
            for task, name in data.MANIFEST_NAMES.items():
                path = output / name
                if task == "plan":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original["plan_manifest_file"], path)
                else:
                    _write_jsonl(path, rows[task])
                index[f"{task}_manifest_file"] = str(path)
                index[f"{task}_manifest_sha256"] = sha256_file(path)
            index["training_data_hash"] = sha256_json(index)
            data.validate_index(index, positions=positions)
            index_file = output / "v7_10_multitask_training_index.json"
            _write_json(index_file, index)
            report.update(training_data_hash=index["training_data_hash"], index_file=str(index_file))
            _write_json(output / "summary.json", report)
            _write_json(output / "artifact_hashes.json", {
                path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()
            })
        reports[window] = report
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
