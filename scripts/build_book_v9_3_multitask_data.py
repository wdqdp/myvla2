#!/usr/bin/env python3
"""Build book V9.3 factual adjustment, need, failure and plan manifests."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts.build_v7_7_multitask_data import _phase_fields, _timeline, _write_json, _write_jsonl
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_stage_a_data import validate_training_index as validate_action_index
from tactile_vla.vla.book_v9_2_adjustment_end_data import load_indexed_manifest_rows
from tactile_vla.vla.book_v9_3_multitask_data import (
    ADJUSTMENT_LABEL_POLICY, DATA_PROFILE, EXPECTED_COUNTS, INDEX_SCHEMA, MANIFEST_SCHEMA,
    NEED_RECOVERY_NEGATIVE_START, validate_index,
)
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt
from tactile_vla.vla.v4_data import load_jsonl, scan_v4_lerobot_frames
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles, scan_selected_qpos
from tactile_vla.vla.v7_7_multitask_data import NEGATIVE_SOURCES, TASK_CYCLE, select_need_rows
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE, helper_identity, sample_episode_history


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "lerobot_data/tactile_vla_rotation_v4")
    parser.add_argument("--v4-dir", type=Path, default=ROOT / "outputs/rotation_v4")
    parser.add_argument("--action-index", type=Path, default=ROOT / "outputs/book_stage_a_v1/book_stage_a_training_index.json")
    parser.add_argument("--adjustment-dir", type=Path, default=ROOT / "outputs/book_adjustment_end_v9_2")
    parser.add_argument("--stage-a-checkpoint", type=Path, default=ROOT / "outputs/stage_a_action/pi05_delta_tac_book_stage_a_v1_no_history/15000")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/book_v9_3_multitask")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load(path: Path):
    return json.loads(path.read_text())


def _base_row(frame, split: str, source: str, *, data_profile=DATA_PROFILE, manifest_schema=MANIFEST_SCHEMA):
    return {
        "schema_version": manifest_schema, "data_profile": data_profile,
        "split": split, "global_index": frame.global_index,
        "episode_id": frame.episode_id, "attempt_id": frame.attempt_id,
        "frame_index": frame.frame_index, "timestamp": frame.ros_timestamp,
        "source": source,
    }


def _validate_timeline_order(frames):
    last = {}
    for frame in sorted(frames, key=lambda item: item.global_index):
        previous = last.get(frame.episode_id)
        if previous is not None and (
            frame.ros_timestamp < previous.ros_timestamp
            or frame.attempt_id < previous.attempt_id
        ):
            raise ValueError(f"Non-monotonic book timeline at global index {frame.global_index}")
        last[frame.episode_id] = frame


def need_negative_source(frame, meta):
    """Only supervise need during execution; book attempt2 execution starts at R."""
    if meta["result"] == "failure" and frame.frame_index < int(meta["shift_frame_index"]):
        return "pre_failure_hard_negative"
    if meta["task"] == "one_success":
        return "one_success_easy_negative"
    if meta["result"] == "success" and frame.attempt_id == 2:
        r = meta["rexecution_frame_index"]
        if r is None:
            raise ValueError("Book recovery attempt2 lacks native R")
        if frame.frame_index >= int(r):
            return "successful_recovery_easy_negative"
    return None


def build(args, *, expected_counts=EXPECTED_COUNTS, data_profile=DATA_PROFILE,
          index_schema=INDEX_SCHEMA, manifest_schema=MANIFEST_SCHEMA,
          validate_stage_a_model=True, need_builder=None, need_rows_complete=False):
    # Defaults preserve V9.3 exactly; incremental versions derive counts from their sources.
    def base_row(frame, split, source):
        return _base_row(frame, split, source, data_profile=data_profile, manifest_schema=manifest_schema)

    if args.seed != 42:
        raise ValueError("Book V9.3 fixes the sampling seed to 42")
    v4_index_path = args.v4_dir / "v4_training_index.json"
    profile_path = args.v4_dir / "profile.json"
    splits_path = args.v4_dir / "splits.json"
    norm_path = args.v4_dir / "norm_stats/norm_stats.json"
    adjustment_index_path = args.adjustment_dir / "adjustment_end_training_index.json"
    adjustment_manifest_path = args.adjustment_dir / "adjustment_end_manifest.jsonl"
    required = [
        v4_index_path, profile_path, splits_path, norm_path, args.action_index,
        adjustment_index_path, adjustment_manifest_path,
    ]
    if validate_stage_a_model:
        stage_a_config_path = args.stage_a_checkpoint.parent / "config.json"
        required += [stage_a_config_path, args.stage_a_checkpoint / "params/_METADATA"]
    for split in ("train", "val", "test"):
        required += [
            args.v4_dir / f"need/{split}.jsonl",
            args.v4_dir / f"reasoning_manifests/failure_reason/{split}.jsonl",
            args.v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl",
        ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    v4_index, profile, action_index, adjustment_index = map(
        _load, (v4_index_path, profile_path, args.action_index, adjustment_index_path)
    )
    if len({v4_index["selection_hash"], action_index["selection_hash"], adjustment_index["selection_hash"]}) != 1:
        raise ValueError("Book V4/Stage A/V9.2 selections differ")
    if adjustment_index["action_training_data_hash"] != action_index["training_data_hash"]:
        raise ValueError("Book V9.2 adjustment index references another Stage A index")
    validate_action_index(action_index, index_path=args.action_index, dataset_dir=args.dataset_dir)
    selected_adjustment_rows = load_indexed_manifest_rows(
        index=adjustment_index, manifest_path=adjustment_manifest_path,
    )
    source_adjustment = load_jsonl(adjustment_manifest_path)
    if (
        len(source_adjustment) != adjustment_index["manifest_identity"]["count"]
        or sha256_json(source_adjustment) != adjustment_index["manifest_identity"]["content_sha256"]
    ):
        raise ValueError("Book V9.2 adjustment manifest content changed")
    if validate_stage_a_model:
        stage_a_config = _load(stage_a_config_path)
        if (
            args.stage_a_checkpoint.name != "15000"
            or stage_a_config.get("data_profile") != "book_stage_a_v1"
            or stage_a_config.get("use_state_history") is not False
            or stage_a_config.get("artifact_identity", {}).get("training_data_hash") != action_index["training_data_hash"]
        ):
            raise ValueError("Book V9.3 must initialize the matching no-history Stage A step 15000")
        if Path(adjustment_index["stage_a_checkpoint"]["path"]).resolve() != args.stage_a_checkpoint.resolve():
            raise ValueError("Book V9.2 adjustment data references another Stage A checkpoint")

    frames = scan_v4_lerobot_frames(args.dataset_dir)
    _validate_timeline_order(frames)
    frame_by_global = {frame.global_index: frame for frame in frames}
    frame_by_key = {frame.key: frame for frame in frames}
    split_lists = _load(splits_path)["original_episode_ids"]
    split_by_episode = {
        int(episode): split for split in ("train", "val", "test")
        for episode in split_lists[split]
    }
    if len(split_by_episode) != sum(len(split_lists[split]) for split in ("train", "val", "test")):
        raise ValueError("Book original episode occurs in multiple splits")
    if {frame.episode_id for frame in frames} != set(split_by_episode):
        raise ValueError("Book V4 frames and episode split identity differ")
    qpos = scan_selected_qpos(dataset_dir=args.dataset_dir, selected_episode_ids=set(split_by_episode))
    _, positions = _timeline(frames, qpos)
    stats = load_state_quantiles(norm_path)

    def phase_fields(frame):
        timeline, position = positions[frame.global_index]
        points, length = sample_episode_history(timeline, position)
        result = _phase_fields(frame, qpos_points=points, effective_length=length, stats=stats)
        if result["history_sources"][-1]["global_index"] != frame.global_index:
            raise AssertionError("Book phase H100 omits current frame")
        return result

    adjustment = []
    for source_index, old in enumerate(source_adjustment):
        frame = frame_by_global[int(old["current_global_index"])]
        if (
            frame.episode_id != int(old["episode_id"])
            or frame.attempt_id != 2
            or frame.frame_index != int(old["frame_index"])
            or split_by_episode[frame.episode_id] != old["split"]
            or bool(old["adjustment_end"]) != (frame.frame_index >= int(old["rexecution_frame"]) - 10)
        ):
            raise ValueError(f"Book V9.2 adjustment row {source_index} identity/label mismatch")
        adjustment.append(base_row(frame, old["split"], "book_v9_2_factual") | {
            "rexecution_frame": int(old["rexecution_frame"]),
            "adjustment_end": bool(old["adjustment_end"]),
            "v9_2_manifest_row_index": source_index,
            "v9_2_selected": source_index in selected_adjustment_rows,
        } | phase_fields(frame))

    profile_attempts = {
        (int(row["episode_id"]), int(row["attempt_id"])): row
        for row in profile["attempts"]
    }
    need_by_split, need_summary = {}, {}
    for split in ("train", "val", "test"):
        if need_builder is None:
            positives = [
                row for row in load_jsonl(args.v4_dir / f"need/{split}.jsonl")
                if bool(row["need_recovery"])
            ]
            negative = {name: [] for name in NEGATIVE_SOURCES}
            for frame in frames:
                if split_by_episode[frame.episode_id] != split:
                    continue
                meta = profile_attempts[frame.attempt_key]
                source = need_negative_source(frame, meta)
                if source is None:
                    continue
                negative[source].append(base_row(frame, split, source) | {"need_recovery": False})
            sampled, need_summary[split] = select_need_rows(positives, negative, seed=args.seed)
        else:
            sampled, need_summary[split] = need_builder(
                frames=frames, profile=profile, split=split, base_row=base_row, seed=args.seed,
                **({"phase_fields": phase_fields} if need_rows_complete else {}),
            )
        if need_rows_complete:
            need_by_split[split] = sampled
            continue
        rows = []
        for old in sampled:
            frame = frame_by_global[int(old["global_index"])]
            if split_by_episode[frame.episode_id] != split:
                raise ValueError("Book need row crosses split")
            rows.append(base_row(frame, split, old.get("source", "failure_active")) | {
                "need_recovery": bool(old["need_recovery"]),
            } | ({"need_boundary": old["need_boundary"]} if "need_boundary" in old else {}) | phase_fields(frame))
        need_by_split[split] = rows
    need = [row for split in ("train", "val", "test") for row in need_by_split[split]]

    failure, plan = [], []
    for split in ("train", "val", "test"):
        for old in load_jsonl(args.v4_dir / f"reasoning_manifests/failure_reason/{split}.jsonl"):
            observation = old["current_observation"]
            key = (int(observation["episode_id"]), int(observation["attempt_id"]), int(old["frame_index"]))
            frame = frame_by_key[key]
            if split_by_episode[frame.episode_id] != split:
                raise ValueError("Book failure row crosses split")
            failure.append(base_row(frame, split, "failure_active") | {
                "frame_offset": int(old["frame_offset"]),
                "target_failure_reason": old["target_failure_reason"],
            } | phase_fields(frame))
        for old in load_jsonl(args.v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl"):
            observation = old["current_observation"]
            key = (int(observation["episode_id"]), int(observation["attempt_id"]), int(old["frame_index"]))
            frame = frame_by_key[key]
            if split_by_episode[frame.episode_id] != split:
                raise ValueError("Book plan row crosses split")
            plan.append(base_row(frame, split, "book_v4_rotation_grasp_interleaved_v2") | {
                "frame_offset": int(old["frame_offset"]),
                "memory_length": int(old["memory_length"]),
                "failure_recovery_memory": old["failure_recovery_memory"],
                "target_recovery_plan": old["target_recovery_plan"],
                "prompt": build_recovery_prompt(
                    instruction=frame.instruction,
                    failed_tactile_caption=frame.tactile_caption,
                    failure_recovery_memory=old["failure_recovery_memory"],
                    prompt_profile=MINIMAL_PROMPT_PROFILE,
                ),
            })
    need_positive_prompts = {
        (row["split"], row["global_index"]): row["prompt"]
        for row in need if row["need_recovery"]
    }
    for row in failure:
        shared_prompt = need_positive_prompts.get((row["split"], row["global_index"]))
        if shared_prompt is None and need_builder is not None:
            # A newer need policy may ignore early failure frames. Keep the LM
            # supervision and check its canonical prefix without inventing a need label.
            shared_prompt = phase_fields(frame_by_global[row["global_index"]])["prompt"]
        if shared_prompt != row["prompt"]:
            raise ValueError("Book need/failure shared phase prompt differs")
    manifests = {"adjustment": adjustment, "need": need, "failure": failure, "plan": plan}
    split_payload = {}
    for split in ("train", "val", "test"):
        split_payload[split] = {"action": {
            "indices": action_index["splits"][split]["execution_indices"],
        }}
        for task, rows in manifests.items():
            chosen = [index for index, row in enumerate(rows) if row["split"] == split]
            if task == "adjustment":
                chosen = [index for index in chosen if rows[index]["v9_2_selected"]]
            if task in {"failure", "plan"} and split != "train":
                chosen = [index for index in chosen if int(rows[index]["frame_offset"]) == 14]
            split_payload[split][task] = {
                "manifest_row_indices": chosen,
                "global_indices": [int(rows[index]["global_index"]) for index in chosen],
                "sample_count": len(chosen),
            }
            if len(chosen) != expected_counts[split][task]:
                raise ValueError(f"Book multitask unexpected {split}/{task} sample count: {len(chosen)}")
    selected_adjustment_globals = {
        split: [source_adjustment[index]["current_global_index"]
                for index in adjustment_index["splits"][split]["manifest_row_indices"]]
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        if selected_adjustment_globals[split] != split_payload[split]["adjustment"]["global_indices"]:
            raise ValueError(f"Book V9.3 {split} adjustment selection differs from V9.2")
    index = {
        "schema_version": index_schema, "data_profile": data_profile,
        "prompt_profile": PROMPT_PROFILE, "task_cycle": list(TASK_CYCLE), "seed": args.seed,
        "selection_hash": v4_index["selection_hash"],
        "dataset_dir": str(args.dataset_dir.resolve()),
        **({"stage_a_checkpoint": str(args.stage_a_checkpoint.resolve())} if validate_stage_a_model else {}),
        "action_index_file": str(args.action_index.resolve()),
        "action_training_data_hash": action_index["training_data_hash"],
        "adjustment_source_index": str(adjustment_index_path.resolve()),
        "adjustment_source_training_data_hash": adjustment_index["training_data_hash"],
        "adjustment_label_policy": ADJUSTMENT_LABEL_POLICY,
        "adjustment_sampling_policy": "reuse_book_v9_2_selection_1_to_2",
        "need_successful_recovery_start": NEED_RECOVERY_NEGATIVE_START,
        "history_policy": helper_identity() | {"idle_perturbation": "none_raw_contiguous"},
        "splits": split_payload,
        "manifest_content_hashes": {task: sha256_json(rows) for task, rows in manifests.items()},
        "source_hashes": {str(path.resolve()): sha256_file(path) for path in required},
        "v4_source_files": v4_index["source_files"],
    }
    summary = {
        "schema_version": "tactile_vla_book_v9_3_summary_v1",
        "counts": {task: {split: sum(row["split"] == split for row in rows)
                           for split in ("train", "val", "test")}
                   for task, rows in manifests.items()},
        "selected_counts": {
            split: {task: split_payload[split][task]["sample_count"]
                    for task in ("adjustment", "need", "failure", "plan")}
            for split in ("train", "val", "test")
        },
        "adjustment_selected_labels": {
            split: dict(Counter(
                "true" if adjustment[index]["adjustment_end"] else "false"
                for index in split_payload[split]["adjustment"]["manifest_row_indices"]
            )) for split in ("train", "val", "test")
        },
        "need": need_summary,
        "failure_target_count": len({row["target_failure_reason"] for row in failure}),
        "plan_target_count": len({row["target_recovery_plan"] for row in plan}),
        "plan_memory_lengths": dict(Counter(str(row["memory_length"]) for row in plan)),
    }
    return index, summary, manifests


def main() -> int:
    args = parse_args()
    index, summary, manifests = build(args)
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    filenames = {
        "adjustment": "adjustment_end_manifest.jsonl", "need": "need_recovery_manifest.jsonl",
        "failure": "failure_reason_manifest.jsonl", "plan": "recovery_plan_manifest.jsonl",
    }
    outputs = {task: args.output_dir / filename for task, filename in filenames.items()}
    outputs |= {
        "index": args.output_dir / "book_v9_3_multitask_training_index.json",
        "summary": args.output_dir / "summary.json",
        "hashes": args.output_dir / "artifact_hashes.json",
    }
    if not args.overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Book V9.3 output exists; pass --overwrite to rebuild")
    for task, rows in manifests.items():
        _write_jsonl(outputs[task], rows)
        index[f"{task}_manifest_file"] = str(outputs[task].resolve())
        index[f"{task}_manifest_sha256"] = sha256_file(outputs[task])
    index["training_data_hash"] = sha256_json(index)
    validate_index(index)
    summary["training_data_hash"] = index["training_data_hash"]
    _write_json(outputs["index"], index)
    _write_json(outputs["summary"], summary)
    _write_json(outputs["hashes"], {"artifacts": {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in outputs.items() if name != "hashes"
    }})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
