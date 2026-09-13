#!/usr/bin/env python3
"""Build V7.7 adjustment/need/failure/plan manifests and unified index."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/src"))

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt
from tactile_vla.vla.v4_data import load_jsonl, scan_v4_lerobot_frames
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles, scan_selected_qpos
from tactile_vla.vla.v7_7_multitask_data import (
    DATA_PROFILE, INDEX_SCHEMA, MANIFEST_SCHEMA, NEGATIVE_SOURCES, TASK_CYCLE, select_need_rows,
)
from tactile_vla.vla.v7_7_phase_prompt import (
    PROMPT_PROFILE, TimelinePoint, build_phase_prompt, discretize_history,
    helper_identity, history_sources, sample_episode_history,
)
from tactile_vla.vla.v7_6_adjustment_end_data import validate_adjustment_end_artifacts

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "lerobot_data/tactile_vla_rotation_v4")
    parser.add_argument("--v4-dir", type=Path, default=ROOT / "outputs/rotation_v4")
    parser.add_argument("--action-index", type=Path, default=ROOT / "outputs/rotation_v7_4_adjustment/v7_4_prompt_training_index.json")
    parser.add_argument("--adjustment-dir", type=Path, default=ROOT / "outputs/rotation_v7_6_adjustment_end_counterfactual_h100")
    parser.add_argument("--boundary-file", type=Path, default=ROOT / "outputs/rotation_v7_2_adjustment/v7_2_boundary_filter.json")
    parser.add_argument("--stage-a-checkpoint", type=Path, default=ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_no_history/15000")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rotation_v7_7_multitask")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _load(path: Path):
    return json.loads(path.read_text())


def _timeline(frames, qpos):
    grouped = defaultdict(list)
    for frame in frames:
        grouped[frame.episode_id].append(TimelinePoint(
            frame.episode_id, frame.attempt_id, frame.frame_index, frame.global_index,
            frame.ros_timestamp, qpos[frame.global_index],
        ))
    positions = {}
    for episode, values in grouped.items():
        values.sort(key=lambda point: (point.timestamp, point.global_index))
        for position, point in enumerate(values):
            positions[point.global_index] = (values, position)
    return grouped, positions


def _phase_fields(frame, *, qpos_points, effective_length, stats, recovery_plan=None):
    discrete = discretize_history(qpos_points, stats)
    plan = recovery_plan if recovery_plan is not None else (
        "none" if frame.attempt_id == 1 else frame.input_recovery_plan
    )
    return {
        "prompt": build_phase_prompt(
            instruction=frame.instruction, tactile_caption=frame.tactile_caption,
            recovery_plan=plan, qpos_h100_11_discrete=discrete,
        ),
        "qpos_h100_11_discrete": discrete.tolist(),
        "history_sources": history_sources(qpos_points),
        "effective_episode_history_length": effective_length,
    }


def _base_row(frame, split, source):
    return {
        "schema_version": MANIFEST_SCHEMA, "data_profile": DATA_PROFILE,
        "split": split, "global_index": frame.global_index,
        "episode_id": frame.episode_id, "attempt_id": frame.attempt_id,
        "frame_index": frame.frame_index, "timestamp": frame.ros_timestamp,
        "source": source,
    }


def main() -> int:
    args = parse_args()
    v4_index_path = args.v4_dir / "v4_training_index.json"
    profile_path = args.v4_dir / "profile.json"
    norm_path = args.v4_dir / "norm_stats/norm_stats.json"
    adjustment_index_path = args.adjustment_dir / "adjustment_end_training_index.json"
    adjustment_manifest_path = args.adjustment_dir / "adjustment_end_manifest.jsonl"
    caption_summary_path = args.v4_dir / "caption_annotation_summary.json"
    stage_a_config_path = args.stage_a_checkpoint.parent / "config.json"
    required = (v4_index_path, profile_path, norm_path, args.action_index,
                adjustment_index_path, adjustment_manifest_path, args.boundary_file,
                caption_summary_path, stage_a_config_path,
                args.stage_a_checkpoint / "params/_METADATA")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    v4_index, profile, action_index, adjustment_index = map(_load, (
        v4_index_path, profile_path, args.action_index, adjustment_index_path
    ))
    hashes = {v4_index["selection_hash"], action_index["selection_hash"], adjustment_index["selection_hash"]}
    if len(hashes) != 1:
        raise ValueError(f"V4/V7.4/V7.6 selection hashes differ: {hashes}")
    if adjustment_index.get("action_training_data_hash") != action_index.get("training_data_hash"):
        raise ValueError("V7.6 adjustment data does not reference this V7.4 action index")
    source_adjustment = load_jsonl(adjustment_manifest_path)
    validate_adjustment_end_artifacts(index=adjustment_index, manifest=source_adjustment)
    frames = scan_v4_lerobot_frames(args.dataset_dir)
    frame_by_global = {frame.global_index: frame for frame in frames}
    frame_by_key = {frame.key: frame for frame in frames}
    qpos = scan_selected_qpos(dataset_dir=args.dataset_dir,
                              selected_episode_ids={frame.episode_id for frame in frames})
    _, positions = _timeline(frames, qpos)
    stats = load_state_quantiles(norm_path)

    def history_for(global_index):
        timeline, position = positions[int(global_index)]
        return sample_episode_history(timeline, position)

    # Adjustment identities/labels/pairs are copied exactly from V7.6. Its
    # physical H100 samples already satisfy the idle perturbation policy; A is
    # >=132 in this dataset, so no row needs an attempt-crossing correction.
    adjustment = []
    for old in source_adjustment:
        frame = frame_by_global[int(old["current_global_index"])]
        points = [positions[int(index)][0][positions[int(index)][1]] for index in old["history_global_indices"]]
        row = _base_row(frame, old["split"], "v7_6_adjustment_replay") | {
            key: value for key, value in old.items() if key not in {
                "schema_version", "data_profile", "prompt_profile", "experiment_kind", "prompt"
            }
        }
        row.update(_phase_fields(
            frame, qpos_points=points, effective_length=min(100, positions[frame.global_index][1] + 1),
            stats=stats,
            recovery_plan=("recovery_plan=move horizontally " + old["source_direction"] + " " +
                           old["prompt_magnitude"] + ", move vertically none moderately."),
        ))
        adjustment.append(row)

    profile_attempts = {(int(row["episode_id"]), int(row["attempt_id"])): row for row in profile["attempts"]}
    boundary = {(int(row["episode_id"]), int(row["attempt_id"])): row for row in _load(args.boundary_file)["attempts"]}
    split_lists = _load(args.v4_dir / "splits.json")["original_episode_ids"]
    all_split_episodes = [int(episode) for split in ("train", "val", "test") for episode in split_lists[split]]
    if len(all_split_episodes) != len(set(all_split_episodes)):
        raise ValueError("an original episode occurs in more than one split")
    split_by_episode = {int(episode): split for split in ("train", "val", "test")
                        for episode in split_lists[split]}
    need_by_split, need_summary = {}, {}
    for split in ("train", "val", "test"):
        source_need = load_jsonl(args.v4_dir / f"need/{split}.jsonl")
        positives = [dict(row) for row in source_need if bool(row["need_recovery"])]
        negative = {name: [] for name in NEGATIVE_SOURCES}
        for frame in frames:
            if split_by_episode[frame.episode_id] != split:
                continue
            meta = profile_attempts[frame.attempt_key]
            if meta["result"] == "failure" and frame.frame_index < int(meta["shift_frame_index"]):
                source = "pre_failure_hard_negative"
            elif meta["task"] == "one_success":
                source = "one_success_easy_negative"
            elif meta["result"] == "success" and frame.attempt_id > 1:
                close = int(boundary[frame.attempt_key]["events"]["gripper_close_start"]["frame_index"])
                if frame.frame_index < close:
                    continue
                source = "successful_recovery_easy_negative"
            else:
                continue
            negative[source].append(_base_row(frame, split, source) | {"need_recovery": False})
        raw_rows, summary = select_need_rows(positives, negative, seed=args.seed)
        rows = []
        for raw in raw_rows:
            frame = frame_by_global[int(raw["global_index"])]
            points, length = history_for(frame.global_index)
            rows.append(_base_row(frame, split, raw.get("source", "failure_active")) |
                        {"need_recovery": bool(raw["need_recovery"])} |
                        _phase_fields(frame, qpos_points=points, effective_length=length, stats=stats))
        need_by_split[split], need_summary[split] = rows, summary
    need = [row for split in ("train", "val", "test") for row in need_by_split[split]]

    failure, plan = [], []
    for split in ("train", "val", "test"):
        for old in load_jsonl(args.v4_dir / f"reasoning_manifests/failure_reason/{split}.jsonl"):
            obs = old["current_observation"]
            frame = frame_by_key[(int(obs["episode_id"]), int(obs["attempt_id"]), int(old["frame_index"]))]
            points, length = history_for(frame.global_index)
            failure.append(_base_row(frame, split, "failure_active") | {
                "frame_offset": int(old["frame_offset"]),
                "target_failure_reason": old["target_failure_reason"],
            } | _phase_fields(frame, qpos_points=points, effective_length=length, stats=stats))
        for old in load_jsonl(args.v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl"):
            obs = old["current_observation"]
            frame = frame_by_key[(int(obs["episode_id"]), int(obs["attempt_id"]), int(old["frame_index"]))]
            plan.append(_base_row(frame, split, "v4_rotation_distance_v1") | {
                "frame_offset": int(old["frame_offset"]), "memory_length": int(old["memory_length"]),
                "failure_recovery_memory": old["failure_recovery_memory"],
                "target_recovery_plan": old["target_recovery_plan"],
                "prompt": build_recovery_prompt(
                    instruction=frame.instruction, failed_tactile_caption=frame.tactile_caption,
                    failure_recovery_memory=old["failure_recovery_memory"],
                    prompt_profile=MINIMAL_PROMPT_PROFILE,
                ),
            })

    manifests = {"adjustment": adjustment, "need": need, "failure": failure, "plan": plan}
    split_payload = {}
    for split in ("train", "val", "test"):
        split_payload[split] = {"action": {
            "indices": action_index["splits"][split]["execution_indices"]
        }}
        for task, rows in manifests.items():
            selected = [i for i, row in enumerate(rows) if row["split"] == split]
            # Main V4 comparison for validation/test uses F+14. Preserve full
            # manifests on disk but keep the training/eval stream comparable.
            if task in {"failure", "plan"} and split != "train":
                selected = [i for i in selected if int(rows[i]["frame_offset"]) == 14]
            split_payload[split][task] = {
                "manifest_row_indices": selected,
                "global_indices": [int(rows[i]["global_index"]) for i in selected],
                "sample_count": len(selected),
            }
    index = {
        "schema_version": INDEX_SCHEMA, "data_profile": DATA_PROFILE,
        "prompt_profile": PROMPT_PROFILE, "task_cycle": list(TASK_CYCLE), "seed": args.seed,
        "selection_hash": v4_index["selection_hash"], "dataset_dir": str(args.dataset_dir.resolve()),
        "stage_a_checkpoint": str(args.stage_a_checkpoint.resolve()),
        "action_index_file": str(args.action_index.resolve()),
        "action_training_data_hash": action_index["training_data_hash"],
        "history_policy": helper_identity(), "splits": split_payload,
        "manifest_content_hashes": {task: sha256_json(rows) for task, rows in manifests.items()},
        "source_hashes": {str(path.resolve()): sha256_file(path) for path in required},
        "v4_source_files": v4_index["source_files"],
    }
    index["training_data_hash"] = sha256_json(index)
    summary = {
        "schema_version": "tactile_vla_v7_7_summary_v1", "need": need_summary,
        "counts": {task: {split: sum(row["split"] == split for row in rows)
                           for split in ("train", "val", "test")} for task, rows in manifests.items()},
        "training_data_hash": index["training_data_hash"],
    }
    if args.dry_run:
        print(json.dumps(summary | {"hash_scope": "dry_run_before_output_file_identities"},
                         indent=2, ensure_ascii=False))
        return 0
    manifest_names = {
        "adjustment": "adjustment_end_manifest.jsonl",
        "need": "need_recovery_manifest.jsonl",
        "failure": "failure_reason_manifest.jsonl",
        "plan": "recovery_plan_manifest.jsonl",
    }
    outputs = {task: args.output_dir / manifest_names[task] for task in manifests}
    outputs |= {"index": args.output_dir / "v7_7_multitask_training_index.json",
                "summary": args.output_dir / "summary.json",
                "hashes": args.output_dir / "artifact_hashes.json"}
    if not args.overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("V7.7 output exists; pass --overwrite")
    for task, rows in manifests.items():
        _write_jsonl(outputs[task], rows)
    for task in manifests:
        index[f"{task}_manifest_file"] = str(outputs[task].resolve())
        index[f"{task}_manifest_sha256"] = sha256_file(outputs[task])
    index["training_data_hash"] = sha256_json({k: v for k, v in index.items() if k != "training_data_hash"})
    _write_json(outputs["index"], index)
    summary["training_data_hash"] = index["training_data_hash"]
    _write_json(outputs["summary"], summary)
    _write_json(outputs["hashes"], {"artifacts": {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in outputs.items() if name != "hashes"
    }})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
