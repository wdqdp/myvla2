#!/usr/bin/env python3
"""Build V7.7.2 manifests from the V7.4.2 new-environment artifacts."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from scripts.build_v7_7_multitask_data import _phase_fields, _timeline, _write_json, _write_jsonl
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt
from tactile_vla.vla.structured_text import recovery_plan_text
from tactile_vla.vla.v4_data import load_jsonl, scan_v4_lerobot_frames
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles, scan_selected_qpos
from tactile_vla.vla.v7_4_2_adjustment_data import validate_boundaries, validate_training_index
from tactile_vla.vla.v7_7_2_multitask_data import (
    DATA_PROFILE, INDEX_SCHEMA, MANIFEST_SCHEMA, factual_adjustment_end,
)
from tactile_vla.vla.v7_7_multitask_data import NEGATIVE_SOURCES, TASK_CYCLE, select_need_rows
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE, helper_identity, sample_episode_history
from tactile_vla.vla.v7_6_adjustment_end_data import select_train_slight_counterfactual_pairs


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "lerobot_data/tactile_vla_rotation_v4_1_new_env")
    parser.add_argument("--v4-dir", type=Path, default=ROOT / "outputs/rotation_v4_1_new_env")
    parser.add_argument("--action-index", type=Path, default=ROOT / "outputs/rotation_v7_4_2_adjustment/v7_4_2_prompt_training_index.json")
    parser.add_argument("--boundary-file", type=Path, default=ROOT / "outputs/rotation_v7_4_2_adjustment/adjustment_boundaries.json")
    parser.add_argument("--stage-a-checkpoint", type=Path, default=ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_2_no_history/10000")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rotation_v7_7_2_multitask")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adjustment-sampling-policy", choices=("retain_all", "ratio_1_to_2"),
                        default="ratio_1_to_2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load(path: Path):
    return json.loads(path.read_text())


def _base_row(frame, split: str, source: str):
    return {
        "schema_version": MANIFEST_SCHEMA, "data_profile": DATA_PROFILE,
        "split": split, "global_index": frame.global_index,
        "episode_id": frame.episode_id, "attempt_id": frame.attempt_id,
        "frame_index": frame.frame_index, "timestamp": frame.ros_timestamp,
        "source": source,
    }


def _validate_attempt_transitions(frames) -> None:
    by_episode = defaultdict(list)
    for frame in frames:
        by_episode[frame.episode_id].append(frame)
    for episode, values in by_episode.items():
        ordered = sorted(values, key=lambda frame: frame.global_index)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.attempt_id != current.attempt_id and (
                current.attempt_id <= previous.attempt_id
                or current.ros_timestamp < previous.ros_timestamp
            ):
                raise ValueError(f"non-monotonic attempt transition in episode {episode}")


def _adjustment_rows(*, frames, events, split_by_episode, history_for, stats):
    by_attempt = defaultdict(dict)
    for frame in frames:
        by_attempt[frame.attempt_key][frame.frame_index] = frame
    rows = []
    for key, (start, stop) in sorted(events.items()):
        episode, _ = key
        split = split_by_episode[episode]
        source_frames = by_attempt[key]
        for p in range(start, stop + 11):
            if p not in source_frames:
                raise ValueError(f"adjustment factual frame is missing: {key}, p={p}")
            frame = source_frames[p]
            points, length = history_for(frame.global_index)
            physical = _base_row(frame, split, "offline_arm_adjustment_stop") | {
                "arm_adjustment_start_frame": start,
                "arm_adjustment_stop_frame": stop,
                "source_direction": frame.horizontal_direction,
                "source_magnitude": frame.horizontal_magnitude,
                "pair_id": frame.global_index,
                "history_idle_policy": "none_raw_contiguous",
            }
            factual_plan = recovery_plan_text(
                frame.horizontal_direction, frame.horizontal_magnitude, "none", "moderately"
            )
            if frame.input_recovery_plan != factual_plan:
                raise ValueError(f"factual recovery plan mismatch: {key}, p={p}")
            factual = physical | {
                "sample_variant": "factual", "prompt_magnitude": frame.horizontal_magnitude,
                "adjustment_end": factual_adjustment_end(p, stop),
            } | _phase_fields(
                frame, qpos_points=points, effective_length=length, stats=stats,
                recovery_plan=factual_plan,
            )
            rows.append(factual)
            if p > stop:
                continue
            if frame.horizontal_magnitude == "moderately":
                target_magnitude = "slightly"
                counterfactual_label = p >= (start + stop) // 2
                variant = "moderately_to_slightly"
            elif frame.horizontal_magnitude == "slightly":
                target_magnitude = "moderately"
                counterfactual_label = False
                variant = "slightly_to_moderately"
            else:
                raise ValueError(f"unknown magnitude {frame.horizontal_magnitude!r}")
            counterfactual_plan = recovery_plan_text(
                frame.horizontal_direction, target_magnitude, "none", "moderately"
            )
            counterfactual = physical | {
                "sample_variant": variant, "prompt_magnitude": target_magnitude,
                "adjustment_end": counterfactual_label,
            } | _phase_fields(
                frame, qpos_points=points, effective_length=length, stats=stats,
                recovery_plan=counterfactual_plan,
            )
            for identity in ("history_sources", "qpos_h100_11_discrete", "effective_episode_history_length"):
                if factual[identity] != counterfactual[identity]:
                    raise AssertionError(f"counterfactual history differs: {key}, p={p}")
            rows.append(counterfactual)
    return rows


def _select_adjustment(rows, *, policy: str, seed: int):
    if policy not in {"retain_all", "ratio_1_to_2"}:
        raise ValueError(policy)
    selected = {"train": [], "val": [], "test": []}
    summary = {}
    for split in selected:
        candidates = [row for row in rows if row["split"] == split]
        sampling_details = {}
        if split != "train" or policy == "retain_all":
            chosen = candidates
        else:
            # Preserve every positive and moderately->slightly pair. Scale
            # V7.6's 975/72 slightly->moderately keep target by the number
            # of new-environment source-slightly train attempts; retain all
            # near-stop hard negatives using the original pair selector.
            core = [row for row in candidates if row["sample_variant"] != "slightly_to_moderately"]
            slight = [row for row in candidates if row["sample_variant"] == "slightly_to_moderately"]
            slight_attempts = {row["episode_id"] for row in slight}
            keep_count = round(975 * len(slight_attempts) / 72)
            selected_pair_ids, slight_summary = select_train_slight_counterfactual_pairs(
                slight, keep_count=keep_count, seed=seed
            )
            retained_slight = [row for row in slight if row["pair_id"] in selected_pair_ids]
            positive = sum(bool(row["adjustment_end"]) for row in core)
            allowed_negatives = 2 * positive
            core_negative = sum(not row["adjustment_end"] for row in core)
            need_drop = core_negative + len(retained_slight) - allowed_negatives
            early_factual = [row for row in core if row["sample_variant"] == "factual"
                             and not row["adjustment_end"]
                             and row["source_magnitude"] == "slightly"
                             and row["pair_id"] not in selected_pair_ids
                             and row["frame_index"] < row["arm_adjustment_stop_frame"] - 20]
            if not 0 <= need_drop <= len(early_factual):
                raise ValueError("cannot reach 1:2 without removing protected boundary rows")
            from tactile_vla.vla.v7_7_multitask_data import deterministic_uniform_select
            dropped = {row["global_index"] for row in deterministic_uniform_select(
                early_factual, need_drop, seed=seed, source="adjustment_early_factual_drop"
            )}
            chosen = [row for row in core if not (
                row["sample_variant"] == "factual" and row["global_index"] in dropped
            )] + retained_slight
            sampling_details = {
                "slightly_counterfactual": slight_summary,
                "early_factual_negative_candidate_count": len(early_factual),
                "early_factual_negative_removed_count": len(dropped),
                "early_factual_negative_selection": f"deterministic_uniform_seed_{seed}",
            }
        factual_pairs = {row["pair_id"] for row in chosen if row["sample_variant"] == "factual"}
        if any(row["pair_id"] not in factual_pairs for row in chosen
               if row["sample_variant"] != "factual"):
            raise AssertionError("counterfactual sample lost its factual pair")
        counts = Counter("true" if row["adjustment_end"] else "false" for row in chosen)
        if split == "train" and policy == "ratio_1_to_2" and counts["false"] != 2 * counts["true"]:
            raise AssertionError("adjustment train ratio is not 1:2")
        selected[split] = chosen
        summary[split] = {
            "candidate_count": len(candidates), "selected_count": len(chosen),
            "false": counts["false"], "true": counts["true"],
            "by_variant": dict(Counter(row["sample_variant"] for row in chosen)),
            **sampling_details,
        }
    return selected, summary


def build(args):
    v4_index_path = args.v4_dir / "v4_training_index.json"
    profile_path = args.v4_dir / "profile.json"
    splits_path = args.v4_dir / "splits.json"
    norm_path = args.v4_dir / "norm_stats/norm_stats.json"
    stage_a_config_path = args.stage_a_checkpoint.parent / "config.json"
    required = [v4_index_path, profile_path, splits_path, norm_path, args.action_index,
                args.boundary_file, stage_a_config_path, args.stage_a_checkpoint / "params/_METADATA"]
    for split in ("train", "val", "test"):
        required += [args.v4_dir / f"need/{split}.jsonl",
                     args.v4_dir / f"reasoning_manifests/failure_reason/{split}.jsonl",
                     args.v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    v4_index, profile, action_index = map(_load, (v4_index_path, profile_path, args.action_index))
    if v4_index["selection_hash"] != action_index["selection_hash"]:
        raise ValueError("V4/V7.4.2 selection hashes differ")
    if action_index["v4_training_data_hash"] != v4_index["training_data_hash"]:
        raise ValueError("V7.4.2 action index points at another V4 training index")
    validate_training_index(action_index, index_path=args.action_index, dataset_dir=args.dataset_dir)
    base_config = _load(stage_a_config_path)
    base_identity = base_config.get("artifact_identity", {})
    if base_config.get("data_profile") != "rotation_phase_v7_4_2_adjustment" or (
        base_identity.get("training_data_hash") != action_index["training_data_hash"]
    ):
        raise ValueError("V7.4.2 step-10000 checkpoint is not bound to this action index")
    frames = scan_v4_lerobot_frames(args.dataset_dir)
    _validate_attempt_transitions(frames)
    frame_by_global = {frame.global_index: frame for frame in frames}
    frame_by_key = {frame.key: frame for frame in frames}
    boundaries = validate_boundaries(_load(args.boundary_file), v4_index_file=v4_index_path, frames=frames)
    split_lists = _load(splits_path)["original_episode_ids"]
    split_by_episode = {int(episode): split for split in ("train", "val", "test")
                        for episode in split_lists[split]}
    if len(split_by_episode) != sum(len(split_lists[split]) for split in ("train", "val", "test")):
        raise ValueError("episode occurs in multiple splits")
    qpos = scan_selected_qpos(dataset_dir=args.dataset_dir,
                              selected_episode_ids=set(split_by_episode))
    _, positions = _timeline(frames, qpos)
    stats = load_state_quantiles(norm_path)

    def history_for(global_index):
        timeline, position = positions[int(global_index)]
        return sample_episode_history(timeline, position)

    adjustment_candidates = _adjustment_rows(
        frames=frames, events=boundaries, split_by_episode=split_by_episode,
        history_for=history_for, stats=stats,
    )
    selected_adjustment, adjustment_summary = _select_adjustment(
        adjustment_candidates, policy=args.adjustment_sampling_policy, seed=args.seed
    )
    adjustment = [row for split in ("train", "val", "test") for row in selected_adjustment[split]]
    profile_attempts = {(int(row["episode_id"]), int(row["attempt_id"])): row
                        for row in profile["attempts"]}
    need_by_split, need_summary = {}, {}
    for split in ("train", "val", "test"):
        positives = [row for row in load_jsonl(args.v4_dir / f"need/{split}.jsonl")
                     if bool(row["need_recovery"])]
        negative = {name: [] for name in NEGATIVE_SOURCES}
        for frame in frames:
            if split_by_episode[frame.episode_id] != split:
                continue
            meta = profile_attempts[frame.attempt_key]
            if meta["result"] == "failure" and frame.frame_index < int(meta["shift_frame_index"]):
                source = "pre_failure_hard_negative"
            elif meta["task"] == "one_success":
                source = "one_success_easy_negative"
            elif meta["result"] == "success" and frame.attempt_id == 2:
                if frame.frame_index <= boundaries[frame.attempt_key][1]:
                    continue
                source = "successful_recovery_easy_negative"
            else:
                continue
            negative[source].append(_base_row(frame, split, source) | {"need_recovery": False})
        raw, need_summary[split] = select_need_rows(positives, negative, seed=args.seed)
        rows = []
        for row in raw:
            frame = frame_by_global[int(row["global_index"])]
            points, length = history_for(frame.global_index)
            rows.append(_base_row(frame, split, row.get("source", "failure_active")) |
                        {"need_recovery": bool(row["need_recovery"])} |
                        _phase_fields(frame, qpos_points=points, effective_length=length, stats=stats))
        need_by_split[split] = rows
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
    need_positive_prompts = {
        (row["split"], row["global_index"]): row["prompt"]
        for row in need if row["need_recovery"]
    }
    for row in failure:
        if need_positive_prompts.get((row["split"], row["global_index"])) != row["prompt"]:
            raise ValueError("need/failure phase prompt differs on the same physical frame")
    manifests = {"adjustment": adjustment, "need": need, "failure": failure, "plan": plan}
    split_payload = {}
    for split in ("train", "val", "test"):
        split_payload[split] = {"action": {"indices": action_index["splits"][split]["execution_indices"]}}
        for task, rows in manifests.items():
            chosen = [i for i, row in enumerate(rows) if row["split"] == split]
            if task in {"failure", "plan"} and split != "train":
                chosen = [i for i in chosen if int(rows[i]["frame_offset"]) == 14]
            split_payload[split][task] = {
                "manifest_row_indices": chosen,
                "global_indices": [int(rows[i]["global_index"]) for i in chosen],
                "sample_count": len(chosen),
            }
    index = {
        "schema_version": INDEX_SCHEMA, "data_profile": DATA_PROFILE,
        "prompt_profile": PROMPT_PROFILE, "task_cycle": list(TASK_CYCLE), "seed": args.seed,
        "selection_hash": v4_index["selection_hash"],
        "dataset_dir": str(args.dataset_dir.resolve()),
        "stage_a_checkpoint": str(args.stage_a_checkpoint.resolve()),
        "action_index_file": str(args.action_index.resolve()),
        "action_training_data_hash": action_index["training_data_hash"],
        "adjustment_sampling_policy": args.adjustment_sampling_policy,
        "adjustment_label_policy": "factual_true_inclusive_[S-10,S+10]",
        "history_policy": helper_identity() | {"idle_perturbation": "none_raw_contiguous"},
        "need_successful_recovery_start": "arm_adjustment_stop_plus_1",
        "splits": split_payload,
        "manifest_content_hashes": {task: sha256_json(rows) for task, rows in manifests.items()},
        "source_hashes": {str(path.resolve()): sha256_file(path) for path in required},
        "v4_source_files": v4_index["source_files"],
    }
    summary = {
        "schema_version": "tactile_vla_v7_7_2_summary_v1",
        "adjustment": adjustment_summary, "need": need_summary,
        "counts": {task: {split: sum(row["split"] == split for row in rows)
                          for split in ("train", "val", "test")}
                   for task, rows in manifests.items()},
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
    outputs |= {"index": args.output_dir / "v7_7_2_multitask_training_index.json",
                "summary": args.output_dir / "summary.json",
                "hashes": args.output_dir / "artifact_hashes.json"}
    if not args.overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("V7.7.2 output exists; pass --overwrite")
    for task, rows in manifests.items():
        _write_jsonl(outputs[task], rows)
        index[f"{task}_manifest_file"] = str(outputs[task].resolve())
        index[f"{task}_manifest_sha256"] = sha256_file(outputs[task])
    index["training_data_hash"] = sha256_json(index)
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
