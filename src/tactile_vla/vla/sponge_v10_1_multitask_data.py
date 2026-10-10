"""Independent Sponge V10.1 five-task data; real captions, pure fz_bias targets."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_2_adjustment_end_data import selected_train_frames
from tactile_vla.vla.book_v9_4_multitask_data import (
    load_captioner_provenance,
    validate_caption,
    validate_upload_metadata,
)
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt
from tactile_vla.vla.sponge_v10_1_stage_a_data import (
    DEFAULT_STAGE_A as DEFAULT_STAGE_A,
    FAILURE_ACTION_POLICY,
    ROOT,
    VERSION_TAG,
    failure_boundary,
    validate_training_index as validate_action_index,
)
from tactile_vla.vla.structured_text import failure_reason_text, recovery_plan_text
from tactile_vla.vla.v4_data import SPLITS, load_jsonl, validate_direct_manifest_rows, validate_v4_index_dataset
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles, scan_selected_qpos
from tactile_vla.vla.v7_7_multitask_data import (
    NEGATIVE_SOURCES,
    TASK_CYCLE,
    allocate_balanced_without_replacement,
    deterministic_uniform_select,
    stable_priority,
)
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE, helper_identity, sample_episode_history

DATA_PROFILE = "sponge_v10_1_five_task_h100"
INDEX_SCHEMA = "tactile_vla_sponge_v10_1_multitask_index_v1"
MANIFEST_SCHEMA = "tactile_vla_sponge_v10_1_multitask_manifest_v1"
DEFAULT_OUTPUT = ROOT / "outputs/sponge_v10_1_multitask"
DEFAULT_INDEX = DEFAULT_OUTPUT / "sponge_v10_1_multitask_training_index.json"
RUN_NAME = "pi05_sponge_v10_1_five_task_h100_no_history"
FAILURE_TARGET = failure_reason_text("none", "appropriate", "left")
PLAN_TARGET = recovery_plan_text("none", "moderately", "up", "moderately")
MEMORY_LENGTHS = (1, 2, 3, 4)
MEMORY_POLICY = {
    "schema_version": "sponge_v10_1_fz_bias_memory_v1",
    "lengths": list(MEMORY_LENGTHS),
    "first_plan": "initial plan",
    "prefix_source": "synthetic_text_only_no_donor_episodes",
    "failure": FAILURE_TARGET,
    "subsequent_plan": PLAN_TARGET,
    "terminal_failure": "current_real_failure",
    "target": "unchanged_adjacent_real_attempt2",
    "runtime_retention": "initial_pair_plus_latest_three",
    "max_recovery_attempts": None,
    "slightly": False,
    "rotation_failures": False,
}
NEED_POLICY = {
    "schema_version": "sponge_v10_1_need_v1",
    "B": "captioner_first_Fz_bias_left",
    "C": "B+3_native_failure_boundary",
    "failed_attempt": "frame<B:false;[B,C):excluded;frame>=C:true",
    "successful_attempt1": "false",
    "successful_attempt2": "frame>=R:false;frame<R:excluded",
    "negative_to_positive_ratio": "3:1",
    "negative_sources": list(NEGATIVE_SOURCES),
    "negative_source_allocation": "as_close_as_possible_to_1:1:1",
    "reserve_hard_negatives": "all_[max(0,B-30),B)_before_uniform_remaining",
    "counterfactual": False,
    "caption": "unchanged_all_six_fields_including_rotation",
    "seed": 42,
}
ADJUSTMENT_POLICY = {
    "range": "inclusive_[A,R]",
    "A": "attempt2_frame_0",
    "positive": "inclusive_[R-10,R]",
    "negative": "inclusive_[A,R-11]",
    "train_sampling": "11_positive+10_hard+8_middle+4_early",
    "val_test": "all_[A,R]",
    "counterfactual": False,
}
REASONING_POLICY = {"train": "inclusive_[C,C+14]", "val_test": "C+14", "failure": FAILURE_TARGET, "plan": PLAN_TARGET}
HISTORY_POLICY = helper_identity() | {"idle_perturbation": "none_raw_contiguous"}


def need_role(frame, meta):
    if meta["result"] == "failure":
        b, c = failure_boundary(meta)
        if frame.frame_index < b:
            return False, NEGATIVE_SOURCES[0]
        if frame.frame_index >= c:
            return True, "failure_active"
        return None, None
    if frame.attempt_id == 1:
        return False, NEGATIVE_SOURCES[1]
    r = meta.get("rexecution_frame_index")
    if r is None:
        raise ValueError("V10.1 successful attempt2 requires native R")
    return (False, NEGATIVE_SOURCES[2]) if frame.frame_index >= int(r) else (None, None)


def select_need(positives, negatives, reserved, *, split):
    count = len(positives)
    if not count:
        raise ValueError("V10.1 need requires real positives in every split")
    allocation = allocate_balanced_without_replacement({s: len(negatives[s]) for s in NEGATIVE_SOURCES}, 3 * count)
    hard = NEGATIVE_SOURCES[0]
    if len(reserved) > allocation[hard]:
        extra = len(reserved) - allocation[hard]
        allocation[hard] += extra
        for source in sorted(NEGATIVE_SOURCES[1:], key=lambda s: allocation[s], reverse=True):
            take = min(extra, allocation[source])
            allocation[source] -= take
            extra -= take
        if extra:
            raise ValueError("Reserved boundary negatives exceed the total need negative budget")
    reserved_ids = {r["global_index"] for r in reserved}
    chosen = list(positives)
    for source in NEGATIVE_SOURCES:
        pool = negatives[source]
        forced = reserved if source == hard else []
        if source == hard:
            pool = [r for r in pool if r["global_index"] not in reserved_ids]
        chosen.extend(forced)
        chosen.extend(
            deterministic_uniform_select(pool, allocation[source] - len(forced), seed=42, source=split + "/" + source)
        )
    chosen.sort(key=lambda r: stable_priority(42, split, "need-v10_1", r["global_index"]))
    if len(chosen) != 4 * count or len({r["global_index"] for r in chosen}) != len(chosen):
        raise ValueError("V10.1 need selection violates the unique 1:3 rule")
    return chosen, {
        "positive": count,
        "negative": 3 * count,
        "reserved_hard_negative": len(reserved),
        "candidate_negative": {s: len(negatives[s]) for s in NEGATIVE_SOURCES},
        "selected_negative": allocation,
    }


def make_memory(length, *, split, global_index):
    if length not in MEMORY_LENGTHS:
        raise ValueError("V10.1 supports memory lengths 1..4")
    variant = "sponge-v10_1-" + sha256_json([42, split, global_index, length, MEMORY_POLICY])[:24]
    return [
        {
            "recovery_plan": "initial plan" if i == 0 else PLAN_TARGET,
            "failure_reason": FAILURE_TARGET,
            "source_type": "real_current_failure" if i == length - 1 else "synthetic",
            "plan_source_type": "initial" if i == 0 else "synthetic",
            "pair_index": i,
            "variant_id": variant,
        }
        for i in range(length)
    ]


def build_artifacts(*, dataset_dir, v4_dir, action_index_file, incremental_state_dir):
    """Data-only construction; no Stage A checkpoint or GPU/model required."""
    from scripts.build_v7_7_multitask_data import _phase_fields, _timeline

    dataset_dir, v4_dir, action_index_file, incremental_state_dir = map(
        Path, (dataset_dir, v4_dir, action_index_file, incremental_state_dir)
    )
    v4_file = v4_dir / "v4_training_index.json"
    v4 = json.loads(v4_file.read_text())
    frames, _ = validate_v4_index_dataset(v4, dataset_dir)
    profile = json.loads((v4_dir / "profile.json").read_text())
    validate_upload_metadata(dataset_dir, profile)
    captioner, provenance_hashes = load_captioner_provenance(incremental_state_dir, profile=profile)
    action = json.loads(action_index_file.read_text())
    validate_action_index(action, index_path=action_index_file, dataset_dir=dataset_dir)
    if action["v4_training_data_hash"] != v4["training_data_hash"]:
        raise ValueError("V10.1 action and multitask data refer to different V4 snapshots")
    norm_path = v4_dir / "norm_stats/norm_stats.json"
    if action["v4_norm_stats_sha256"] != sha256_file(norm_path):
        raise ValueError("V10.1 norm stats mismatch")
    metadata = {(m["episode_id"], m["attempt_id"]): m for m in profile["attempts"]}
    for f in frames:
        validate_caption(f.tactile_caption)
    qpos = scan_selected_qpos(dataset_dir=dataset_dir, selected_episode_ids={f.episode_id for f in frames})
    _, positions = _timeline(frames, qpos)
    stats = load_state_quantiles(norm_path)
    by_key = {f.key: f for f in frames}

    def identity(f, split, source):
        return {
            "schema_version": MANIFEST_SCHEMA,
            "data_profile": DATA_PROFILE,
            "split": split,
            "global_index": f.global_index,
            "episode_id": f.episode_id,
            "attempt_id": f.attempt_id,
            "frame_index": f.frame_index,
            "timestamp": f.ros_timestamp,
            "source": source,
        }

    def phase_fields(f):
        timeline, p = positions[f.global_index]
        points, length = sample_episode_history(timeline, p)
        return _phase_fields(f, qpos_points=points, effective_length=length, stats=stats)

    adjustment, need, failure, plan = [], [], [], []
    need_summary, boundaries = {}, []
    for split in SPLITS:
        positive, negative, reserved = [], {s: [] for s in NEGATIVE_SOURCES}, []
        for f in frames:
            m = metadata[f.attempt_key]
            if m["split"] != split:
                continue
            if f.attempt_id == 2:
                r = int(m["rexecution_frame_index"])
                if f.frame_index <= r:
                    selected = selected_train_frames(r, episode_id=f.episode_id) if split == "train" else None
                    adjustment.append(
                        identity(f, split, "real_adjustment")
                        | {
                            "rexecution_frame": r,
                            "adjustment_end": f.frame_index >= r - 10,
                            "selected": split != "train" or f.frame_index in selected,
                        }
                        | phase_fields(f)
                    )
            label, source = need_role(f, m)
            if label is None:
                continue
            row = identity(f, split, source) | {"need_recovery": label}
            if label:
                positive.append(row)
            else:
                negative[source].append(row)
                if source == NEGATIVE_SOURCES[0]:
                    b, _ = failure_boundary(m)
                    if max(0, b - 30) <= f.frame_index < b:
                        reserved.append(row)
        selected_need, need_summary[split] = select_need(positive, negative, reserved, split=split)
        for row in selected_need:
            row.update(phase_fields(by_key[row["episode_id"], row["attempt_id"], row["frame_index"]]))
        need.extend(selected_need)

        source_failure = validate_direct_manifest_rows(
            load_jsonl(v4_dir / f"reasoning_manifests/failure_reason/{split}.jsonl"),
            split=split,
            task="failure",
            frame_lookup=by_key,
            profile_attempts=metadata,
            failure_window_length=15,
        )
        source_plan = validate_direct_manifest_rows(
            load_jsonl(v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl"),
            split=split,
            task="plan",
            frame_lookup=by_key,
            profile_attempts=metadata,
            failure_window_length=15,
        )
        anchors = {(r["current_observation"]["episode_id"], r["frame_index"]): r for r in source_plan}
        for old in source_failure:
            obs = old["current_observation"]
            f = by_key[obs["episode_id"], obs["attempt_id"], old["frame_index"]]
            anchor = anchors[f.episode_id, f.frame_index]
            if old["target_failure_reason"] != FAILURE_TARGET or anchor["target_recovery_plan"] != PLAN_TARGET:
                raise ValueError("V10.1 requires real rotate-none/fz_bias-left -> up-moderately labels")
            b, c = failure_boundary(metadata[f.attempt_key])
            if old["window_start"] != c or not c <= f.frame_index <= c + 14:
                raise ValueError("V10.1 reasoning must use actual [C,C+14] source observations")
            row = identity(f, split, "real_C_window") | {
                "frame_offset": old["frame_offset"],
                "B": b,
                "C": c,
            }
            failure.append(row | {"target_failure_reason": FAILURE_TARGET} | phase_fields(f))
            for length in MEMORY_LENGTHS:
                memory = make_memory(length, split=split, global_index=f.global_index)
                plan.append(
                    row
                    | {
                        "memory_length": length,
                        "failure_recovery_memory": memory,
                        "variant_id": memory[0]["variant_id"],
                        "target_recovery_plan": PLAN_TARGET,
                        "target_source": anchor["target_source"],
                        "prompt": build_recovery_prompt(
                            instruction=f.instruction,
                            failed_tactile_caption=f.tactile_caption,
                            failure_recovery_memory=memory,
                            prompt_profile=MINIMAL_PROMPT_PROFILE,
                        ),
                    }
                )
    for m in profile["attempts"]:
        if m["result"] == "failure":
            b, c = failure_boundary(m)
            boundaries.append(
                {
                    "episode_id": m["episode_id"],
                    "split": m["split"],
                    "B": b,
                    "C": c,
                    "R": metadata[m["episode_id"], 2]["rexecution_frame_index"],
                    "frame_count": m["frame_count"],
                }
            )
    manifests = {"adjustment": adjustment, "need": need, "failure": failure, "plan": plan}
    # Use exactly the training tokenizer and normalized current state; overflow is an error.
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.transforms import Normalize

    tokenizer = PaligemmaTokenizer(512)
    normalizer = Normalize(normalize.load(v4_dir / "norm_stats"), use_quantiles=True)
    token_summary = {}
    for task, rows in manifests.items():
        lengths = {}
        for row in rows:
            state = normalizer({"state": np.asarray(qpos[row["global_index"]], dtype=np.float32)})["state"]
            target_text = row.get("target_failure_reason", row.get("target_recovery_plan"))
            target = None if target_text is None else tokenizer.encode_text(target_text, add_eos=True)
            _, mask, _, _, prefix = tokenizer.tokenize_structured_response(
                row["prompt"], state, target, max_len=320 if task == "plan" else 512
            )
            row["token_lengths"] = {
                "prefix": int(prefix),
                "total": int(mask.sum()),
                "target": 0 if target is None else len(target),
            }
            lengths.setdefault(str(row.get("memory_length", "all")), []).append(int(mask.sum()))
        token_summary[task] = {k: {"count": len(v), "min": min(v), "max": max(v)} for k, v in lengths.items()}
    splits = {}
    for split in SPLITS:
        splits[split] = {"action": {"indices": action["splits"][split]["execution_indices"]}}
        for task, rows in manifests.items():
            selected = [
                i
                for i, r in enumerate(rows)
                if r["split"] == split
                and (task != "adjustment" or r["selected"])
                and (task not in {"failure", "plan"} or split == "train" or r["frame_offset"] == 14)
            ]
            splits[split][task] = {
                "manifest_row_indices": selected,
                "global_indices": [rows[i]["global_index"] for i in selected],
                "sample_count": len(selected),
            }
            if not selected:
                raise ValueError(f"V10.1 empty {split}/{task} stream")
    sources = {str(p.resolve()): sha256_file(p) for p in (v4_file, action_index_file, norm_path)} | provenance_hashes
    index = {
        "schema_version": INDEX_SCHEMA,
        "experiment_version": VERSION_TAG,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "task_cycle": list(TASK_CYCLE),
        "seed": 42,
        "dataset_dir": str(dataset_dir.resolve()),
        "v4_dir": str(v4_dir.resolve()),
        "incremental_state_dir": str(incremental_state_dir.resolve()),
        "action_index_file": str(action_index_file.resolve()),
        "action_training_data_hash": action["training_data_hash"],
        "source_scope": action["source_scope"],
        "selection_hash": v4["selection_hash"],
        "fz_bias_directions": ["left"],
        "include_fz_bias_failure_grammar": True,
        "failure_action_policy": FAILURE_ACTION_POLICY,
        "need_policy": NEED_POLICY,
        "adjustment_policy": ADJUSTMENT_POLICY,
        "reasoning_policy": REASONING_POLICY,
        "history_policy": HISTORY_POLICY,
        "memory_policy": MEMORY_POLICY,
        "captioner_identity": captioner,
        "model_dependency": "none_data_only",
        "training_target_coverage": {"failure_reason": [FAILURE_TARGET], "recovery_plan": [PLAN_TARGET]},
        "splits": splits,
        "source_hashes": sources,
        "token_validation": token_summary,
        "manifest_content_hashes": {t: sha256_json(r) for t, r in manifests.items()},
        "boundary_audit": boundaries,
    }
    summary = {
        "experiment_version": VERSION_TAG,
        "selected_counts": {
            s: {"action": len(splits[s]["action"]["indices"])} | {t: splits[s][t]["sample_count"] for t in manifests}
            for s in SPLITS
        },
        "adjustment_labels": {
            s: dict(
                Counter(
                    "true" if adjustment[i]["adjustment_end"] else "false"
                    for i in splits[s]["adjustment"]["manifest_row_indices"]
                )
            )
            for s in SPLITS
        },
        "need": need_summary,
        "token_validation": token_summary,
        "caption_rotation_note": "all_Fz_bias_left_captions_are_counterclockwise_in_current_snapshot_real_failure_stays_rotate_none",
        "model_dependency": "none_data_only",
    }
    return index, summary, manifests


def validate_index(index):
    if (
        index.get("schema_version") != INDEX_SCHEMA
        or index.get("experiment_version") != VERSION_TAG
        or index.get("data_profile") != DATA_PROFILE
    ):
        raise ValueError("V10.1 multitask index header mismatch")
    if index.get("training_data_hash") != sha256_json({k: v for k, v in index.items() if k != "training_data_hash"}):
        raise ValueError("V10.1 multitask index hash mismatch")
    for path, digest in index["source_hashes"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"V10.1 source changed: {path}")
    expected, _, rows = build_artifacts(
        dataset_dir=Path(index["dataset_dir"]),
        v4_dir=Path(index["v4_dir"]),
        action_index_file=Path(index["action_index_file"]),
        incremental_state_dir=Path(index["incremental_state_dir"]),
    )
    for key, value in expected.items():
        if index.get(key) != value:
            raise ValueError(f"V10.1 source-derived index {key} mismatch")
    for task, expected_rows in rows.items():
        path = Path(index[f"{task}_manifest_file"])
        if sha256_file(path) != index[f"{task}_manifest_sha256"] or load_jsonl(path) != expected_rows:
            raise ValueError(f"V10.1 {task} manifest differs from source semantics/prompts/history")


def validate_stage_a_for_training(checkpoint, index):
    checkpoint = Path(checkpoint).resolve()
    config_path = checkpoint.parent / "config.json"
    metadata = checkpoint / "params/_METADATA"
    if checkpoint.name != "15000" or not metadata.is_file():
        raise ValueError("V10.1 requires completed sponge Stage A step15000")
    config = json.loads(config_path.read_text())
    action = json.loads(Path(index["action_index_file"]).read_text())
    if (
        config.get("experiment_version") != VERSION_TAG
        or config.get("use_state_history") is not False
        or config.get("data_profile") != "book_stage_a_v1"
        or config.get("num_steps") != 15000
        or config.get("failure_action_policy") != FAILURE_ACTION_POLICY
        or config.get("artifact_identity", {}).get("training_data_hash") != index["action_training_data_hash"]
        or config.get("source_scope") != action["source_scope"]
        or config.get("artifact_identity", {}).get("index_sha256") != sha256_file(index["action_index_file"])
        or config.get("artifact_identity", {}).get("v4_norm_stats_sha256") != action["v4_norm_stats_sha256"]
    ):
        raise ValueError("V10.1 Stage A version, dataset, C filter or norm identity mismatch")
    return {
        "path": str(checkpoint),
        "step": 15000,
        "experiment_version": VERSION_TAG,
        "config_sha256": sha256_file(config_path),
        "params_metadata_sha256": sha256_file(metadata),
        "action_training_data_hash": index["action_training_data_hash"],
    }
