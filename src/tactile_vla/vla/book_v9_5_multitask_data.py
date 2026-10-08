"""Book V9.5: later C, train rotation-none negatives, real C-window reasoning."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from tactile_vla.vla import book_v9_4_multitask_data as common
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.book_v9_5_need_data import (
    LABEL_POLICY,
    SAMPLING_POLICY,
    failure_boundary,
    positive_counts,
    validate_need_rows,
    EXPECTED_ROTATION,
    rotation,
)
from tactile_vla.vla.book_v9_5_stage_a_data import reject_archive, validate_training_index as validate_action_index
from tactile_vla.vla.book_v9_4_memory import expand_plan_rows
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt
from tactile_vla.vla.v4_data import SPLITS, load_jsonl, scan_v4_lerobot_frames

ROOT = common.ROOT
VERSION_TAG = "book_v9_5"
DATA_PROFILE = "book_v9_5_five_task_h100"
INDEX_SCHEMA = "tactile_vla_book_v9_5_multitask_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_book_v9_5_multitask_manifest_v1"
DEFAULT_ACTION_INDEX = ROOT / "outputs/book_stage_a_v9_5/book_stage_a_training_index.json"
DEFAULT_ADJUSTMENT_DIR = ROOT / "outputs/book_adjustment_end_v9_5"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_book_stage_a_v9_5_no_history/15000"
DEFAULT_MULTITASK_DIR = ROOT / "outputs/book_v9_5_multitask"
DEFAULT_INDEX = DEFAULT_MULTITASK_DIR / "book_v9_5_multitask_training_index.json"
DEFAULT_OUTPUT = ROOT / "outputs/multitask_v9_5"
RUN_NAME = "pi05_book_v9_5_five_task_h100_no_history"
REASONING_WINDOW_POLICY = {
    "schema_version": "book_v9_5_reasoning_window_policy_v1",
    "C": "stable_reasoning_window_start_not_first_need_true",
    "direction_offsets_frames": {"left": 27, "right": 8},
    "train": "[C,C+14]_inclusive",
    "val_test": "C+14_all_plan_memory_lengths",
    "window_frames": 15,
    "short_attempt": "error_no_clipping_or_cross_attempt_padding",
    "failure_target": "unchanged_real_failed_attempt_metadata",
    "plan_target": "unchanged_adjacent_real_attempt2",
}


def reasoning_window(meta):
    f, c = failure_boundary(meta)
    end = c + REASONING_WINDOW_POLICY["window_frames"] - 1
    if end >= int(meta["frame_count"]):
        raise ValueError(f"V9.5 episode{meta['episode_id']} cannot support [C,C+14] within its failed attempt")
    return {
        "failure_reference_frame": f,
        "need_stable_window_start_frame": c,
        "reasoning_window_start": c,
        "reasoning_window_end": end,
        "source_anchor_frame_index": f,
    }


def expected_counts(v4_dir: Path, *, expand_plan=True):
    counts = common.expected_counts(v4_dir, expand_plan=expand_plan)
    profile = json.loads((v4_dir / "profile.json").read_text())
    positives = positive_counts(
        profile,
        scan_v4_lerobot_frames(Path(json.loads((v4_dir / "v4_training_index.json").read_text())["dataset_dir"])),
    )
    for split in SPLITS:
        counts[split]["need"] = positives[split] * 4
    attempts = Counter()
    for meta in profile["attempts"]:
        if meta["result"] == "failure":
            reasoning_window(meta)
            attempts[meta["split"]] += 1
    for split in SPLITS:
        counts[split]["failure"] = attempts[split] * (15 if split == "train" else 1)
        counts[split]["plan"] = counts[split]["failure"] * (4 if expand_plan else 1)
    return counts


def source_plan_anchors(v4_dir):
    """Original F observations supply target semantics, not the new observation inputs."""
    anchors = {}
    for split in SPLITS:
        for row in load_jsonl(v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl"):
            if row["frame_offset"] != 0:
                continue
            obs = row["current_observation"]
            key = (split, obs["episode_id"], obs["attempt_id"])
            if key in anchors:
                raise ValueError("Duplicate V9.5 real target anchor")
            anchors[key] = row
    return anchors


def build_reasoning_rows(*, profile, frame_by_key, anchors, phase_fields, base_row):
    """Re-read current frames/qpos/captions; never move old F-window inputs by renumbering."""
    failure, plans, sources = [], [], {}
    for meta in profile["attempts"]:
        if meta["result"] != "failure":
            continue
        fields = reasoning_window(meta)
        key = (meta["split"], meta["episode_id"], meta["attempt_id"])
        anchor = anchors[key]
        target = f"failure_reason=rotate {meta['rotation_direction']},grasp {meta['grasp_position']}."
        if anchor["current_observation"]["failure_reason"] != target:
            raise ValueError("V9.5 original real failure target disagrees with metadata")
        if int(anchor["frame_index"]) != fields["failure_reference_frame"]:
            raise ValueError("V9.5 target anchor is not the original F frame")
        for offset in range(15):
            frame_index = fields["reasoning_window_start"] + offset
            frame = frame_by_key.get((meta["episode_id"], meta["attempt_id"], frame_index))
            if frame is None:
                raise ValueError(f"V9.5 missing real frame {key}/{frame_index}")
            if rotation(frame.tactile_caption) != EXPECTED_ROTATION[meta["rotation_direction"]]:
                raise ValueError(f"V9.5 reasoning window has none/wrong rotation: {key}/{frame_index}; audit data")
            identity = base_row(frame, meta["split"], "book_v9_5_C_window") | fields | {"frame_offset": offset}
            failure.append(identity | {"target_failure_reason": target} | phase_fields(frame))
            memory = [{"recovery_plan": "initial plan", "failure_reason": target}]
            plans.append(
                identity
                | {
                    "memory_length": 1,
                    "failure_recovery_memory": memory,
                    "target_recovery_plan": anchor["target_recovery_plan"],
                    "prompt": build_recovery_prompt(
                        instruction=frame.instruction,
                        failed_tactile_caption=frame.tactile_caption,
                        failure_recovery_memory=memory,
                        prompt_profile=MINIMAL_PROMPT_PROFILE,
                    ),
                }
            )
            # Retain the explicit F anchor ID and adjacent real attempt2 provenance.
            # This derived observation is genuinely read from C+offset, not claimed to be in V4's F window.
            sources[meta["split"], frame.global_index] = anchor | {
                "current_observation": anchor["current_observation"]
                | {
                    "frame_index": frame_index,
                    "frame_offset": offset,
                    "window_start": fields["reasoning_window_start"],
                    "window_end": fields["reasoning_window_end"],
                    "ros_timestamp": frame.ros_timestamp,
                    "tactile_caption": frame.tactile_caption,
                },
            }
    return failure, expand_plan_rows(plans, failure, sources=sources)


def validate_reasoning_rows(index, manifests, profile):
    attempts = {
        (m["split"], m["episode_id"], m["attempt_id"]): m for m in profile["attempts"] if m["result"] == "failure"
    }
    all_meta = {(m["episode_id"], m["attempt_id"]): m for m in profile["attempts"]}
    anchors = source_plan_anchors(Path(index["v4_dir"]))
    need = {(r["split"], r["global_index"]): r for r in manifests["need"] if r["need_recovery"]}
    for task in ("failure", "plan"):
        observed = set()
        for row in manifests[task]:
            key = (row["split"], row["episode_id"], row["attempt_id"])
            meta = attempts[key]
            fields = reasoning_window(meta)
            frame = row["frame_index"]
            if (
                any(row.get(k) != v for k, v in fields.items())
                or not fields["reasoning_window_start"] <= frame <= fields["reasoning_window_end"]
            ):
                raise ValueError("V9.5 failure/plan frame is not in its C window")
            if row["frame_offset"] != frame - fields["reasoning_window_start"]:
                raise ValueError("V9.5 frame_offset must be relative to C, not F")
            identity = (*key, frame, row["memory_length"] if task == "plan" else 0)
            if identity in observed:
                raise ValueError("Duplicate V9.5 reasoning frame/variant")
            observed.add(identity)
            target = f"failure_reason=rotate {meta['rotation_direction']},grasp {meta['grasp_position']}."
            positive = need.get((row["split"], row["global_index"]))
            if positive is None or positive["frame_index"] != frame or positive["episode_id"] != row["episode_id"]:
                raise ValueError("V9.5 reasoning frames must coincide with actual need positives")
            if task == "failure":
                if row["target_failure_reason"] != target or row["prompt"] != positive["prompt"]:
                    raise ValueError("V9.5 failure target/shared need prefix mismatch")
            else:
                anchor = anchors[key]
                recovery = all_meta[row["episode_id"], row["attempt_id"] + 1]
                plan = (
                    f"recovery_plan=move horizontally {recovery['horizontal_direction']} "
                    f"{recovery['horizontal_magnitude']}, move vertically {recovery['vertical_direction']} "
                    f"{recovery['vertical_magnitude']}."
                )
                ts = anchor["target_source"]
                if (
                    row["current_real_failure"] != target
                    or row["target_recovery_plan"] != plan
                    or row["target_recovery_plan"] != anchor["target_recovery_plan"]
                    or row.get("target_source") != ts
                    or row.get("source_variant_id") != anchor["variant_id"]
                    or ts["source_type"] != "real"
                    or ts["episode_id"] != row["episode_id"]
                    or ts["failed_attempt_id"] != row["attempt_id"]
                    or ts["plan_attempt_id"] != row["attempt_id"] + 1
                ):
                    raise ValueError("V9.5 changed real failure/adjacent attempt2 target provenance")
        expected = {
            (*key, frame, length)
            for key, meta in attempts.items()
            for frame in range(
                reasoning_window(meta)["reasoning_window_start"], reasoning_window(meta)["reasoning_window_end"] + 1
            )
            for length in ((1, 2, 3, 4) if task == "plan" else (0,))
        }
        if observed != expected:
            raise ValueError("V9.5 C window must contain all 15 frames and every plan history variant")
        for split in SPLITS:
            chosen = [
                i
                for i, row in enumerate(manifests[task])
                if row["split"] == split and (split == "train" or row["frame_offset"] == 14)
            ]
            if index["splits"][split][task]["manifest_row_indices"] != chosen:
                raise ValueError("V9.5 train uses all C frames; val/test must select C+14")
    return {
        "schema_version": "book_v9_5_reasoning_boundary_audit_v1",
        "policy": REASONING_WINDOW_POLICY,
        "attempts": [
            {
                "split": key[0],
                "episode_id": key[1],
                "attempt_id": key[2],
                "rotation_direction": meta["rotation_direction"],
                **reasoning_window(meta),
                "eval_frame": reasoning_window(meta)["reasoning_window_end"],
                "frame_count": 15,
            }
            for key, meta in sorted(attempts.items())
        ],
    }


def validate_manifest_rows(index, manifests):
    common.validate_manifest_rows(
        index, manifests, data_profile=DATA_PROFILE, manifest_schema=MANIFEST_SCHEMA, need_variant_key="need_variant"
    )
    profile = json.loads((Path(index["v4_dir"]) / "profile.json").read_text())
    need_audit = validate_need_rows(profile, manifests["need"], scan_v4_lerobot_frames(Path(index["dataset_dir"])))
    need_audit["label_policy"] = LABEL_POLICY
    for split in SPLITS:
        expected = [i for i, r in enumerate(manifests["need"]) if r["split"] == split]
        if index["splits"][split]["need"]["manifest_row_indices"] != expected:
            raise ValueError("V9.5 must retain the entire new V9.5 need stream")
    return need_audit, validate_reasoning_rows(index, manifests, profile)


def _validate_plan_sources(index, manifests, profile):
    validate_reasoning_rows(index, manifests, profile)
    frames = {f.global_index: f for f in scan_v4_lerobot_frames(Path(index["dataset_dir"]))}
    from scripts.build_book_v9_3_multitask_data import _phase_fields, _timeline
    from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles, scan_selected_qpos
    from tactile_vla.vla.v7_7_phase_prompt import sample_episode_history

    dataset_dir = Path(index["dataset_dir"])
    qpos = scan_selected_qpos(dataset_dir=dataset_dir, selected_episode_ids={f.episode_id for f in frames.values()})
    _, positions = _timeline(list(frames.values()), qpos)
    stats = load_state_quantiles(Path(index["v4_dir"]) / "norm_stats/norm_stats.json")
    for row in manifests["need"]:
        frame = frames[row["global_index"]]
        timeline, position = positions[frame.global_index]
        points, length = sample_episode_history(timeline, position)
        canonical = _phase_fields(frame, qpos_points=points, effective_length=length, stats=stats)
        cf = row.get("need_counterfactual")
        if cf is not None:
            if cf["original_tactile_caption"] != frame.tactile_caption or cf["original_prompt"] != canonical["prompt"]:
                raise ValueError("V9.5 counterfactual original caption differs from actual source")
            caption = cf["input_tactile_caption"]
        else:
            caption = frame.tactile_caption
            if row["prompt"] != canonical["prompt"]:
                raise ValueError("V9.5 real need prompt changed from source")
        if any(row[name] != canonical[name] for name in canonical if name != "prompt"):
            raise ValueError("V9.5 need qpos/H100 history differs from source")
        if (
            frame.key != (row["episode_id"], row["attempt_id"], row["frame_index"])
            or frame.ros_timestamp != row["timestamp"]
            or f"\nTouch: {caption}\n" not in row["prompt"]
        ):
            raise ValueError("V9.5 need caption/frame source mismatch")
    for task in ("failure", "plan"):
        for row in manifests[task]:
            frame = frames[row["global_index"]]
            if (
                frame.key != (row["episode_id"], row["attempt_id"], row["frame_index"])
                or frame.ros_timestamp != row["timestamp"]
                or frame.tactile_caption not in row["prompt"]
            ):
                raise ValueError("V9.5 reasoning input must use the actual new C-window observation")


def validate_index(index):
    if (
        index.get("experiment_version") != VERSION_TAG
        or index.get("need_label_policy") != LABEL_POLICY
        or index.get("need_sampling_policy") != SAMPLING_POLICY
        or index.get("reasoning_window_policy") != REASONING_WINDOW_POLICY
    ):
        raise ValueError("V9.5 requires failure/plan [C,C+14]; rebuild old data")
    for name in ("dataset_dir", "v4_dir", "incremental_state_dir", "action_index_file", "adjustment_source_index"):
        reject_archive(Path(index[name]))
    for path in index["source_hashes"]:
        reject_archive(Path(path))
    action_path = Path(index["action_index_file"])
    action = json.loads(action_path.read_text())
    if index.get("source_scope") != action.get("source_scope"):
        raise ValueError("V9.5 multitask and Stage A source scopes differ")
    validate_action_index(
        action,
        index_path=action_path,
        dataset_dir=Path(index["dataset_dir"]),
        book_root=Path(index["source_scope"]["book_root"]),
    )
    common.validate_index(
        index,
        data_profile=DATA_PROFILE,
        index_schema=INDEX_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
        counts_fn=expected_counts,
        plan_source_validator=_validate_plan_sources,
        need_variant_key="need_variant",
    )
    manifests = {
        task: load_jsonl(Path(index[f"{task}_manifest_file"])) for task in ("adjustment", "need", "failure", "plan")
    }
    audits = validate_manifest_rows(index, manifests)
    for name, audit in zip(("need_boundary_audit", "reasoning_boundary_audit"), audits, strict=True):
        path = Path(index[f"{name}_file"])
        if sha256_file(path) != index[f"{name}_sha256"] or json.loads(path.read_text()) != audit:
            raise ValueError(f"V9.5 {name} differs from actual frames")


def validate_stage_a_for_training(checkpoint, index):
    checkpoint = reject_archive(checkpoint)
    identity = common.validate_stage_a_for_training(checkpoint, index)
    config = json.loads((checkpoint.parent / "config.json").read_text())
    if config.get("experiment_version") != VERSION_TAG or config.get("source_scope") != index["source_scope"]:
        raise ValueError("V9.5 requires matching V9.5 Stage A version and current source scope")
    return identity
