"""Bottle V11.1 Stage A: native R phases and Fy-boundary-safe raw H30."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from tactile_vla.vla import book_stage_a_data as common
from tactile_vla.vla.artifacts import action_indices_identity, sha256_file, sha256_json
from tactile_vla.vla.v4_data import SPLITS, load_jsonl, validate_v4_index_dataset
from tactile_vla.vla.bottle_v11_1_grammar import bottle_recovery_plans
from tactile_vla.vla.prompts import build_phase_prompt
from tactile_vla.vla.structured_text import recovery_plan_text

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/bottle")
VERSION_TAG = "bottle_v11_1"
DATA_PROFILE = common.DATA_PROFILE  # Reuse the proven phase_v2/H30 action transforms.
EXPERIMENT_KIND = common.EXPERIMENT_KIND
RUN_NAME = "pi05_delta_tac_bottle_stage_a_v11_1_no_history"
DEFAULT_INDEX = ROOT / "outputs/bottle_stage_a_v11_1/stage_a_training_index.json"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action" / RUN_NAME / "15000"
INSTRUCTION = "Open the bottle"
CAPTIONER_CHECKPOINT = Path(
    "/data1/qxh/tac_vla_new/tac_data/tac_cap_data/tactile_captioner/tcn_v4_w30_masked_six_head_mixed_2/best.pt"
)
CAPTIONER_SHA256 = "8f896cc7b48e1bef98405047e5a7f03e481eb8f9851cb9fee953a8c0a43e6763"
FAILURE_ACTION_POLICY = {
    "schema_version": "bottle_v11_1_failure_safe_action_v1",
    "F": "first_Fy_positive_segment_at_least_18_frames_after_W30_warmup",
    "C": "native_V4_failure_boundary_frame_index_F_plus_3",
    "captioner_checkpoint_sha256": CAPTIONER_SHA256,
    "failed_attempt_keep": "s+29<C",
    "failed_attempt_exclude": "cross_C_or_start_at_or_after_C",
    "raw_frames": "unchanged_keep_for_H100_and_other_tasks",
    "norm_stats": "reuse_unchanged_source_V4_stats",
}
POLICY = common.POLICY | {"failure_action_filter": FAILURE_ACTION_POLICY}


def failure_boundary(meta):
    """F/C are offline labels from mixed_2; pending/drop are not inferred."""
    if (
        meta["result"] != "failure"
        or meta.get("failure_boundary_source") != "captioner_fy"
        or meta.get("failure_boundary_status") != "resolved"
        or meta.get("fz_bias_direction", "none") != "none"
        or meta.get("fy_change") != "raise"
        or meta.get("rotation_direction") != "none"
        or meta.get("grasp_position") != "appropriate"
    ):
        raise ValueError("V11.1 requires a resolved pure Fy-raise failure with captioner_fy boundary")
    if (
        meta.get("failure_boundary_rule_version") != "captioner_fy_positive_offset3_v1"
        or meta.get("failure_boundary_checkpoint_sha256") != CAPTIONER_SHA256
    ):
        raise ValueError("V11.1 failure boundary rule/captioner identity mismatch")
    f, c = int(meta["captioner_fy_positive_start_frame"]), int(meta["failure_boundary_frame_index"])
    if f < 29 or c != f + 3 or c != int(meta["failure_window_start"]) or c + 14 >= int(meta["frame_count"]):
        raise ValueError("V11.1 requires F after W30 warm-up, C=F+3 and complete [C,C+14]")
    return f, c


def failure_safe_chunk(start, boundary, horizon=30):
    return int(start) + int(horizon) - 1 < int(boundary)


def action_token_summary(rows, frames, *, dataset_dir, norm_stats_dir):
    """Audit the exact Pi05 Action: prefix before the legacy tokenizer can truncate."""
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.transforms import Normalize
    from tactile_vla.vla.v5_3_adjustment_end_data import scan_selected_qpos

    qpos = scan_selected_qpos(dataset_dir=dataset_dir, selected_episode_ids={f.episode_id for f in frames})
    by_global = {f.global_index: f for f in frames}
    normalizer = Normalize(normalize.load(norm_stats_dir), use_quantiles=True)
    tokenizer = PaligemmaTokenizer(200)
    lengths = {}
    for row in rows:
        if not row["trainable"]:
            continue
        frame = by_global[row["global_index"]]
        prompt = (
            build_phase_prompt(
                phase=row["phase"],
                instruction=frame.instruction,
                recovery_plan=frame.input_recovery_plan,
                prompt_profile="phase_v2",
            )
            .strip()
            .replace("_", " ")
            .replace("\n", " ")
        )
        state = normalizer({"state": np.asarray(qpos[frame.global_index], dtype=np.float32)})["state"]
        state = np.asarray(state, dtype=np.float32)
        discrete = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_text = " ".join(map(str, discrete))
        length = len(tokenizer.encode_text(f"Task: {prompt}, State: {state_text};\nAction: ", add_bos=True))
        if length > 200:
            raise ValueError(f"V11.1 action prefix exceeds 200 tokens at {frame.key}: {length}")
        lengths.setdefault(row["split"] + "/" + row["phase"], []).append(length)
    return {key: {"count": len(values), "min": min(values), "max": max(values)} for key, values in lengths.items()}


def build_artifacts(*, dataset_dir: Path, v4_index_file: Path, norm_stats_dir: Path):
    rows, index, _ = common.build_artifacts(
        dataset_dir=dataset_dir, v4_index_file=v4_index_file, norm_stats_dir=norm_stats_dir
    )
    v4 = json.loads(v4_index_file.read_text())
    profile_path = Path(v4["source_files"]["profile"]["path"])
    profile = json.loads(profile_path.read_text())
    metadata = {(m["episode_id"], m["attempt_id"]): m for m in profile["attempts"]}
    frames, _ = validate_v4_index_dataset(v4, dataset_dir)
    if any(f.task_name != "bottle" or f.instruction != INSTRUCTION for f in frames):
        raise ValueError("V11.1 requires bottle data with instruction 'Open the bottle'")
    for m in metadata.values():
        if m["result"] == "failure":
            failure_boundary(m)
        elif (
            m["result"] != "success"
            or m.get("rotation_direction") != "none"
            or m.get("grasp_position") != "appropriate"
            or m.get("fz_bias_direction", "none") != "none"
            or m.get("fy_change", "none") != "none"
        ):
            raise ValueError("V11.1 successful attempts must have neutral failure labels")
        if m["attempt_id"] == 2:
            plan = recovery_plan_text(
                m["horizontal_direction"],
                m["horizontal_magnitude"],
                m["vertical_direction"],
                m["vertical_magnitude"],
                m.get("rotate_direction", "none"),
                m.get("rotate_magnitude", "moderately"),
            )
            if plan not in bottle_recovery_plans():
                raise ValueError("V11.1 recovery metadata must contain a nonempty three-axis plan")
            if any(f.input_recovery_plan != plan for f in frames if f.attempt_key == (m["episode_id"], 2)):
                raise ValueError("V11.1 attempt2 action prompt differs from the full real recovery plan")
    counts = {split: Counter() for split in SPLITS}
    for row in rows:
        m = metadata[row["episode_id"], row["attempt_id"]]
        row["experiment_version"] = VERSION_TAG
        row["failure_boundary_frame"] = None
        if m["result"] == "failure":
            _, c = failure_boundary(m)
            row["failure_boundary_frame"] = c
            if not failure_safe_chunk(row["frame_index"], c):
                row.update(
                    trainable=False,
                    chunk_phase_pure=False,
                    raw_chunk_phase_pure=False,
                    exclusion_reason="failure_at_or_after_C" if row["frame_index"] >= c else "crosses_failure_C",
                )
        counts[row["split"]]["candidate_count"] += 1
        counts[row["split"]][row["phase"] + "_candidates"] += 1
        if row["trainable"]:
            counts[row["split"]]["trainable_count"] += 1
            counts[row["split"]][row["phase"] + "_trainable"] += 1
        else:
            counts[row["split"]][row["exclusion_reason"]] += 1
    splits = {
        s: {"execution_indices": [r["global_index"] for r in rows if r["split"] == s and r["trainable"]]}
        for s in SPLITS
    }
    scope = {
        "schema_version": "bottle_v11_1_source_scope_v1",
        "dataset_dir": str(dataset_dir.resolve()),
        "v4_index_file": str(v4_index_file.resolve()),
        "selection": "explicit_selected_dataset_and_hashed_V4_sources_no_recursive_archive_scan",
    }
    summary = {
        "schema_version": "bottle_v11_1_stage_a_summary_v1",
        "experiment_version": VERSION_TAG,
        "policy": POLICY,
        "source_scope": scope,
        "splits": {s: dict(counts[s]) for s in SPLITS},
        "attempt2_count": sum(m["attempt_id"] == 2 for m in profile["attempts"]),
        "token_validation": action_token_summary(rows, frames, dataset_dir=dataset_dir, norm_stats_dir=norm_stats_dir),
    }
    index.update(
        experiment_version=VERSION_TAG,
        target_policy=POLICY,
        source_scope=scope,
        splits=splits,
        action_indices_identity=action_indices_identity(splits),
        summary=summary,
        data_config_hash=sha256_json({"source_data_config_hash": index["data_config_hash"], "policy": POLICY}),
        action_phase_manifest_identity={"count": len(rows), "content_sha256": sha256_json(rows)},
    )
    index["source_files"]["profile"] = {"path": str(profile_path), "sha256": sha256_file(profile_path)}
    return rows, index, summary


def validate_training_index(payload, *, index_path: Path, dataset_dir: Path):
    if payload.get("experiment_version") != VERSION_TAG or payload.get("target_policy") != POLICY:
        raise ValueError("V11.1 needs its own failure-safe Stage A index")
    if not index_path.is_file() or payload.get("training_data_hash") != sha256_json(
        {k: v for k, v in payload.items() if k != "training_data_hash"}
    ):
        raise ValueError("V11.1 Stage A index missing or hash mismatch")
    sources = payload["source_files"]
    for name, source in sources.items():
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError(f"V11.1 Stage A {name} hash mismatch")
    expected_rows, expected, _ = build_artifacts(
        dataset_dir=dataset_dir,
        v4_index_file=Path(sources["v4_training_index"]["path"]),
        norm_stats_dir=Path(sources["norm_stats"]["path"]).parent,
    )
    rows = load_jsonl(Path(sources["action_phase_manifest"]["path"]))
    if rows != expected_rows:
        raise ValueError("V11.1 Stage A action rows differ from source-derived C/R rules")
    for key, value in expected.items():
        if key not in {"source_files", "action_phase_manifest_identity"} and payload.get(key) != value:
            raise ValueError(f"V11.1 Stage A {key} mismatch")
    if {k: sources[k] for k in expected["source_files"]} != expected["source_files"]:
        raise ValueError("V11.1 Stage A source identity mismatch")
    identity = payload["action_phase_manifest_identity"]
    if identity != expected["action_phase_manifest_identity"] | {
        "file_sha256": sources["action_phase_manifest"]["sha256"]
    }:
        raise ValueError("V11.1 Stage A manifest identity mismatch")
    return rows, {r["global_index"]: r for r in rows if r["trainable"]}
