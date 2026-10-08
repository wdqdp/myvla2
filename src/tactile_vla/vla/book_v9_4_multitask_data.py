"""Book V9.4 incremental data counts, captioner provenance and training contract."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re

from tactile_vla.common.labels_v4 import LABEL_FIELDS, LABEL_MAPS, LABEL_SCHEMA_VERSION
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_3_multitask_data import ADJUSTMENT_LABEL_POLICY, NEED_RECOVERY_NEGATIVE_START
from tactile_vla.vla.book_v9_4_memory import (
    MEMORY_LENGTHS,
    MEMORY_POLICY,
    PLAN_MAX_TOKEN_LEN,
    validate_balanced_variants,
    validate_plan_row,
)
from tactile_vla.vla.v4_data import SPLITS, load_jsonl
from tactile_vla.vla.v7_7_multitask_data import TASK_CYCLE
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE, helper_identity


ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/book")
DEFAULT_ACTION_INDEX = ROOT / "outputs/book_stage_a_v9_4/book_stage_a_training_index.json"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_book_stage_a_v9_4_no_history/15000"
DEFAULT_ADJUSTMENT_DIR = ROOT / "outputs/book_adjustment_end_v9_4"
DEFAULT_MULTITASK_DIR = ROOT / "outputs/book_v9_4_multitask"
DEFAULT_INDEX = DEFAULT_MULTITASK_DIR / "book_v9_4_multitask_training_index.json"
DATA_PROFILE = "book_v9_4_five_task_h100"
INDEX_SCHEMA = "tactile_vla_book_v9_4_multitask_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_book_v9_4_multitask_manifest_v1"
CAPTION_KEYS = dict(zip(("area", "Fx", "Fy", "Fz", "Fz_bias", "rotation"), LABEL_FIELDS, strict=True))


def validate_schedule(*, num_steps: int, eval_interval: int, save_interval: int, keep_period: int) -> None:
    for name, value in locals().copy().items():
        if value <= 0 or value % len(TASK_CYCLE):
            raise ValueError(f"{name} must be positive and a multiple of 5 (five-task cycle)")
    if keep_period % save_interval:
        raise ValueError("keep_period must be a multiple of save_interval")


def validate_caption(caption: str) -> None:
    if not caption.startswith("Touch[") or not caption.endswith("]"):
        raise ValueError("V9.4 requires a six-field Touch caption")
    fields = {}
    for item in caption[6:-1].split(";"):
        key, sep, value = item.strip().partition("=")
        if not sep or key in fields:
            raise ValueError("Malformed/duplicate tactile caption field")
        fields[key] = value
    if set(fields) != set(CAPTION_KEYS):
        raise ValueError("V9.4 caption must include Fz_bias and all six fields")
    for key, value in fields.items():
        if value not in LABEL_MAPS[CAPTION_KEYS[key]]:
            raise ValueError(f"Invalid tactile caption {key}={value}")


def load_captioner_provenance(state_dir: Path, *, profile: dict) -> tuple[dict, dict[str, str]]:
    report_path, rebuild_path = state_dir / "caption_report.json", state_dir / "rebuild.json"
    report = json.loads(report_path.read_text())
    rebuild = json.loads(rebuild_path.read_text())
    checkpoint = rebuild["inputs"]["checkpoint"]
    attempts = profile["attempts"]
    frames = sum(row["frame_count"] for row in attempts)
    expected = {
        "checkpoint_sha256": checkpoint["sha256"],
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "window_size": 30,
        "selected_attempts": len(attempts),
        "annotated_attempts": len(attempts),
        "skipped_attempts": 0,
        "annotated_frames": frames,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Caption report does not match the new profile/rebuild; upload may be incomplete")
    if (
        rebuild.get("phase") != "complete"
        or rebuild.get("attempts") != len(attempts)
        or checkpoint.get("label_schema_version") != LABEL_SCHEMA_VERSION
        or checkpoint.get("window_size") != 30
        or tuple(checkpoint.get("label_fields", ())) != LABEL_FIELDS
        or not re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"])
    ):
        raise ValueError("V9.4 requires a complete six-head W30 caption rebuild")
    identity = {
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_fields": list(LABEL_FIELDS),
        "window_size": 30,
        "warmup_policy": report["warmup_policy"],
    }
    return identity, {str(path.resolve()): sha256_file(path) for path in (report_path, rebuild_path)}


def validate_upload_metadata(dataset_dir: Path, profile: dict) -> None:
    """Fast preflight; full Parquet/source hashes are still checked by the V4 builder."""
    info = json.loads((dataset_dir / "meta/info.json").read_text())
    attempts = profile["attempts"]
    if (
        info.get("total_episodes") != len(attempts)
        or info.get("total_frames") != sum(row["frame_count"] for row in attempts)
        or info.get("fps") != 30
    ):
        raise ValueError("LeRobot metadata/profile are not aligned; finish the upload before building V9.4")


def adjustment_counts(profile: dict) -> dict[str, tuple[int, int, int]]:
    counts = {}
    for split in SPLITS:
        attempts = [row for row in profile["attempts"] if row["attempt_id"] == 2 and row["split"] == split]
        if not attempts or any(row["rexecution_frame_index"] < 44 for row in attempts):
            raise ValueError(f"{split}: adjustment attempts must support the original 11/22 sampling buckets")
        positives = 11 * len(attempts)
        negatives = (
            22 * len(attempts) if split == "train" else sum(row["rexecution_frame_index"] - 10 for row in attempts)
        )
        counts[split] = (len(attempts), positives, negatives)
    return counts


def expected_counts(v4_dir: Path, *, expand_plan: bool = True) -> dict[str, dict[str, int]]:
    profile = json.loads((v4_dir / "profile.json").read_text())
    adjustment = adjustment_counts(profile)
    counts = {}
    for split in SPLITS:
        positives = sum(bool(row["need_recovery"]) for row in load_jsonl(v4_dir / f"need/{split}.jsonl"))
        counts[split] = {"adjustment": sum(adjustment[split][1:]), "need": 4 * positives}
        for task, folder in (("failure", "failure_reason"), ("plan", "reasoning")):
            rows = load_jsonl(v4_dir / f"reasoning_manifests/{folder}/{split}.jsonl")
            counts[split][task] = sum(split == "train" or row["frame_offset"] == 14 for row in rows)
            if task == "plan" and expand_plan:
                counts[split][task] *= len(MEMORY_LENGTHS)
        if min(counts[split].values()) <= 0:
            raise ValueError(f"Empty V9.4 task stream in {split}")
    return counts


def target_coverage(manifests: dict[str, list[dict]]) -> dict[str, list[str]]:
    return {
        key: sorted({row[field] for row in manifests[task] if row["split"] == "train"})
        for key, task, field in (
            ("failure_reason", "failure", "target_failure_reason"),
            ("recovery_plan", "plan", "target_recovery_plan"),
        )
    }


def validate_manifest_rows(index: dict, manifests: dict[str, list[dict]], *,
                           data_profile=DATA_PROFILE, manifest_schema=MANIFEST_SCHEMA) -> None:
    failures = {(row["split"], row["global_index"]): row["target_failure_reason"] for row in manifests["failure"]}
    for task, rows in manifests.items():
        if sha256_json(rows) != index["manifest_content_hashes"][task]:
            raise ValueError(f"V9.4 {task} manifest content hash mismatch")
        seen = set()
        for row in rows:
            if row["schema_version"] != manifest_schema or row["data_profile"] != data_profile:
                raise ValueError("V9.4 manifest header mismatch")
            identity = (row["split"], row["global_index"])
            if task == "plan":
                identity += (row["memory_length"],)
                validate_plan_row(row, failures[row["split"], row["global_index"]])
            if identity in seen:
                raise ValueError(f"Duplicate V9.4 {task} frame")
            seen.add(identity)
            touch = next((line[7:] for line in row["prompt"].splitlines() if line.startswith("Touch: ")), None)
            if touch is None and task == "plan":
                # Plan prompts use a different heading; validate their embedded Touch string.
                match = re.search(r"Touch\[[^\]]+\]", row["prompt"])
                touch = match.group() if match else None
            validate_caption(touch or "")
        for split in SPLITS:
            stream = index["splits"][split][task]
            positions, globals_ = stream["manifest_row_indices"], stream["global_indices"]
            if (
                len(positions) != stream["sample_count"]
                or len(globals_) != len(positions)
                or len(set(positions)) != len(positions)
            ):
                raise ValueError(f"V9.4 {split}/{task} stream length/duplicate mismatch")
            selected = []
            for position, global_index in zip(positions, globals_, strict=True):
                if not 0 <= position < len(rows):
                    raise ValueError("V9.4 manifest row outside bounds")
                row = rows[position]
                if row["split"] != split or row["global_index"] != global_index:
                    raise ValueError("V9.4 manifest selected frame identity mismatch")
                selected.append(row)
            if task == "plan":
                validate_balanced_variants(selected)
            if task in ("adjustment", "need"):
                key = "adjustment_end" if task == "adjustment" else "need_recovery"
                labels = Counter(bool(row[key]) for row in selected)
                if (task == "need" and labels[False] != 3 * labels[True]) or (
                    task == "adjustment" and split == "train" and labels[False] != 2 * labels[True]
                ):
                    raise ValueError(f"V9.4 {split}/{task} sampling ratio mismatch")
    if target_coverage(manifests) != index["training_target_coverage"]:
        raise ValueError("V9.4 training target coverage mismatch")
    validate_balanced_variants(manifests["plan"])


def validate_index(index: dict, *, data_profile=DATA_PROFILE, index_schema=INDEX_SCHEMA,
                   manifest_schema=MANIFEST_SCHEMA, counts_fn=expected_counts) -> None:
    expected = {
        "schema_version": index_schema,
        "data_profile": data_profile,
        "prompt_profile": PROMPT_PROFILE,
        "history_policy": helper_identity() | {"idle_perturbation": "none_raw_contiguous"},
        "adjustment_label_policy": ADJUSTMENT_LABEL_POLICY,
        "need_successful_recovery_start": NEED_RECOVERY_NEGATIVE_START,
        "plan_memory_policy": MEMORY_POLICY,
        "model_dependency": "none_data_only",
    }
    if any(index.get(key) != value for key, value in expected.items()) or tuple(index["task_cycle"]) != TASK_CYCLE:
        raise ValueError("V9.4 index training protocol mismatch")
    if "stage_a_checkpoint" in index:
        raise ValueError("Rebuild V9.4 data without binding a Stage A model")
    if index.get("training_data_hash") != sha256_json({k: v for k, v in index.items() if k != "training_data_hash"}):
        raise ValueError("V9.4 training data hash mismatch")
    for name, digest in index["source_hashes"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"V9.4 source changed: {name}")
    v4_dir = Path(index["v4_dir"])
    profile = json.loads((v4_dir / "profile.json").read_text())
    identity, provenance_hashes = load_captioner_provenance(Path(index["incremental_state_dir"]), profile=profile)
    if index["captioner_identity"] != identity or any(
        index["source_hashes"].get(k) != v for k, v in provenance_hashes.items()
    ):
        raise ValueError("V9.4 captioner provenance mismatch")
    counts = counts_fn(v4_dir)
    for split in SPLITS:
        for task, count in counts[split].items():
            if index["splits"][split][task]["sample_count"] != count:
                raise ValueError(f"V9.4 {split}/{task} source-derived sample count mismatch")
    manifests = {}
    for task in ("adjustment", "need", "failure", "plan"):
        path = Path(index[f"{task}_manifest_file"])
        if sha256_file(path) != index[f"{task}_manifest_sha256"]:
            raise ValueError(f"V9.4 {task} manifest file hash mismatch")
        manifests[task] = load_jsonl(path)
    validate_manifest_rows(index, manifests, data_profile=data_profile, manifest_schema=manifest_schema)
    source_plans = {}
    for split in SPLITS:
        for row in load_jsonl(v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl"):
            obs = row["current_observation"]
            key = (split, obs["episode_id"], obs["attempt_id"], row["frame_index"])
            if key in source_plans:
                raise ValueError("Duplicate V9.4 source plan observation")
            source_plans[key] = row
    for row in manifests["plan"]:
        source = source_plans[(row["split"], row["episode_id"], row["attempt_id"], row["frame_index"])]
        target_source = source["target_source"]
        if (
            row["target_recovery_plan"] != source["target_recovery_plan"]
            or row.get("target_source") != target_source
            or row.get("source_variant_id") != source["variant_id"]
            or row["current_real_failure"] != source["current_observation"]["failure_reason"]
            or source["current_observation"]["tactile_caption"] not in row["prompt"]
            or target_source["source_type"] != "real"
            or target_source["episode_id"] != row["episode_id"]
            or target_source["failed_attempt_id"] != 1
            or target_source["plan_attempt_id"] != 2
        ):
            raise ValueError("V9.4 memory changed current observation or adjacent real target provenance")
    token_summary = index["plan_token_validation"]
    if (
        token_summary.get("max_len") != PLAN_MAX_TOKEN_LEN
        or token_summary.get("includes") != "current_qpos_answer_target_eos"
        or token_summary.get("truncation") != "error"
    ):
        raise ValueError("V9.4 plan token validation policy mismatch")
    observed = {}
    for length in MEMORY_LENGTHS:
        lengths = []
        for row in manifests["plan"]:
            if row["memory_length"] != length:
                continue
            tokens = row["plan_token_lengths"]
            if (
                tokens["prefix"] <= 0
                or tokens["target"] <= 0
                or tokens["total"] != tokens["prefix"] + tokens["target"]
                or tokens["total"] > PLAN_MAX_TOKEN_LEN
            ):
                raise ValueError("V9.4 plan prompt/target exceeds token limit or has invalid lengths")
            lengths.append(tokens["total"])
        observed[str(length)] = {"count": len(lengths), "min": min(lengths), "max": max(lengths)}
    if token_summary.get("by_memory_length") != observed:
        raise ValueError("V9.4 plan token summary mismatch")


def validate_stage_a_for_training(checkpoint: Path, index: dict) -> dict:
    """Bind the actual initialization model at training time, never during data construction."""
    checkpoint = checkpoint.expanduser().resolve()
    config_path = checkpoint.parent / "config.json"
    metadata_path = checkpoint / "params/_METADATA"
    if checkpoint.name != "15000" or not metadata_path.is_file():
        raise ValueError("Book V9.4 training requires the Stage A step 15000 checkpoint")
    config = json.loads(config_path.read_text())
    if (
        config.get("data_profile") != "book_stage_a_v1"
        or config.get("num_steps") != 15000
        or config.get("use_state_history") is not False
        or config.get("artifact_identity", {}).get("training_data_hash") != index["action_training_data_hash"]
    ):
        raise ValueError("Book V9.4 Stage A config/data identity differs from the selected training index")
    return {
        "path": str(checkpoint),
        "step": 15000,
        "config_sha256": sha256_file(config_path),
        "params_metadata_sha256": sha256_file(metadata_path),
        "action_training_data_hash": index["action_training_data_hash"],
    }
