"""Exact no-touch ablation of V7.7.2; no relabeling, resampling or new Stage A."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from tactile_vla.vla import v7_7_2_multitask_data as source
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v7_7_multitask_data import TASK_CYCLE, load_jsonl
from tactile_vla.vla.v7_8_phase_prompt import INPUT_POLICY, PROMPT_PROFILE, remove_touch, validate_no_touch

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
VERSION_TAG = "v7_8"
DATA_PROFILE = "rotation_phase_v7_8_five_task_h100_visual"
INDEX_SCHEMA = "tactile_vla_v7_8_multitask_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_v7_8_multitask_manifest_v1"
DEFAULT_SOURCE_INDEX = ROOT / "outputs/rotation_v7_7_2_multitask/v7_7_2_multitask_training_index.json"
DEFAULT_OUTPUT = ROOT / "outputs/rotation_v7_8_multitask"
DEFAULT_INDEX = DEFAULT_OUTPUT / "v7_8_multitask_training_index.json"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_2_no_history/10000"
RUN_NAME = "pi05_rotation_v7_8_five_task_h100_no_history"
TASKS = TASK_CYCLE[1:]
SPLITS = ("train", "val", "test")
MANIFEST_NAMES = {
    "adjustment": "adjustment_end_manifest.jsonl",
    "need": "need_recovery_manifest.jsonl",
    "failure": "failure_reason_manifest.jsonl",
    "plan": "recovery_plan_manifest.jsonl",
}


def load_source(path):
    index = json.loads(Path(path).read_text())
    source.validate_index(index)
    rows = {}
    for task in TASKS:
        manifest = Path(index[f"{task}_manifest_file"])
        rows[task] = load_jsonl(manifest)
        if (sha256_file(manifest) != index[f"{task}_manifest_sha256"]
                or sha256_json(rows[task]) != index["manifest_content_hashes"][task]):
            raise ValueError(f"V7.8 source V7.7.2 {task} manifest hash mismatch")
        for split in SPLITS:
            stream = index["splits"][split][task]
            for position, global_index in zip(stream["manifest_row_indices"], stream["global_indices"], strict=True):
                row = rows[task][position]
                if row["split"] != split or row["global_index"] != global_index:
                    raise ValueError(f"V7.8 source {split}/{task} manifest selection mismatch")
    return index, rows


def convert_manifests(original):
    rows = copy.deepcopy(original)
    for task in TASKS:
        for row in rows[task]:
            row.update(schema_version=MANIFEST_SCHEMA, data_profile=DATA_PROFILE)
            row["prompt"] = remove_touch(row["prompt"], task=task)
            row.pop("token_lengths", None)
    return rows


def norm_stats_file(index):
    paths = [Path(p) for p in index["source_hashes"] if Path(p).name == "norm_stats.json"]
    if len(paths) != 1:
        raise ValueError("V7.8 requires exactly one source V7.7.2 norm stats identity")
    return paths[0]


def validate_tokens(index, rows):
    """Measure actual visual/state prompts plus targets, with the source normalizer."""
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.transforms import Normalize
    from tactile_vla.vla.v5_3_adjustment_end_data import scan_selected_qpos

    qpos = scan_selected_qpos(
        dataset_dir=Path(index["dataset_dir"]),
        selected_episode_ids={r["episode_id"] for items in rows.values() for r in items},
    )
    tokenizer = PaligemmaTokenizer(512)
    normalizer = Normalize(normalize.load(norm_stats_file(index).parent), use_quantiles=True)
    report = {}
    for task, items in rows.items():
        lengths = {}
        for row in items:
            validate_no_touch(row["prompt"])
            state = normalizer({"state": np.asarray(qpos[row["global_index"]], dtype=np.float32)})["state"]
            text = row.get("target_failure_reason", row.get("target_recovery_plan"))
            target = None if text is None else tokenizer.encode_text(text, add_eos=True)
            _, mask, _, _, prefix = tokenizer.tokenize_structured_response(
                row["prompt"], state, target, max_len=320 if task == "plan" else 512,
            )
            row["token_lengths"] = {
                "prefix": int(prefix), "total": int(mask.sum()),
                "target": 0 if target is None else len(target),
            }
            lengths.setdefault(str(row.get("memory_length", "all")), []).append(int(mask.sum()))
        report[task] = {k: {"count": len(v), "min": min(v), "max": max(v)} for k, v in lengths.items()}
    return report


def derive(original, original_rows, source_path):
    rows = convert_manifests(original_rows)
    tokens = validate_tokens(original, rows)
    index = copy.deepcopy(original)
    index.pop("training_data_hash", None)
    for task in TASKS:
        index.pop(f"{task}_manifest_file", None)
        index.pop(f"{task}_manifest_sha256", None)
    path = str(Path(source_path).resolve())
    index.update(
        schema_version=INDEX_SCHEMA, experiment_version=VERSION_TAG,
        data_profile=DATA_PROFILE, prompt_profile=PROMPT_PROFILE,
        input_policy=copy.deepcopy(INPUT_POLICY), captioner_identity_scope="offline_label_provenance_only",
        history_policy=original["history_policy"] | {"prompt_profile": PROMPT_PROFILE},
        source_multitask_index_file=path, source_multitask_index_sha256=sha256_file(path),
        source_multitask_training_data_hash=original["training_data_hash"],
        token_validation=tokens,
        manifest_content_hashes={task: sha256_json(items) for task, items in rows.items()},
    )
    index["source_hashes"][path] = sha256_file(path)
    # Source manifests are part of the ablation identity, even if their locations are unchanged.
    for task in TASKS:
        index["source_hashes"][original[f"{task}_manifest_file"]] = original[f"{task}_manifest_sha256"]
    return index, rows


def validate_index(index):
    if (index.get("schema_version"), index.get("experiment_version"), index.get("data_profile")) != (
        INDEX_SCHEMA, VERSION_TAG, DATA_PROFILE,
    ):
        raise ValueError("V7.8 requires its own visual/state index")
    if index.get("training_data_hash") != sha256_json({k: v for k, v in index.items() if k != "training_data_hash"}):
        raise ValueError("V7.8 index hash mismatch")
    path = Path(index["source_multitask_index_file"])
    if sha256_file(path) != index["source_multitask_index_sha256"]:
        raise ValueError("V7.8 source V7.7.2 index changed")
    original, original_rows = load_source(path)
    expected, rows = derive(original, original_rows, path)
    excluded = {"training_data_hash"}
    for task in TASKS:
        excluded.update((f"{task}_manifest_file", f"{task}_manifest_sha256"))
        manifest = Path(index[f"{task}_manifest_file"])
        if sha256_file(manifest) != index[f"{task}_manifest_sha256"] or load_jsonl(manifest) != rows[task]:
            raise ValueError(f"V7.8 {task} differs from the exact V7.7.2 no-touch conversion")
    if {k: v for k, v in index.items() if k not in excluded} != expected:
        raise ValueError("V7.8 labels/selection/history/action/initialization differ from the source ablation")


def validate_stage_a_for_training(checkpoint, index):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint != Path(index["stage_a_checkpoint"]).resolve():
        raise ValueError("V7.8 must use the same Stage A checkpoint as its source V7.7.2 index")
    config_file, metadata = checkpoint.parent / "config.json", checkpoint / "params/_METADATA"
    config = json.loads(config_file.read_text())
    action_file = Path(index["action_index_file"])
    action = json.loads(action_file.read_text())
    identity = config.get("artifact_identity", {})
    norm_file = norm_stats_file(index)
    if (config.get("data_profile") != "rotation_phase_v7_4_2_adjustment"
            or config.get("use_state_history") is not False
            or config.get("prompt_profile") != "phase_v2"
            or identity.get("training_data_hash") != index["action_training_data_hash"]
            or identity.get("index_sha256") != sha256_file(action_file)
            or identity.get("v4_norm_stats_sha256") != sha256_file(norm_file)
            or action.get("training_data_hash") != index["action_training_data_hash"]
            or not metadata.is_file()):
        raise ValueError("V7.8 requires source V7.4.2 Stage A action/data/norm identity, not multitask weights")
    return {
        "path": str(checkpoint), "step": int(checkpoint.name),
        "config_sha256": sha256_file(config_file), "params_metadata_sha256": sha256_file(metadata),
        "action_training_data_hash": index["action_training_data_hash"],
    }


def validate_prompts(index):
    for task in TASKS:
        for row in load_jsonl(Path(index[f"{task}_manifest_file"])):
            validate_no_touch(row["prompt"])


def summary(index, rows):
    return {
        "experiment_version": VERSION_TAG, "input_policy": INPUT_POLICY,
        "source_multitask_training_data_hash": index["source_multitask_training_data_hash"],
        "stage_a_checkpoint": index["stage_a_checkpoint"],
        "selected_counts": {
            split: {"action": len(index["splits"][split]["action"]["indices"])}
            | {task: index["splits"][split][task]["sample_count"] for task in TASKS}
            for split in SPLITS
        },
        "manifest_counts": {task: len(items) for task, items in rows.items()},
        "adjustment_labels": {
            split: {
                "positive": sum(bool(rows["adjustment"][i]["adjustment_end"]) for i in stream["manifest_row_indices"]),
                "negative": sum(not rows["adjustment"][i]["adjustment_end"] for i in stream["manifest_row_indices"]),
            }
            for split, tasks in index["splits"].items() for stream in [tasks["adjustment"]]
        },
        "token_validation": index["token_validation"],
    }
