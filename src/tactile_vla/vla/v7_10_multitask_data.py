"""Frozen V7.7.2 samples with shorter qpos histories for V7.10 retraining."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

import numpy as np

from tactile_vla.vla import v7_7_2_multitask_data as source
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v5_3_adjustment_end_data import load_state_quantiles
from tactile_vla.vla.v7_7_multitask_data import TASK_CYCLE, load_jsonl
from tactile_vla.vla.v7_7_phase_prompt import TimelinePoint, discretize_history, history_sources
from tactile_vla.vla.v7_10_phase_prompt import (
    history_policy, prompt_profile, replace_history, sample_history,
)

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data/black_box")
VERSION_TAG = "v7_10"
INDEX_SCHEMA = "tactile_vla_v7_10_multitask_training_index_v1"
MANIFEST_SCHEMA = "tactile_vla_v7_10_multitask_manifest_v1"
DEFAULT_SOURCE_INDEX = ROOT / "outputs/rotation_v7_7_2_multitask/v7_7_2_multitask_training_index.json"
DEFAULT_OUTPUT = ROOT / "outputs/rotation_v7_10_multitask"
DEFAULT_STAGE_A = ROOT / "outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_2_no_history/10000"
TASKS = TASK_CYCLE[1:]
PHASE_TASKS = ("adjustment", "need", "failure")
SPLITS = ("train", "val", "test")
MANIFEST_NAMES = {
    "adjustment": "adjustment_end_manifest.jsonl", "need": "need_recovery_manifest.jsonl",
    "failure": "failure_reason_manifest.jsonl", "plan": "recovery_plan_manifest.jsonl",
}
COUNTERFACTUAL_POLICY = "unchanged_v7_7_2_train_val_test_including_counterfactuals"


def data_profile(window):
    return f"rotation_phase_v7_10_five_task_{window}_qpos_history"


def default_index(window):
    return DEFAULT_OUTPUT / window / "v7_10_multitask_training_index.json"


def run_name(window):
    return f"pi05_rotation_v7_10_five_task_{window}_no_history"


def norm_stats_file(index):
    paths = [Path(path) for path in index["source_hashes"] if Path(path).name == "norm_stats.json"]
    if len(paths) != 1:
        raise ValueError("V7.10 requires exactly one source norm stats identity")
    return paths[0]


def load_source(path):
    index = json.loads(Path(path).read_text())
    source.validate_index(index)
    rows = {}
    for task in TASKS:
        file = Path(index[f"{task}_manifest_file"])
        rows[task] = load_jsonl(file)
        if (sha256_file(file) != index[f"{task}_manifest_sha256"]
                or sha256_json(rows[task]) != index["manifest_content_hashes"][task]):
            raise ValueError(f"V7.10 source {task} manifest hash mismatch")
        for split in SPLITS:
            stream = index["splits"][split][task]
            for position, global_index in zip(stream["manifest_row_indices"], stream["global_indices"], strict=True):
                row = rows[task][position]
                if row["split"] != split or row["global_index"] != global_index:
                    raise ValueError(f"V7.10 source {split}/{task} selection mismatch")
    return index, rows


def load_timelines(index):
    """Read the raw 30 Hz qpos, not the already sampled H100 points."""
    import pyarrow.parquet as pq

    root = Path(index["dataset_dir"])
    expected_files = index["v4_source_files"]["lerobot_parquet"]
    actual_files = {str(path.relative_to(root)) for path in (root / "data").glob("chunk-*/episode_*.parquet")}
    if actual_files != set(expected_files):
        raise ValueError("V7.10 raw LeRobot parquet file set differs from V7.7.2")
    columns = ["index", "episode_id", "attempt_id", "frame_index", "ros_timestamp", "observation.state"]
    timelines = defaultdict(list)
    for relative, expected_hash in sorted(expected_files.items()):
        file = root / relative
        if sha256_file(file) != expected_hash:
            raise ValueError(f"V7.10 source qpos parquet changed: {file}")
        values = pq.read_table(file, columns=columns).to_pydict()
        for i, global_index in enumerate(values["index"]):
            qpos = np.asarray(values["observation.state"][i], dtype=np.float64)
            timestamp = float(values["ros_timestamp"][i])
            if qpos.shape != (7,) or not np.isfinite(qpos).all() or not np.isfinite(timestamp):
                raise ValueError(f"V7.10 non-finite qpos/time at global index {global_index}")
            episode = int(values["episode_id"][i])
            timelines[episode].append(TimelinePoint(
                episode, int(values["attempt_id"][i]), int(values["frame_index"][i]),
                int(global_index), timestamp, qpos,
            ))
    positions = {}
    for timeline in timelines.values():
        timeline.sort(key=lambda point: (point.timestamp, point.global_index))
        for i, point in enumerate(timeline):
            if point.global_index in positions:
                raise ValueError("V7.10 duplicate raw global index")
            if i and (timeline[i - 1].global_index >= point.global_index
                      or timeline[i - 1].attempt_id > point.attempt_id):
                raise ValueError("V7.10 non-monotonic episode/attempt timeline")
            positions[point.global_index] = (timeline, i)
    return positions


def convert_manifests(original_rows, positions, stats, window):
    rows = copy.deepcopy(original_rows)
    cache = {}
    for task in PHASE_TASKS:
        for row in rows[task]:
            global_index = int(row["global_index"])
            timeline, position = positions[global_index]
            current = timeline[position]
            if (current.episode_id, current.attempt_id, current.frame_index, current.timestamp) != (
                row["episode_id"], row["attempt_id"], row["frame_index"], row["timestamp"],
            ):
                raise ValueError(f"V7.10 raw qpos/manifest identity mismatch: {global_index}")
            if global_index not in cache:
                points, length = sample_history(timeline, position, window)
                cache[global_index] = (discretize_history(points, stats).tolist(), history_sources(points), length)
            discrete, sources, length = cache[global_index]
            row["prompt"] = replace_history(row["prompt"], discrete, window)
            row.update(
                schema_version=MANIFEST_SCHEMA, data_profile=data_profile(window),
                qpos_history_11_discrete=discrete, history_sources=sources,
                effective_episode_history_length=length,
            )
            row.pop("qpos_h100_11_discrete")
    return rows


def derive(original, original_rows, source_path, window, *, positions=None):
    policy = history_policy(window)
    if positions is None:
        positions = load_timelines(original)
    rows = convert_manifests(original_rows, positions, load_state_quantiles(norm_stats_file(original)), window)
    index = copy.deepcopy(original)
    index.pop("training_data_hash", None)
    for task in TASKS:
        index.pop(f"{task}_manifest_file", None)
        index.pop(f"{task}_manifest_sha256", None)
    path = str(Path(source_path).resolve())
    index.update(
        schema_version=INDEX_SCHEMA, experiment_version=VERSION_TAG,
        data_profile=data_profile(window), prompt_profile=prompt_profile(window),
        history_window=window, history_policy=policy, counterfactual_policy=COUNTERFACTUAL_POLICY,
        source_multitask_index_file=path, source_multitask_index_sha256=sha256_file(path),
        source_multitask_training_data_hash=original["training_data_hash"],
        manifest_content_hashes={task: sha256_json(items) for task, items in rows.items()},
    )
    index["source_hashes"][path] = sha256_file(path)
    for task in TASKS:
        index["source_hashes"][original[f"{task}_manifest_file"]] = original[f"{task}_manifest_sha256"]
    return index, rows


def validate_index(index, *, positions=None):
    window = index.get("history_window")
    policy = history_policy(window)
    if (index.get("schema_version"), index.get("experiment_version"), index.get("data_profile")) != (
        INDEX_SCHEMA, VERSION_TAG, data_profile(window),
    ):
        raise ValueError("V7.10 requires a window-specific V7.10 index")
    if index.get("training_data_hash") != sha256_json({k: v for k, v in index.items() if k != "training_data_hash"}):
        raise ValueError("V7.10 index hash mismatch")
    if index.get("history_policy") != policy:
        raise ValueError("V7.10 history policy mismatch")
    path = Path(index["source_multitask_index_file"])
    if sha256_file(path) != index["source_multitask_index_sha256"]:
        raise ValueError("V7.10 source V7.7.2 index changed")
    original, original_rows = load_source(path)
    expected, rows = derive(original, original_rows, path, window, positions=positions)
    excluded = {"training_data_hash"}
    for task in TASKS:
        excluded.update((f"{task}_manifest_file", f"{task}_manifest_sha256"))
        file = Path(index[f"{task}_manifest_file"])
        if sha256_file(file) != index[f"{task}_manifest_sha256"] or load_jsonl(file) != rows[task]:
            raise ValueError(f"V7.10 {task} differs from the exact source history conversion")
    if {k: v for k, v in index.items() if k not in excluded} != expected:
        raise ValueError("V7.10 changes source labels/sampling/action/plan or history identity")


def validate_stage_a_for_training(checkpoint, index):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint != Path(index["stage_a_checkpoint"]).resolve():
        raise ValueError("V7.10 must initialize from the source V7.4.2 Stage A checkpoint")
    config_file, metadata = checkpoint.parent / "config.json", checkpoint / "params/_METADATA"
    config = json.loads(config_file.read_text())
    identity = config.get("artifact_identity", {})
    action_file = Path(index["action_index_file"])
    if (config.get("data_profile") != "rotation_phase_v7_4_2_adjustment"
            or config.get("use_state_history") is not False or config.get("prompt_profile") != "phase_v2"
            or identity.get("training_data_hash") != index["action_training_data_hash"]
            or identity.get("index_sha256") != sha256_file(action_file)
            or identity.get("v4_norm_stats_sha256") != sha256_file(norm_stats_file(index))
            or not metadata.is_file()):
        raise ValueError("V7.10 requires matching V7.4.2 Stage A data/norm/config/params")
    return {
        "path": str(checkpoint), "step": int(checkpoint.name), "config_sha256": sha256_file(config_file),
        "params_metadata_sha256": sha256_file(metadata),
        "action_training_data_hash": index["action_training_data_hash"],
    }


def summary(index, rows):
    history_audit = {}
    for task in PHASE_TASKS:
        history_audit[task] = {}
        for split in SPLITS:
            selected = [rows[task][i] for i in index["splits"][split][task]["manifest_row_indices"]]
            spans = [float(row["timestamp"] - row["history_sources"][0]["timestamp"]) for row in selected]
            history_audit[task][split] = {
                "sample_count": len(selected),
                "padded_count": sum(row["effective_episode_history_length"] < index["history_policy"]["history_frames"]
                                    for row in selected),
                "cross_attempt_count": sum(row["history_sources"][0]["attempt_id"] != row["attempt_id"]
                                           for row in selected),
                "actual_span_seconds": {"min": min(spans), "p50": float(np.median(spans)), "max": max(spans)},
            }
    return {
        "experiment_version": VERSION_TAG, "history_policy": index["history_policy"],
        "counterfactual_policy": COUNTERFACTUAL_POLICY, "stage_a_checkpoint": index["stage_a_checkpoint"],
        "source_multitask_training_data_hash": index["source_multitask_training_data_hash"],
        "selected_counts": {
            split: {"action": len(index["splits"][split]["action"]["indices"])}
            | {task: index["splits"][split][task]["sample_count"] for task in TASKS} for split in SPLITS
        },
        "adjustment_variants": {
            split: dict(Counter(rows["adjustment"][i]["sample_variant"]
                                for i in index["splits"][split]["adjustment"]["manifest_row_indices"]))
            for split in SPLITS
        },
        "history_audit": history_audit,
    }


def audit_tokens(index, rows):
    """Check actual state+prompt+answer budgets, including all source rows."""
    from openpi.models.tokenizer import PaligemmaTokenizer
    from tactile_vla.vla.v5_3_adjustment_end_data import scan_selected_qpos
    from tactile_vla.vla.v5_3_phase_change import normalize_state_qpos

    qpos = scan_selected_qpos(
        dataset_dir=Path(index["dataset_dir"]),
        selected_episode_ids={row["episode_id"] for items in rows.values() for row in items},
    )
    stats = load_state_quantiles(norm_stats_file(index))
    tokenizer = PaligemmaTokenizer(512)
    report = {}
    for task, items in rows.items():
        lengths = []
        for row in items:
            state = normalize_state_qpos(qpos[row["global_index"]].astype(np.float32), stats)
            text = row.get("target_failure_reason", row.get("target_recovery_plan"))
            target = None if text is None else tokenizer.encode_text(text, add_eos=True)
            _, mask, _, _, _ = tokenizer.tokenize_structured_response(
                row["prompt"], state, target, max_len=320 if task == "plan" else 512,
            )
            lengths.append(int(mask.sum()))
        report[task] = {"count": len(lengths), "min": min(lengths), "max": max(lengths)}
    return report
