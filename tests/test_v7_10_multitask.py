"""Check causal history windows and isolate them from the frozen five-task data."""
# ruff: noqa: E402
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "openpi/src")]

from tactile_vla.vla import v7_10_multitask_data as data
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v7_7_phase_prompt import (
    TimelinePoint, build_phase_prompt, discretize_history, helper_identity, history_sources, sample_episode_history,
)
from tactile_vla.vla.v7_10_phase_prompt import (
    SOURCE_HISTORY_PREFIX, WINDOW_FRAMES, history_policy, history_prefix, sample_history,
)

CAPTION = "Touch[area=full; Fx=near_zero; Fy=near_zero; Fz=negative; rotation=clockwise]"
PLAN = "recovery_plan=move horizontally left moderately, move vertically none moderately."
SLIGHT_PLAN = PLAN.replace("moderately", "slightly", 1)
FAILURE = "failure_reason=rotate left,grasp appropriate."


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def timeline(episode=1, count=120):
    return [TimelinePoint(episode, 1 if i < 80 else 2, i if i < 80 else i - 80,
                          episode * 1000 + i, episode * 100.0 + i / 30, np.full(7, i / 200))
            for i in range(count)]


@pytest.mark.parametrize("window,first,step", [("0s", 110, 0), ("1s", 80, 3), ("2s", 50, 6)])
def test_raw_uniform_sampling_and_attempt_continuity(window, first, step):
    values = timeline()
    points, length = sample_history(values, 110, window)
    assert [point.global_index for point in points] == [1000 + first + step * i for i in range(11)]
    assert length == WINDOW_FRAMES[window]
    assert points[-1] is values[110]
    if window == "2s":
        assert points[0].attempt_id == 1 and points[-1].attempt_id == 2


@pytest.mark.parametrize("window", tuple(WINDOW_FRAMES))
def test_padding_stays_inside_episode_and_never_uses_future(window):
    values = timeline(episode=9)
    points, length = sample_history(values, 2, window)
    expected_offsets = np.linspace(0, WINDOW_FRAMES[window] - 1, 11, dtype=np.int64)
    expected = np.maximum(0, 2 - WINDOW_FRAMES[window] + 1 + expected_offsets)
    assert [point.global_index for point in points] == [values[int(i)].global_index for i in expected]
    assert length == min(3, WINDOW_FRAMES[window])
    assert all(point.timestamp <= values[2].timestamp for point in points)
    assert all(point.episode_id == 9 for point in points)


def test_invalid_history_and_cross_episode_rejected():
    values = timeline()
    with pytest.raises(ValueError, match="history window"):
        sample_history(values, 10, "3s")
    with pytest.raises(IndexError):
        sample_history(values, -1, "1s")
    mixed = timeline(episode=2, count=60) + timeline(episode=3, count=60)
    with pytest.raises(ValueError, match="episode"):
        sample_history(mixed, 70, "2s")


@pytest.fixture
def source_artifacts(tmp_path):
    dataset = tmp_path / "dataset"
    parquet_hashes, rows = {}, {task: [] for task in data.TASKS}
    splits = {}
    stats = StateQuantileStats(q01=[-1] * 7, q99=[1] * 7)
    for episode, split in enumerate(data.SPLITS, 1):
        values = timeline(episode)
        relative = f"data/chunk-000/episode_{episode:06d}.parquet"
        file = dataset / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "index": [point.global_index for point in values], "episode_id": [episode] * len(values),
            "attempt_id": [point.attempt_id for point in values], "frame_index": [point.frame_index for point in values],
            "ros_timestamp": [point.timestamp for point in values], "observation.state": [p.qpos.tolist() for p in values],
        }), file)
        parquet_hashes[relative] = sha256_file(file)
        splits[split] = {"action": {"indices": [values[110].global_index]}}
        for task in data.TASKS:
            selected = []
            for position in ([110, 2] if task == "need" else [110]):
                current = values[position]
                points, length = sample_episode_history(values, position)
                history = discretize_history(points, stats).tolist()
                row = {
                    "schema_version": data.source.MANIFEST_SCHEMA, "data_profile": data.source.DATA_PROFILE,
                    "split": split, "episode_id": episode, "attempt_id": current.attempt_id,
                    "frame_index": current.frame_index, "global_index": current.global_index,
                    "timestamp": current.timestamp, "source": "test_source",
                    "prompt": build_phase_prompt(instruction="Transfer the object.", tactile_caption=CAPTION,
                                                 recovery_plan=PLAN, qpos_h100_11_discrete=history),
                    "qpos_h100_11_discrete": history, "history_sources": history_sources(points),
                    "effective_episode_history_length": length,
                }
                variants = ["factual", "moderately_to_slightly", "slightly_to_moderately"] if task == "adjustment" else [None]
                for variant in variants:
                    item = copy.deepcopy(row)
                    if task == "adjustment":
                        item.update(sample_variant=variant, adjustment_end=variant != "slightly_to_moderately", pair_id=current.global_index)
                        if variant == "moderately_to_slightly":
                            item["prompt"] = item["prompt"].replace(PLAN, SLIGHT_PLAN)
                    elif task == "need":
                        item["need_recovery"] = position == 110
                    elif task == "failure":
                        item.update(target_failure_reason=FAILURE, frame_offset=14)
                    else:
                        item = {key: value for key, value in item.items() if key not in {
                            "qpos_h100_11_discrete", "history_sources", "effective_episode_history_length"
                        }}
                        item.update(prompt="Mode: reasoning. Touch: " + CAPTION + " Memory: " + FAILURE,
                                    target_recovery_plan=PLAN, frame_offset=14, memory_length=1)
                    selected.append(len(rows[task]))
                    rows[task].append(item)
            splits[split][task] = {
                "manifest_row_indices": selected, "global_indices": [rows[task][i]["global_index"] for i in selected],
                "sample_count": len(selected),
            }
    norm = tmp_path / "norm_stats.json"
    write_json(norm, {"norm_stats": {"state": {"q01": [-1] * 7, "q99": [1] * 7}}})
    action = tmp_path / "action.json"
    write_json(action, {"training_data_hash": "action_hash"})
    stage_a = tmp_path / "stage_a" / "10000"
    metadata = stage_a / "params/_METADATA"
    write_json(metadata, {})
    config = stage_a.parent / "config.json"
    write_json(config, {
        "data_profile": "rotation_phase_v7_4_2_adjustment", "use_state_history": False, "prompt_profile": "phase_v2",
        "artifact_identity": {"training_data_hash": "action_hash", "index_sha256": sha256_file(action),
                              "v4_norm_stats_sha256": sha256_file(norm)},
    })
    index = {
        "schema_version": data.source.INDEX_SCHEMA, "data_profile": data.source.DATA_PROFILE,
        "prompt_profile": data.source.PROMPT_PROFILE, "task_cycle": list(data.TASK_CYCLE), "seed": 42,
        "dataset_dir": str(dataset), "stage_a_checkpoint": str(stage_a), "action_index_file": str(action),
        "action_training_data_hash": "action_hash", "selection_hash": "selection",
        "adjustment_sampling_policy": "ratio_1_to_2", "adjustment_label_policy": "factual_true_inclusive_[S-10,S+10]",
        "need_successful_recovery_start": "arm_adjustment_stop_plus_1",
        "history_policy": helper_identity() | {"idle_perturbation": "none_raw_contiguous"},
        "source_hashes": {str(file): sha256_file(file) for file in [norm, action, config, metadata]},
        "v4_source_files": {"lerobot_parquet": parquet_hashes}, "splits": splits,
        "manifest_content_hashes": {task: sha256_json(items) for task, items in rows.items()},
    }
    for task, items in rows.items():
        file = tmp_path / f"source_{task}.jsonl"
        write_rows(file, items)
        index[f"{task}_manifest_file"], index[f"{task}_manifest_sha256"] = str(file), sha256_file(file)
    index["training_data_hash"] = sha256_json(index)
    file = tmp_path / "source.json"
    write_json(file, index)
    return index, rows, file


def persist(source_artifacts, window="1s"):
    source, source_rows, source_path = source_artifacts
    index, rows = data.derive(source, source_rows, source_path, window)
    for task, items in rows.items():
        file = source_path.parent / window / data.MANIFEST_NAMES[task]
        write_rows(file, items)
        index[f"{task}_manifest_file"], index[f"{task}_manifest_sha256"] = str(file), sha256_file(file)
    index["training_data_hash"] = sha256_json(index)
    return index, rows


@pytest.mark.parametrize("window", tuple(WINDOW_FRAMES))
def test_conversion_changes_only_history_and_keeps_counterfactuals(source_artifacts, window):
    source, original, source_path = source_artifacts
    snapshot = copy.deepcopy((source, original))
    index, rows = data.derive(source, original, source_path, window)
    assert index["splits"] == source["splits"]
    assert rows["plan"] == original["plan"]
    for task in data.PHASE_TASKS:
        for old, new in zip(original[task], rows[task], strict=True):
            excluded = {"schema_version", "data_profile", "prompt", "qpos_h100_11_discrete",
                        "qpos_history_11_discrete", "history_sources", "effective_episode_history_length"}
            assert {k: v for k, v in new.items() if k not in excluded} == {
                k: v for k, v in old.items() if k not in excluded}
            assert old["prompt"].partition(SOURCE_HISTORY_PREFIX)[0] == new["prompt"].partition(history_prefix(window))[0]
            assert new["history_sources"][-1]["global_index"] == old["global_index"]
            assert len(new["qpos_history_11_discrete"]) == 11
            if window == "0s":
                assert all(point == old["qpos_h100_11_discrete"][-1] for point in new["qpos_history_11_discrete"])
    assert [row["sample_variant"] for row in rows["adjustment"]] == [row["sample_variant"] for row in original["adjustment"]]
    for split in data.SPLITS:
        variants = [row for row in rows["adjustment"] if row["split"] == split]
        assert all(row["history_sources"] == variants[0]["history_sources"] for row in variants)
    assert (source, original) == snapshot


@pytest.mark.parametrize("tamper", ["label", "history", "selection", "action", "plan", "counterfactual"])
def test_rehashed_changes_outside_history_policy_rejected(source_artifacts, tamper):
    index, rows = persist(source_artifacts)
    data.validate_index(index)
    task = "adjustment"
    if tamper == "label":
        rows[task][0]["adjustment_end"] = False
    elif tamper == "history":
        rows[task][0]["qpos_history_11_discrete"][0][0] += 1
    elif tamper == "selection":
        index["splits"]["train"][task]["manifest_row_indices"].reverse()
    elif tamper == "action":
        index["splits"]["train"]["action"]["indices"] = []
    elif tamper == "plan":
        task = "plan"
        rows[task][0]["target_recovery_plan"] = SLIGHT_PLAN
    else:
        rows[task].pop(1)
    file = Path(index[f"{task}_manifest_file"])
    write_rows(file, rows[task])
    index[f"{task}_manifest_sha256"] = sha256_file(file)
    index["manifest_content_hashes"][task] = sha256_json(rows[task])
    index["training_data_hash"] = sha256_json({k: v for k, v in index.items() if k != "training_data_hash"})
    with pytest.raises(ValueError):
        data.validate_index(index)


def test_raw_qpos_hash_change_rejected(source_artifacts):
    source, _, _ = source_artifacts
    file = Path(source["dataset_dir"]) / next(iter(source["v4_source_files"]["lerobot_parquet"]))
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="parquet changed"):
        data.load_timelines(source)


def test_builder_all_windows_and_no_overwrite(source_artifacts, tmp_path, monkeypatch):
    from scripts import build_v7_10_multitask_data as builder

    monkeypatch.setattr(data, "audit_tokens", lambda *args: {"checked": True})
    original, _, file = source_artifacts
    output = tmp_path / "built"
    arguments = ["--source-index", str(file), "--output-dir", str(output)]
    assert builder.main([*arguments, "--dry-run"]) == 0
    assert not output.exists()
    assert builder.main(arguments) == 0
    for window in WINDOW_FRAMES:
        index = json.loads((output / window / "v7_10_multitask_training_index.json").read_text())
        data.validate_index(index)
        assert Path(index["plan_manifest_file"]).read_bytes() == Path(original["plan_manifest_file"]).read_bytes()
        assert data.validate_stage_a_for_training(Path(index["stage_a_checkpoint"]), index)["step"] == 10000
    with pytest.raises(FileExistsError):
        builder.main(arguments)


@pytest.fixture
def entry(monkeypatch):
    from scripts import train_vla_multitask_v7_10 as training

    for obj in (training.trainer, training.base):
        for name, value in list(vars(obj).items()):
            if not name.startswith("__"):
                monkeypatch.setattr(obj, name, value)
    training.configure_version()
    return training


@pytest.mark.parametrize("window", tuple(WINDOW_FRAMES))
def test_training_windows_defaults_and_manual_parameters(entry, window):
    args = entry.parse_args([f"--history-window={window}"])
    assert (args.num_steps, args.batch_size, args.fsdp_devices) == (5000, 8, 2)
    assert (args.eval_interval, args.save_interval, args.keep_period, args.eval_max_need_samples) == (1000, 1000, 1000, 4096)
    assert args.stage_a_checkpoint == data.DEFAULT_STAGE_A
    assert args.index_file == data.default_index(window)
    assert args.run_name == data.run_name(window)
    assert args.prompt_profile == history_policy(window)["prompt_profile"]
    assert not args.use_state_history and args.state_history_len == 0
    custom = entry.parse_args(["--history-window", window, "--num-steps=2000", "--batch-size=16", "--fsdp-devices=1"])
    assert (custom.num_steps, custom.batch_size, custom.fsdp_devices) == (2000, 16, 1)
    entry.configure_training()
    assert entry.base.TASK_CYCLE == data.TASK_CYCLE
    assert entry.base.evaluate_adjustment is entry.evaluate_adjustment
    assert entry.base.StageBV3Model is entry.trainer.V77MultitaskModel


def test_resume_rejects_other_window_and_export_records_history(entry, tmp_path, monkeypatch):
    args = entry.parse_args(["--history-window", "1s"])
    entry.configure_training()
    entry.base.EXTRA_CONFIG.update(history_window="1s", history_policy=history_policy("1s"))
    saved = vars(args) | copy.deepcopy(entry.base.EXTRA_CONFIG) | {"precision": "float32"}
    entry.validate_resume(saved, args, actual_precision="float32")
    saved["history_window"] = "2s"
    with pytest.raises(ValueError, match="resume"):
        entry.validate_resume(saved, args, actual_precision="float32")
    step_dir = tmp_path / "1000"
    step_dir.mkdir()
    monkeypatch.setattr(entry, "_BASE_EXPORT", lambda *args: write_json(step_dir / "v7_10_export.json", {"exports": {}}))
    entry.export_checkpoint(tmp_path, None, 1000, None)
    metadata = json.loads((step_dir / "v7_10_export.json").read_text())
    assert metadata["history_window"] == "1s"
    assert metadata["history_policy"]["history_frames"] == 31
    assert metadata["data_profile"] == data.data_profile("1s")


def test_cli_requires_explicit_window_and_rejects_dense_history(entry):
    with pytest.raises(SystemExit):
        entry.parse_args([])
    with pytest.raises(ValueError, match="protocol mismatch"):
        entry.parse_args(["--history-window", "1s", "--use-state-history"])
