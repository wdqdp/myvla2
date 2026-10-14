"""V7.8 removes only current Touch, retaining the exact V7.7.2 experiment."""

# ruff: noqa: E402
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src")]

from tactile_vla.vla import v7_8_multitask_data as data
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt as tactile_plan
from tactile_vla.vla.v7_7_multitask_data import V77ManifestDataset
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt as tactile_phase, helper_identity
from tactile_vla.vla.v7_8_phase_prompt import (
    INPUT_POLICY, PROMPT_PROFILE, build_phase_prompt, build_recovery_prompt, remove_touch, validate_no_touch,
)

CAPTION = "Touch[area=full; Fx=near_zero; Fy=near_zero; Fz=negative; rotation=counterclockwise]"
INSTRUCTION = "Pick up and transfer the object stably."
FAILURE = "failure_reason=rotate left,grasp appropriate."
PLAN = "recovery_plan=move horizontally left moderately, move vertically none moderately."
HISTORY = np.arange(77).reshape(11, 7).tolist()


def memory(length):
    return [{"recovery_plan": "initial plan" if i == 0 else PLAN, "failure_reason": FAILURE}
            for i in range(length)]


def phase(plan=PLAN):
    return tactile_phase(instruction=INSTRUCTION, tactile_caption=CAPTION, recovery_plan=plan,
                         qpos_h100_11_discrete=HISTORY)


@pytest.mark.parametrize("task", ["need", "failure", "adjustment"])
def test_phase_strip_preserves_task_plan_and_h100(task):
    assert remove_touch(phase(), task=task) == build_phase_prompt(
        instruction=INSTRUCTION, recovery_plan=PLAN, qpos_h100_11_discrete=HISTORY,
    )


@pytest.mark.parametrize("length", [1, 2, 4])
def test_plan_strip_preserves_failure_plan_semantics(length):
    original = tactile_plan(instruction=INSTRUCTION, failed_tactile_caption=CAPTION,
                            failure_recovery_memory=memory(length), prompt_profile=MINIMAL_PROMPT_PROFILE)
    result = remove_touch(original, task="plan")
    assert result == build_recovery_prompt(instruction=INSTRUCTION, failure_recovery_memory=memory(length))
    assert result.count(FAILURE) == length
    assert "rotate left" in result and "Touch" not in result


@pytest.mark.parametrize("prompt", [CAPTION, "Touch: none", "Touch:", "rotation=none", "Fy=positive",
                                    "tactile_caption=none", "Touch [area=none]"])
def test_neutral_and_hidden_touch_are_rejected(prompt):
    with pytest.raises(ValueError, match="tactile"):
        validate_no_touch(prompt)


def test_missing_duplicate_touch_and_unknown_tasks_fail():
    for prompt in ("Mode: phase\nTask: rotation", phase() + "\nTouch: " + CAPTION + "\n"):
        with pytest.raises(ValueError):
            remove_touch(prompt, task="need")
    with pytest.raises(ValueError):
        remove_touch("Mode: reasoning. " + CAPTION + " Failure-recovery memory: " + CAPTION, task="plan")
    with pytest.raises(ValueError, match="Unsupported"):
        remove_touch(phase(), task="action")


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


@pytest.fixture
def source_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "validate_tokens", lambda index, rows: {"checked": True})
    rows = {task: [] for task in data.TASKS}
    splits = {split: {"action": {"indices": [10, 20]}} for split in data.SPLITS}
    for split in splits:
        for task in data.TASKS:
            g = len(rows[task]) * 10
            row = {
                "schema_version": data.source.MANIFEST_SCHEMA, "data_profile": data.source.DATA_PROFILE,
                "split": split, "global_index": g, "episode_id": 271, "attempt_id": 2,
                "frame_index": 100, "timestamp": 123.0, "source": "offline_real",
                "prompt": phase(), "qpos_h100_11_discrete": HISTORY,
                "history_sources": [{"global_index": g, "frame_index": 100}],
                "effective_episode_history_length": 100,
            }
            if task == "adjustment":
                row.update(adjustment_end=False, arm_adjustment_start_frame=50, arm_adjustment_stop_frame=200,
                           sample_variant="factual", pair_id=g, prompt_magnitude="slightly")
            elif task == "need":
                row["need_recovery"] = True
            elif task == "failure":
                row.update(target_failure_reason=FAILURE, frame_offset=14)
            else:
                row.update(memory_length=2, failure_recovery_memory=memory(2), target_recovery_plan=PLAN,
                           frame_offset=14)
                row["prompt"] = tactile_plan(instruction=INSTRUCTION, failed_tactile_caption=CAPTION,
                                              failure_recovery_memory=memory(2),
                                              prompt_profile=MINIMAL_PROMPT_PROFILE)
            position = len(rows[task])
            rows[task].append(row)
            if task == "adjustment":
                rows[task].append(row | {"sample_variant": "slightly_to_moderately", "prompt_magnitude": "moderately",
                                        "prompt": phase(PLAN)})
            selected = list(range(position, len(rows[task])))
            splits[split][task] = {
                "manifest_row_indices": selected, "global_indices": [g] * len(selected),
                "sample_count": len(selected),
            }
    stage_a = tmp_path / "stage_a" / "10000"
    (stage_a / "params").mkdir(parents=True)
    (stage_a / "params/_METADATA").write_text("{}")
    norm = tmp_path / "norm_stats.json"
    norm.write_text("{}")
    action = tmp_path / "action.json"
    action.write_text(json.dumps({"training_data_hash": "a" * 64}))
    config = {
        "data_profile": "rotation_phase_v7_4_2_adjustment", "use_state_history": False,
        "prompt_profile": "phase_v2",
        "artifact_identity": {"training_data_hash": "a" * 64, "index_sha256": sha256_file(action),
                              "v4_norm_stats_sha256": sha256_file(norm)},
    }
    config_file = stage_a.parent / "config.json"
    config_file.write_text(json.dumps(config))
    index = {
        "schema_version": data.source.INDEX_SCHEMA, "data_profile": data.source.DATA_PROFILE,
        "prompt_profile": data.source.PROMPT_PROFILE, "task_cycle": list(data.TASK_CYCLE),
        "seed": 42, "dataset_dir": str(tmp_path / "dataset"), "selection_hash": "selection",
        "stage_a_checkpoint": str(stage_a), "action_index_file": str(action),
        "action_training_data_hash": "a" * 64,
        "adjustment_sampling_policy": "ratio_1_to_2",
        "adjustment_label_policy": "factual_true_inclusive_[S-10,S+10]",
        "need_successful_recovery_start": "arm_adjustment_stop_plus_1",
        "history_policy": helper_identity() | {"idle_perturbation": "none_raw_contiguous"},
        "splits": splits, "manifest_content_hashes": {t: sha256_json(r) for t, r in rows.items()},
        "source_hashes": {str(p): sha256_file(p) for p in [norm, action, config_file, stage_a / "params/_METADATA"]},
    }
    for task in data.TASKS:
        file = tmp_path / ("source_" + task + ".jsonl")
        write_rows(file, rows[task])
        index[f"{task}_manifest_file"], index[f"{task}_manifest_sha256"] = str(file), sha256_file(file)
    index["training_data_hash"] = sha256_json(index)
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(index))
    return index, rows, source_file


def derived_artifacts(original, rows, source_file):
    index, converted = data.derive(original, rows, source_file)
    for task in data.TASKS:
        file = source_file.parent / ("visual_" + task + ".jsonl")
        write_rows(file, converted[task])
        index[f"{task}_manifest_file"], index[f"{task}_manifest_sha256"] = str(file), sha256_file(file)
    index["training_data_hash"] = sha256_json(index)
    return index, converted


def test_conversion_keeps_original_pairs_labels_history_memory_and_order(source_artifacts):
    index, rows, source_file = source_artifacts
    before_index, before_rows = copy.deepcopy(index), copy.deepcopy(rows)
    converted = data.convert_manifests(rows)
    excluded = {"schema_version", "data_profile", "prompt", "token_lengths"}
    for task in data.TASKS:
        for original, result in zip(rows[task], converted[task], strict=True):
            assert {k: v for k, v in original.items() if k not in excluded} == {
                k: v for k, v in result.items() if k not in excluded
            }
            assert result["prompt"] == remove_touch(original["prompt"], task=task)
    derived, _ = data.derive(index, rows, source_file)
    for key in ("splits", "stage_a_checkpoint", "action_index_file", "action_training_data_hash",
                "adjustment_sampling_policy", "adjustment_label_policy", "need_successful_recovery_start"):
        assert derived[key] == index[key]
    assert derived["history_policy"] == index["history_policy"] | {"prompt_profile": PROMPT_PROFILE}
    assert index == before_index and rows == before_rows


@pytest.mark.parametrize("tamper", ["label", "history", "memory", "pair", "selection", "action", "policy"])
def test_semantic_validator_rejects_rehashed_tampering(source_artifacts, tamper):
    original, rows, source_file = source_artifacts
    index, converted = derived_artifacts(original, rows, source_file)
    data.validate_index(index)
    task = "adjustment"
    if tamper == "label":
        converted[task][0]["adjustment_end"] = True
    elif tamper == "history":
        converted[task][0]["qpos_h100_11_discrete"][0][0] += 1
    elif tamper == "memory":
        task = "plan"
        converted[task][0]["failure_recovery_memory"][0]["failure_reason"] = "other"
    elif tamper == "pair":
        converted[task][1]["pair_id"] += 1
    elif tamper == "selection":
        index["splits"]["train"][task]["manifest_row_indices"].reverse()
    elif tamper == "action":
        index["splits"]["train"]["action"]["indices"].reverse()
    else:
        index["input_policy"]["tactile_description"] = "neutral"
    file = Path(index[f"{task}_manifest_file"])
    write_rows(file, converted[task])
    index[f"{task}_manifest_sha256"] = sha256_file(file)
    index["manifest_content_hashes"][task] = sha256_json(converted[task])
    index["training_data_hash"] = sha256_json({k: v for k, v in index.items() if k != "training_data_hash"})
    with pytest.raises(ValueError):
        data.validate_index(index)


def test_source_manifest_change_is_rejected(source_artifacts):
    original, _, source_file = source_artifacts
    data.load_source(source_file)
    Path(original["need_manifest_file"]).write_text("{}\n")
    with pytest.raises(ValueError, match="manifest hash"):
        data.load_source(source_file)


@pytest.mark.parametrize("tamper", ["profile", "weights", "norm", "checkpoint"])
def test_stage_a_requires_same_v742_initialization(source_artifacts, tamper):
    original, rows, source_file = source_artifacts
    index, _ = data.derive(original, rows, source_file)
    checkpoint = Path(index["stage_a_checkpoint"])
    assert data.validate_stage_a_for_training(checkpoint, index)["step"] == 10000
    file = checkpoint.parent / "config.json"
    config = json.loads(file.read_text())
    if tamper == "profile":
        config["data_profile"] = data.source.DATA_PROFILE
    elif tamper == "weights":
        config["artifact_identity"]["training_data_hash"] = "other"
    elif tamper == "norm":
        config["artifact_identity"]["v4_norm_stats_sha256"] = "other"
    else:
        checkpoint = checkpoint.parent / "5000"
    file.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Stage A"):
        data.validate_stage_a_for_training(checkpoint, index)


def test_raw_caption_metadata_does_not_reach_model():
    row = {"episode_id": 271, "attempt_id": 2, "frame_index": 100,
           "prompt": remove_touch(phase(), task="need"), "need_recovery": False}
    item = {"index": 0, "episode_id": 271, "attempt_id": 2, "frame_index": 100,
            "observation.images.front": np.zeros((3, 2, 2)), "observation.images.left": np.zeros((3, 2, 2)),
            "observation.state": np.zeros(7), "tactile_caption": CAPTION, "Fx": 99}
    raw = V77ManifestDataset(rows=[row], row_indices=[0], global_indices=[0], task="need", lerobot_dataset=[item])[0]
    assert "tactile_caption" not in raw and "Fx" not in raw
    validate_no_touch(raw["prompt"])


@pytest.fixture
def training_entry(monkeypatch):
    from scripts import train_vla_multitask_v7_8 as entry

    for obj in (entry.trainer, entry.base):
        for name, value in list(vars(obj).items()):
            if not name.startswith("__"):
                monkeypatch.setattr(obj, name, value)
    entry.configure_version()
    return entry


def test_training_reuses_v772_defaults_model_loaders_and_grammar(training_entry):
    entry = training_entry
    args = entry.parse_args([])
    assert (args.data_profile, args.prompt_profile) == (data.DATA_PROFILE, PROMPT_PROFILE)
    assert args.stage_a_checkpoint == data.DEFAULT_STAGE_A
    assert args.dataset_dir.name == "tactile_vla_rotation_v4_1_new_env"
    assert (args.num_steps, args.batch_size, args.eval_interval, args.save_interval, args.keep_period) == (
        20000, 8, 1000, 5000, 5000,
    )
    assert (args.lr, args.weight_decay, args.grad_clip, args.seed) == (1e-4, 1e-4, 1.0, 42)
    assert args.output_dir == data.ROOT / "outputs/multitask_v7_8"
    entry.configure_training()
    assert entry.base.TASK_CYCLE == data.TASK_CYCLE
    assert entry.base.StageBV3Model is entry.trainer.V77MultitaskModel
    assert entry.base.trainable_filter is entry.trainer.trainable_filter
    assert entry.base.EXTRA_CONFIG["input_policy"] == INPUT_POLICY
    assert entry.base.CHECKPOINT_EXPORT_HOOK is entry.export_checkpoint
    assert entry.base.build_loaders is entry.build_loaders
    # Full V3 output grammar remains selected by the unchanged base engine.
    assert args.grammar_profile == "v3_full_v1"


def test_cli_equals_and_explicit_training_schedule(training_entry):
    args = training_entry.parse_args(["--num-steps=5000", "--save-interval=1000", "--fsdp-devices=2"])
    assert (args.num_steps, args.save_interval, args.fsdp_devices) == (5000, 1000, 2)


@pytest.mark.parametrize("flags", [["--prompt-profile=phase_v7_7_h100_11_answer"], ["--use-state-history"],
                                 ["--max-token-len=256"], ["--no-norm"]])
def test_incompatible_model_input_settings_fail(training_entry, flags):
    with pytest.raises(ValueError, match="input/model protocol"):
        training_entry.parse_args(flags)


@pytest.mark.parametrize("tamper", ["input_policy", "source_multitask_training_data_hash", "lr"])
def test_resume_rejects_ablation_or_training_changes(training_entry, tamper):
    entry = training_entry
    args = entry.parse_args([])
    entry.configure_training()
    saved = vars(args) | copy.deepcopy(entry.base.EXTRA_CONFIG) | {"precision": "float32"}
    entry.validate_resume(saved, args, actual_precision="float32")
    saved[tamper] = "changed"
    with pytest.raises(ValueError, match="resume"):
        entry.validate_resume(saved, args, actual_precision="float32")


def test_export_records_independent_version_and_source(training_entry, tmp_path, monkeypatch):
    entry = training_entry
    entry.configure_training()
    step_dir = tmp_path / "1000"
    step_dir.mkdir()

    def exported(*args):
        (step_dir / "v7_8_export.json").write_text(json.dumps({"exports": {"delta_params": {}, "full_params": {}}}))

    monkeypatch.setattr(entry, "_BASE_EXPORT", exported)
    entry.export_checkpoint(tmp_path, None, 1000, None)
    result = json.loads((step_dir / "v7_8_export.json").read_text())
    assert result["data_profile"] == data.DATA_PROFILE and result["prompt_profile"] == PROMPT_PROFILE
    assert result["input_policy"] == INPUT_POLICY
    assert set(result["exports"]) == {"delta_params", "full_params"}


def test_build_dry_run_and_existing_output_never_overwrite(source_artifacts, monkeypatch, tmp_path):
    from scripts import build_v7_8_multitask_data as builder

    _, _, source = source_artifacts
    output = tmp_path / "output"
    builder.main(["--source-index", str(source), "--output-dir", str(output), "--dry-run"])
    assert not output.exists()
    builder.main(["--source-index", str(source), "--output-dir", str(output)])
    index = json.loads((output / "v7_8_multitask_training_index.json").read_text())
    data.validate_index(index)
    with pytest.raises(FileExistsError):
        builder.main(["--source-index", str(source), "--output-dir", str(output)])
