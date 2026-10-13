"""Strict no-touch ablation, with the V11.2 compressed timelines unchanged."""

# ruff: noqa: E402
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src"), str(PROJECT / "openpi/src")]

from tactile_vla.vla import bottle_v11_3_multitask_data as data
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.bottle_v11_3_prompts import (
    INPUT_POLICY,
    PROMPT_PROFILE,
    remove_touch,
    validate_no_touch,
    build_phase_prompt,
    build_recovery_prompt,
)
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt as tactile_plan
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt as tactile_phase
from tactile_vla.vla.v7_7_multitask_data import V77ManifestDataset

CAPTION = "Touch[area=small; Fx=negative; Fy=positive; Fz=negative; Fz_bias=none; rotation=none]"
INSTRUCTION = "Open the bottle"


def phase():
    return tactile_phase(
        instruction=INSTRUCTION,
        tactile_caption=CAPTION,
        recovery_plan="none",
        qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32),
    )


@pytest.mark.parametrize("task", ["need", "failure", "adjustment"])
def test_phase_strip_preserves_every_other_prompt_field(task):
    assert remove_touch(phase(), task=task) == build_phase_prompt(
        instruction=INSTRUCTION, recovery_plan="none", qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32)
    )


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_plan_strip_keeps_fy_reason_and_rotate_plan_memory(length):
    memory = data.source.previous.make_memory(length, split="train", global_index=100)
    original = tactile_plan(
        instruction=INSTRUCTION,
        failed_tactile_caption=CAPTION,
        failure_recovery_memory=memory,
        prompt_profile=MINIMAL_PROMPT_PROFILE,
    )
    result = remove_touch(original, task="plan")
    assert result == build_recovery_prompt(instruction=INSTRUCTION, failure_recovery_memory=memory)
    assert result.count("fy_change raise") == length
    assert "Touch" not in result


@pytest.mark.parametrize(
    "prompt", [CAPTION, "Touch: none", "Touch:", "rotation=none", "Fy=positive", "tactile_caption=none"]
)
def test_no_touch_guard_rejects_neutral_and_hidden_caption_fields(prompt):
    with pytest.raises(ValueError):
        validate_no_touch(prompt)


def test_missing_duplicate_caption_and_touch_memory_are_rejected():
    for prompt in ("Mode: phase\nTask: bottle", phase() + "\nTouch: " + CAPTION + "\n"):
        with pytest.raises(ValueError):
            remove_touch(prompt, task="need")
    with pytest.raises(ValueError):
        remove_touch("Mode: plan " + CAPTION + " Failure-recovery memory: " + CAPTION, task="plan")


def fixture():
    rows = {task: [] for task in data.TASKS}
    splits = {split: {"action": {"indices": [10, 20, 30]}} for split in ("train", "val", "test")}
    for split in splits:
        for task in data.TASKS:
            g = len(rows[task]) * 10
            row = {
                "split": split,
                "global_index": g,
                "episode_id": 21,
                "attempt_id": 2,
                "frame_index": 100,
                "timestamp": 123.0,
                "source": "real",
                "schema_version": "v11_2",
                "data_profile": data.source.DATA_PROFILE,
                "prompt": phase(),
                "qpos_h100_11_discrete": np.zeros((11, 7), dtype=np.int32).tolist(),
                "history_sources": [{"global_index": g, "frame_index": 100}],
                "effective_episode_history_length": 100,
                "token_lengths": {"total": 400},
            }
            if task == "need":
                row["need_recovery"] = True
            elif task == "adjustment":
                row.update(adjustment_end=False, selected=True, rexecution_frame=200)
            elif task == "failure":
                row.update(target_failure_reason=data.source.previous.FAILURE_TARGET, F=97, C=100, frame_offset=0)
            else:
                memory = data.source.previous.make_memory(4, split=split, global_index=g)
                row.update(
                    memory_length=4,
                    failure_recovery_memory=memory,
                    target_recovery_plan=data.source.previous.PLAN_TARGET,
                )
                row["prompt"] = tactile_plan(
                    instruction=INSTRUCTION,
                    failed_tactile_caption=CAPTION,
                    failure_recovery_memory=memory,
                    prompt_profile=MINIMAL_PROMPT_PROFILE,
                )
            position = len(rows[task])
            rows[task].append(row)
            splits[split][task] = {"manifest_row_indices": [position], "global_indices": [g], "sample_count": 1}
    index = {
        "splits": splits,
        "training_data_hash": "a" * 64,
        "source_hashes": {},
        "action_index_file": "v11_2_action.json",
        "action_training_data_hash": "b" * 64,
        "action_target_policy": {"compressed": True, "settle_frames": 5},
        "idle_gap_audit": {"attempts": [{"removed_start": 30, "removed_stop": 60, "removed_frame_count": 30}]},
    }
    return index, rows


def test_conversion_preserves_labels_history_memory_order_and_original():
    _, rows = fixture()
    before = copy.deepcopy(rows)
    converted = data.convert_manifests(rows)
    assert rows == before
    ignored = {"schema_version", "data_profile", "token_lengths", "prompt"}
    for task in data.TASKS:
        for original, result in zip(rows[task], converted[task], strict=True):
            assert {k: v for k, v in original.items() if k not in ignored} == {
                k: v for k, v in result.items() if k not in ignored
            }
            assert result["prompt"] == remove_touch(original["prompt"], task=task)
            assert "token_lengths" not in result


def test_derive_retains_action_compression_and_history_without_rebuilding(tmp_path, monkeypatch):
    original, rows = fixture()
    snapshot = copy.deepcopy(original)
    path = tmp_path / "source.json"
    path.write_text(json.dumps(original))
    monkeypatch.setattr(data, "validate_tokens", lambda *args: {"checked": True})
    index, _ = data.derive(original, rows, path)
    assert original == snapshot
    for key in ("splits", "action_index_file", "action_training_data_hash", "action_target_policy", "idle_gap_audit"):
        assert index[key] == original[key]
    assert index["history_policy"]["adjustment_end"] == data.source.HISTORY_POLICY["adjustment_end"]
    assert index["input_policy"] == INPUT_POLICY and index["prompt_profile"] == PROMPT_PROFILE
    assert index["source_multitask_training_data_hash"] == original["training_data_hash"]


@pytest.mark.parametrize("tamper", ["label", "idle_gap"])
def test_semantic_validator_rejects_forged_consistent_hashes(tmp_path, monkeypatch, tamper):
    original, rows = fixture()
    source = tmp_path / "source.json"
    source.write_text(json.dumps(original))
    monkeypatch.setattr(data, "load_source", lambda path: (original, rows))
    monkeypatch.setattr(data, "validate_tokens", lambda *args: {})
    index, converted = data.derive(original, rows, source)
    for task in data.TASKS:
        file = tmp_path / (task + ".jsonl")
        file.write_text("\n".join(json.dumps(row) for row in converted[task]) + "\n")
        index[f"{task}_manifest_file"], index[f"{task}_manifest_sha256"] = str(file), sha256_file(file)
    index["training_data_hash"] = sha256_json(index)
    data.validate_index(index)
    if tamper == "label":
        converted["need"][0]["need_recovery"] = False
        file = Path(index["need_manifest_file"])
        file.write_text("\n".join(json.dumps(row) for row in converted["need"]) + "\n")
        index["need_manifest_sha256"] = sha256_file(file)
    else:
        index["idle_gap_audit"]["attempts"][0]["removed_start"] = 40
    index["training_data_hash"] = sha256_json({k: v for k, v in index.items() if k != "training_data_hash"})
    with pytest.raises(ValueError):
        data.validate_index(index)


def test_grammar_is_full_bottle_space_and_version_guarded():
    class Tokenizer:
        def encode_text(self, text, *, add_eos):
            return [ord(c) for c in text] + ([2] if add_eos else [])

    args = SimpleNamespace(data_profile=data.DATA_PROFILE)
    index = {"experiment_version": data.VERSION_TAG, "text_schema_policy": data.TEXT_SCHEMA_POLICY}
    failure, plan = data.build_training_grammars(Tokenizer(), args=args, index=index)
    assert len(failure.texts) == 179 and len(plan.texts) == 636
    index["experiment_version"] = "bottle_v11_2"
    with pytest.raises(ValueError):
        data.build_training_grammars(Tokenizer(), args=args, index=index)


def test_raw_caption_metadata_never_reaches_model_inputs():
    row = {
        "global_index": 0,
        "episode_id": 1,
        "attempt_id": 2,
        "frame_index": 3,
        "prompt": remove_touch(phase(), task="need"),
        "need_recovery": False,
    }
    item = {
        "index": 0,
        "episode_id": 1,
        "attempt_id": 2,
        "frame_index": 3,
        "observation.images.front": np.zeros((3, 2, 2)),
        "observation.images.left": np.zeros((3, 2, 2)),
        "observation.state": np.zeros(7),
        "tactile_caption": CAPTION,
        "Fy": 99,
    }
    dataset = V77ManifestDataset(rows=[row], row_indices=[0], global_indices=[0], task="need", lerobot_dataset=[item])
    result = dataset[0]
    assert "tactile_caption" not in result and "Fy" not in result
    validate_no_touch(result["prompt"])


def test_stage_a_validation_delegates_to_v11_2_not_multitask(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        data.source,
        "validate_stage_a_for_training",
        lambda checkpoint, index: calls.append((checkpoint, index)) or {"step": 7500},
    )
    assert data.validate_stage_a_for_training(data.DEFAULT_STAGE_A, {}) == {"step": 7500}
    assert calls == [(data.DEFAULT_STAGE_A, {})]
    assert data.DEFAULT_STAGE_A.parent.name == "pi05_delta_tac_bottle_stage_a_v11_2_no_history"


def test_training_entry_uses_visual_profile_and_v11_2_checkpoint(monkeypatch):
    from scripts import train_vla_multitask_bottle_v11_3 as entry

    for obj in (entry.engine, entry.engine.trainer, entry.engine.base):
        for name, value in list(vars(obj).items()):
            if not name.startswith("__"):
                monkeypatch.setattr(obj, name, value)
    entry.configure_version()
    args = entry.parse_args([])
    assert args.stage_a_checkpoint == data.DEFAULT_STAGE_A
    assert args.data_profile == data.DATA_PROFILE and args.prompt_profile == PROMPT_PROFILE
    assert (args.num_steps, args.batch_size, args.fsdp_devices) == (2000, 8, 1)
    assert args.output_dir == data.ROOT / "outputs/multitask_v11_3"
    entry.configure_training()
    assert entry.engine.base.TEXT_GRAMMAR_FACTORY is data.build_training_grammars
    assert entry.engine.base.EXTRA_CONFIG["input_policy"] == INPUT_POLICY
    assert entry.engine.base.EXTRA_CONFIG["action_sampling_policy"]["ratio"] == "1:1_per_batch"
