from __future__ import annotations

# ruff: noqa: E402
import copy
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from tactile_vla.common.labels_v4 import LABEL_FIELDS, LABEL_SCHEMA_VERSION
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_2_adjustment_end_data import (
    EXPECTED_COUNTS as OLD_ADJUSTMENT_COUNTS,
    selected_train_frames,
)
from tactile_vla.vla.book_v9_3_multitask_data import EXPECTED_COUNTS as OLD_MULTITASK_COUNTS
from tactile_vla.vla.book_v9_4_multitask_data import (
    DATA_PROFILE,
    INDEX_SCHEMA,
    MANIFEST_SCHEMA,
    adjustment_counts,
    expected_counts,
    load_captioner_provenance,
    target_coverage,
    validate_caption,
    validate_index,
    validate_manifest_rows,
    validate_schedule,
    validate_upload_metadata,
)
from tactile_vla.vla.book_v9_3_multitask_data import ADJUSTMENT_LABEL_POLICY, NEED_RECOVERY_NEGATIVE_START
from tactile_vla.vla.v7_7_multitask_data import TASK_CYCLE
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE, helper_identity
from tactile_vla.vla.book_v9_4_memory import (
    MEMORY_POLICY,
    expand_plan_rows,
    validate_plan_tokens,
)

CAPTION = "Touch[area=none; Fx=near_zero; Fy=near_zero; Fz=near_zero; Fz_bias=balanced; rotation=none]"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def profile():
    return {
        "attempts": [
            {"episode_id": number, "attempt_id": 2, "split": split, "rexecution_frame_index": r, "frame_count": 300}
            for number, (split, r) in enumerate((("train", 150), ("val", 160), ("test", 170)), 1)
        ]
    }


@pytest.mark.parametrize("steps", [5, 4000, 8000, 8500, 10000])
def test_configurable_steps_accept_cycle_boundaries(steps):
    validate_schedule(num_steps=steps, eval_interval=500, save_interval=500, keep_period=1000)


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_steps", 0),
        ("num_steps", -5),
        ("num_steps", 4001),
        ("eval_interval", 0),
        ("eval_interval", 801),
        ("save_interval", 799),
        ("keep_period", 750),
    ],
)
def test_invalid_schedule_is_rejected(key, value):
    values = dict(num_steps=4000, eval_interval=500, save_interval=500, keep_period=1000)
    values[key] = value
    with pytest.raises(ValueError):
        validate_schedule(**values)


def test_counts_are_derived_without_changing_old_versions(tmp_path):
    data = profile()
    assert adjustment_counts(data) == {"train": (1, 11, 22), "val": (1, 11, 150), "test": (1, 11, 160)}
    write_json(tmp_path / "profile.json", data)
    for split in ("train", "val", "test"):
        write_rows(tmp_path / f"need/{split}.jsonl", [{"need_recovery": True}] * 7 + [{"need_recovery": False}] * 20)
        for folder in ("failure_reason", "reasoning"):
            write_rows(
                tmp_path / f"reasoning_manifests/{folder}/{split}.jsonl", [{"frame_offset": 0}, {"frame_offset": 14}]
            )
    counts = expected_counts(tmp_path)
    assert counts["train"] == {"adjustment": 33, "need": 28, "failure": 2, "plan": 8}
    assert counts["val"] == {"adjustment": 161, "need": 28, "failure": 1, "plan": 4}
    assert expected_counts(tmp_path, expand_plan=False)["train"]["plan"] == 2
    assert counts["test"]["adjustment"] == 171
    assert OLD_ADJUSTMENT_COUNTS["train"] == (24, 264, 528)
    assert OLD_MULTITASK_COUNTS["train"]["need"] == 9480
    data["attempts"][0]["rexecution_frame_index"] = 43
    with pytest.raises(ValueError, match="buckets"):
        adjustment_counts(data)


def test_six_field_caption_validation():
    validate_caption(CAPTION)
    validate_caption(CAPTION.replace("rotation=none", "rotation=clockwise"))
    for caption in (
        CAPTION.replace("; Fz_bias=balanced", ""),
        CAPTION.replace("balanced", "invalid"),
        CAPTION.replace("rotation=none", "rotation=none; rotation=none"),
    ):
        with pytest.raises(ValueError):
            validate_caption(caption)


def test_partial_upload_is_rejected(tmp_path):
    write_json(tmp_path / "meta/info.json", {"total_episodes": 2, "total_frames": 600, "fps": 30})
    with pytest.raises(ValueError, match="finish the upload"):
        validate_upload_metadata(tmp_path, profile())
    write_json(tmp_path / "meta/info.json", {"total_episodes": 3, "total_frames": 900, "fps": 30})
    validate_upload_metadata(tmp_path, profile())


def test_caption_provenance_requires_complete_matching_rebuild(tmp_path):
    checkpoint = {
        "path": "/offline/best.pt",
        "sha256": "a" * 64,
        "window_size": 30,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_fields": list(LABEL_FIELDS),
    }
    report = {
        "checkpoint_sha256": "a" * 64,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "window_size": 30,
        "selected_attempts": 3,
        "annotated_attempts": 3,
        "skipped_attempts": 0,
        "annotated_frames": 900,
        "warmup_policy": "neutral",
    }
    write_json(tmp_path / "caption_report.json", report)
    write_json(tmp_path / "rebuild.json", {"phase": "complete", "attempts": 3, "inputs": {"checkpoint": checkpoint}})
    identity, hashes = load_captioner_provenance(tmp_path, profile=profile())
    assert identity["checkpoint_sha256"] == "a" * 64
    assert len(hashes) == 2
    # Offline data training does not require the raw captioner weights on this host.
    write_json(tmp_path / "caption_report.json", report | {"checkpoint_sha256": "b" * 64})
    with pytest.raises(ValueError, match="upload may be incomplete"):
        load_captioner_provenance(tmp_path, profile=profile())


def make_manifests():
    manifests = {task: [] for task in ("adjustment", "need", "failure", "plan")}
    streams = {split: {} for split in ("train", "val", "test")}
    for split in streams:
        for task in manifests:
            labels = [True, False, False] if task == "adjustment" else [True, False, False, False]
            if task in ("failure", "plan"):
                labels = [True, True]
            start = len(manifests[task])
            for number, positive in enumerate(labels):
                row = {
                    "schema_version": MANIFEST_SCHEMA,
                    "data_profile": DATA_PROFILE,
                    "split": split,
                    "global_index": start + number,
                    "prompt": "Touch: " + CAPTION,
                }
                if task in ("adjustment", "need"):
                    row["adjustment_end" if task == "adjustment" else "need_recovery"] = positive
                else:
                    direction = "left" if number == 0 else "right"
                    if task == "failure":
                        row["target_failure_reason"] = f"failure_reason=rotate {direction},grasp appropriate."
                    else:
                        row["target_recovery_plan"] = (
                            f"recovery_plan=move horizontally {direction} moderately, move vertically none moderately."
                        )
                manifests[task].append(row)
            positions = list(range(start, len(manifests[task])))
            streams[split][task] = {
                "manifest_row_indices": positions,
                "global_indices": positions.copy(),
                "sample_count": len(positions),
            }
    expand_fixture_plans(manifests, streams)
    index = {
        "splits": streams,
        "manifest_content_hashes": {k: sha256_json(v) for k, v in manifests.items()},
        "training_target_coverage": target_coverage(manifests),
    }
    return index, manifests


def expand_fixture_plans(manifests, streams, *, sources=None):
    failures = {(row["split"], row["global_index"]): row["target_failure_reason"] for row in manifests["failure"]}
    old_selected = {split: set(streams[split]["plan"]["global_indices"]) for split in streams}
    for row in manifests["plan"]:
        row.update(
            {
                "memory_length": 1,
                "failure_recovery_memory": [
                    {"recovery_plan": "initial plan", "failure_reason": failures[row["split"], row["global_index"]]}
                ],
                "prompt": "Mode: reasoning. Task: Book. " + CAPTION + " Failure-recovery memory: old",
            }
        )
    manifests["plan"] = expand_plan_rows(manifests["plan"], manifests["failure"], sources=sources)
    for split in streams:
        positions = [
            i
            for i, row in enumerate(manifests["plan"])
            if row["split"] == split and row["global_index"] in old_selected[split]
        ]
        streams[split]["plan"] = {
            "manifest_row_indices": positions,
            "global_indices": [manifests["plan"][i]["global_index"] for i in positions],
            "sample_count": len(positions),
        }

    class Tokenizer:
        def encode_text(self, text, *, add_eos):
            return [1, 2]

        def tokenize_structured_response(self, prompt, state, target, *, max_len):
            import numpy as np

            prefix = len(prompt.split())
            return None, np.ones(prefix + len(target), dtype=bool), None, None, prefix

    return validate_plan_tokens(
        manifests["plan"],
        tokenizer=Tokenizer(),
        normalized_states={row["global_index"]: [0] * 7 for row in manifests["plan"]},
    )


def test_selected_manifest_identity_ratios_and_left_right_coverage():
    index, manifests = make_manifests()
    validate_manifest_rows(index, manifests)
    assert len(index["training_target_coverage"]["failure_reason"]) == 2
    broken = copy.deepcopy(index)
    broken["splits"]["train"]["need"]["global_indices"][0] = -1
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_manifest_rows(broken, manifests)
    broken = copy.deepcopy(manifests)
    broken["need"][0]["need_recovery"] = False
    index["manifest_content_hashes"]["need"] = sha256_json(broken["need"])
    with pytest.raises(ValueError, match="ratio mismatch"):
        validate_manifest_rows(index, broken)


@pytest.mark.parametrize(
    "arguments,expected", [([], 4000), (["--num-steps", "8500"], 8500), (["--num-steps=10000"], 10000)]
)
def test_real_training_cli_accepts_steps_without_reading_upload(monkeypatch, arguments, expected):
    from scripts import train_vla_multitask_book_v9_4 as entry

    # The version wrapper configures the reused trainer inside this process only.
    names = (
        "DATA_PROFILE",
        "DEFAULT_INDEX",
        "DEFAULT_STAGE_A",
        "DEFAULT_OUTPUT",
        "DEFAULT_DATASET",
        "DEFAULT_NORM_STATS",
        "RUN_NAME",
        "VERSION_TAG",
        "validate_index",
        "validate_v7_4_adjustment_training_index",
        "parse_args",
        "configure",
        "ensure_index",
        "ROOT",
        "_ARGS",
    )
    for name in names:
        monkeypatch.setattr(entry.trainer, name, getattr(entry.trainer, name))
    monkeypatch.setattr(entry.training_base, "evaluate_text", entry.training_base.evaluate_text)
    monkeypatch.setattr(sys, "argv", ["train_vla_multitask_book_v9_4.py", *arguments])
    entry.configure_version()
    args = entry.parse_args()
    assert args.num_steps == expected
    assert args.data_profile == DATA_PROFILE
    assert "v9_4" in str(args.stage_a_checkpoint)
    assert "v9_4" in str(args.index_file)
    assert "v9_4" in args.run_name
    assert args.eval_interval == 500


def test_complete_index_validation_and_source_change_detection(tmp_path):
    v4_dir, state_dir = tmp_path / "v4", tmp_path / "state"
    data = profile()
    for row in data["attempts"]:
        row["rexecution_frame_index"] = 44
    write_json(v4_dir / "profile.json", data)
    for split in ("train", "val", "test"):
        write_rows(v4_dir / f"need/{split}.jsonl", [{"need_recovery": True}] * 2)
        for folder in ("failure_reason", "reasoning"):
            write_rows(
                v4_dir / f"reasoning_manifests/{folder}/{split}.jsonl", [{"frame_offset": 0}, {"frame_offset": 14}]
            )
    checkpoint = {
        "path": "/cached/best.pt",
        "sha256": "a" * 64,
        "window_size": 30,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_fields": list(LABEL_FIELDS),
    }
    report = {
        "checkpoint_sha256": "a" * 64,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "window_size": 30,
        "selected_attempts": 3,
        "annotated_attempts": 3,
        "skipped_attempts": 0,
        "annotated_frames": 900,
        "warmup_policy": "neutral",
    }
    write_json(state_dir / "caption_report.json", report)
    write_json(state_dir / "rebuild.json", {"phase": "complete", "attempts": 3, "inputs": {"checkpoint": checkpoint}})
    identity, _ = load_captioner_provenance(state_dir, profile=data)
    plan_sources = {}
    for episode, split in enumerate(("train", "val", "test"), 1):
        source_rows = []
        for frame in (0, 14):
            direction = "left" if frame == 0 else "right"
            source = {
                "frame_offset": frame,
                "frame_index": frame,
                "variant_id": f"source-{episode}-{frame}",
                "current_observation": {
                    "episode_id": episode,
                    "attempt_id": 1,
                    "failure_reason": f"failure_reason=rotate {direction},grasp appropriate.",
                    "tactile_caption": CAPTION,
                },
                "target_recovery_plan": f"recovery_plan=move horizontally {direction} moderately, move vertically none moderately.",
                "target_source": {
                    "source_type": "real",
                    "episode_id": episode,
                    "failed_attempt_id": 1,
                    "plan_attempt_id": 2,
                },
            }
            source_rows.append(source)
            plan_sources[split, episode * 1000 + frame] = source
        write_rows(v4_dir / f"reasoning_manifests/reasoning/{split}.jsonl", source_rows)
    sources = {str(path.resolve()): sha256_file(path) for path in tmp_path.rglob("*.json*")}
    manifests = {task: [] for task in ("adjustment", "need", "failure", "plan")}
    streams = {split: {} for split in ("train", "val", "test")}
    for episode, split in enumerate(streams, 1):
        for task in manifests:
            if task == "adjustment":
                frames = list(range(45))
                chosen = selected_train_frames(44, episode_id=episode) if split == "train" else set(frames)
            elif task == "need":
                frames, chosen = list(range(8)), set(range(8))
            else:
                frames = [0, 14]
                chosen = set(frames) if split == "train" else {14}
            positions, global_indices = [], []
            for frame in frames:
                row = {
                    "schema_version": MANIFEST_SCHEMA,
                    "data_profile": DATA_PROFILE,
                    "split": split,
                    "global_index": episode * 1000 + frame,
                    "episode_id": episode,
                    "attempt_id": 1,
                    "frame_index": frame,
                    "prompt": "Touch: " + CAPTION,
                }
                if task == "adjustment":
                    row["adjustment_end"] = frame >= 34
                elif task == "need":
                    row["need_recovery"] = frame < 2
                elif task == "failure":
                    row["target_failure_reason"] = (
                        f"failure_reason=rotate {'left' if frame == 0 else 'right'},grasp appropriate."
                    )
                else:
                    row["prompt"] = "Failed tactile observation: " + CAPTION
                    row["target_recovery_plan"] = (
                        f"recovery_plan=move horizontally {'left' if frame == 0 else 'right'} moderately, move vertically none moderately."
                    )
                if frame in chosen:
                    positions.append(len(manifests[task]))
                    global_indices.append(row["global_index"])
                manifests[task].append(row)
            streams[split][task] = {
                "sample_count": len(positions),
                "manifest_row_indices": positions,
                "global_indices": global_indices,
            }
    token_summary = expand_fixture_plans(manifests, streams, sources=plan_sources)
    index = {
        "schema_version": INDEX_SCHEMA,
        "data_profile": DATA_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "task_cycle": list(TASK_CYCLE),
        "history_policy": helper_identity() | {"idle_perturbation": "none_raw_contiguous"},
        "adjustment_label_policy": ADJUSTMENT_LABEL_POLICY,
        "need_successful_recovery_start": NEED_RECOVERY_NEGATIVE_START,
        "plan_memory_policy": MEMORY_POLICY,
        "model_dependency": "none_data_only",
        "plan_token_validation": token_summary,
        "source_hashes": sources,
        "v4_dir": str(v4_dir),
        "incremental_state_dir": str(state_dir),
        "captioner_identity": identity,
        "splits": streams,
        "training_target_coverage": target_coverage(manifests),
        "manifest_content_hashes": {key: sha256_json(rows) for key, rows in manifests.items()},
    }
    for task, rows in manifests.items():
        path = tmp_path / "manifests" / f"{task}.jsonl"
        write_rows(path, rows)
        index[f"{task}_manifest_file"] = str(path)
        index[f"{task}_manifest_sha256"] = sha256_file(path)
    index["training_data_hash"] = sha256_json(index)
    validate_index(index)
    write_rows(v4_dir / "need/val.jsonl", [{"need_recovery": True}])
    with pytest.raises(ValueError, match="source changed"):
        validate_index(index)
