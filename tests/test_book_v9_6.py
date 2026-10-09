from __future__ import annotations

# ruff: noqa: E402, SLF001
import copy
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "openpi/src"), str(ROOT / "openpi/packages/openpi-client/src")]

from tactile_vla.vla import book_v9_6_multitask_data as data
from tactile_vla.vla import book_v9_6_runtime as deployment
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.book_v9_6_prompts import (
    INPUT_POLICY,
    PROMPT_PROFILE,
    build_phase_prompt,
    build_recovery_prompt,
    remove_touch,
    validate_no_touch,
)
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt as tactile_plan
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt as tactile_phase

CAPTION = "Touch[area=small; Fx=negative; Fy=positive; Fz=negative; Fz_bias=left; rotation=clockwise]"
FAILURE = "failure_reason=rotate left,grasp appropriate."
PLAN = "recovery_plan=move horizontally left moderately, move vertically none moderately."


def phase_prompt(caption=CAPTION):
    return tactile_phase(
        instruction=deployment.BOOK_INSTRUCTION,
        tactile_caption=caption,
        recovery_plan="",
        qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32),
    )


@pytest.mark.parametrize("task", ["need", "failure", "adjustment"])
def test_phase_prompt_removes_only_touch(task):
    actual = remove_touch(phase_prompt(), task=task)
    assert actual == build_phase_prompt(
        instruction=deployment.BOOK_INSTRUCTION,
        recovery_plan="",
        qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32),
    )
    assert "Recovery plan: none" in actual and "H100 sampled to 11" in actual


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_plan_prompt_preserves_memory_and_punctuation(length):
    memory = [{"recovery_plan": "initial plan", "failure_reason": FAILURE}]
    memory += [{"recovery_plan": PLAN, "failure_reason": FAILURE}] * (length - 1)
    original = tactile_plan(
        instruction=deployment.BOOK_INSTRUCTION,
        failed_tactile_caption=CAPTION,
        failure_recovery_memory=memory,
        prompt_profile=MINIMAL_PROMPT_PROFILE,
    )
    actual = build_recovery_prompt(instruction=deployment.BOOK_INSTRUCTION, failure_recovery_memory=memory)
    assert actual == remove_touch(original, task="plan")
    assert actual.count("failure_reason=") == length
    assert "Touch" not in actual


@pytest.mark.parametrize("text", [CAPTION, "Touch: none", "rotation=none", "Fz_bias=left", "tactile_caption=none"])
def test_no_touch_guard(text):
    with pytest.raises(ValueError):
        validate_no_touch(text)


@pytest.mark.parametrize("prompt", ["Mode: phase\nTask: book", phase_prompt() + "\nTouch: " + CAPTION + "\n"])
def test_conversion_rejects_missing_or_duplicate_touch(prompt):
    with pytest.raises(ValueError):
        remove_touch(prompt, task="need")


def source_fixture():
    rows = {task: [] for task in data.TASKS}
    splits = {split: {} for split in ("train", "val", "test")}
    for split in splits:
        for task in data.TASKS:
            row = {
                "split": split,
                "global_index": len(rows[task]) * 10,
                "episode_id": 1,
                "attempt_id": 1,
                "frame_index": 110,
                "timestamp": 123.0,
                "prompt": phase_prompt(),
                "qpos_h100_11_discrete": np.zeros((11, 7), dtype=np.int32).tolist(),
                "schema_version": "old",
                "data_profile": "old",
            }
            if task == "need":
                row.update(need_variant="real", need_recovery=True, need_pair_id="old_pair")
            if task == "failure":
                row["target_failure_reason"] = FAILURE
            if task == "plan":
                row.update(memory_length=1, target_recovery_plan=PLAN, plan_token_lengths={"total": 100})
                row["prompt"] = tactile_plan(
                    instruction=deployment.BOOK_INSTRUCTION,
                    failed_tactile_caption=CAPTION,
                    failure_recovery_memory=[{"recovery_plan": "initial plan", "failure_reason": FAILURE}],
                    prompt_profile=MINIMAL_PROMPT_PROFILE,
                )
            start = len(rows[task])
            rows[task].append(row)
            if task == "need" and split == "train":
                synthetic = copy.deepcopy(row)
                synthetic.update(
                    need_variant="rotation_none", need_recovery=False, need_counterfactual={"input": "none"}
                )
                synthetic["prompt"] = phase_prompt(CAPTION.replace("clockwise", "none"))
                rows[task].append(synthetic)
                negative = copy.deepcopy(row)
                negative.update(global_index=999, need_recovery=False)
                rows[task].append(negative)
            positions = list(range(start, len(rows[task])))
            splits[split][task] = {
                "manifest_row_indices": positions,
                "global_indices": [rows[task][p]["global_index"] for p in positions],
                "sample_count": len(positions),
            }
        splits[split]["action"] = {"indices": [10, 20, 30]}
    return {"splits": splits}, rows


def test_exact_real_conversion_no_resampling_no_label_or_history_changes():
    index, original = source_fixture()
    snapshot = copy.deepcopy(original)
    converted, splits, removed = data.convert_manifests(index, original)
    assert original == snapshot
    assert len(removed) == 1 and len(converted["need"]) == len(original["need"]) - 1
    for task in data.TASKS:
        kept = [r for r in original[task] if task != "need" or r["need_variant"] == "real"]
        for before, after in zip(kept, converted[task], strict=True):
            for key in before.keys() - {
                "prompt",
                "schema_version",
                "data_profile",
                "need_pair_id",
                "plan_token_lengths",
            }:
                assert before[key] == after[key]
            assert "need_pair_id" not in after and "need_counterfactual" not in after
            validate_no_touch(after["prompt"])
    assert splits["train"]["need"]["global_indices"] == [0, 999]
    for split in ("train", "val", "test"):
        assert splits[split]["action"] == index["splits"][split]["action"]
        if split != "train":
            assert splits[split]["need"]["sample_count"] == index["splits"][split]["need"]["sample_count"]


@pytest.mark.parametrize("field,value", [("need_variant", "other"), ("need_recovery", True), ("split", "val")])
def test_unexpected_counterfactual_rejected(field, value):
    index, rows = source_fixture()
    rows["need"][1][field] = value
    with pytest.raises(ValueError):
        data.convert_manifests(index, rows)


def test_builder_has_no_checkpoint_or_overwrite_and_protects_source(tmp_path):
    from scripts.build_book_v9_6_multitask_data import parse_args, main

    args = parse_args([])
    assert args.source_index == data.DEFAULT_SOURCE_INDEX
    assert not hasattr(args, "stage_a_checkpoint") and not hasattr(args, "overwrite")
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(args.source_index.parent)])
    with pytest.raises(FileExistsError):
        main(["--output-dir", str(tmp_path)])


def test_initialization_remains_matching_v9_5(monkeypatch):
    assert "v9_5" in str(data.DEFAULT_STAGE_A) and data.DEFAULT_STAGE_A.name == "15000"
    monkeypatch.setattr(
        data.source, "validate_stage_a_for_training", lambda checkpoint, index: {"path": checkpoint, "step": 15000}
    )
    assert data.validate_stage_a_for_training("checkpoint", {}) == {
        "path": "checkpoint",
        "step": 15000,
        "experiment_version": "book_v9_5",
    }


@pytest.fixture(scope="module")
def client():
    path = ROOT / "openpi/inference/agilex/inference/agilex_inference_book_v9_6_asyn.py"
    spec = importlib.util.spec_from_file_location("book_v96_async_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visual_ros_operator_subscribes_only_camera_and_qpos(client):
    topics = []
    operator = object.__new__(client.VisualRosOperator)
    operator.args = SimpleNamespace(
        img_front_topic="front", img_left_topic="wrist", puppet_arm_topic="qpos", puppet_arm_cmd_topic="cmd"
    )
    operator.Image = operator.JointState = object
    operator.rospy = SimpleNamespace(
        init_node=lambda *a, **k: None,
        Subscriber=lambda topic, *a, **k: topics.append(topic),
        Publisher=lambda *a, **k: "publisher",
    )
    operator._init_ros()
    assert topics == ["front", "wrist", "qpos"] and operator.tactile is None


def test_client_needs_no_captioner_file_or_tactile_arguments(client, tmp_path):
    norm = tmp_path / "norm.json"
    norm.write_text("{}")
    args, parser = client.get_arguments(
        ["--noise-seed", "42", "--gripper-min", "0.001", "--norm-stats-file", str(norm)]
    )
    client.validate_args(args, parser)
    assert not hasattr(args, "captioner_checkpoint") and not hasattr(args, "tactile_window_size")
    assert args.need_recovery_rate_hz == args.adjustment_end_rate_hz == 7.0


def test_runtime_memory_unlimited_initial_plus_latest_three(client):
    memory = []
    for i in range(20):
        payload, memory = client.plan_payload_after_failure(
            instruction=deployment.BOOK_INSTRUCTION,
            executed_recovery_plan="" if i == 0 else PLAN,
            failure_reason=FAILURE,
            memory=memory,
            image=np.zeros((2, 2, 3)),
            wrist_image=np.zeros((2, 2, 3)),
            qpos=np.zeros(7),
        )
        assert len(memory) == min(i + 1, 4) and memory[0]["recovery_plan"] == "initial plan"
        validate_no_touch(payload["prompt"])


def test_capture_requires_only_fresh_camera_qpos(client, monkeypatch):
    client.runtime.shutdown_event.clear()
    calls = []
    joint = SimpleNamespace(position=np.zeros(7), header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 10.0)))
    operator = SimpleNamespace(
        is_shutdown=lambda: False,
        rate=lambda _: SimpleNamespace(sleep=lambda: None),
        get_frame=lambda **kw: (calls.append(kw) or (np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), joint)),
    )
    args = SimpleNamespace(observation_poll_rate=200, phase_change_timeout_seconds=1)
    result = client._capture_action_observation(args, operator, after_timestamp=9.0)
    assert calls == [{"after_timestamp": 9.0}]
    assert result.timestamp == 10.0 and result.tactile_caption == ""


def test_server_rejects_tactile_and_legacy_prompt(client):
    from scripts import serve_tactile_vla_book_v9_6 as server

    policy = object.__new__(server.BookV96Policy)
    with pytest.raises(ValueError, match="tactile"):
        next(policy.infer_events({"mode": "phase", "prompt": phase_prompt()}))
    prompt = build_phase_prompt(
        instruction=deployment.BOOK_INSTRUCTION, recovery_plan="", qpos_h100_11_discrete=np.zeros((11, 7))
    )
    with pytest.raises(ValueError, match="tactile/caption"):
        next(policy.infer_events({"prompt": prompt, "tactile_caption": "none"}))
    args = server.parse_args([])
    assert "multitask_v9_6" in str(args.checkpoint) and not hasattr(args, "captioner_checkpoint_sha256")


def test_thresholds_reject_old_version_calibration():
    kwargs = dict(
        step=2000,
        full_params_sha="a" * 64,
        norm_sha="b" * 64,
        training_data_hash="c" * 64,
        need_override=None,
        adjustment_override=None,
    )
    thresholds, overrides = deployment.resolve_thresholds(calibration=None, **kwargs)
    assert thresholds == {"need_recovery": 0.5, "adjustment_end": 0.5} and not any(overrides.values())
    with pytest.raises(ValueError):
        deployment.resolve_thresholds(calibration={"schema_version": "book_v9_5_val_thresholds_v1"}, **kwargs)


def test_training_defaults_in_isolated_process():
    code = """import sys
from scripts import train_vla_multitask_book_v9_6 as entry
entry.configure_version()
sys.argv = ["train9_6", "--data-only-dry-run"]
args = entry.parse_args()
assert args.num_steps == 2000 and args.eval_interval == args.save_interval == args.keep_period == 1000
assert args.dry_run and args.data_only_dry_run
assert "v9_5" in str(args.stage_a_checkpoint) and "v9_6" in str(args.index_file)
assert args.prompt_profile == entry.PROMPT_PROFILE
assert args.batch_size == 8 and args.lr == 1e-4
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def valid_identity():
    return {
        "training_data_hash": "a" * 64,
        "source_scope": {"policy": deployment.SCOPE_POLICY},
        "stage_a_initialization_identity": {
            "step": 15000,
            "experiment_version": "book_v9_5",
            "action_training_data_hash": "b" * 64,
            "config_sha256": "c" * 64,
            "params_metadata_sha256": "d" * 64,
        },
        "need_boundary_audit_sha256": "e" * 64,
        "reasoning_boundary_audit_sha256": "f" * 64,
    }


def valid_metadata():
    return (
        valid_identity()
        | deployment.deployment_policy_fields("book_v9_6")
        | {
            "data_profile": data.DATA_PROFILE,
            "experiment_version": "book_v9_6",
            "name": deployment.SERVER_NAME,
            "phase_prompt_profile": PROMPT_PROFILE,
            "action_prompt_profile": "phase_v2",
            "supports_streamed_phase_events": True,
            "supports_action_noise": True,
            "requires_action_noise": True,
            "supports_failure_generation": True,
            "supports_recovery_generation": True,
            "action_horizon": 30,
            "action_dim": 32,
            "output_action_dim": 7,
            "use_state_history": False,
            "state_history_len": 0,
            "state_history_fps": 30.0,
            "qpos_h100_sample_offsets": deployment.HISTORY_OFFSETS,
            "max_memory_pairs": 4,
            "memory_policy": deployment.MEMORY_POLICY,
            "max_supported_attempts": None,
            "classification_qpos_policy": "raw_no_gripper_remap",
            "episode_start_padding": "left_pad_episode_frame_0",
            "requires_captioner": False,
            "requires_tactile_topics": False,
            "need_recovery_threshold": 0.5,
            "adjustment_end_threshold": 0.5,
            "thresholds_status": "default_0_5",
        }
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_policy", {}),
        ("requires_captioner", True),
        ("requires_tactile_topics", True),
        ("max_memory_pairs", 1),
        ("data_profile", "book_v9_5_five_task_h100"),
    ],
)
def test_metadata_identity_rejects_touch_and_old_version(field, value):
    metadata = valid_metadata()
    deployment.validate_server_metadata(metadata)
    with pytest.raises(ValueError):
        deployment.validate_server_metadata(metadata | {field: value})


def test_config_binds_visual_input_and_v9_5_stage_a_without_captioner():
    config = (
        valid_identity()
        | deployment.deployment_policy_fields("book_v9_6")
        | {
            "data_profile": data.DATA_PROFILE,
            "experiment_version": "book_v9_6",
            "prompt_profile": PROMPT_PROFILE,
            "action_horizon": 30,
            "action_dim": 32,
            "max_token_len": 512,
            "reasoning_max_token_len": 320,
            "use_state_history": False,
            "state_history_len": 0,
            "history_hidden_dim": 0,
            "state_history_dim": 7,
            "state_history_fps": 30.0,
            "grammar_profile": "v3_full_v1",
            "phase_prefill_protocol": "need_failure_shared_kv_v1",
            "plan_memory_policy": deployment.TRAINING_MEMORY_POLICY,
            "artifact_identity": {
                "data_profile": "book_stage_a_v1",
                "v4_norm_stats_sha256": "c" * 64,
                "book_v9_6_training_data_hash": "a" * 64,
                "training_data_hash": "b" * 64,
            },
        }
    )
    deployment.validate_training_config(config, "c" * 64)
    config["stage_a_initialization_identity"]["experiment_version"] = "book_v9_6"
    with pytest.raises(ValueError, match="book_v9_5"):
        deployment.validate_training_config(config, "c" * 64)


def test_phase_assessment_uses_fresh_visual_observation_and_h100_only(client, monkeypatch):
    from tactile_vla.vla.v5_3_phase_change import StateQuantileStats

    observation = SimpleNamespace(
        timestamp=10.0,
        qpos=np.zeros(7),
        img_front=np.zeros((2, 2, 3), dtype=np.uint8),
        img_left=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    captures, payloads, handled = [], [], []
    monkeypatch.setattr(client, "_capture_action_observation", lambda *a, **kw: captures.append(kw) or observation)
    history = np.linspace(0, 1, 100)[:, None] * np.ones((1, 7))
    operator = SimpleNamespace(
        state_history=SimpleNamespace(snapshot=lambda **kw: (history.copy(), np.ones(100, dtype=bool), {}))
    )
    args = SimpleNamespace(
        state_history_fps=30.0,
        episode_start_timestamp=0.0,
        episode_start_qpos=np.zeros(7),
        state_history_max_gap_seconds=0.02,
        instruction=deployment.BOOK_INSTRUCTION,
    )
    connection = SimpleNamespace(
        events=lambda payload: payloads.append(payload)
        or iter(
            [
                {"event": "phase_decision", "need_recovery": False, "need_recovery_probs": [1.0, 0.0]},
            ]
        )
    )
    result = client._assess_phase(
        args=args,
        client=connection,
        operator=operator,
        stats=StateQuantileStats(q01=np.zeros(7), q99=np.ones(7)),
        gate=SimpleNamespace(handle_phase_event=handled.append),
        phase="execution",
        generation=0,
        attempt_id=1,
        request_id="test",
        captured_step=1,
        after_timestamp=9.0,
        recovery_plan="",
        submitted_monotonic=0.0,
    )
    assert captures == [{"after_timestamp": 9.0}] and len(handled) == 1
    assert result.qpos_h100_11_discrete[-1] == [255] * 7
    validate_no_touch(payloads[0]["prompt"])
    assert not any("tactile" in key for key in payloads[0])
    assert result.synchronized_timestamps == {"qpos_timestamp": 10.0}


@pytest.mark.parametrize("mutation", ["label", "prompt", "selection", "audit", "hash", "version"])
def test_index_validator_rejects_even_rehashed_conversion_corruption(tmp_path, monkeypatch, mutation):
    source_path = tmp_path / "source.json"
    source_path.write_text("{}")
    template = {
        "experiment_version": data.VERSION_TAG,
        "schema_version": data.INDEX_SCHEMA,
        "source_multitask_index_file": str(source_path),
        "source_multitask_index_sha256": sha256_file(source_path),
        "input_policy": INPUT_POLICY,
        "splits": {"train": {"need": {"sample_count": 1}}},
    }
    manifests = {task: [{"prompt": "Mode: phase\nTask: book", "need_recovery": True}] for task in data.TASKS}
    audits = {"need_boundary_audit": {"removed": 1}, "reasoning_boundary_audit": {"C": 27}}
    monkeypatch.setattr(data, "load_source", lambda _: ({}, {}))
    monkeypatch.setattr(
        data,
        "derive",
        lambda *a: (
            copy.deepcopy(template),
            copy.deepcopy(manifests),
            audits["need_boundary_audit"],
            audits["reasoning_boundary_audit"],
        ),
    )
    index = copy.deepcopy(template)
    for task, rows in manifests.items():
        path = tmp_path / f"{task}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        index[f"{task}_manifest_file"], index[f"{task}_manifest_sha256"] = str(path), sha256_file(path)
    for name, audit in audits.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(audit))
        index[f"{name}_file"], index[f"{name}_sha256"] = str(path), sha256_file(path)
    index["training_data_hash"] = sha256_json(index)
    data.validate_index(index)
    if mutation in {"label", "prompt"}:
        path = Path(index["need_manifest_file"])
        row = dict(manifests["need"][0])
        row["need_recovery" if mutation == "label" else "prompt"] = False if mutation == "label" else "Task: changed"
        path.write_text(json.dumps(row) + "\n")
        index["need_manifest_sha256"] = sha256_file(path)
    elif mutation == "selection":
        index["splits"]["train"]["need"]["sample_count"] = 2
    elif mutation == "audit":
        path = Path(index["need_boundary_audit_file"])
        path.write_text('{"removed":0}')
        index["need_boundary_audit_sha256"] = sha256_file(path)
    elif mutation == "version":
        index["experiment_version"] = "book_v9_5"
    index["training_data_hash"] = (
        "invalid" if mutation == "hash" else sha256_json({k: v for k, v in index.items() if k != "training_data_hash"})
    )
    with pytest.raises(ValueError):
        data.validate_index(index)


def test_visual_control_loop_recovers_seven_times_without_touch(client, monkeypatch, tmp_path):
    norm = tmp_path / "norm.json"
    norm.write_text("{}")
    args, _ = client.get_arguments(
        ["--noise-seed", "42", "--gripper-min", "0.025", "--no-publish", "--norm-stats-file", str(norm)]
    )
    metadata = valid_metadata() | {"norm_stats_sha256": sha256_file(norm), "instruction": deployment.BOOK_INSTRUCTION}
    observation = SimpleNamespace(
        qpos=np.zeros(7),
        timestamp=1.0,
        tactile_caption="",
        img_front=np.zeros((2, 2, 3), dtype=np.uint8),
        img_left=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    logs, plans = [], []

    class ImmediateExecutor:
        def __init__(self, **kwargs):
            pass

        def submit(self, function, *args, **kwargs):
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except Exception as exc:
                future.set_exception(exc)
            return future

        def shutdown(self, **kwargs):
            pass

    def phase_worker(**kwargs):
        assert "captioner" not in kwargs
        decision = {
            "event": "phase_decision",
            "request_id": kwargs["request_id"],
            "phase": kwargs["phase"],
            "need_recovery": kwargs["phase"] == "execution",
            "adjustment_end": kwargs["phase"] == "adjustment",
            "adjustment_end_probs": [0.1, 0.9],
        }
        kwargs["gate"].handle_phase_event(decision)
        events = [decision]
        if kwargs["phase"] == "execution":
            failure = {"event": "failure_reason", "request_id": kwargs["request_id"], "failure_reason": FAILURE}
            kwargs["gate"].handle_phase_event(failure)
            events.append(failure)
        return client.PhaseAssessmentResult(
            kwargs["request_id"],
            kwargs["phase"],
            kwargs["generation"],
            kwargs["attempt_id"],
            kwargs["captured_step"],
            observation,
            {},
            "visual phase",
            [[0] * 7] * 11,
            events,
            0.0,
            0.0,
        )

    def request_plan(payload, **kwargs):
        validate_no_touch(payload["prompt"])
        plans.append(payload)
        return {"recovery_plan": PLAN}

    operator = SimpleNamespace(
        reset_state_history=lambda: None,
        state_history=SimpleNamespace(push=lambda *a: None),
        is_shutdown=lambda: False,
        rate=lambda *a: SimpleNamespace(sleep=lambda: None),
    )
    monkeypatch.setattr(client, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(client, "load_state_quantiles", lambda *a: None)
    monkeypatch.setattr(client, "_run_phase_assessment", phase_worker)
    monkeypatch.setattr(client, "_request_action_chunk", lambda **kw: np.zeros((30, 7)))
    monkeypatch.setattr(client, "_capture_action_observation", lambda *a, **kw: observation)
    monkeypatch.setattr(client.v52, "_latest_feedback", lambda *a: (np.zeros(7), 1.0))
    monkeypatch.setattr(client.v53, "_wait_feedback_after", lambda *a: (np.zeros(7), 2.0))
    keyboard = SimpleNamespace(get_key=lambda: args.success_key if len(plans) >= 7 else None)
    action = SimpleNamespace(
        get_server_metadata=lambda: metadata, infer_with_timeout=request_plan, _ws=SimpleNamespace(close=lambda: None)
    )
    phase = SimpleNamespace(metadata=metadata, close=lambda: None)
    client.runtime.shutdown_event.clear()
    try:
        with pytest.raises(client.OperatorStopError, match="operator_success"):
            client.run_book_v9_6_async(args, operator, action, phase, keyboard, SimpleNamespace(record=logs.append))
    finally:
        client.runtime.shutdown_event.clear()
    assert len(plans) == 7
    assert all(
        len(row["memory"]) <= 4 and row["memory"][0]["recovery_plan"] == "initial plan"
        for row in logs
        if row["event"] == "failure_and_plan"
    )
    assert max(row["attempt_id"] for row in logs if row["event"] == "phase_transition") >= 7


def test_visual_server_warmup_preserves_streaming_output_protocol():
    from scripts import serve_tactile_vla_book_v9_6 as server

    class Policy:
        _need_threshold = 0.7

        def infer_events(self, payload):
            validate_no_touch(payload["prompt"])
            if payload["mode"] == "phase":
                yield {"event": "phase_decision"}
                if payload["phase"] == "execution":
                    assert self._need_threshold == 0.0
                    yield {"event": "failure_reason", "failure_reason": FAILURE}
            elif payload["mode"] == "execution":
                yield {"actions": np.zeros((30, 7))}
            else:
                yield {"recovery_plan": PLAN}

    policy = Policy()
    result = server.warm_up(policy)
    assert policy._need_threshold == 0.7
    assert result["action_shape"] == [30, 7] and len(result["plan"]) == 8
    assert [event["event"] for event in result["execution"]] == ["phase_decision", "failure_reason"]


def test_existing_robot_yaml_ignores_tactile_and_preserves_cli_topics(client, monkeypatch):
    args, parser = client.get_arguments(
        [
            "--noise-seed",
            "42",
            "--gripper-min",
            "0.025",
            "--config_path",
            "/tmp/robot.yaml",
            "--img_left_topic",
            "cli_wrist",
        ]
    )
    monkeypatch.setattr(
        client.runtime,
        "_read_yaml",
        lambda _: {
            "dataInfo": {
                "camera": {"color": {"names": ["front", "left"], "topics": ["yaml_front", "yaml_wrist"]}},
                "arm": {"jointState": {"names": ["puppetRight", "masterRight"], "topics": ["yaml_qpos", "yaml_cmd"]}},
                "tactile": {"force": {"names": ["left", "right"], "topics": ["tactile_left", "tactile_right"]}},
            }
        },
    )
    client.apply_visual_yaml_defaults(args, parser)
    assert args.img_front_topic == "yaml_front" and args.img_left_topic == "cli_wrist"
    assert args.puppet_arm_topic == "yaml_qpos" and args.puppet_arm_cmd_topic == "yaml_cmd"
    assert not any("tactile" in name or "captioner" in name for name in vars(args))
