from __future__ import annotations

# ruff: noqa: E402, SLF001
import copy
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

from tactile_vla.vla import sponge_v10_2_multitask_data as data
from tactile_vla.vla import sponge_v10_2_runtime as deployment
from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.sponge_v10_2_prompts import (
    INPUT_POLICY,
    PROMPT_PROFILE,
    build_phase_prompt,
    build_recovery_prompt,
    remove_touch,
    validate_no_touch,
)
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt as tactile_plan
from tactile_vla.vla.structured_text import legal_failure_reasons, legal_recovery_plans
from tactile_vla.vla.v7_7_phase_prompt import build_phase_prompt as tactile_phase

CAPTION = "Touch[area=small; Fx=negative; Fy=positive; Fz=negative; Fz_bias=left; rotation=counterclockwise]"
INSTRUCTION = deployment.SPONGE_INSTRUCTION


def phase_prompt():
    return tactile_phase(
        instruction=INSTRUCTION,
        tactile_caption=CAPTION,
        recovery_plan="",
        qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32),
    )


@pytest.mark.parametrize("task", ["need", "failure", "adjustment"])
def test_phase_removes_only_touch_and_matches_online_builder(task):
    assert remove_touch(phase_prompt(), task=task) == build_phase_prompt(
        instruction=INSTRUCTION, recovery_plan="", qpos_h100_11_discrete=np.zeros((11, 7), dtype=np.int32)
    )


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_plan_keeps_fz_failure_memory_without_caption(length):
    memory = data.source.make_memory(length, split="train", global_index=714)
    original = tactile_plan(
        instruction=INSTRUCTION,
        failed_tactile_caption=CAPTION,
        failure_recovery_memory=memory,
        prompt_profile=MINIMAL_PROMPT_PROFILE,
    )
    actual = build_recovery_prompt(instruction=INSTRUCTION, failure_recovery_memory=memory)
    assert actual == remove_touch(original, task="plan")
    assert actual.count(data.FAILURE_TARGET) == length and "Touch" not in actual


@pytest.mark.parametrize("text", [CAPTION, "Touch: none", "rotation=none", "Fz_bias=left", "tactile_caption=none"])
def test_no_touch_guard_rejects_neutral_too(text):
    with pytest.raises(ValueError):
        validate_no_touch(text)


@pytest.mark.parametrize("text", ["Mode: phase\nTask: sponge", phase_prompt() + "\nTouch: " + CAPTION + "\n"])
def test_missing_duplicate_touch_is_not_silently_converted(text):
    with pytest.raises(ValueError):
        remove_touch(text, task="need")


def source_fixture():
    rows = {task: [] for task in data.TASKS}
    splits = {split: {"action": {"indices": [10, 20, 30]}} for split in ("train", "val", "test")}
    for split in splits:
        for task in data.TASKS:
            g = len(rows[task]) * 10
            row = {
                "split": split,
                "global_index": g,
                "episode_id": 21,
                "attempt_id": 1,
                "frame_index": 714,
                "timestamp": 123.0,
                "source": "real",
                "schema_version": "old",
                "data_profile": "old",
                "prompt": phase_prompt(),
                "qpos_h100_11_discrete": np.zeros((11, 7), dtype=np.int32).tolist(),
                "history_sources": [{"global_index": g}],
                "effective_episode_history_length": 100,
                "token_lengths": {"total": 400},
            }
            if task == "need":
                row["need_recovery"] = True
            elif task == "adjustment":
                row.update(adjustment_end=False, selected=True, rexecution_frame=200)
            elif task == "failure":
                row.update(target_failure_reason=data.FAILURE_TARGET, B=711, C=714, frame_offset=0)
            else:
                memory = data.source.make_memory(4, split=split, global_index=g)
                row.update(memory_length=4, failure_recovery_memory=memory, target_recovery_plan=data.PLAN_TARGET)
                row["prompt"] = tactile_plan(
                    instruction=INSTRUCTION,
                    failed_tactile_caption=CAPTION,
                    failure_recovery_memory=memory,
                    prompt_profile=MINIMAL_PROMPT_PROFILE,
                )
            position = len(rows[task])
            rows[task].append(row)
            splits[split][task] = {"manifest_row_indices": [position], "global_indices": [g], "sample_count": 1}
    return {"splits": splits, "training_data_hash": "a" * 64, "source_hashes": {}}, rows


def test_conversion_keeps_every_row_label_history_memory_and_order():
    _, original = source_fixture()
    snapshot = copy.deepcopy(original)
    converted = data.convert_manifests(original)
    assert original == snapshot
    for task in data.TASKS:
        for before, after in zip(original[task], converted[task], strict=True):
            assert {
                k: v
                for k, v in before.items()
                if k not in {"prompt", "schema_version", "data_profile", "token_lengths"}
            } == {
                k: v for k, v in after.items() if k not in {"prompt", "schema_version", "data_profile", "token_lengths"}
            }
            validate_no_touch(after["prompt"])
            assert "token_lengths" not in after


def test_derive_has_exact_same_splits_and_offline_provenance(tmp_path, monkeypatch):
    original, rows = source_fixture()
    path = tmp_path / "source.json"
    path.write_text(json.dumps(original))
    monkeypatch.setattr(data, "validate_tokens", lambda *_: {"validated": True})
    index, _ = data.derive(original, rows, path)
    assert index["splits"] == original["splits"]
    assert index["source_multitask_index_sha256"] == sha256_file(path)
    assert index["source_multitask_training_data_hash"] == original["training_data_hash"]
    assert (
        index["input_policy"] == INPUT_POLICY and index["captioner_identity_scope"] == "offline_label_provenance_only"
    )


@pytest.mark.parametrize("mutation", ["label", "prompt", "history", "split", "memory"])
def test_manifest_semantic_tampering_rejected_even_after_rehash(tmp_path, monkeypatch, mutation):
    original, source_rows = source_fixture()
    path = tmp_path / "source.json"
    path.write_text(json.dumps(original))
    monkeypatch.setattr(data, "validate_tokens", lambda *_: {"validated": True})
    monkeypatch.setattr(data, "load_source", lambda _: (original, source_rows))
    index, rows = data.derive(original, source_rows, path)
    for task in data.TASKS:
        manifest = tmp_path / f"{task}.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows[task]))
        index[f"{task}_manifest_file"] = str(manifest)
        index[f"{task}_manifest_sha256"] = sha256_file(manifest)
    index["training_data_hash"] = sha256_json(index)
    data.validate_index(index)
    task = "plan" if mutation == "memory" else "need"
    edited = copy.deepcopy(rows[task])
    if mutation == "label":
        edited[0]["need_recovery"] = False
    elif mutation == "prompt":
        edited[0]["prompt"] += "\nTouch: none"
    elif mutation == "history":
        edited[0]["qpos_h100_11_discrete"][0][0] = 42
    elif mutation == "split":
        edited[0]["split"] = "test"
    else:
        edited[0]["failure_recovery_memory"][0]["failure_reason"] = "wrong"
    manifest = Path(index[f"{task}_manifest_file"])
    manifest.write_text("".join(json.dumps(r) + "\n" for r in edited))
    index[f"{task}_manifest_sha256"] = sha256_file(manifest)
    index["manifest_content_hashes"][task] = sha256_json(edited)
    index.pop("training_data_hash")
    index["training_data_hash"] = sha256_json(index)
    with pytest.raises(ValueError):
        data.validate_index(index)


def test_builder_checkpoint_free_and_refuses_overwrite(tmp_path):
    from scripts.build_sponge_v10_2_multitask_data import parse_args, main

    args = parse_args([])
    assert args.source_index == data.DEFAULT_SOURCE_INDEX and args.output_dir == data.DEFAULT_OUTPUT
    assert not hasattr(args, "stage_a_checkpoint")
    with pytest.raises(FileExistsError):
        main(["--output-dir", str(tmp_path)])


def test_stage_a_validator_reuses_v10_1(monkeypatch):
    monkeypatch.setattr(
        data.source, "validate_stage_a_for_training", lambda *_: {"experiment_version": "sponge_v10_1", "step": 15000}
    )
    assert data.validate_stage_a_for_training("checkpoint", {})["experiment_version"] == "sponge_v10_1"
    assert "v10_1" in str(data.DEFAULT_STAGE_A)


def test_training_defaults_and_configuration_in_isolated_process():
    code = """from scripts import train_vla_multitask_sponge_v10_2 as entry
entry.configure_version()
args = entry.parse_args(["--data-only-dry-run"])
assert args.num_steps == 2000 and args.eval_interval == args.save_interval == args.keep_period == 1000
assert args.dry_run and args.data_only_dry_run and args.batch_size == 8 and args.lr == 1e-4
assert "v10_1" in str(args.stage_a_checkpoint) and "v10_2" in str(args.index_file)
assert args.prompt_profile == entry.PROMPT_PROFILE
entry.configure_training()
assert entry.base.EXTRA_CONFIG["input_policy"] == entry.INPUT_POLICY
assert entry.base.EXTRA_CONFIG["include_fz_bias_failure_grammar"] is True
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def client():
    path = ROOT / "openpi/inference/agilex/inference/agilex_inference_sponge_v10_2_asyn.py"
    spec = importlib.util.spec_from_file_location("sponge_v102_async_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visual_ros_has_only_camera_and_qpos_subscriptions(client):
    topics = []
    operator = object.__new__(client.VisualRosOperator)
    operator.args = SimpleNamespace(
        img_front_topic="front", img_left_topic="wrist", puppet_arm_topic="qpos", puppet_arm_cmd_topic="cmd"
    )
    operator.Image = operator.JointState = object
    operator.img_front_callback = operator.img_left_callback = operator.puppet_arm_callback = lambda _: None
    operator.rospy = SimpleNamespace(
        init_node=lambda *a, **k: None,
        Subscriber=lambda topic, *a, **k: topics.append(topic),
        Publisher=lambda *a, **k: object(),
    )
    operator._init_ros()
    assert topics == ["front", "wrist", "qpos"] and operator.tactile is None


def test_client_arguments_require_no_captioner(client, tmp_path):
    norm = tmp_path / "norm.json"
    norm.write_text("{}")
    args, parser = client.get_arguments(
        ["--noise-seed", "42", "--gripper-min", "0.015", "--norm-stats-file", str(norm)]
    )
    client.validate_args(args, parser)
    assert not hasattr(args, "captioner_checkpoint") and not hasattr(args, "tactile_window_size")
    assert args.need_recovery_rate_hz == args.adjustment_end_rate_hz == 7.0


def test_capture_is_visual_qpos_only_and_requests_fresh_frame(client):
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
    assert calls == [{"after_timestamp": 9.0}] and result.timestamp == 10.0 and result.tactile_caption == ""


def test_predicted_fz_failure_memory_unlimited(client):
    memory = []
    for i in range(20):
        payload, memory = client.plan_payload_after_failure(
            instruction=INSTRUCTION,
            executed_recovery_plan="" if i == 0 else data.PLAN_TARGET,
            failure_reason=data.FAILURE_TARGET,
            memory=memory,
            image=np.zeros((2, 2, 3)),
            wrist_image=np.zeros((2, 2, 3)),
            qpos=np.zeros(7),
        )
        assert len(memory) == min(i + 1, 4) and memory[0]["recovery_plan"] == "initial plan"
        validate_no_touch(payload["prompt"])
    assert memory[-1]["failure_reason"] == data.FAILURE_TARGET


def test_true_need_immediately_holds_and_rejects_stale_action(client):
    held = []
    gate = client.SpongeV102AsyncControlGate(hold=lambda: held.append(True), reset_hold=lambda: None)
    gate.begin_phase_request("req")
    generation = gate.snapshot()[3]
    gate.handle_phase_event(
        {"event": "phase_decision", "request_id": "req", "phase": "execution", "need_recovery": True}
    )
    assert held and not gate.may_publish_action(generation)


def test_server_rejects_touch_prompts_and_payloads():
    from scripts import serve_tactile_vla_sponge_v10_2 as server

    policy = object.__new__(server.SpongeV102Policy)
    with pytest.raises(ValueError, match="tactile"):
        next(policy.infer_events({"mode": "phase", "prompt": phase_prompt()}))
    prompt = remove_touch(phase_prompt(), task="need")
    with pytest.raises(ValueError, match="tactile"):
        next(policy.infer_events({"mode": "phase", "prompt": prompt, "observation/tactile": [1]}))
    assert not hasattr(server.parse_args([]), "captioner_checkpoint_sha256")


def valid_identity():
    return {
        "training_data_hash": "a" * 64,
        "source_multitask_training_data_hash": "f" * 64,
        "source_scope": {
            "schema_version": "sponge_v10_1_source_scope_v1",
            "selection": "explicit_selected_dataset_and_hashed_V4_sources_no_recursive_archive_scan",
            "dataset_dir": "/dataset",
            "v4_index_file": "/index",
        },
        "stage_a_initialization_identity": {
            "step": 15000,
            "experiment_version": "sponge_v10_1",
            "action_training_data_hash": "b" * 64,
            "config_sha256": "c" * 64,
            "params_metadata_sha256": "d" * 64,
        },
        "boundary_audit_sha256": "e" * 64,
    }


def valid_config():
    return (
        valid_identity()
        | deployment.deployment_policy_fields(data.VERSION_TAG)
        | {
            "data_profile": data.DATA_PROFILE,
            "experiment_version": data.VERSION_TAG,
            "prompt_profile": PROMPT_PROFILE,
            "dataset_dir": "/dataset",
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
            "failure_grammar": list(legal_failure_reasons(include_fz_bias=True)),
            "recovery_grammar": list(legal_recovery_plans()),
            "artifact_identity": {
                "data_profile": "book_stage_a_v1",
                "v4_norm_stats_sha256": "c" * 64,
                "sponge_v10_2_training_data_hash": "a" * 64,
                "training_data_hash": "b" * 64,
            },
        }
    )


def test_deployment_config_keeps_extended_grammar_and_v10_1_stage_a():
    config = valid_config()
    deployment.validate_training_config(config, "c" * 64)
    config["stage_a_initialization_identity"]["experiment_version"] = "sponge_v10_2"
    with pytest.raises(ValueError, match="V10.1 Stage A"):
        deployment.validate_training_config(config, "c" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_policy", {}),
        ("prompt_profile", "phase_v7_7_h100_11_answer"),
        ("data_profile", "sponge_v10_1_five_task_h100"),
        ("include_fz_bias_failure_grammar", False),
        ("source_multitask_training_data_hash", "wrong"),
    ],
)
def test_config_rejects_wrong_version_input_and_provenance(field, value):
    with pytest.raises(ValueError):
        deployment.validate_training_config(valid_config() | {field: value}, "c" * 64)


def test_thresholds_require_v10_2_identity():
    kwargs = dict(
        step=1000,
        full_params_sha="a" * 64,
        norm_sha="b" * 64,
        training_data_hash="c" * 64,
        need_override=None,
        adjustment_override=None,
    )
    thresholds, overrides = deployment.resolve_thresholds(calibration=None, **kwargs)
    assert thresholds == {"need_recovery": 0.5, "adjustment_end": 0.5} and not any(overrides.values())
    with pytest.raises(ValueError):
        deployment.resolve_thresholds(calibration={"schema_version": "sponge_v10_1_val_thresholds_v1"}, **kwargs)


def test_warmup_has_no_caption_and_all_memory_lengths():
    from scripts import serve_tactile_vla_sponge_v10_2 as server

    class FakePolicy:
        _need_threshold = 0.5

        def __init__(self):
            self.requests = []

        def infer_events(self, request):
            self.requests.append(request)
            validate_no_touch(request["prompt"])
            if request["mode"] == "phase":
                yield {"event": "phase_decision"}
                if request["phase"] == "execution":
                    yield {"event": "failure_reason", "failure_reason": data.FAILURE_TARGET}
            elif request["mode"] == "reasoning":
                yield {"recovery_plan": data.PLAN_TARGET}
            else:
                yield {"actions": np.zeros((30, 7))}

    policy = FakePolicy()
    result = server.warm_up(policy)
    assert [p["memory_length"] for p in result["plan"]] == [1, 2, 3, 4]
    assert result["action_shape"] == [30, 7] and policy._need_threshold == 0.5


def test_phase_assessment_fresh_visual_only_with_h100(client, monkeypatch):
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
        state_history_max_gap_seconds=0.04,
        instruction=INSTRUCTION,
    )
    connection = SimpleNamespace(
        events=lambda payload: payloads.append(payload)
        or iter([{"event": "phase_decision", "need_recovery": False, "need_recovery_probs": [1.0, 0.0]}])
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
    assert not any("tactile" in key or "caption" in key for key in payloads[0])


def test_model_loader_uses_visual_profile_without_mutating_v10_1(monkeypatch):
    from scripts import serve_tactile_vla_sponge_v10_2 as server

    seen = []
    original_profile, original_prompt = server.base.DATA_PROFILE, server.base.PROMPT_PROFILE

    def fake_init(self, **kwargs):
        seen.append((server.base.DATA_PROFILE, server.base.PROMPT_PROFILE))
        self._model, self._metadata = object(), {}

    monkeypatch.setattr(server.base.V77Policy, "__init__", fake_init)
    monkeypatch.setattr(server.base.v3.nnx, "state", lambda _: {})
    monkeypatch.setattr(server, "parameter_tree_sha256", lambda _: "a" * 64)
    server.SpongeV102Policy(
        args=SimpleNamespace(),
        config=valid_config() | {"captioner_identity": {"offline_only": True}},
        model_config=None,
        norm_stats=None,
        identity={"full_params_sha256": "a" * 64},
        deployment={},
    )
    assert seen == [(data.DATA_PROFILE, PROMPT_PROFILE)]
    assert (server.base.DATA_PROFILE, server.base.PROMPT_PROFILE) == (original_profile, original_prompt)


def metadata_fixture():
    return (
        valid_identity()
        | deployment.deployment_policy_fields(data.VERSION_TAG)
        | {
            "data_profile": data.DATA_PROFILE,
            "experiment_version": data.VERSION_TAG,
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
            "max_supported_attempts": None,
            "classification_qpos_policy": "raw_no_gripper_remap",
            "episode_start_padding": "left_pad_episode_frame_0",
            "requires_captioner": False,
            "requires_tactile_topics": False,
            "need_recovery_threshold": 0.5,
            "adjustment_end_threshold": 0.8,
            "thresholds_status": "explicit_manual_override",
        }
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("requires_captioner", True),
        ("requires_tactile_topics", True),
        ("input_policy", {}),
        ("max_memory_pairs", 1),
        ("source_multitask_training_data_hash", "wrong"),
    ],
)
def test_client_metadata_contract(field, value):
    metadata = metadata_fixture()
    deployment.validate_server_metadata(metadata)
    with pytest.raises(ValueError):
        deployment.validate_server_metadata(metadata | {field: value})
