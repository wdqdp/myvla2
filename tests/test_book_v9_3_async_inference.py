from __future__ import annotations

# ruff: noqa: E402, SLF001
import asyncio
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "openpi/src"),
                str(PROJECT_ROOT / "openpi/packages/openpi-client/src")]

from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.book_v9_3_multitask_data import DATA_PROFILE
from tactile_vla.vla.book_v9_3_runtime import (
    BOOK_INSTRUCTION, HISTORY_OFFSETS, MEMORY_POLICY, SERVER_NAME, THRESHOLD_SCHEMA,
    latest_failure_memory, resolve_thresholds, validate_server_metadata, validate_training_config,
)
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats
from tactile_vla.vla.v7_7_phase_prompt import PROMPT_PROFILE
from scripts import serve_tactile_vla_book_v9_3 as server
from scripts.calibrate_book_v9_3_thresholds import collect_predictions, select_from_val


FAILURE = "failure_reason=rotate right,grasp appropriate."
PLAN = "recovery_plan=move horizontally right moderately, move vertically none moderately."


@pytest.fixture(scope="module")
def client():
    path = PROJECT_ROOT / "openpi/inference/agilex/inference/agilex_inference_book_v9_3_asyn.py"
    spec = importlib.util.spec_from_file_location("book_v93_async_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_metadata():
    return {
        "name": SERVER_NAME, "data_profile": DATA_PROFILE, "phase_prompt_profile": PROMPT_PROFILE,
        "action_prompt_profile": "phase_v2", "supports_streamed_phase_events": True,
        "supports_action_noise": True, "requires_action_noise": True,
        "supports_failure_generation": True, "supports_recovery_generation": True,
        "action_horizon": 30, "action_dim": 32, "output_action_dim": 7,
        "use_state_history": False, "state_history_len": 0, "state_history_fps": 30.0,
        "qpos_h100_sample_offsets": HISTORY_OFFSETS, "max_memory_pairs": 1,
        "memory_policy": MEMORY_POLICY, "max_supported_attempts": None,
        "classification_qpos_policy": "raw_no_gripper_remap",
        "episode_start_padding": "left_pad_episode_frame_0", "captioner_window_size": 30,
        "need_recovery_threshold": 0.6, "adjustment_end_threshold": 0.7,
        "thresholds_status": "calibrated_on_book_val", "instruction": BOOK_INSTRUCTION,
        "threshold_manual_overrides": {"need_recovery": False, "adjustment_end": False},
    }


def calibration():
    return {
        "schema_version": THRESHOLD_SCHEMA, "data_profile": DATA_PROFILE,
        "prompt_profile": PROMPT_PROFILE, "checkpoint_step": 4000,
        "full_params_sha256": "a" * 64, "norm_stats_sha256": "b" * 64,
        "training_data_hash": "c" * 64, "selection_split": "val",
        "thresholds_status": "calibrated_on_book_val",
        "thresholds": {"need_recovery": 0.6, "adjustment_end": 0.7},
    }


def resolve(document, **kwargs):
    return resolve_thresholds(calibration=document, step=4000, full_params_sha="a" * 64,
                              norm_sha="b" * 64, training_data_hash="c" * 64,
                              need_override=kwargs.get("need"), adjustment_override=kwargs.get("adjustment"))


def test_thresholds_require_calibration_or_explicit_override():
    with pytest.raises(ValueError, match="placeholders"):
        resolve(None)
    thresholds, overrides = resolve(calibration())
    assert thresholds == {"need_recovery": 0.6, "adjustment_end": 0.7}
    assert not any(overrides.values())
    thresholds, overrides = resolve(None, need=0.4, adjustment=0.8)
    assert thresholds["adjustment_end"] == 0.8 and all(overrides.values())
    with pytest.raises(ValueError, match="adjustment_end"):
        resolve(None, need=0.4)


@pytest.mark.parametrize("field,value", [("checkpoint_step", 3500), ("full_params_sha256", "d" * 64),
                                         ("norm_stats_sha256", "d" * 64), ("selection_split", "test")])
def test_threshold_identity_prevents_cross_checkpoint_or_test_selection(field, value):
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve(calibration() | {field: value})


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_manual_thresholds_rejected(threshold):
    with pytest.raises(ValueError, match="finite"):
        resolve(calibration(), need=threshold)


def test_metadata_rejects_v9_2_history_or_multi_pair_memory():
    metadata = valid_metadata()
    validate_server_metadata(metadata)
    for field, value in (("qpos_h100_sample_offsets", [0, 10, 20, 30, 40, 50, 59, 69, 79, 89, 99]),
                         ("max_memory_pairs", 4), ("max_supported_attempts", 5),
                         ("thresholds_status", "uncalibrated_placeholders_not_for_robot")):
        with pytest.raises(ValueError):
            validate_server_metadata(metadata | {field: value})


def test_config_checks_book_norm_and_profile():
    config = {
        "data_profile": DATA_PROFILE, "prompt_profile": PROMPT_PROFILE,
        "action_horizon": 30, "action_dim": 32, "max_token_len": 512,
        "reasoning_max_token_len": 320, "use_state_history": False, "state_history_len": 0,
        "grammar_profile": "v3_full_v1", "phase_prefill_protocol": "need_failure_shared_kv_v1",
        "artifact_identity": {"data_profile": "book_stage_a_v1", "v4_norm_stats_sha256": "a" * 64,
                              "book_v9_3_training_data_hash": "c" * 64},
    }
    validate_training_config(config, "a" * 64)
    with pytest.raises(ValueError, match="norm_stats"):
        validate_training_config(config, "b" * 64)


def test_memory_replaced_for_arbitrarily_many_failures(client):
    memory = []
    for attempt in range(20):
        payload, memory = client.plan_payload_after_failure(
            instruction=BOOK_INSTRUCTION, tactile_caption="Touch[test]",
            executed_recovery_plan="" if attempt == 0 else PLAN, failure_reason=FAILURE,
            memory=memory, image=np.zeros((2, 2, 3)), wrist_image=np.zeros((2, 2, 3)), qpos=np.zeros(7),
        )
        assert len(memory) == 1
        assert payload["prompt"].count("failure_reason=") == 1
        assert memory == latest_failure_memory(executed_recovery_plan="" if attempt == 0 else PLAN,
                                               failure_reason=FAILURE)
        assert "initial plan" in payload["prompt"] if attempt == 0 else "initial plan" not in payload["prompt"]


def test_h100_floor_offsets_and_start_padding(client):
    stats = StateQuantileStats(q01=np.zeros(7), q99=np.ones(7))
    history = client.ContinuousH100()
    for index in range(100):
        history.append(np.full(7, index / 100))
    actual = history.sampled_discrete(stats)
    expected = client.discretize_state_qpos(np.array([np.full(7, index / 100) for index in HISTORY_OFFSETS]), stats)
    np.testing.assert_array_equal(actual, expected)
    history.reset_episode()
    history.append(np.full(7, 0.1))
    history.append(np.full(7, 0.2))
    padded = history.sampled_discrete(stats)
    np.testing.assert_array_equal(padded[:10], np.repeat(padded[:1], 10, axis=0))
    assert not np.array_equal(padded[0], padded[-1])


def test_frequency_flags_and_unlimited_attempt_defaults(client):
    args, _ = client.get_arguments(["--noise-seed", "42", "--gripper-min", "0.032",
                                    "--need-recovery-rate-hz", "4", "--adjustment-end-rate-hz", "9"])
    assert args.need_recovery_rate_hz == 4 and args.adjustment_end_rate_hz == 9
    assert args.max_attempts is None and args.instruction == BOOK_INSTRUCTION
    assert args.publish_rate == 30


def test_client_validation_forbids_remap_and_invalid_frequency(client, tmp_path):
    args, parser = client.get_arguments(["--noise-seed", "42", "--gripper-min", "0.032"])
    args.norm_stats_file = args.captioner_checkpoint = tmp_path / "exists"
    args.norm_stats_file.touch()
    client.validate_args(args, parser)
    args.need_recovery_rate_hz = 0
    with pytest.raises(SystemExit):
        client.validate_args(args, parser)
    args.need_recovery_rate_hz = 7
    args.classification_gripper_open_threshold = 0.09
    with pytest.raises(SystemExit):
        client.validate_args(args, parser)


def test_true_event_stops_mid_chunk_and_late_result_is_discarded(client):
    held, published = [], []
    gate = client.BookV93AsyncControlGate(hold=lambda: held.append(True))
    gate.begin_phase_request("p")
    def poll():
        if len(published) == 6:
            gate.handle_phase_event({"event": "phase_decision", "request_id": "p",
                                     "phase": "execution", "need_recovery": True})
    assert gate.publish_chunk(range(30), generation=0, publish=published.append, before_each=poll) == 6
    assert held and not gate.accept_action_result(0, [1])
    assert gate.handle_phase_event({"event": "failure_reason", "request_id": "p", "failure_reason": FAILURE})
    gate.switch_phase("adjustment", increment_attempt=True)
    assert not gate.handle_phase_event({"event": "phase_decision", "request_id": "p", "phase": "execution",
                                       "need_recovery": True})
    generation = gate.begin_action_request()
    assert gate.snapshot()[-1]
    assert not gate.release_hold_with_fresh_actions(generation - 1, [1])
    assert gate.release_hold_with_fresh_actions(generation, [1])


def test_val_threshold_selection_ignores_test_labels():
    val = [{"label": 0, "probability": 0.1}, {"label": 0, "probability": 0.2},
           {"label": 1, "probability": 0.8}, {"label": 1, "probability": 0.9}]
    test = [{"label": 0, "probability": 0.95}, {"label": 1, "probability": 0.05}]
    first = select_from_val(val, test, maximum_negative_fpr=0.01)
    second = select_from_val(val, [row | {"label": 1 - row["label"]} for row in test], maximum_negative_fpr=0.01)
    assert first["threshold"] == second["threshold"] == 0.8
    assert first["val"]["recall"] == 1 and first["test"]["recall"] == 0


def test_server_worker_streams_decision_before_decode_without_blocking_event_loop():
    from openpi_client import msgpack_numpy
    events = []
    main_thread = threading.get_ident()
    class FakePolicy:
        metadata = {"name": SERVER_NAME}
        def infer_events(self, request):
            assert threading.get_ident() != main_thread
            yield {"event": "phase_decision", "need_recovery": True}
            assert "phase_decision_sent" in events
            events.append("decode_started")
            yield {"event": "failure_reason", "failure_reason": FAILURE}
    class Socket:
        recv_count = 0
        async def recv(self):
            self.recv_count += 1
            if self.recv_count > 1:
                raise server.websockets.ConnectionClosedOK(None, None)
            return msgpack_numpy.Packer().pack({"mode": "phase"})
        async def send(self, payload):
            assert threading.get_ident() == main_thread
            value = msgpack_numpy.unpackb(payload)
            if value.get("event") == "phase_decision":
                events.append("phase_decision_sent")
            elif value.get("event") == "failure_reason":
                events.append("failure_sent")
        async def close(self, **kwargs):
            pass
    async def check():
        serving = server.WorkerStreamingPolicyServer(FakePolicy(), "localhost", 0)
        try:
            await asyncio.wait_for(serving._handler(Socket()), timeout=5)
        finally:
            serving._executor.shutdown(wait=True)
    asyncio.run(check())
    assert events == ["phase_decision_sent", "decode_started", "failure_sent"]


def test_real_control_loop_recovers_beyond_five_attempts(client, monkeypatch, tmp_path):
    args, _ = client.get_arguments(["--noise-seed", "42", "--gripper-min", "0.032", "--no-publish"])
    args.norm_stats_file = args.captioner_checkpoint = tmp_path / "identity"
    args.norm_stats_file.touch()
    metadata = valid_metadata() | {"norm_stats_sha256": sha256_file(args.norm_stats_file),
                                   "captioner_checkpoint_sha256": sha256_file(args.captioner_checkpoint)}
    observation = SimpleNamespace(qpos=np.zeros(7), timestamp=1.0, tactile_caption="Touch[test]",
                                  img_front=np.zeros((2, 2, 3), dtype=np.uint8),
                                  img_left=np.zeros((2, 2, 3), dtype=np.uint8))
    logs, plans = [], []
    class ImmediateExecutor:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, *args, **kwargs):
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except Exception as exc:
                future.set_exception(exc)
            return future
    def phase_worker(**kwargs):
        decision = {"event": "phase_decision", "request_id": kwargs["request_id"], "phase": kwargs["phase"],
                    "need_recovery": kwargs["phase"] == "execution",
                    "adjustment_end": kwargs["phase"] == "adjustment", "adjustment_end_probs": [0.1, 0.9]}
        events = [decision]
        kwargs["gate"].handle_phase_event(decision)
        if kwargs["phase"] == "execution":
            failure = {"event": "failure_reason", "request_id": kwargs["request_id"], "failure_reason": FAILURE}
            kwargs["gate"].handle_phase_event(failure)
            events.append(failure)
        return client.PhaseAssessmentResult(kwargs["request_id"], kwargs["phase"], kwargs["generation"],
                                            kwargs["attempt_id"], kwargs["captured_step"], observation, {},
                                            "phase prompt", [[0] * 7] * 11, events, 0.0, 0.0)
    def request_plan(payload, **kwargs):
        assert payload["prompt"].count("failure_reason=") == 1
        plans.append(payload)
        return {"recovery_plan": PLAN}
    class Keyboard:
        def get_key(self):
            return args.success_key if len(plans) >= 7 else None
    operator = SimpleNamespace(reset_state_history=lambda: None,
                               state_history=SimpleNamespace(push=lambda *args: None),
                               is_shutdown=lambda: False, rate=lambda *args: SimpleNamespace(sleep=lambda: None))
    monkeypatch.setattr(client, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(client, "load_state_quantiles", lambda *args: None)
    monkeypatch.setattr(client, "_run_phase_assessment", phase_worker)
    monkeypatch.setattr(client, "_request_action_chunk", lambda **kwargs: np.zeros((30, 7)))
    monkeypatch.setattr(client.v52, "_capture_observation", lambda *args, **kwargs: observation)
    monkeypatch.setattr(client.v52, "_latest_feedback", lambda *args: (np.zeros(7), 1.0))
    monkeypatch.setattr(client.v53, "_wait_feedback_after", lambda *args: (np.zeros(7), 2.0))
    client.runtime.shutdown_event.clear()
    client.run_book_v9_3_async(args, operator,
                              SimpleNamespace(get_server_metadata=lambda: metadata, infer_with_timeout=request_plan),
                              SimpleNamespace(metadata=metadata), None, Keyboard(),
                              SimpleNamespace(record=logs.append))
    assert len(plans) == 7
    transitions = [row for row in logs if row["event"] == "phase_transition"]
    assert max(row["attempt_id"] for row in transitions) == 8
    assert all(len(row["memory"]) == 1 for row in logs if row["event"] == "failure_and_plan")


def test_next_event_handles_empty_iterator():
    assert server.next_event(iter([])) == (True, None)


def test_calibration_collects_complete_manifest_with_runtime_prefill_route():
    manifests = {"need": [{"episode_id": 1, "attempt_id": 1, "frame_index": i,
                           "prompt": "phase prompt", "need_recovery": bool(i == 2),
                           "source": "pre_failure_hard_negative"} for i in range(3)]}
    index = {"splits": {"val": {"need": {"manifest_row_indices": [0, 1, 2],
                                          "global_indices": [0, 1, 2], "sample_count": 3}}}}
    dataset = [{"index": i, "episode_id": 1, "attempt_id": 1, "frame_index": i,
                "observation.images.front": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation.images.left": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation.state": np.zeros(7)} for i in range(3)]
    def transform(raw):
        return {"image": {"front": raw["observation/image"]}, "image_mask": {"front": np.bool_(True)},
                "state": raw["observation/state"]}
    batch_sizes = []
    def prefill(observation, compact):
        batch_sizes.append(observation.state.shape[0])
        return np.tile([0.0, 1.0], (observation.state.shape[0], 1)), None, None, None, None
    policy = SimpleNamespace(_assessment_transform=transform, _assessment_prefill=prefill,
                             _failure_grammar=SimpleNamespace(compact_token_ids=[0, 1]))
    rows = collect_predictions(policy, index, "need", "val", manifests, dataset,
                               SimpleNamespace(batch_size=2, num_workers=0))
    assert batch_sizes == [2, 1] and len(rows) == 3
    assert [row["label"] for row in rows] == [0, 0, 1]
    assert [row["global_index"] for row in rows] == [0, 1, 2]


def test_warm_up_checks_all_five_tasks_and_restores_threshold():
    class Policy:
        _need_threshold = 0.75
        seen = []
        def infer_events(self, request):
            assert BOOK_INSTRUCTION in request["prompt"]
            self.seen.append((request["mode"], request.get("phase")))
            if request["mode"] == "phase":
                yield {"event": "phase_decision"}
                if request["phase"] == "execution":
                    assert self._need_threshold == 0.0
                    yield {"event": "failure_reason", "failure_reason": FAILURE}
            elif request["mode"] == "execution":
                assert request["action_noise"].shape == (30, 32)
                yield {"actions": np.zeros((30, 7))}
            else:
                yield {"recovery_plan": PLAN}
    policy = Policy()
    result = server.warm_up(policy)
    assert policy._need_threshold == 0.75
    assert policy.seen == [("phase", "execution"), ("phase", "adjustment"),
                           ("reasoning", None), ("execution", None)]
    assert result["action_shape"] == [30, 7]


def test_streaming_phase_timeout_leaves_control_stopped(client):
    gate = client.BookV93AsyncControlGate(hold=lambda: None)
    gate.begin_phase_request("timeout-need")
    decision = {"event": "phase_decision", "request_id": "timeout-need", "phase": "execution",
                "need_recovery": True}
    gate.handle_phase_event(decision)
    assert gate.snapshot()[-1]
    # A missing failure/plan must never release this stop; only a fresh new-stage chunk may do so.
    assert not gate.accept_action_result(0, np.zeros((30, 7)))
    assert not gate.release_hold_with_fresh_actions(0, np.zeros((30, 7)))


def test_local_websocket_decision_is_delivered_while_decode_is_pending():
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client as ws_client
    import websockets.asyncio.server as ws_server
    decode_started = threading.Event()
    release_decode = threading.Event()
    class Policy:
        metadata = {"name": SERVER_NAME}
        def infer_events(self, request):
            yield {"event": "phase_decision", "request_id": request["request_id"], "phase": "execution",
                   "need_recovery": True, "need_recovery_probs": [0.1, 0.9]}
            decode_started.set()
            if not release_decode.wait(timeout=5):
                raise TimeoutError("Test failure continuation was not released")
            yield {"event": "failure_reason", "request_id": request["request_id"], "phase": "execution",
                   "failure_reason": FAILURE}
    async def check():
        worker_server = server.WorkerStreamingPolicyServer(Policy(), "localhost", 0)
        try:
            async with ws_server.serve(worker_server._handler, "127.0.0.1", 0) as listening:
                port = listening.sockets[0].getsockname()[1]
                async with ws_client.connect(f"ws://127.0.0.1:{port}", proxy=None) as ws:
                    metadata = msgpack_numpy.unpackb(await ws.recv())
                    assert metadata["name"] == SERVER_NAME
                    await ws.send(msgpack_numpy.Packer().pack({"mode": "phase", "request_id": "stream-test"}))
                    decision = msgpack_numpy.unpackb(await asyncio.wait_for(ws.recv(), timeout=3))
                    assert decision["event"] == "phase_decision" and decision["need_recovery"]
                    assert not release_decode.is_set()
                    # Network loop remains responsive while the model worker is blocked.
                    pong = await ws.ping()
                    await asyncio.wait_for(pong, timeout=1)
                    release_decode.set()
                    failure = msgpack_numpy.unpackb(await asyncio.wait_for(ws.recv(), timeout=3))
                    assert failure["event"] == "failure_reason" and failure["request_id"] == decision["request_id"]
        finally:
            release_decode.set()
            worker_server._executor.shutdown(wait=True)
    asyncio.run(check())
    assert decode_started.is_set()


def test_phase_worker_sends_raw_qpos_and_records_floor_sampled_h100(client, monkeypatch):
    args, _ = client.get_arguments(["--noise-seed", "42", "--gripper-min", "0.032"])
    dense = np.stack([np.full(7, index / 200) for index in range(100)])
    dense[:, 6] = 0.099
    observation = SimpleNamespace(qpos=dense[-1].copy(), timestamp=4.0, tactile_caption="Touch[test]",
                                  img_front=np.zeros((2, 2, 3), dtype=np.uint8),
                                  img_left=np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(client.v53, "_capture_classification_observation",
                        lambda *args, **kwargs: (observation, {}))
    payloads = []
    class PhaseClient:
        def events(self, payload):
            payloads.append(payload)
            yield {"event": "phase_decision", "request_id": "raw-history", "phase": "adjustment",
                   "adjustment_end": False, "adjustment_end_probs": [0.9, 0.1]}
    gate = client.BookV93AsyncControlGate(hold=lambda: None)
    gate.state.phase = "adjustment"
    gate.begin_phase_request("raw-history")
    stats = StateQuantileStats(q01=np.zeros(7), q99=np.ones(7))
    result = client._run_phase_assessment(
        args=args, client=PhaseClient(), operator=SimpleNamespace(
            state_history=SimpleNamespace(snapshot=lambda **kwargs: (dense, np.ones(100, dtype=bool)))),
        captioner=None, stats=stats, gate=gate, phase="adjustment", generation=0, attempt_id=2,
        request_id="raw-history", captured_step=0, after_timestamp=3.0, recovery_plan=PLAN,
        submitted_monotonic=0.0,
    )
    np.testing.assert_array_equal(payloads[0]["observation/state"], observation.qpos.astype(np.float32))
    assert "observation/state_history" not in payloads[0]
    np.testing.assert_array_equal(result.qpos_h100_raw, dense)
    np.testing.assert_array_equal(result.qpos_h100_11_discrete,
                                  client.discretize_state_qpos(dense[HISTORY_OFFSETS], stats))
    assert result.decision_received_monotonic is not None and not gate.snapshot()[-1]
