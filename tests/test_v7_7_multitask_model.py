from __future__ import annotations

import asyncio
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.train_vla_stage_b_v3 import pad_eval_batch


def test_stage_b_eval_tail_padding_preserves_valid_count():
    raw = {
        "image": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        "structured_target_text_index": np.asarray([3, 7], dtype=np.int32),
    }

    padded, valid_count = pad_eval_batch(raw, multiple=4)

    assert valid_count == 2
    assert padded["image"].shape[0] == 4
    assert padded["structured_target_text_index"].tolist() == [3, 7, 7, 7]


def test_v77_trainable_filter_names_both_heads():
    source = Path("src/tactile_vla/vla/v7_7_multitask_model.py").read_text()
    assert 'PathRegex(".*need_head.*")' in source
    assert 'PathRegex(".*adjustment_head.*")' in source
    assert 'PathRegex(".*backbone.*llm.*lora.*")' in source


def test_v77_training_cycle_and_v74_initialization_are_pinned():
    source = Path("scripts/train_vla_multitask_v7_7.py").read_text()
    assert "pi05_delta_tac_rotation_phase_v7_4_no_history/15000" in source
    assert "base.TASK_CYCLE = TASK_CYCLE" in source
    assert '"delta_params"' in source and '"full_params"' in source


def test_v77_server_uses_client_supplied_action_noise():
    from scripts.serve_tactile_vla_v7_7 import V77Policy

    policy = object.__new__(V77Policy)
    policy._model = SimpleNamespace(backbone=SimpleNamespace(action_horizon=30, action_dim=32))
    policy._rng = None
    policy._num_inference_steps = 10
    policy._output_action_dim = 7
    received = {}

    def sample(_rng, _observation, *, num_steps, noise):
        received["num_steps"] = num_steps
        received["noise"] = np.asarray(noise)
        return noise

    policy._sample_actions = sample
    policy._output_transform = lambda values: values
    noise = np.arange(30 * 32, dtype=np.float32).reshape(30, 32)
    raw, actions = policy._actions_with_noise(
        {"state": np.zeros(7, dtype=np.float32)}, object(), noise,
    )
    assert received["num_steps"] == 10
    np.testing.assert_array_equal(received["noise"], noise[None])
    np.testing.assert_array_equal(raw, noise)
    np.testing.assert_array_equal(actions, noise[:, :7])


def test_v77_server_restores_raw_full_params_export(tmp_path, monkeypatch):
    from scripts import serve_tactile_vla_v7_7 as server

    full_params = tmp_path / "10000" / "full_params"
    full_params.mkdir(parents=True)
    expected = {"backbone": {"weight": np.asarray([1.0], dtype=np.float32)}}
    captured = {}

    class FakeCheckpointer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def metadata(self, path):
            assert path == full_params.resolve()
            return {"backbone": {"weight": object()}}

        def restore(self, path, args):
            captured["path"] = path
            captured["args"] = args
            return expected

    monkeypatch.setattr(server.ocp, "PyTreeCheckpointer", FakeCheckpointer)
    actual = server._restore_exported_full_params(
        full_params,
        dtype="bfloat16",
        sharding=object(),
    )

    assert actual is expected
    assert captured["path"] == full_params.resolve()
    restore_args = captured["args"].restore_args["backbone"]["weight"]
    assert isinstance(restore_args.sharding, server.jax.sharding.SingleDeviceSharding)
    assert restore_args.restore_type is server.jax.Array


def test_v77_server_rejects_non_full_parameter_tree(tmp_path):
    from scripts.serve_tactile_vla_v7_7 import _restore_exported_full_params

    params = tmp_path / "10000" / "params"
    params.mkdir(parents=True)
    with pytest.raises(ValueError, match="requires a full_params export"):
        _restore_exported_full_params(params)


def test_v77_warmup_covers_all_routes_and_restores_threshold():
    from scripts.serve_tactile_vla_v7_7 import warm_up_v77

    class FakePolicy:
        _need_threshold = 0.8

        def __init__(self):
            self.requests = []

        def infer_events(self, request):
            self.requests.append((request, self._need_threshold))
            if request["mode"] == "phase" and request["phase"] == "execution":
                yield {"event": "phase_decision"}
                yield {"event": "failure_reason"}
            else:
                yield {"actions": np.zeros((30, 7), dtype=np.float32)}

    policy = FakePolicy()
    warm_up_v77(policy)
    assert policy._need_threshold == 0.8
    assert [(item[0]["mode"], item[0].get("phase")) for item in policy.requests] == [
        ("execution", None), ("phase", "execution"),
        ("phase", "adjustment"), ("reasoning", None),
    ]
    assert all(threshold == -1.0 for _, threshold in policy.requests)


def test_v77_model_executor_keeps_websocket_event_loop_responsive():
    from scripts.serve_tactile_vla_v7_7 import StreamingPolicyServer
    from openpi_client import msgpack_numpy

    class FakePolicy:
        metadata = {"name": "test"}

        def infer_events(self, _request):
            time.sleep(0.15)
            yield {"event": "phase_decision"}

    class FakeWebsocket:
        def __init__(self):
            self.sent = []
            self.recv_count = 0

        async def send(self, payload):
            self.sent.append(payload)

        async def recv(self):
            self.recv_count += 1
            if self.recv_count == 1:
                return msgpack_numpy.Packer().pack({"mode": "phase"})
            await asyncio.Event().wait()

    async def exercise():
        server = StreamingPolicyServer(FakePolicy(), "localhost", 0)
        websocket = FakeWebsocket()
        handler = asyncio.create_task(server._handler(websocket))
        try:
            await asyncio.sleep(0.02)
            assert len(websocket.sent) == 1  # metadata was sent before the slow model call
            tick_start = time.monotonic()
            await asyncio.sleep(0.02)
            assert time.monotonic() - tick_start < 0.08
            await asyncio.wait_for(_wait_for_second_message(websocket), timeout=1.0)
        finally:
            handler.cancel()
            try:
                await handler
            except asyncio.CancelledError:
                pass
            server._model_executor.shutdown(wait=True)

    async def _wait_for_second_message(websocket):
        while len(websocket.sent) < 2:
            await asyncio.sleep(0.01)

    asyncio.run(exercise())
