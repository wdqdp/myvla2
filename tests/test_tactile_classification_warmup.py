from __future__ import annotations

# ruff: noqa: E402, SLF001

from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "openpi/inference/agilex/inference")]

import agilex_inference_book_v9_4_asyn as book
import agilex_inference_forced_phase_anlation_5_3 as v53
from tactile_vla.runtime.tactile_buffer import TactileTopics, TactileWindowBuffer


KEYS = tuple(vars(TactileTopics()))
GRID = np.zeros((35, 20, 3), dtype=np.float32)


class Tactile:
    def __init__(self):
        self.buffer = TactileWindowBuffer()
        self.topics = TactileTopics()

    @property
    def ready(self):
        return self.buffer.ready

    def caption(self, predictor):
        return self.buffer.caption(predictor)

    def push(self, timestamp=101.0):
        for key in KEYS:
            self.buffer.push_topic(key, GRID, timestamp=timestamp)


class Predictor:
    window_size = 30
    schema_version = book.LABEL_SCHEMA_VERSION
    label_fields = book.LABEL_FIELDS

    def __init__(self):
        self.calls = []

    def predict(self, mesh_motion, force):
        self.calls.append((mesh_motion.shape, force.shape))
        return SimpleNamespace(caption=book.neutral_caption())


@pytest.fixture(autouse=True)
def shutdown_state():
    v53.runtime.shutdown_event.clear()
    yield
    v53.runtime.shutdown_event.clear()


def make_operator(monkeypatch, *, frames=0, update=None, timeout=1.0):
    tactile = Tactile()
    for _ in range(frames):
        tactile.push()
    clock = [0.0]
    sleeps = []

    def sleep():
        clock[0] += 0.01
        sleeps.append(clock[0])
        if update is not None:
            update(tactile)

    def message(timestamp, **kwargs):
        return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: timestamp)), **kwargs)

    operator = SimpleNamespace(
        tactile=tactile,
        img_front_deque=[message(101.1, image=np.zeros((2, 2, 3), dtype=np.uint8))],
        img_left_deque=[message(101.2, image=np.zeros((2, 2, 3), dtype=np.uint8))],
        puppet_arm_deque=[message(101.3, position=np.zeros(7, dtype=np.float32))],
        bridge=SimpleNamespace(imgmsg_to_cv2=lambda msg, _encoding: msg.image),
        is_shutdown=lambda: False,
        rate=lambda _hz: SimpleNamespace(sleep=sleep),
    )
    args = SimpleNamespace(observation_poll_rate=200, phase_change_timeout_seconds=timeout, use_state_history=False)
    monkeypatch.setattr(v53.time, "monotonic", lambda: clock[0])
    return args, operator, sleeps


@pytest.mark.parametrize("initial_frames", [0, 1, 29])
def test_classification_waits_for_complete_w30(monkeypatch, initial_frames):
    args, operator, sleeps = make_operator(monkeypatch, frames=initial_frames, update=lambda t: t.push())
    predictor = Predictor()
    observation, timestamps = v53._capture_classification_observation(
        args, operator, predictor, after_timestamp=100.0,
    )
    assert len(sleeps) == 30 - initial_frames
    assert predictor.calls == [((30, 35, 20, 12), (30, 35, 20, 6))]
    assert observation.tactile_caption == book.neutral_caption()
    assert min(timestamps.values()) > 100.0


def test_healthy_topic_traffic_can_still_require_more_than_30_updates():
    tactile = Tactile()
    for _ in range(30):
        for key in KEYS[:-1]:
            tactile.buffer.push_topic(key, GRID, timestamp=101.0)
    assert len(tactile.buffer) == 0
    status = tactile.buffer.status()
    assert status["missing_topics"] == [KEYS[-1]]
    assert not tactile.ready
    tactile.buffer.push_topic(KEYS[-1], GRID, timestamp=101.0)
    assert len(tactile.buffer) == 1 and not tactile.ready


def test_classification_timeout_reports_warmup_progress(monkeypatch):
    args, operator, sleeps = make_operator(monkeypatch, frames=29, timeout=0.025)
    predictor = Predictor()
    with pytest.raises(v53.FailClosedError, match="'frames': 29"):
        v53._capture_classification_observation(args, operator, predictor, after_timestamp=100.0)
    assert len(sleeps) == 3
    assert predictor.calls == []


def test_complete_but_stale_window_is_not_accepted(monkeypatch):
    args, operator, _ = make_operator(monkeypatch, frames=30, timeout=0.025)
    for key in ("img_front_deque", "img_left_deque", "puppet_arm_deque"):
        getattr(operator, key)[-1].header.stamp.to_sec = lambda: 103.0
    predictor = Predictor()
    with pytest.raises(v53.FailClosedError, match="after_timestamp=102.0"):
        v53._capture_classification_observation(args, operator, predictor, after_timestamp=102.0)
    assert predictor.calls == []


def test_replay_advances_frames_during_classification_warmup(monkeypatch):
    args, fake, sleeps = make_operator(monkeypatch)
    operator = v53.runtime.ReplayOperator.__new__(v53.runtime.ReplayOperator)
    operator.tactile = fake.tactile
    operator.is_shutdown = fake.is_shutdown
    operator.rate = fake.rate
    captured = []

    def replay_observation(*args, **kwargs):
        operator.tactile.push()
        captured.append(True)
        return np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), fake.puppet_arm_deque[-1]

    monkeypatch.setattr(v53.runtime, "get_ros_observation", replay_observation)
    predictor = Predictor()
    v53._capture_classification_observation(args, operator, predictor, after_timestamp=100.0)
    assert len(captured) == 30 and len(sleeps) == 29
    assert len(predictor.calls) == 1


def test_startup_warmup_times_out_with_missing_ros_topic(monkeypatch):
    def partial(tactile):
        for key in KEYS[:-1]:
            tactile.buffer.push_topic(key, GRID, timestamp=101.0)

    args, operator, _ = make_operator(monkeypatch, update=partial, timeout=0.025)
    with pytest.raises(v53.FailClosedError, match=operator.tactile.topics.right_mesh_3d_flow):
        v53._wait_for_tactile_ready(args, operator, Predictor())


def test_startup_warmup_waits_and_does_not_infer_caption(monkeypatch, capsys):
    args, operator, sleeps = make_operator(monkeypatch, frames=29, update=lambda t: t.push())
    predictor = Predictor()
    v53._wait_for_tactile_ready(args, operator, predictor)
    assert len(sleeps) == 1 and operator.tactile.ready
    assert predictor.calls == []
    output = capsys.readouterr().out
    assert "warming up" in output and "ready" in output


@pytest.mark.parametrize("missing", ["captioner", "tactile"])
def test_missing_configuration_fails_immediately(monkeypatch, missing):
    args, operator, sleeps = make_operator(monkeypatch)
    predictor = Predictor()
    if missing == "captioner":
        predictor = None
    else:
        operator.tactile = None
    with pytest.raises(v53.FailClosedError, match="requires"):
        v53._capture_classification_observation(args, operator, predictor, after_timestamp=100.0)
    assert sleeps == []


def test_warmup_is_interruptible(monkeypatch):
    args, operator, sleeps = make_operator(monkeypatch, update=lambda t: v53.runtime.shutdown_event.set())
    with pytest.raises(RuntimeError, match="Stopped"):
        v53._wait_for_tactile_ready(args, operator, Predictor())
    assert len(sleeps) == 1


@pytest.mark.parametrize("frames,expect_timeout", [(0, True), (29, False)])
def test_book_startup_warms_tactile_before_action_requests(monkeypatch, tmp_path, frames, expect_timeout):
    args, operator, _ = make_operator(
        monkeypatch, frames=frames, timeout=0.025,
        update=None if expect_timeout else lambda t: t.push(),
    )
    args.instruction = book.BOOK_INSTRUCTION
    args.norm_stats_file = args.captioner_checkpoint = tmp_path / "identity"
    args.norm_stats_file.touch()
    args.publish_rate = 30
    args.gripper_min = 0.032
    args.no_publish = False
    args.need_recovery_rate_hz = args.adjustment_end_rate_hz = 7.0
    args.quit_key, args.success_key = "q", "s"
    args.max_publish_step = 1
    resets, captures, commands, requests, pushes = [], [], [], [], []
    metadata = {"instruction": args.instruction, "norm_stats_sha256": "hash", "captioner_checkpoint_sha256": "hash"}
    operator.puppet_arm_publish = lambda qpos: commands.append(qpos.copy())
    operator.reset_state_history = lambda: resets.append(operator.tactile.ready)
    operator.state_history = SimpleNamespace(push=lambda timestamp, qpos: pushes.append(timestamp))

    class ImmediateExecutor:
        def submit(self, function, *args, **kwargs):
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except Exception as error:
                future.set_exception(error)
            return future

    @contextmanager
    def executor(**kwargs):
        yield ImmediateExecutor()

    def capture(*args):
        captures.append(operator.tactile.ready)
        return SimpleNamespace(qpos=np.zeros(7, dtype=np.float32), timestamp=float(len(captures)))

    def request_action(**kwargs):
        requests.append(operator.tactile.ready)
        raise RuntimeError("test reached action request")

    monkeypatch.setattr(book, "validate_server_metadata", lambda metadata: None)
    monkeypatch.setattr(book, "sha256_file", lambda path: "hash")
    monkeypatch.setattr(book, "load_state_quantiles", lambda path: None)
    monkeypatch.setattr(book, "inference_executor", executor)
    monkeypatch.setattr(book, "_capture_action_observation", capture)
    monkeypatch.setattr(book, "_request_action_chunk", request_action)
    with pytest.raises(RuntimeError, match="Timed out" if expect_timeout else "test reached action request"):
        book.run_book_v9_4_async(
            args, operator, SimpleNamespace(get_server_metadata=lambda: metadata),
            SimpleNamespace(metadata=metadata), Predictor(),
            SimpleNamespace(get_key=lambda: None), SimpleNamespace(record=lambda record: None),
        )
    if expect_timeout:
        assert requests == [] and resets == [] and pushes == []
        assert captures == [False]
    else:
        assert captures == [False, True] and requests == [True] and resets == [True]
        assert pushes == [2.0]
    assert commands
    assert all(np.all(qpos[:6] == 0) and qpos[6] == np.float32(0.032) for qpos in commands)
