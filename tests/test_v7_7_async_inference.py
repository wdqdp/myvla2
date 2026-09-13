from __future__ import annotations

from tactile_vla.vla.v7_7_async_state import AsyncPhaseState


def test_true_decision_discards_chunk_and_invalidates_action_generation():
    state = AsyncPhaseState(pending_actions=list(range(30)))
    old_generation = state.begin_action_request()
    state.begin_phase_request("phase-1")
    assert state.apply_phase_decision({
        "event": "phase_decision", "request_id": "phase-1", "phase": "execution",
        "need_recovery": True,
    })
    assert state.stop_latched and state.pending_actions == []
    assert not state.accept_action_result(old_generation, [1, 2])


def test_failure_event_survives_action_generation_invalidation():
    state = AsyncPhaseState()
    state.begin_phase_request("phase-2")
    state.apply_phase_decision({
        "event": "phase_decision", "request_id": "phase-2", "phase": "execution",
        "need_recovery": True,
    })
    assert state.accept_failure_event({
        "event": "failure_reason", "request_id": "phase-2", "phase": "execution",
    })


def test_adjustment_true_holds_until_fresh_execution_chunk():
    state = AsyncPhaseState(phase="adjustment", pending_actions=list(range(10)))
    state.begin_phase_request("phase-3")
    state.apply_phase_decision({
        "event": "phase_decision", "request_id": "phase-3", "phase": "adjustment",
        "adjustment_end": True,
    })
    state.switch_phase("execution")
    generation = state.begin_action_request()
    assert state.stop_latched
    assert not state.release_hold_with_fresh_actions(generation - 1, [1])
    assert state.release_hold_with_fresh_actions(generation, [1])
    assert not state.stop_latched


def test_gate_stops_h30_at_injected_decision(monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path("openpi/inference/agilex/inference/agilex_inference_v7_7_asyn.py")
    spec = importlib.util.spec_from_file_location("v77_async", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    held, published = [], []
    gate = module.V77AsyncControlGate(hold=lambda: held.append(True))
    gate.state.begin_phase_request("p")
    ticks = 0
    def poll():
        nonlocal ticks
        ticks += 1
        if ticks == 7:
            gate.handle_phase_event({"event": "phase_decision", "request_id": "p",
                                     "phase": "execution", "need_recovery": True})
    count = gate.publish_chunk(range(30), generation=0, publish=published.append, before_each=poll)
    assert count == 6 and published == list(range(6)) and held


def test_gate_releases_only_with_new_phase_actions():
    import importlib.util
    from pathlib import Path
    path = Path("openpi/inference/agilex/inference/agilex_inference_v7_7_asyn.py")
    spec = importlib.util.spec_from_file_location("v77_async_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gate = module.V77AsyncControlGate(hold=lambda: None)
    gate.begin_phase_request("need-1")
    gate.handle_phase_event({
        "event": "phase_decision", "request_id": "need-1",
        "phase": "execution", "need_recovery": True,
    })
    stopped_generation = gate.snapshot()[2]
    gate.switch_phase("adjustment", increment_attempt=True)
    fresh_generation = gate.begin_action_request()
    assert fresh_generation > stopped_generation
    assert not gate.release_hold_with_fresh_actions(fresh_generation - 1, [1])
    assert gate.release_hold_with_fresh_actions(fresh_generation, [1])
    assert gate.snapshot() == ("adjustment", 2, fresh_generation, 1, False)


def test_v77_server_metadata_requires_deterministic_action_noise():
    import importlib.util
    from pathlib import Path
    path = Path("openpi/inference/agilex/inference/agilex_inference_v7_7_asyn.py")
    spec = importlib.util.spec_from_file_location("v77_async_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = {
        "name": "tactile_vla_v7_7",
        "data_profile": module.DATA_PROFILE,
        "phase_prompt_profile": module.PROMPT_PROFILE,
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
        "need_recovery_threshold": 0.5,
        "adjustment_end_threshold": 0.5,
    }
    module.validate_server_metadata(metadata)
    metadata["requires_action_noise"] = False
    try:
        module.validate_server_metadata(metadata)
    except ValueError as exc:
        assert "requires_action_noise" in str(exc)
    else:
        raise AssertionError("missing deterministic-noise protocol was accepted")
