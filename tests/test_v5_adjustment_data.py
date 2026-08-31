from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset  # noqa: E402
from tactile_vla.vla.prompts import build_phase_prompt  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import DetectedRexecution  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import FKJoint, PiperFKChain  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import PhaseBoundaryError  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import apply_h30_terminal_hold  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import detect_rexecution_frame  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import load_v2_phase_overrides  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import phase_for_rexecution_frame  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import piper_fk_positions  # noqa: E402
from tactile_vla.vla.v5_adjustment_data import resolve_v2_phase_override  # noqa: E402


def test_phase_v2_prompts_are_exact_and_reposition_is_rejected() -> None:
    assert build_phase_prompt(
        phase="execution", instruction="Pick object", prompt_profile="phase_v2"
    ) == "Mode: execution. Task: Pick object."
    assert build_phase_prompt(
        phase="adjustment",
        instruction="Pick object.",
        recovery_plan="move horizontally right slightly",
        prompt_profile="phase_v2",
    ) == (
        "Mode: adjustment. Task: Pick object.\nPut the object back, and follow this "
        "recovery plan: move horizontally right slightly."
    )
    with pytest.raises(ValueError, match="reposition"):
        build_phase_prompt(
            phase="reposition", instruction="Pick object", prompt_profile="phase_v2"
        )


def test_two_phase_boundary_is_half_open() -> None:
    assert phase_for_rexecution_frame(1, 999, None) == "execution"
    assert phase_for_rexecution_frame(2, 0, 120) == "adjustment"
    assert phase_for_rexecution_frame(2, 119, 120) == "adjustment"
    assert phase_for_rexecution_frame(2, 120, 120) == "execution"


def test_terminal_hold_repeats_last_adjustment_action_without_mutating_input() -> None:
    raw = np.arange(30 * 7, dtype=np.float32).reshape(30, 7)
    held = apply_h30_terminal_hold(raw, terminal_hold_from_offset=5)
    np.testing.assert_array_equal(held[:5], raw[:5])
    np.testing.assert_array_equal(held[5:], np.repeat(raw[4:5], 25, axis=0))
    assert not np.shares_memory(raw, held)
    np.testing.assert_array_equal(raw, np.arange(30 * 7, dtype=np.float32).reshape(30, 7))
    with pytest.raises(ValueError, match=r"\[1,29\]"):
        apply_h30_terminal_hold(raw, terminal_hold_from_offset=0)


def test_generic_fk_applies_revolute_joint_before_child_offset(tmp_path: Path) -> None:
    joint = FKJoint(
        name="joint1",
        joint_type="revolute",
        origin=np.eye(4),
        axis=np.array([0.0, 0.0, 1.0]),
    )
    fixed_origin = np.eye(4)
    fixed_origin[0, 3] = 1.0
    fixed = FKJoint(name="tool", joint_type="fixed", origin=fixed_origin, axis=None)
    chain = PiperFKChain(
        source_path=tmp_path / "robot.urdf",
        source_sha256="0" * 64,
        base_link="base",
        end_link="tool",
        joints=(joint, fixed),
        revolute_joint_count=1,
    )
    positions = piper_fk_positions([[0.0], [np.pi / 2]], chain)
    np.testing.assert_allclose(positions[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(positions[1], [0.0, 1.0, 0.0], atol=1e-12)


def test_rexecution_detector_finds_first_sustained_post_apex_descent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = 220
    release, regrasp = 40, 180
    xyz = np.zeros((frames, 3), dtype=np.float64)
    xyz[release:81, 2] = np.linspace(0.0, 0.060, 81 - release)
    xyz[81:131, 2] = 0.060
    xyz[131:161, 2] = np.linspace(0.060, 0.0, 30)
    xyz[161:, 2] = 0.0
    xyz[80:121, 0] = np.linspace(0.0, 0.050, 41)
    xyz[121:, 0] = 0.050
    qpos = np.zeros((frames, 7), dtype=np.float64)
    qpos[:, 6] = 0.035
    qpos[55:151, 6] = 0.095
    qpos[151:171, 6] = np.linspace(0.095, 0.040, 20)
    timestamp = 1_780_000_000.0 + np.arange(frames) / 30.0

    monkeypatch.setattr(
        "tactile_vla.vla.v5_adjustment_data.piper_fk_positions",
        lambda _qpos, _chain: xyz,
    )
    fake_chain = PiperFKChain(
        source_path=tmp_path / "robot.urdf",
        source_sha256="0" * 64,
        base_link="base",
        end_link="tool",
        joints=(),
        revolute_joint_count=6,
    )
    detected = detect_rexecution_frame(
        puppet_right=qpos,
        timestamp=timestamp,
        release_frame=release,
        regrasp_frame=regrasp,
        fk_chain=fake_chain,
    )
    assert isinstance(detected, DetectedRexecution)
    assert 125 <= detected.rexecution_frame <= 140
    assert detected.last_effective_translation_frame == detected.rexecution_frame - 1
    assert detected.apex_frame <= detected.rexecution_frame
    assert detected.boundary_trigger == "sustained_descent_10mm"
    assert detected.horizontal_displacement_m == pytest.approx(0.050, abs=0.002)
    assert detected.lift_height_m >= 0.055
    assert detected.z_drop_over_confirmation_window_m <= -0.010
    assert detected.downward_velocity_fraction >= 0.80
    assert detected.descent_confirmation_frames == 10
    assert detected.minimum_z_drop_m == 0.010


def test_rexecution_detector_rejects_small_horizontal_displacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = 100
    qpos = np.zeros((frames, 7), dtype=np.float64)
    qpos[:, 6] = np.r_[np.full(30, 0.04), np.full(40, 0.09), np.full(30, 0.04)]
    xyz = np.zeros((frames, 3), dtype=np.float64)
    xyz[:, 0] = np.linspace(0.0, 0.005, frames)
    xyz[20:50, 2] = np.linspace(0.0, 0.05, 30)
    monkeypatch.setattr(
        "tactile_vla.vla.v5_adjustment_data.piper_fk_positions",
        lambda _qpos, _chain: xyz,
    )
    fake_chain = PiperFKChain(
        source_path=tmp_path / "robot.urdf",
        source_sha256="0" * 64,
        base_link="base",
        end_link="tool",
        joints=(),
        revolute_joint_count=6,
    )
    with pytest.raises(PhaseBoundaryError, match="below 0.010"):
        detect_rexecution_frame(
            puppet_right=qpos,
            timestamp=np.arange(frames, dtype=np.float64),
            release_frame=20,
            regrasp_frame=80,
            fk_chain=fake_chain,
        )


def test_rexecution_detector_rejects_when_no_sustained_descent_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = 200
    release, regrasp = 30, 170
    xyz = np.zeros((frames, 3), dtype=np.float64)
    xyz[release:71, 2] = np.linspace(0.0, 0.05, 41)
    xyz[71:, 2] = 0.05
    xyz[70:131, 0] = np.linspace(0.0, 0.050, 61)
    xyz[131:, 0] = 0.050
    qpos = np.zeros((frames, 7), dtype=np.float64)
    qpos[:, 6] = 0.035
    qpos[45:116, 6] = 0.095
    qpos[116:142, 6] = np.linspace(0.095, 0.040, 26)
    monkeypatch.setattr(
        "tactile_vla.vla.v5_adjustment_data.piper_fk_positions",
        lambda _qpos, _chain: xyz,
    )
    fake_chain = PiperFKChain(
        source_path=tmp_path / "robot.urdf",
        source_sha256="0" * 64,
        base_link="base",
        end_link="tool",
        joints=(),
        revolute_joint_count=6,
    )
    with pytest.raises(PhaseBoundaryError, match="No sustained post-apex descent"):
        detect_rexecution_frame(
            puppet_right=qpos,
            timestamp=np.arange(frames, dtype=np.float64) / 30.0,
            release_frame=release,
            regrasp_frame=regrasp,
            fk_chain=fake_chain,
        )


def test_v2_override_preserves_contact_values_and_accepts_rexecution_timestamp(
    tmp_path: Path,
) -> None:
    timestamps = 1_785_658_883.0 + np.arange(20, dtype=np.float64) / 30.0
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "tactile_vla_v5_phase_boundary_overrides_v2",
                "version": 2,
                "overrides": [
                    {
                        "episode_id": 51,
                        "attempt_id": 2,
                        "release_frame": 3,
                        "regrasp_frame": 16,
                        "rexecution_frame": float(timestamps[12] + 0.004),
                    }
                ],
            }
        )
    )
    _, lookup = load_v2_phase_overrides(path)
    resolved, audit = resolve_v2_phase_override(
        lookup[(51, 2)], timestamps, attempt_key=(51, 2)
    )
    assert resolved == {"release_frame": 3, "regrasp_frame": 16, "rexecution_frame": 12}
    assert audit["rexecution_frame_input_kind"] == "approximate_timestamp"


class _FakeLeRobotDataset:
    def __init__(self, item: dict):
        self.item = item

    def __getitem__(self, _index: int) -> dict:
        return self.item


def test_stage_a_loader_applies_terminal_hold_before_transforms() -> None:
    action = np.arange(30 * 7, dtype=np.float32).reshape(30, 7)
    item = {
        "index": 7,
        "episode_id": 2,
        "attempt_id": 2,
        "frame_index": 115,
        "instruction": "Pick object",
        "input_recovery_plan": "move horizontally left moderately",
        "tactile_caption": "ignored",
        "case_id": "ignored",
        "observation.state": np.zeros(7, dtype=np.float32),
        "observation.images.front": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.left": np.zeros((8, 8, 3), dtype=np.uint8),
        "action": action,
    }
    dataset = object.__new__(TactileVLAFrameDataset)
    dataset.indices = [7]
    dataset.stage = "execution"
    dataset.state_history_len = 0
    dataset.reasoning_source_indices = None
    dataset.prompt_profile = "phase_v2"
    dataset.action_phase_by_global_index = {
        7: {
            "episode_id": 2,
            "attempt_id": 2,
            "frame_index": 115,
            "phase": "adjustment",
            "terminal_hold_from_offset": 5,
        }
    }
    dataset._dataset = _FakeLeRobotDataset(item)
    sample = dataset[0]
    np.testing.assert_array_equal(sample["actions"][:5], action[:5])
    np.testing.assert_array_equal(sample["actions"][5:], np.repeat(action[4:5], 25, axis=0))
    assert sample["prompt"].startswith("Mode: adjustment.")


def _load_stage_a_script():
    spec = importlib.util.spec_from_file_location(
        "v52_stage_a_test_module",
        PROJECT_ROOT / "scripts/train_vla_stage_a_openpi.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_a_v2_protocol_is_version_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_stage_a_script()
    monkeypatch.setattr(module, "DEFAULT_BASE_CHECKPOINT", Path("/models/pi05_base/params"))
    args = SimpleNamespace(
        data_profile="rotation_phase_v5_adjustment_v2",
        prompt_profile="phase_v2",
        experiment_kind="phase_prompt_h30_terminal_hold",
        no_norm=False,
        split="train",
        batch_size=8,
        num_steps=15_000,
        lr=5e-5,
        lr_final=5e-7,
        lr_transition_steps=7_000,
        save_interval=1_000,
        keep_period=5_000,
        action_horizon=30,
        action_dim=32,
        use_state_history=True,
        state_history_len=60,
        state_history_dim=7,
        history_hidden_dim=256,
        max_token_len=200,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        train_lora_only=True,
        allow_random_init=False,
        seed=42,
        checkpoint="/models/pi05_base/params",
    )
    module.validate_v5_args(args)
    module.validate_v4_training_protocol(args)
    args.prompt_profile = "phase_v1"
    with pytest.raises(ValueError, match="phase_v2"):
        module.validate_v5_args(args)
