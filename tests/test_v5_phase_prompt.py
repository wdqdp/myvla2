from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset  # noqa: E402
from tactile_vla.vla.prompts import build_phase_prompt  # noqa: E402
from tactile_vla.vla.v5_phase_data import PhaseBoundaryError  # noqa: E402
from tactile_vla.vla.v5_phase_data import adaptive_contact_force  # noqa: E402
from tactile_vla.vla.v5_phase_data import detect_phase_boundaries  # noqa: E402
from tactile_vla.vla.v5_phase_data import load_phase_overrides  # noqa: E402
from tactile_vla.vla.v5_phase_data import load_raw_attempt_streams  # noqa: E402
from tactile_vla.vla.v5_phase_data import phase_for_frame  # noqa: E402
from tactile_vla.vla.v5_phase_data import resolve_phase_override  # noqa: E402


def _signals(*, maximum_opening: float = 0.080, asymmetric: bool = False):
    frames = 190
    puppet = np.full(frames, 0.035)
    master = np.full(frames, 0.030)
    puppet[35:65] = np.linspace(0.035, maximum_opening, 30)
    master[35:65] = np.linspace(0.030, maximum_opening - 0.004, 30)
    puppet[65:120] = maximum_opening
    master[65:120] = maximum_opening - 0.004
    puppet[120:145] = np.linspace(maximum_opening, 0.040, 25)
    master[120:145] = np.linspace(maximum_opening - 0.004, 0.034, 25)
    puppet[145:] = 0.040
    master[145:] = 0.034

    rng = np.random.default_rng(4)
    left = 8.0 + rng.normal(0.0, 0.002, frames)
    right = -5.0 + rng.normal(0.0, 0.003, frames)
    left[:55] += 3.0
    left[140:] += 3.0
    if asymmetric:
        right[:55] -= 0.15
        right[140:] -= 0.15
    else:
        right[:55] -= 2.0
        right[140:] -= 2.0
    return puppet, master, left, right


def test_phase_v1_prompts_are_exact_and_tactile_free() -> None:
    assert build_phase_prompt(
        phase="execution", instruction="Pick object", prompt_profile="phase_v1"
    ) == "Mode: execution. Task: Pick object."
    assert build_phase_prompt(
        phase="reposition", instruction="Pick object.", prompt_profile="phase_v1"
    ) == (
        "Mode: reposition. Task: Pick object. "
        "Description: Put the object back to its original position."
    )
    adjustment = build_phase_prompt(
        phase="adjustment",
        instruction="Pick object",
        recovery_plan="move horizontally right slightly",
        prompt_profile="phase_v1",
    )
    assert adjustment == (
        "Mode: adjustment. Task: Pick object. "
        "Recovery plan: move horizontally right slightly."
    )
    assert "Touch[" not in adjustment


@pytest.mark.parametrize("asymmetric", [False, True])
def test_fz_detector_handles_bias_asymmetry_and_opening_below_fixed_threshold(asymmetric: bool) -> None:
    puppet, master, left, right = _signals(maximum_opening=0.080, asymmetric=asymmetric)
    # A one-frame collision during no contact and one-frame loss after regrasp
    # must not manufacture either stable boundary.
    left[90] += 8.0
    left[165] = 8.0
    right[165] = -5.0
    detected = detect_phase_boundaries(
        puppet_gripper=puppet,
        master_gripper=master,
        fz_left=left,
        fz_right=right,
        tactile_captions=["Touch[area=none; Fz=near_zero]"] * len(left),
    )
    assert 50 <= detected.release_frame < 80
    assert 140 <= detected.regrasp_frame < 165
    assert detected.thresholds.force_on_threshold > detected.thresholds.force_off_threshold
    assert detected.caption_support["available"] is True
    assert max(puppet) < 0.095


def test_force_distributions_that_do_not_separate_are_rejected() -> None:
    puppet, master, left, right = _signals()
    left[:] = 1.0
    right[:] = -2.0
    with pytest.raises(PhaseBoundaryError, match="cannot be reliably separated"):
        adaptive_contact_force(
            puppet_gripper=puppet,
            master_gripper=master,
            fz_left=left,
            fz_right=right,
        )


def test_missing_fz_source_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("arm/jointStatePosition/puppetRight", data=np.zeros((40, 7)))
        handle.create_dataset("arm/jointStatePosition/masterRight", data=np.zeros((40, 7)))
        handle.create_dataset("tactile/force_resultant/left", data=np.zeros((40, 6)))
        handle.create_dataset("timestamp", data=np.arange(40))
    with pytest.raises(PhaseBoundaryError, match="fz_right"):
        load_raw_attempt_streams(path)


def test_override_is_versioned_reviewed_and_ordered(tmp_path: Path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "tactile_vla_v5_phase_boundary_overrides_v1",
                "version": 1,
                "overrides": [
                    {
                        "episode_id": 51,
                        "attempt_id": 2,
                        "release_frame": 176,
                        "regrasp_frame": 367,
                        "reason": "manual force/gripper review",
                        "reviewer": "reviewer-id",
                    }
                ],
            }
        )
    )
    _, lookup = load_phase_overrides(path)
    assert lookup[(51, 2)]["release_frame"] == 176
    assert phase_for_frame(2, 176, 176, 367) == "reposition"
    assert phase_for_frame(2, 177, 176, 367) == "adjustment"
    assert phase_for_frame(2, 367, 176, 367) == "adjustment"
    assert phase_for_frame(2, 368, 176, 367) == "execution"
    assert phase_for_frame(1, 99, None, None) == "execution"


def test_override_accepts_approximate_timestamps_without_review_metadata(tmp_path: Path) -> None:
    timestamps = 1_785_658_883.0 + np.arange(20, dtype=np.float64) / 30.0
    path = tmp_path / "timestamp-overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "tactile_vla_v5_phase_boundary_overrides_v1",
                "version": 1,
                "overrides": [
                    {
                        "episode_id": 51,
                        "attempt_id": 2,
                        "release_frame": float(timestamps[5] + 0.004),
                        "regrasp_frame": float(timestamps[14] - 0.005),
                    }
                ],
            }
        )
    )

    _, lookup = load_phase_overrides(path)
    release, regrasp, resolution = resolve_phase_override(
        lookup[(51, 2)], timestamps, attempt_key=(51, 2)
    )

    assert (release, regrasp) == (5, 14)
    assert resolution["release_frame_input_kind"] == "approximate_timestamp"
    assert resolution["regrasp_frame_input_kind"] == "approximate_timestamp"
    assert resolution["resolved_release_frame"] == 5
    assert resolution["resolved_regrasp_frame"] == 14
    # Epoch-sized float timestamps have roughly sub-microsecond precision.
    assert resolution["release_frame_timestamp_error_seconds"] == pytest.approx(
        0.004, abs=1e-6
    )
    assert resolution["regrasp_frame_timestamp_error_seconds"] == pytest.approx(
        0.005, abs=1e-6
    )


def test_override_keeps_integer_frame_compatibility_and_rejects_bad_timestamp() -> None:
    timestamps = 1_785_658_883.0 + np.arange(20, dtype=np.float64) / 30.0
    release, regrasp, resolution = resolve_phase_override(
        {"release_frame": 5, "regrasp_frame": 14},
        timestamps,
        attempt_key=(51, 2),
    )
    assert (release, regrasp) == (5, 14)
    assert resolution["release_frame_input_kind"] == "attempt_local_frame"

    with pytest.raises(ValueError, match="outside attempt range"):
        resolve_phase_override(
            {"release_frame": 1_700_000_000.0, "regrasp_frame": float(timestamps[14])},
            timestamps,
            attempt_key=(51, 2),
        )


def test_stage_a_dataset_uses_global_index_phase_only_for_prompt() -> None:
    dataset = object.__new__(TactileVLAFrameDataset)
    dataset.stage = "execution"
    dataset.prompt_profile = "phase_v1"
    dataset.action_phase_by_global_index = {
        7: {
            "episode_id": 2,
            "attempt_id": 2,
            "frame_index": 10,
            "phase": "adjustment",
            "chunk_phase_pure": False,
        }
    }
    item = {
        "index": 7,
        "episode_id": 2,
        "attempt_id": 2,
        "frame_index": 10,
        "instruction": "Pick object",
        "input_recovery_plan": "move horizontally left moderately",
    }
    assert dataset._prompt(item) == (
        "Mode: adjustment. Task: Pick object. "
        "Recovery plan: move horizontally left moderately."
    )


def _load_stage_a_script():
    spec = importlib.util.spec_from_file_location(
        "v5_stage_a_test_module",
        PROJECT_ROOT / "scripts" / "train_vla_stage_a_openpi.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v5_stage_a_args(**overrides):
    values = dict(
        data_profile="rotation_phase_v5",
        prompt_profile="phase_v1",
        experiment_kind="phase_prompt_only",
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
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stage_a_v5_protocol_is_prompt_only_and_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_stage_a_script()
    monkeypatch.setattr(module, "DEFAULT_BASE_CHECKPOINT", Path("/models/pi05_base/params"))
    args = _v5_stage_a_args()
    module.validate_v5_args(args)
    module.validate_v4_training_protocol(args)
    with pytest.raises(ValueError, match="phase_v1"):
        module.validate_v5_args(_v5_stage_a_args(prompt_profile="minimal_v1"))
    with pytest.raises(ValueError, match="phase_prompt_only"):
        module.validate_v5_args(_v5_stage_a_args(experiment_kind="stage_b"))
    with pytest.raises(ValueError, match="protocol mismatch"):
        module.validate_v4_training_protocol(_v5_stage_a_args(seed=7))


def _load_eval_script():
    spec = importlib.util.spec_from_file_location(
        "v5_counterfactual_eval_test_module",
        PROJECT_ROOT / "scripts" / "evaluate_v5_phase_prompt_counterfactual.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counterfactual_acceptance_uses_only_phase_pure_chunks() -> None:
    module = _load_eval_script()
    rows = []
    for phase in ("execution", "reposition", "adjustment"):
        for _ in range(5):
            losses = {name: 2.0 for name in ("execution", "reposition", "adjustment")}
            losses[phase] = 1.0
            rows.append(
                {
                    "true_phase": phase,
                    "phase_losses": losses,
                    "v4_loss": 1.0,
                    "chunk_phase_pure": True,
                }
            )
    rows.append(
        {
            "true_phase": "execution",
            "phase_losses": {"execution": 99.0, "reposition": 0.0, "adjustment": 0.0},
            "v4_loss": 1.0,
            "chunk_phase_pure": False,
        }
    )
    summary = module.summarize_counterfactual_losses(rows)
    assert summary["all_acceptance_gates_passed"] is True
    assert summary["excluded_cross_phase_chunk_count"] == 1
    assert all(value["correct_prompt_win_rate"] == 1.0 for value in summary["phases"].values())
