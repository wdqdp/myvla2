from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi/inference/agilex/inference"))

from agilex_inference_forced_phase_anlation_5_3 import validate_server_metadata
from agilex_inference_tactile_vla_sync_single import build_payload
from scripts.serve_tactile_vla_v7 import V7Policy
from scripts.serve_tactile_vla_v7 import _restore_tree
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v7_adjustment_end_data import LABEL_POLICY
from tactile_vla.vla.v7_adjustment_end_data import is_adjustment_end_positive
from tactile_vla.vla.v7_adjustment_end_data import is_adjustment_end_valid
from tactile_vla.vla.v7_adjustment_end_evaluation import relative_probability_profile


def test_v7_label_window_is_r_minus_10_through_r_only() -> None:
    rexecution = 100
    assert not is_adjustment_end_positive(89, rexecution)
    assert is_adjustment_end_positive(90, rexecution)
    assert is_adjustment_end_positive(100, rexecution)
    assert not is_adjustment_end_positive(101, rexecution)
    assert is_adjustment_end_valid(100, rexecution)
    assert not is_adjustment_end_valid(101, rexecution)
    assert LABEL_POLICY["boundary"] == "native_reexecution_frame_index"


def test_v7_relative_profile_ends_at_r() -> None:
    rows = [
        {
            "episode_id": episode,
            "frame_index": rexecution + relative,
            "rexecution_frame": rexecution,
            "label": int(relative >= -10),
            "probability": (relative + 30) / 30,
        }
        for episode, rexecution in ((1, 100), (2, 200))
        for relative in range(-30, 1)
    ]
    result = relative_probability_profile(rows)
    assert result["range"] == {"start_inclusive": -30, "end_inclusive": 0}
    assert [(row["relative_frame_start_inclusive"], row["relative_frame_end_inclusive"]) for row in result["bins"]] == [
        (-30, -26), (-25, -21), (-20, -16), (-15, -11), (-10, -6), (-5, -1), (0, 0)
    ]
    assert [row["sample_count"] for row in result["bins"]] == [10, 10, 10, 10, 10, 10, 2]


def test_no_history_payload_omits_both_history_keys() -> None:
    payload = build_payload(
        mode="execution",
        img_front_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        img_left_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        qpos=np.zeros(7, dtype=np.float32),
        state_history=None,
        state_history_mask=None,
        prompt="Mode: execution.",
    )
    assert "observation/state_history" not in payload
    assert "observation/state_history_mask" not in payload


def test_v7_server_rejects_history_fields() -> None:
    with pytest.raises(ValueError, match="forbidden fields"):
        V7Policy._clean_inputs({"observation/state_history": np.zeros((60, 7))})


def test_restore_tree_supplies_concrete_inference_sharding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jax
    import orbax.checkpoint as ocp

    checkpoint = tmp_path / "params"
    checkpoint.mkdir()
    captured = {}

    class FakeCheckpointer:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def metadata(self, path):
            assert path == checkpoint.resolve()
            return {"weight": object()}

        def restore(self, path, args):
            assert path == checkpoint.resolve()
            captured["args"] = args
            return {"weight": jax.numpy.arange(6, dtype=jax.numpy.float32).reshape(2, 3)}

    monkeypatch.setattr(ocp, "PyTreeCheckpointer", FakeCheckpointer)

    restored = _restore_tree(checkpoint)

    np.testing.assert_array_equal(restored["weight"], np.arange(6, dtype=np.float32).reshape(2, 3))
    restore_args = captured["args"].restore_args["weight"]
    assert isinstance(restore_args.sharding, jax.sharding.SingleDeviceSharding)
    assert restore_args.restore_type is jax.Array


def test_v7_client_accepts_only_no_history_metadata(tmp_path: Path) -> None:
    captioner = tmp_path / "captioner.pt"
    captioner.write_bytes(b"v7 captioner")
    args = SimpleNamespace(
        expected_data_profile="rotation_phase_v7_adjustment",
        phase_change_timeout_seconds=10.0,
        captioner_checkpoint=captioner,
        allow_experimental_adjustment_end=False,
    )
    metadata = {
        "supports_action_noise": True,
        "requires_action_noise": True,
        "supports_adjustment_end": True,
        "prompt_profile": "phase_v2",
        "data_profile": "rotation_phase_v7_adjustment",
        "experiment_kind": "phase_prompt_h30_terminal_hold_native_reexecution",
        "stage_a_protocol": "v7_no_state_history",
        "phase_change_prompt_profile": "phase_change_v1",
        "phase_change_max_token_len": 512,
        "qpos_h30_sample_offsets": [0, 3, 6, 10, 13, 16, 19, 23, 26, 29],
        "qpos_bin_count": 256,
        "qpos_discretization_extra_clip": False,
        "captioner_window_size": 30,
        "captioner_checkpoint_sha256": sha256_file(captioner),
        "action_horizon": 30,
        "action_dim": 32,
        "output_action_dim": 7,
        "state_history_len": 0,
        "state_history_dim": 7,
        "use_state_history": False,
        "adjustment_end_threshold": 0.7,
        "phase_change_timeout_seconds": 10.0,
        "adjustment_end_experimental_override": False,
    }
    validate_server_metadata(args, metadata)
    metadata["use_state_history"] = True
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_server_metadata(args, metadata)
