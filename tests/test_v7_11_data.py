"""Tests for the direct-tactile ablation's supervision and prompts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tactile_vla.vla.v7_11_data import (
    direct_recovery_prompt, factual_selection, phase_prompt_without_plan,
)
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v7_11_stage_a import checkpoint_prompt_identity, validate_stage_a_checkpoint


class DirectTactileDataTest(unittest.TestCase):
    def test_phase_removes_only_plan(self):
        prompt = ("Mode: phase\nTask: move the object\nTouch: Touch[area=small]\n"
                  "Recovery plan: recovery_plan=move left moderately\n"
                  "State history H100 sampled to 11 points: [[1,2,3]]")
        result = phase_prompt_without_plan(prompt)
        self.assertEqual(result, "Mode: phase\nTask: move the object\nTouch: Touch[area=small]\n"
                                 "State history H100 sampled to 11 points: [[1,2,3]]")
        with self.assertRaises(ValueError):
            phase_prompt_without_plan(result)

    def test_recovery_uses_current_touch_and_has_no_plan(self):
        result = direct_recovery_prompt("move the object", "Touch[area=small]")
        self.assertEqual(result, "Mode: adjustment. Task: move the object.\nTouch: Touch[area=small]")
        self.assertNotIn("plan", result.lower())
        with self.assertRaises(ValueError):
            direct_recovery_prompt("move", " ")

    def test_only_selected_factual_rows_remain(self):
        rows = [
            {"global_index": 2, "sample_variant": "factual"},
            {"global_index": 2, "sample_variant": "moderately_to_slightly"},
            {"global_index": 3, "sample_variant": "slightly_to_moderately"},
            {"global_index": 4, "sample_variant": "factual"},
        ]
        stream = {"manifest_row_indices": [0, 1, 2], "global_indices": [2, 2, 3]}
        self.assertEqual(factual_selection(rows, stream), ([0], [2]))
        with self.assertRaises(ValueError):
            factual_selection(rows, {"manifest_row_indices": [0], "global_indices": [5]})

    def test_stage_b_rejects_legacy_or_mismatched_stage_a(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            action, norm = root / "action.json", root / "norm_stats.json"
            action.write_text("action")
            norm.write_text("norm")
            step = root / "run" / "10000"
            (step / "params").mkdir(parents=True)
            (step / "params/_METADATA").write_text("metadata")
            index = {"action_index_file": str(action), "action_training_data_hash": "train-hash",
                     "source_hashes": {str(norm): sha256_file(norm)}}
            config = {
                "data_profile": "rotation_phase_v7_4_2_adjustment", "prompt_profile": "phase_v2",
                "stage_a_protocol": "v7_4_2_finetune_no_state_history", "use_state_history": False,
                "state_history_len": 0, "history_hidden_dim": 0, "action_horizon": 30,
                "action_dim": 32, "train_lora_only": True, "paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m_lora",
                "artifact_identity": {"training_data_hash": "train-hash",
                                      "index_sha256": sha256_file(action),
                                      "v4_norm_stats_sha256": sha256_file(norm)},
            }
            config_path = step.parent / "config.json"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "direct-tactile protocol"):
                validate_stage_a_checkpoint(step, index)
            config["v7_11_stage_a"] = checkpoint_prompt_identity()
            config_path.write_text(json.dumps(config))
            self.assertEqual(validate_stage_a_checkpoint(step, index)["step"], 10000)
            config["artifact_identity"]["training_data_hash"] = "wrong"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "training_data_hash"):
                validate_stage_a_checkpoint(step, index)


if __name__ == "__main__":
    unittest.main()
