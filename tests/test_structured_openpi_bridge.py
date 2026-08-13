from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "openpi" / "src"))

from tactile_vla.vla.openpi_bridge import TokenizeStructuredPrompt  # noqa: E402
from tactile_vla.vla.openpi_bridge import TokenizeStructuredResponse  # noqa: E402
from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset  # noqa: E402
from tactile_vla.vla.structured_text import ConstrainedTokenGrammar  # noqa: E402


class _FakeTokenizer:
    def tokenize_structured_response(self, prompt, state, target_tokens, *, max_len):
        del prompt, state
        prefix = [2, 3, 4]
        answer = [] if target_tokens is None else list(target_tokens)
        tokens = prefix + answer
        padding = [0] * (max_len - len(tokens))
        return (
            np.asarray(tokens + padding, dtype=np.int32),
            np.asarray([True] * len(tokens) + [False] * len(padding)),
            np.asarray([0] * len(prefix) + [1] * len(answer) + [0] * len(padding), dtype=np.int32),
            np.asarray([False] * len(prefix) + [True] * len(answer) + [False] * len(padding)),
            len(prefix),
        )


def test_structured_response_aligns_targets_with_previous_token_logits() -> None:
    grammar = ConstrainedTokenGrammar(("left", "right"), encode=lambda text: [10 if text == "left" else 11, 1])
    transform = TokenizeStructuredResponse(
        tokenizer=_FakeTokenizer(),
        grammar=grammar,
        max_len=8,
    )

    result = transform(
        {
            "prompt": "diagnose",
            "target_text": "left",
            "state": np.zeros((7,), dtype=np.float32),
        }
    )

    targets = result["structured_target_compact_ids"]
    allowed = result["structured_allowed_token_mask"]
    assert targets.shape == (7,)
    assert np.flatnonzero(targets >= 0).tolist() == [2, 3]
    assert allowed[2].sum() == 2
    assert allowed[3].sum() == 1


def test_structured_prompt_contains_no_teacher_forcing_tokens() -> None:
    result = TokenizeStructuredPrompt(
        tokenizer=_FakeTokenizer(),
        max_len=8,
    )(
        {
            "prompt": "diagnose",
            "state": np.zeros((7,), dtype=np.float32),
        }
    )

    assert result["tokenized_prompt_mask"].sum() == 3
    assert not result["token_ar_mask"].any()
    assert not result["token_loss_mask"].any()


def test_execution_dataset_does_not_parse_v3_structured_failure_reason() -> None:
    dataset = object.__new__(TactileVLAFrameDataset)
    dataset.indices = [0]
    dataset.reasoning_source_indices = None
    dataset.stage = "execution"
    dataset.state_history_len = 0
    dataset._dataset = [
        {
            "observation.images.front": np.zeros((3, 8, 8), dtype=np.float32),
            "observation.images.left": np.zeros((3, 8, 8), dtype=np.float32),
            "observation.state": np.zeros((7,), dtype=np.float32),
            "action": np.zeros((30, 7), dtype=np.float32),
            "instruction": "Pick up the object.",
            "tactile_caption": "Touch[area=medium; rotation=clockwise]",
            "input_recovery_plan": "none",
            "failure_reason": "failure_reason=rotate front,grasp appropriate.",
            "need_recovery": True,
            "case_id": "not-a-model-input",
            "index": 123,
            "episode_id": 4,
            "attempt_id": 1,
            "frame_index": 56,
        }
    ]

    result = dataset[0]

    assert result["actions"].shape == (30, 7)
    assert "failure_reason_label" not in result
    assert "failure_reason_mask" not in result
    assert "recovery_plan_label" not in result
    assert "need_recovery_label" not in result


def test_execution_dataset_returns_masked_state_history_for_action_training() -> None:
    dataset = object.__new__(TactileVLAFrameDataset)
    dataset.indices = [0]
    dataset.reasoning_source_indices = None
    dataset.stage = "execution"
    dataset.state_history_len = 3
    history = np.arange(21, dtype=np.float32).reshape(3, 7)
    dataset._dataset = [
        {
            "observation.images.front": np.zeros((3, 8, 8), dtype=np.float32),
            "observation.images.left": np.zeros((3, 8, 8), dtype=np.float32),
            "observation.state": history,
            "observation.state_is_pad": np.asarray([True, False, False]),
            "action": np.zeros((30, 7), dtype=np.float32),
            "instruction": "Pick up the object.",
            "tactile_caption": "Touch[area=medium; rotation=none]",
            "input_recovery_plan": "none",
            "failure_reason": "",
            "need_recovery": False,
            "case_id": "not-a-model-input",
            "index": 123,
            "episode_id": 4,
            "attempt_id": 1,
            "frame_index": 2,
        }
    ]

    result = dataset[0]

    np.testing.assert_array_equal(result["observation/state"], history[-1])
    np.testing.assert_array_equal(result["observation/state_history"], history)
    np.testing.assert_array_equal(
        result["observation/state_history_mask"],
        np.asarray([False, True, True]),
    )
