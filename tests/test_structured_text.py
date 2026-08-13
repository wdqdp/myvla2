from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vla.vla.structured_text import ConstrainedTokenGrammar  # noqa: E402
from tactile_vla.vla.structured_text import legal_failure_reasons  # noqa: E402
from tactile_vla.vla.structured_text import legal_recovery_plans  # noqa: E402


def _encode_characters(text: str) -> list[int]:
    return [ord(character) for character in text] + [1]


def test_failure_grammar_excludes_success_state() -> None:
    values = legal_failure_reasons()
    assert len(values) == 19
    assert "failure_reason=rotate none,grasp appropriate." not in values
    assert "failure_reason=rotate left,grasp missing." in values


def test_recovery_grammar_supports_three_non_none_vertical_magnitudes() -> None:
    values = set(legal_recovery_plans())
    assert (
        "recovery_plan=move horizontally left slightly, "
        "move vertically up significantly."
    ) in values
    assert (
        "recovery_plan=move horizontally none moderately, "
        "move vertically none moderately."
    ) in values
    assert not any("horizontally none slightly" in value for value in values)
    assert not any("vertically none significantly" in value for value in values)


def test_constrained_grammar_masks_each_next_token() -> None:
    grammar = ConstrainedTokenGrammar(("ab", "ac"), encode=_encode_characters)
    encoded = grammar.encode_target("ab", output_length=3)

    compact = list(encoded.compact_token_ids)
    first_allowed = set(np.asarray(encoded.compact_token_ids)[encoded.allowed_token_mask[0]])
    second_allowed = set(np.asarray(encoded.compact_token_ids)[encoded.allowed_token_mask[1]])
    assert first_allowed == {ord("a")}
    assert second_allowed == {ord("b"), ord("c")}
    assert compact[encoded.target_compact_ids[1]] == ord("b")
    assert grammar.is_complete(_encode_characters("ab"))


def test_constrained_grammar_rejects_unknown_target() -> None:
    grammar = ConstrainedTokenGrammar(("ab",), encode=_encode_characters)
    with pytest.raises(ValueError, match="outside the constrained grammar"):
        grammar.encode_target("ac", output_length=3)
