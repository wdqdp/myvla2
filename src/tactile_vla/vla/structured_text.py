"""Fixed V3 text schemas and token-level constrained decoding helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np


ROTATION_DIRECTIONS = ("left", "right", "front", "back", "none")
GRASP_POSITIONS = ("appropriate", "missing", "too_high", "too_low")
HORIZONTAL_DIRECTIONS = ("left", "right", "front", "back", "none")
MAGNITUDES = ("slightly", "moderately", "significantly")
VERTICAL_DIRECTIONS = ("up", "down", "none")


def failure_reason_text(rotation: str, grasp: str) -> str:
    return f"failure_reason=rotate {rotation},grasp {grasp}."


def recovery_plan_text(
    horizontal_direction: str,
    horizontal_magnitude: str,
    vertical_direction: str,
    vertical_magnitude: str,
) -> str:
    return (
        "recovery_plan=move horizontally "
        f"{horizontal_direction} {horizontal_magnitude}, move vertically "
        f"{vertical_direction} {vertical_magnitude}."
    )


def legal_failure_reasons() -> tuple[str, ...]:
    return tuple(
        failure_reason_text(rotation, grasp)
        for rotation in ROTATION_DIRECTIONS
        for grasp in GRASP_POSITIONS
        if not (rotation == "none" and grasp == "appropriate")
    )


def legal_recovery_plans() -> tuple[str, ...]:
    values: list[str] = []
    for horizontal_direction in HORIZONTAL_DIRECTIONS:
        horizontal_magnitudes = (
            ("moderately",) if horizontal_direction == "none" else MAGNITUDES
        )
        for horizontal_magnitude in horizontal_magnitudes:
            for vertical_direction in VERTICAL_DIRECTIONS:
                vertical_magnitudes = (
                    ("moderately",) if vertical_direction == "none" else MAGNITUDES
                )
                for vertical_magnitude in vertical_magnitudes:
                    values.append(
                        recovery_plan_text(
                            horizontal_direction,
                            horizontal_magnitude,
                            vertical_direction,
                            vertical_magnitude,
                        )
                    )
    return tuple(values)


@dataclass(frozen=True)
class GrammarEncoding:
    compact_token_ids: np.ndarray
    target_compact_ids: np.ndarray
    allowed_token_mask: np.ndarray


class ConstrainedTokenGrammar:
    """A trie over complete SentencePiece token sequences, including EOS."""

    def __init__(
        self,
        texts: Iterable[str],
        *,
        encode: Callable[[str], Sequence[int]],
    ) -> None:
        text_values = tuple(dict.fromkeys(str(text) for text in texts))
        if not text_values:
            raise ValueError("A constrained grammar requires at least one legal output")

        sequences: dict[str, tuple[int, ...]] = {}
        next_tokens: dict[tuple[int, ...], set[int]] = {}
        for text in text_values:
            sequence = tuple(int(token) for token in encode(text))
            if not sequence:
                raise ValueError(f"Tokenizer produced no tokens for legal output: {text!r}")
            sequences[text] = sequence
            for index, token in enumerate(sequence):
                next_tokens.setdefault(sequence[:index], set()).add(token)

        self._texts = text_values
        self._sequences = sequences
        self._text_by_sequence = {sequence: text for text, sequence in sequences.items()}
        self._next_tokens = next_tokens
        self._complete_sequences = frozenset(sequences.values())
        self._compact_token_ids = tuple(
            sorted({token for sequence in sequences.values() for token in sequence})
        )
        self._compact_index = {
            token: index for index, token in enumerate(self._compact_token_ids)
        }

    @property
    def texts(self) -> tuple[str, ...]:
        return self._texts

    @property
    def compact_token_ids(self) -> tuple[int, ...]:
        return self._compact_token_ids

    @property
    def max_target_tokens(self) -> int:
        return max(len(sequence) for sequence in self._sequences.values())

    def sequence_for_text(self, text: str) -> tuple[int, ...]:
        try:
            return self._sequences[text]
        except KeyError as exc:
            raise ValueError(f"Target is outside the constrained grammar: {text!r}") from exc

    def allowed_next(self, prefix: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted(self._next_tokens.get(tuple(int(token) for token in prefix), ())))

    def is_complete(self, sequence: Sequence[int]) -> bool:
        return tuple(int(token) for token in sequence) in self._complete_sequences

    def text_for_sequence(self, sequence: Sequence[int]) -> str:
        key = tuple(int(token) for token in sequence)
        try:
            return self._text_by_sequence[key]
        except KeyError as exc:
            raise ValueError(f"Token sequence is not a complete legal output: {key}") from exc

    def encode_target(self, text: str, *, output_length: int) -> GrammarEncoding:
        sequence = self.sequence_for_text(text)
        if output_length < len(sequence):
            raise ValueError(
                f"output_length={output_length} is shorter than target tokens={len(sequence)}"
            )

        compact_targets = np.full((output_length,), -100, dtype=np.int32)
        allowed = np.zeros(
            (output_length, len(self._compact_token_ids)),
            dtype=np.bool_,
        )
        for index, token in enumerate(sequence):
            compact_targets[index] = self._compact_index[token]
            for candidate in self.allowed_next(sequence[:index]):
                allowed[index, self._compact_index[candidate]] = True
            if not allowed[index, compact_targets[index]]:
                raise AssertionError("Grammar target token is not allowed by its own prefix")
        return GrammarEncoding(
            compact_token_ids=np.asarray(self._compact_token_ids, dtype=np.int32),
            target_compact_ids=compact_targets,
            allowed_token_mask=allowed,
        )


def failure_grammar(encode: Callable[[str], Sequence[int]]) -> ConstrainedTokenGrammar:
    return ConstrainedTokenGrammar(legal_failure_reasons(), encode=encode)


def recovery_grammar(encode: Callable[[str], Sequence[int]]) -> ConstrainedTokenGrammar:
    return ConstrainedTokenGrammar(legal_recovery_plans(), encode=encode)
