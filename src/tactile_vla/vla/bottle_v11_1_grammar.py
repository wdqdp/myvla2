"""Complete bottle text formats: four failure fields and three recovery axes."""

from functools import cache

from tactile_vla.vla.structured_text import (
    ConstrainedTokenGrammar,
    legal_failure_reasons,
    legal_recovery_plans,
)

TEXT_SCHEMA_POLICY = {
    "schema_version": "bottle_v11_1_text_schema_v1",
    "failure_fields": ["rotate", "grasp", "fz_bias", "fy_change"],
    "recovery_axes": ["horizontal", "vertical", "rotate"],
    "output_space": "complete_legal_combinations_not_single_training_target",
    "empty_recovery": "excluded",
}


@cache
def bottle_failure_reasons() -> tuple[str, ...]:
    return tuple(text for text in legal_failure_reasons(include_fy_change=True) if ",fy_change " in text)


@cache
def bottle_recovery_plans() -> tuple[str, ...]:
    empty = "recovery_plan=move horizontally none moderately, move vertically none moderately, rotate none moderately."
    return tuple(text for text in legal_recovery_plans(include_rotate=True) if ", rotate " in text and text != empty)


def build_training_grammars(tokenizer, *, args, index):
    if args.data_profile != "bottle_v11_1_five_task_h100" or index.get("text_schema_policy") != TEXT_SCHEMA_POLICY:
        raise ValueError("Bottle V11.1 text grammar requires its versioned data/schema contract")

    def encode(text):
        return tokenizer.encode_text(text, add_eos=True)

    return (
        ConstrainedTokenGrammar(bottle_failure_reasons(), encode=encode),
        ConstrainedTokenGrammar(bottle_recovery_plans(), encode=encode),
    )
