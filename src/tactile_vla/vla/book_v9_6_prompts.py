"""Book V9.6 prompts: the V9.5 information protocol with all touch text removed."""

from __future__ import annotations

import re

from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt as original_plan_prompt
from tactile_vla.vla.v7_7_phase_prompt import compact_history

PROMPT_PROFILE = "phase_book_v9_6_visual_h100_11_answer"
INPUT_POLICY = {
    "schema_version": "book_v9_6_input_policy_v1",
    "images": ["front", "wrist"],
    "current_qpos": True,
    "discrete_qpos_history": "H100_to_11_episode_continuous",
    "failure_plan_history": "initial_plus_latest_three_pairs",
    "tactile_description": "absent_not_neutral",
    "captioner_at_runtime": False,
    "tactile_topics_at_runtime": False,
    "supervision": "unchanged_v9_5_real_labels_offline_only",
}


def validate_no_touch(prompt: str) -> None:
    if re.search(r"Touch\s*[:\[]|tactile_caption|(?:area|Fx|Fy|Fz|Fz_bias|rotation)\s*=", prompt, re.IGNORECASE):
        raise ValueError("V9.6 model prompts must not contain tactile descriptions")


def remove_touch(prompt: str, *, task: str) -> str:
    """Remove exactly the original current caption; never change task/history text."""
    if task == "plan":
        result, count = re.subn(r" Touch\[[^\]\n]*\]", "", prompt)
    else:
        result, count = re.subn(r"(?m)^Touch: Touch\[[^\]\n]*\]\n", "", prompt)
    if count != 1:
        raise ValueError(f"V9.6 expected exactly one original Touch caption for {task}")
    validate_no_touch(result)
    return result


def build_phase_prompt(*, instruction, recovery_plan, qpos_h100_11_discrete):
    if not str(instruction).strip():
        raise ValueError("instruction must be non-empty")
    prompt = "\n".join(
        (
            "Mode: phase",
            f"Task: {str(instruction).strip()}",
            f"Recovery plan: {str(recovery_plan).strip() or 'none'}",
            "State history H100 sampled to 11 points: " + compact_history(qpos_h100_11_discrete),
        )
    )
    validate_no_touch(prompt)
    return prompt


def build_recovery_prompt(*, instruction, failure_recovery_memory):
    # Keep punctuation and memory serialization identical to the original protocol.
    prompt = original_plan_prompt(
        instruction=instruction,
        failed_tactile_caption="",
        failure_recovery_memory=failure_recovery_memory,
        prompt_profile=MINIMAL_PROMPT_PROFILE,
    ).replace("  Failure-recovery memory:", " Failure-recovery memory:")
    validate_no_touch(prompt)
    return prompt
