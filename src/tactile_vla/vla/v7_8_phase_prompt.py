"""V7.8 visual/state prompts: strip current Touch, preserve task semantics."""

from __future__ import annotations

import re

from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE, build_recovery_prompt as original_plan_prompt
from tactile_vla.vla.v7_7_phase_prompt import compact_history

PROMPT_PROFILE = "phase_v7_8_visual_h100_11_answer"
INPUT_POLICY = {
    "schema_version": "v7_8_input_policy_v1",
    "images": ["front", "wrist"],
    "current_qpos": True,
    "discrete_qpos_history": "unchanged_v7_7_2_raw_episode_continuous_H100_to_11",
    "action": "unchanged_v7_7_2_raw_contiguous_H30_no_touch",
    "failure_plan_history": "unchanged_v7_7_2_rotation_distance_v1",
    "tactile_description": "absent_not_neutral",
    "captioner_at_runtime": False,
    "tactile_topics_at_runtime": False,
    "supervision": "unchanged_v7_7_2_offline_labels_and_counterfactual_pairs",
}


def validate_no_touch(prompt: str) -> None:
    if re.search(r"Touch\s*[:\[]|tactile_caption|(?:area|Fx|Fy|Fz|Fz_bias|rotation)\s*=", prompt, re.IGNORECASE):
        raise ValueError("V7.8 prompts must not contain tactile descriptions or neutral Touch placeholders")


def remove_touch(prompt: str, *, task: str) -> str:
    if task not in {"adjustment", "need", "failure", "plan"}:
        raise ValueError(f"Unsupported V7.8 task: {task}")
    if task == "plan":
        result, count = re.subn(r" Touch\[[^\]\n]*\]", "", prompt)
    else:
        result, count = re.subn(r"(?m)^Touch: Touch\[[^\]\n]*\]\n", "", prompt)
    if count != 1:
        raise ValueError(f"V7.8 expected exactly one original Touch caption for {task}")
    validate_no_touch(result)
    return result


def build_phase_prompt(*, instruction, recovery_plan, qpos_h100_11_discrete):
    if not str(instruction).strip():
        raise ValueError("instruction must be non-empty")
    result = "\n".join((
        "Mode: phase",
        f"Task: {str(instruction).strip()}",
        f"Recovery plan: {str(recovery_plan).strip() or 'none'}",
        "State history H100 sampled to 11 points: " + compact_history(qpos_h100_11_discrete),
    ))
    validate_no_touch(result)
    return result


def build_recovery_prompt(*, instruction, failure_recovery_memory):
    result = original_plan_prompt(
        instruction=instruction, failed_tactile_caption="",
        failure_recovery_memory=failure_recovery_memory, prompt_profile=MINIMAL_PROMPT_PROFILE,
    ).replace("  Failure-recovery memory:", " Failure-recovery memory:")
    validate_no_touch(result)
    return result
