from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vla.vla.prompts import build_assessment_prompt  # noqa: E402
from tactile_vla.vla.prompts import build_execution_prompt  # noqa: E402
from tactile_vla.vla.prompts import build_failure_prompt  # noqa: E402
from tactile_vla.vla.prompts import build_monitor_prompt  # noqa: E402
from tactile_vla.vla.prompts import build_recovery_prompt  # noqa: E402
from tactile_vla.vla.prompts import build_reasoning_prompt  # noqa: E402
from tactile_vla.vla.prompts import format_memory  # noqa: E402
from tactile_vla.vla.prompts import MINIMAL_PROMPT_PROFILE  # noqa: E402
from tactile_vla.vla.prompts import resolve_prompt_profile  # noqa: E402
from tactile_vla.vla.prompts import update_failure_recovery_memory  # noqa: E402


def test_v3_memory_format_has_no_attempt_identifiers_or_duplicate_prefixes() -> None:
    text = format_memory(
        [
            {
                "recovery_plan": "initial plan",
                "failure_reason": (
                    "failure_reason=rotate clockwise,grasp appropriate."
                ),
            },
            {
                "recovery_plan": (
                    "recovery_plan=move horizontally right moderately, "
                    "move vertically none moderately."
                ),
                "failure_reason": (
                    "failure_reason=rotate counterclockwise,grasp appropriate."
                ),
            },
        ]
    )
    assert "attempt" not in text.lower()
    assert "recovery_plan=recovery_plan=" not in text
    assert "failure_reason=failure_reason=" not in text
    assert text.count("recovery_plan=") == 2
    assert text.count("failure_reason=") == 2


def test_prompts_exclude_case_and_attempt_ids() -> None:
    execution = build_execution_prompt(
        instruction="test",
        tactile_caption="Touch[rotation=none]",
        input_recovery_plan="",
        case_id="case_secret",
        attempt_id=4,
    )
    reasoning = build_reasoning_prompt(
        instruction="test",
        failed_tactile_caption="Touch[rotation=clockwise]",
        failure_recovery_memory=[
            {
                "recovery_plan": "initial plan",
                "failure_reason": (
                    "failure_reason=rotate clockwise,grasp appropriate."
                ),
            }
        ],
        case_id="case_secret",
        failed_attempt_id=4,
    )
    for prompt in (execution, reasoning):
        assert "case_secret" not in prompt
        assert "Attempt:" not in prompt
        assert "Failed attempt:" not in prompt


def test_v3_need_and_failure_share_the_exact_assessment_prompt() -> None:
    monitor = build_monitor_prompt(
        instruction="test",
        tactile_caption="Touch[rotation=none]",
        input_recovery_plan="initial plan",
    )
    failure = build_failure_prompt(
        instruction="test",
        tactile_caption="Touch[rotation=clockwise]",
        input_recovery_plan="initial plan",
    )
    assessment = build_assessment_prompt(
        instruction="test",
        tactile_caption="Touch[rotation=none]",
        input_recovery_plan="initial plan",
    )
    recovery = build_recovery_prompt(
        instruction="test",
        failed_tactile_caption="Touch[rotation=clockwise]",
        failure_recovery_memory=[
            {
                "recovery_plan": "initial plan",
                "failure_reason": "failure_reason=rotate right,grasp appropriate.",
            }
        ],
    )

    assert monitor == assessment
    assert failure != assessment  # Different touch input above, not a different task prefix.
    same_observation_failure = build_failure_prompt(
        instruction="test",
        tactile_caption="Touch[rotation=none]",
        input_recovery_plan="initial plan",
    )
    assert same_observation_failure == monitor
    assert "Mode: tactile assessment." in monitor
    assert "output only the structured failure_reason." in failure
    assert "Mode: reasoning." in recovery
    for prompt in (assessment, monitor, failure, recovery):
        assert "Attempt:" not in prompt
        assert "Case:" not in prompt


def test_minimal_v1_prompt_snapshots() -> None:
    memory = [
        {
            "recovery_plan": "initial plan",
            "failure_reason": "failure_reason=rotate right,grasp appropriate.",
        }
    ]
    execution = build_execution_prompt(
        instruction="pick object",
        tactile_caption="Touch[rotation=clockwise]",
        input_recovery_plan="recovery_plan=move right",
        prompt_profile=MINIMAL_PROMPT_PROFILE,
    )
    assessment = build_assessment_prompt(
        instruction="pick object",
        tactile_caption="Touch[rotation=clockwise]",
        input_recovery_plan="initial plan",
        prompt_profile=MINIMAL_PROMPT_PROFILE,
    )
    reasoning = build_reasoning_prompt(
        instruction="pick object",
        failed_tactile_caption="Touch[rotation=clockwise]",
        failure_recovery_memory=memory,
        prompt_profile=MINIMAL_PROMPT_PROFILE,
    )

    assert execution == (
        "Mode: execution. Task: pick object "
        "Recovery plan: recovery_plan=move right."
    )
    assert "Touch[" not in execution
    assert assessment == (
        "Mode: tactile assessment. Task: pick object Touch[rotation=clockwise] "
        "Recovery plan: initial plan."
    )
    assert reasoning == (
        "Mode: reasoning. Task: pick object Touch[rotation=clockwise] "
        "Failure-recovery memory: recovery_plan=initial plan; "
        "failure_reason=rotate right,grasp appropriate."
    )


def test_omitted_checkpoint_prompt_profile_falls_back_to_legacy() -> None:
    assert resolve_prompt_profile(None) == "legacy"
    legacy = build_execution_prompt(
        instruction="test",
        tactile_caption="Touch[rotation=none]",
    )
    assert "Touch[rotation=none]" in legacy
    assert legacy.endswith("monitor whether recovery is needed.")


def test_minimal_runtime_memory_replaces_the_previous_entry() -> None:
    old = [{"recovery_plan": "old", "failure_reason": "old reason"}]
    latest = {"recovery_plan": "new", "failure_reason": "new reason"}
    assert update_failure_recovery_memory(
        old,
        latest,
        prompt_profile=MINIMAL_PROMPT_PROFILE,
    ) == [latest]
    assert update_failure_recovery_memory(
        old,
        latest,
        prompt_profile="legacy",
    ) == [*old, latest]
