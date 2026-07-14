"""Centralized LeRobot field names used by the tactile VLA pipeline."""

OBS_STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_KEYS = ("observation.images.left", "observation.images.front")

ROTATION_STATE_ID_KEY = "rotation_state_id"
TACTILE_CAPTION_KEY = "tactile_caption"
NEED_RECOVERY_KEY = "need_recovery"
FAILURE_REASON_KEY = "failure_reason"
RECOVERY_PLAN_KEY = "recovery_plan"
INPUT_RECOVERY_PLAN_KEY = "input_recovery_plan"
FAILURE_RECOVERY_MEMORY_KEY = "failure_recovery_memory"

REASONING_KEYS = (
    "reasoning_has_sample",
    "reasoning_failed_attempt_id",
    "reasoning_failed_frame_index",
    "reasoning_failed_tactile_caption",
    "reasoning_failure_reason",
    "reasoning_failure_recovery_memory",
    "reasoning_recovery_plan",
)
