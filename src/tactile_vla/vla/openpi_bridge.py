"""OpenPI data bridge for the local tactile VLA LeRobot dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
import torch

from tactile_vla.vla import labels as vla_labels
from tactile_vla.vla.prompts import build_execution_prompt
from tactile_vla.vla.prompts import build_failure_prompt
from tactile_vla.vla.prompts import build_monitor_prompt
from tactile_vla.vla.prompts import build_phase_prompt
from tactile_vla.vla.prompts import build_recovery_prompt
from tactile_vla.vla.prompts import build_reasoning_prompt
from tactile_vla.vla.prompts import PHASE_PROMPT_PROFILE
from tactile_vla.vla.prompts import resolve_prompt_profile
from tactile_vla.vla.structured_text import ConstrainedTokenGrammar
from tactile_vla.vla.structured_text import legal_failure_reasons
from tactile_vla.vla.structured_text import legal_recovery_plans
from tactile_vla.vla.artifacts import sha256_file
from tactile_vla.vla.v4_data import V4_NEED_SCHEMA
from tactile_vla.vla.v4_data import V4_REASONING_SCHEMA


StageName = Literal["execution", "status", "reasoning"]
V3StageBTask = Literal["need_recovery", "failure_reason"]
V4StageBTask = Literal["need", "failure", "plan"]
_V4_FAILURE_RE = re.compile(
    r"^failure_reason=rotate (right|left|front|back),grasp appropriate\.$"
)
_V4_PLAN_RE = re.compile(
    r"^recovery_plan=move horizontally (right|left|front|back) "
    r"(slightly|moderately), move vertically none moderately\.$"
)


def _ensure_openpi_imports() -> None:
    try:
        import openpi  # noqa: F401
        import lerobot  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "OpenPI/LeRobot imports failed. Run these scripts with the OpenPI uv environment, for example: "
            "`cd openpi && UV_CACHE_DIR=../.uv-cache uv run python ../scripts/<script>.py`."
        ) from exc


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_image(image: Any) -> np.ndarray:
    image = _to_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        if image.max(initial=0.0) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _scalar(value: Any) -> int | float | bool:
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.item()
    return value


def _text(value: Any) -> str:
    return str(_scalar(value))


@dataclass(frozen=True)
class TactileVLAOpenPIInputs:
    """Convert a raw local LeRobot row to OpenPI's model input dictionary."""

    right_wrist_valid: bool = False
    use_state_history: bool = False
    state_history_len: int = 60
    state_history_dim: int = 7

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        result = {
            "state": _to_numpy(data["observation/state"]).astype(np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.bool_(self.right_wrist_valid),
            },
            "prompt": str(data["prompt"]),
        }
        if self.use_state_history:
            if "observation/state_history" not in data or "observation/state_history_mask" not in data:
                raise ValueError("State-history model requires observation/state_history and its mask")
            history = _to_numpy(data["observation/state_history"]).astype(np.float32)
            history_mask = _to_numpy(data["observation/state_history_mask"]).astype(np.bool_)
            if history.shape != (self.state_history_len, self.state_history_dim):
                raise ValueError(
                    f"Expected state history [{self.state_history_len},{self.state_history_dim}], got {history.shape}"
                )
            if history_mask.shape != (self.state_history_len,):
                raise ValueError(f"Expected state history mask [{self.state_history_len}], got {history_mask.shape}")
            result["state_history"] = history
            result["state_history_mask"] = history_mask
        if "actions" in data:
            result["actions"] = _to_numpy(data["actions"]).astype(np.float32)
        if "target_text" in data:
            result["target_text"] = str(data["target_text"])
        for key in (
            "need_recovery_label",
            "failure_reason_label",
            "failure_reason_mask",
            "recovery_plan_label",
            "recovery_plan_mask",
            "global_index",
            "episode_id",
            "attempt_id",
            "frame_index",
        ):
            if key in data:
                result[key] = np.asarray(data[key])
        return result


@dataclass(frozen=True)
class TokenizeStructuredResponse:
    tokenizer: Any
    grammar: ConstrainedTokenGrammar
    max_len: int

    def __call__(self, data: dict) -> dict:
        prompt = data.pop("prompt", None)
        target_text = data.pop("target_text", None)
        if prompt is None or target_text is None:
            raise ValueError("Structured response training requires prompt and target_text")
        state = np.asarray(data["state"], dtype=np.float32)
        target_sequence = self.grammar.sequence_for_text(str(target_text))
        tokens, token_mask, ar_mask, token_loss_mask, prefix_length = (
            self.tokenizer.tokenize_structured_response(
                str(prompt),
                state,
                target_sequence,
                max_len=self.max_len,
            )
        )

        grammar_encoding = self.grammar.encode_target(
            str(target_text),
            output_length=len(target_sequence),
        )
        prediction_length = self.max_len - 1
        targets = np.full((prediction_length,), -100, dtype=np.int32)
        allowed = np.zeros(
            (prediction_length, len(self.grammar.compact_token_ids)),
            dtype=np.bool_,
        )
        for target_offset in range(len(target_sequence)):
            prediction_index = prefix_length + target_offset - 1
            if not 0 <= prediction_index < prediction_length:
                raise ValueError(
                    "Structured target cannot be aligned to next-token logits: "
                    f"prefix_length={prefix_length}, target_offset={target_offset}, max_len={self.max_len}"
                )
            targets[prediction_index] = grammar_encoding.target_compact_ids[target_offset]
            allowed[prediction_index] = grammar_encoding.allowed_token_mask[target_offset]

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": token_loss_mask,
            "structured_target_compact_ids": targets,
            "structured_allowed_token_mask": allowed,
            "structured_target_text_index": np.asarray(
                self.grammar.texts.index(str(target_text)),
                dtype=np.int32,
            ),
        }


@dataclass(frozen=True)
class TokenizeStructuredPrompt:
    tokenizer: Any
    max_len: int

    def __call__(self, data: dict) -> dict:
        prompt = data.pop("prompt", None)
        if prompt is None:
            raise ValueError("Structured generation requires a prompt")
        tokens, token_mask, ar_mask, token_loss_mask, _ = (
            self.tokenizer.tokenize_structured_response(
                str(prompt),
                np.asarray(data["state"], dtype=np.float32),
                None,
                max_len=self.max_len,
            )
        )
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": token_loss_mask,
        }


@dataclass(frozen=True)
class CastStateActionFloat32:
    def __call__(self, data: dict) -> dict:
        data["state"] = np.asarray(data["state"], dtype=np.float32)
        if "state_history" in data:
            data["state_history"] = np.asarray(data["state_history"], dtype=np.float32)
        if "state_history_mask" in data:
            data["state_history_mask"] = np.asarray(data["state_history_mask"], dtype=np.bool_)
        if "actions" in data:
            data["actions"] = np.asarray(data["actions"], dtype=np.float32)
        return data


@dataclass(frozen=True)
class NormalizeStateHistory:
    """Normalize qpos history with the same statistics as the current state."""

    norm_stats: dict | None
    use_quantiles: bool = True

    def __call__(self, data: dict) -> dict:
        if self.norm_stats is None or "state_history" not in data:
            return data
        stats = self.norm_stats.get("state")
        if stats is None:
            raise KeyError("norm_stats does not contain the state statistics required by state_history")
        history = np.asarray(data["state_history"], dtype=np.float32)
        if self.use_quantiles:
            if stats.q01 is None or stats.q99 is None:
                raise ValueError("Quantile normalization requested, but state q01/q99 are missing")
            q01 = np.asarray(stats.q01)[..., : history.shape[-1]]
            q99 = np.asarray(stats.q99)[..., : history.shape[-1]]
            history = (history - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            mean = np.asarray(stats.mean)[..., : history.shape[-1]]
            std = np.asarray(stats.std)[..., : history.shape[-1]]
            history = (history - mean) / (std + 1e-6)
        data["state_history"] = history.astype(np.float32)
        return data


class TactileVLAFrameDataset(torch.utils.data.Dataset):
    """Random-access dataset over selected local LeRobot frame indices."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        indices: Sequence[int],
        stage: StageName,
        reasoning_source_indices: Sequence[int] | None = None,
        action_horizon: int = 30,
        state_history_len: int = 0,
        fps: float = 30.0,
        video_backend: str = "pyav",
        prompt_profile: str | None = None,
        action_phase_by_global_index: Mapping[int, Mapping[str, Any]] | None = None,
        dataset_repo_id: str = "tactile_vla",
        lerobot_dataset: Any | None = None,
    ) -> None:
        _ensure_openpi_imports()
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.dataset_dir = Path(dataset_dir)
        self.indices = [int(index) for index in indices]
        self.reasoning_source_indices = (
            [int(index) for index in reasoning_source_indices] if reasoning_source_indices is not None else None
        )
        if self.reasoning_source_indices is not None and len(self.reasoning_source_indices) != len(self.indices):
            raise ValueError("reasoning_source_indices must have the same length as indices")
        self.stage = stage
        self.action_horizon = action_horizon
        self.state_history_len = int(state_history_len)
        if self.state_history_len < 0:
            raise ValueError(f"state_history_len must be non-negative, got {self.state_history_len}")
        self.fps = float(fps)
        self.prompt_profile = prompt_profile
        self.action_phase_by_global_index = (
            {int(index): dict(row) for index, row in action_phase_by_global_index.items()}
            if action_phase_by_global_index is not None
            else None
        )
        if stage == "execution" and resolve_prompt_profile(prompt_profile) == PHASE_PROMPT_PROFILE:
            if self.action_phase_by_global_index is None:
                raise ValueError("phase_v1 execution requires the V5 action-phase manifest")
            missing_phase = sorted(set(self.indices) - set(self.action_phase_by_global_index))
            if missing_phase:
                raise ValueError(
                    f"V5 action-phase manifest lacks requested global indices: {missing_phase[:20]}"
                )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        delta_timestamps = {}
        if self.state_history_len > 0:
            delta_timestamps["observation.state"] = [
                step / self.fps for step in range(-(self.state_history_len - 1), 1)
            ]
        if stage == "execution":
            delta_timestamps["action"] = [step / self.fps for step in range(action_horizon)]
        self._dataset = lerobot_dataset or LeRobotDataset(
            dataset_repo_id,
            root=self.dataset_dir,
            delta_timestamps=delta_timestamps or None,
            download_videos=False,
            video_backend=video_backend,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _prompt(self, item: dict) -> str:
        if self.stage == "reasoning":
            return build_reasoning_prompt(
                instruction=str(item["instruction"]),
                failed_tactile_caption=str(item["reasoning_failed_tactile_caption"]),
                failure_recovery_memory=str(item["reasoning_failure_recovery_memory"]),
                case_id=str(item["case_id"]),
                failed_attempt_id=int(_scalar(item["reasoning_failed_attempt_id"])),
                prompt_profile=getattr(self, "prompt_profile", None),
            )
        if resolve_prompt_profile(getattr(self, "prompt_profile", None)) == PHASE_PROMPT_PROFILE:
            global_index = int(_scalar(item["index"]))
            if self.action_phase_by_global_index is None or global_index not in self.action_phase_by_global_index:
                raise ValueError(f"No V5 phase label for global_index={global_index}")
            phase_row = self.action_phase_by_global_index[global_index]
            frame_identity = (
                int(_scalar(item["episode_id"])),
                int(_scalar(item["attempt_id"])),
                int(_scalar(item["frame_index"])),
            )
            manifest_identity = (
                int(phase_row["episode_id"]),
                int(phase_row["attempt_id"]),
                int(phase_row["frame_index"]),
            )
            if frame_identity != manifest_identity:
                raise ValueError(
                    f"V5 phase/LeRobot frame identity mismatch at global_index={global_index}: "
                    f"manifest={manifest_identity}, lerobot={frame_identity}"
                )
            return build_phase_prompt(
                phase=str(phase_row["phase"]),
                instruction=str(item["instruction"]),
                recovery_plan=str(item["input_recovery_plan"]),
                prompt_profile=self.prompt_profile,
            )
        return build_execution_prompt(
            instruction=str(item["instruction"]),
            tactile_caption=str(item["tactile_caption"]),
            input_recovery_plan=str(item["input_recovery_plan"]),
            case_id=str(item["case_id"]),
            attempt_id=int(_scalar(item["attempt_id"])),
            prompt_profile=getattr(self, "prompt_profile", None),
        )

    def __getitem__(self, dataset_index: int) -> dict:
        item = self._dataset[self.indices[dataset_index]]
        reasoning_item = item
        if self.stage == "reasoning" and self.reasoning_source_indices is not None:
            reasoning_item = self._dataset[self.reasoning_source_indices[dataset_index]]
        prompt_item = reasoning_item if self.stage == "reasoning" else item

        state = item["observation.state"]
        result = {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": state[-1] if self.state_history_len > 0 else state,
            "prompt": self._prompt(prompt_item),
            "global_index": int(_scalar(item["index"])),
            "episode_id": int(_scalar(item["episode_id"])),
            "attempt_id": int(_scalar(item["attempt_id"])),
            "frame_index": int(_scalar(item["frame_index"])),
        }
        # Execution streams optimize only the action target. In particular,
        # V3 stores a structured failure_reason string that is intentionally
        # incompatible with the legacy V2 classification label map. Do not
        # parse or carry unused auxiliary labels into Stage A / action replay.
        if self.stage != "execution":
            need_recovery = bool(_scalar(item["need_recovery"]))
            failure_reason_label = -100
            failure_reason_mask = False
            failure_reason = str(item["failure_reason"]).strip()
            if failure_reason:
                failure_reason_label = vla_labels.failure_reason_to_id(failure_reason)
                failure_reason_mask = True

            recovery_plan_label = -100
            recovery_plan_mask = False
            recovery_plan = (
                str(reasoning_item["reasoning_recovery_plan"]).strip()
                if self.stage == "reasoning"
                else ""
            )
            if self.stage == "reasoning" and recovery_plan:
                recovery_plan_label = vla_labels.recovery_plan_to_id(recovery_plan)
                recovery_plan_mask = True
            result.update(
                {
                    "need_recovery_label": int(need_recovery),
                    "failure_reason_label": int(failure_reason_label),
                    "failure_reason_mask": bool(failure_reason_mask),
                    "recovery_plan_label": int(recovery_plan_label),
                    "recovery_plan_mask": bool(recovery_plan_mask),
                }
            )
        if self.state_history_len > 0:
            state_history = _to_numpy(state).astype(np.float32)
            history_is_pad = _to_numpy(item["observation.state_is_pad"]).astype(np.bool_)
            expected_shape = (self.state_history_len, state_history.shape[-1])
            if state_history.shape != expected_shape or history_is_pad.shape != (self.state_history_len,):
                raise ValueError(
                    "Unexpected state history returned by LeRobot: "
                    f"history={state_history.shape}, pad={history_is_pad.shape}, expected={expected_shape}"
                )
            result["observation/state_history"] = state_history
            result["observation/state_history_mask"] = np.logical_not(history_is_pad)
        if self.stage == "execution":
            result["actions"] = item["action"]
        return result


class V3StageBFrameDataset(torch.utils.data.Dataset):
    """V3 monitor/diagnosis rows selected by the persisted frame index."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        indices: Sequence[int],
        task: V3StageBTask,
        video_backend: str = "pyav",
        prompt_profile: str | None = None,
        dataset_repo_id: str = "tactile_vla_v3",
        lerobot_dataset: Any | None = None,
    ) -> None:
        _ensure_openpi_imports()
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.indices = [int(index) for index in indices]
        self.task = task
        self.prompt_profile = prompt_profile
        self._dataset = lerobot_dataset or LeRobotDataset(
            dataset_repo_id,
            root=Path(dataset_dir),
            download_videos=False,
            video_backend=video_backend,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, dataset_index: int) -> dict:
        item = self._dataset[self.indices[dataset_index]]
        instruction = str(item["instruction"])
        tactile_caption = str(item["tactile_caption"])
        input_recovery_plan = str(item["input_recovery_plan"])
        result = {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": item["observation.state"],
            "global_index": int(_scalar(item["index"])),
            "episode_id": int(_scalar(item["episode_id"])),
            "attempt_id": int(_scalar(item["attempt_id"])),
            "frame_index": int(_scalar(item["frame_index"])),
        }
        if self.task == "need_recovery":
            result.update(
                {
                    "prompt": build_monitor_prompt(
                        instruction=instruction,
                        tactile_caption=tactile_caption,
                        input_recovery_plan=input_recovery_plan,
                        prompt_profile=self.prompt_profile,
                    ),
                    "need_recovery_label": int(bool(_scalar(item["need_recovery"]))),
                }
            )
            return result

        failure_reason = str(item["failure_reason"]).strip()
        if not failure_reason:
            raise ValueError(
                "failure_reason task received an empty target at "
                f"global_index={result['global_index']}"
            )
        result.update(
            {
                "prompt": build_failure_prompt(
                    instruction=instruction,
                    tactile_caption=tactile_caption,
                    input_recovery_plan=input_recovery_plan,
                    prompt_profile=self.prompt_profile,
                ),
                "target_text": failure_reason,
            }
        )
        return result


class V3RecoveryManifestDataset(torch.utils.data.Dataset):
    """Expand each V3 reasoning manifest sample over its fixed observation window."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        manifest_file: str | Path,
        frame_lookup: dict[tuple[int, int, int], int],
        reasoning_window_frames: int,
        training: bool,
        video_backend: str = "pyav",
        prompt_profile: str | None = None,
        dataset_repo_id: str = "tactile_vla_v3",
        lerobot_dataset: Any | None = None,
    ) -> None:
        _ensure_openpi_imports()
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        if reasoning_window_frames <= 0:
            raise ValueError(
                f"reasoning_window_frames must be positive, got {reasoning_window_frames}"
            )
        manifest_file = Path(manifest_file)
        samples = [
            json.loads(line)
            for line in manifest_file.read_text().splitlines()
            if line.strip()
        ]
        offsets = range(reasoning_window_frames) if training else (reasoning_window_frames - 1,)
        expanded: list[tuple[dict[str, Any], int]] = []
        for sample in samples:
            observation = sample["current_observation"]
            episode_id = int(observation["episode_id"])
            attempt_id = int(observation["attempt_id"])
            start_frame = int(observation["frame_index"])
            for offset in offsets:
                key = (episode_id, attempt_id, start_frame + int(offset))
                if key not in frame_lookup:
                    raise ValueError(
                        "V3 recovery window is incomplete for "
                        f"episode_id={episode_id}, attempt_id={attempt_id}, "
                        f"start_frame={start_frame}, window_frames={reasoning_window_frames}; "
                        f"missing frame={key[2]}"
                    )
                expanded.append((sample, int(frame_lookup[key])))

        self.samples = expanded
        self.prompt_profile = prompt_profile
        self._dataset = lerobot_dataset or LeRobotDataset(
            dataset_repo_id,
            root=Path(dataset_dir),
            download_videos=False,
            video_backend=video_backend,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, dataset_index: int) -> dict:
        sample, global_index = self.samples[dataset_index]
        item = self._dataset[global_index]
        target = str(sample["target_recovery_plan"]).strip()
        if not target or not bool(sample.get("target_recovery_plan_mask", True)):
            raise ValueError(f"Manifest sample has no valid recovery target at row {dataset_index}")
        return {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": item["observation.state"],
            "prompt": build_recovery_prompt(
                instruction=str(item["instruction"]),
                failed_tactile_caption=str(item["tactile_caption"]),
                failure_recovery_memory=sample["failure_recovery_memory"],
                prompt_profile=self.prompt_profile,
            ),
            "target_text": target,
            "global_index": int(_scalar(item["index"])),
            "episode_id": int(_scalar(item["episode_id"])),
            "attempt_id": int(_scalar(item["attempt_id"])),
            "frame_index": int(_scalar(item["frame_index"])),
        }


class V4DirectManifestDataset(torch.utils.data.Dataset):
    """Read exactly one V4 manifest row and one LeRobot frame per sample.

    V4 failure/reasoning manifests are already expanded to one row per real
    frame.  In contrast to :class:`V3RecoveryManifestDataset`, this dataset
    never expands a row into a time window.
    """

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        manifest_file: str | Path,
        manifest_row_indices: Sequence[int],
        global_indices: Sequence[int],
        task: V4StageBTask,
        expected_manifest_sha256: str | None = None,
        expected_split: str | None = None,
        video_backend: str = "pyav",
        prompt_profile: str | None = None,
        dataset_repo_id: str = "tactile_vla_rotation_v4",
        lerobot_dataset: Any | None = None,
    ) -> None:
        if task not in {"need", "failure", "plan"}:
            raise ValueError(f"Unsupported V4 direct manifest task: {task!r}")
        manifest_file = Path(manifest_file)
        if expected_manifest_sha256 is not None:
            actual = sha256_file(manifest_file)
            if actual != expected_manifest_sha256:
                raise ValueError(
                    f"V4 {task} manifest hash mismatch: expected={expected_manifest_sha256}, actual={actual}"
                )
        all_rows = [
            json.loads(line)
            for line in manifest_file.read_text().splitlines()
            if line.strip()
        ]
        row_indices = [int(value) for value in manifest_row_indices]
        indices = [int(value) for value in global_indices]
        if len(row_indices) != len(indices):
            raise ValueError("V4 manifest_row_indices and global_indices lengths differ")
        samples: list[tuple[dict[str, Any], int, int]] = []
        for row_index, global_index in zip(row_indices, indices, strict=True):
            if not 0 <= row_index < len(all_rows):
                raise ValueError(f"V4 {task} manifest row index is out of range: {row_index}")
            row = all_rows[row_index]
            if not isinstance(row, dict):
                raise ValueError(f"V4 {task} manifest row {row_index} is not an object")
            split = str(row.get("split", ""))
            if expected_split is not None and split != expected_split:
                raise ValueError(
                    f"V4 {task} row {row_index} split={split!r}, expected={expected_split!r}"
                )
            if task == "need":
                if row.get("schema_version") != V4_NEED_SCHEMA:
                    raise ValueError(f"V4 need row {row_index} has wrong schema")
                if int(row.get("global_index", -1)) != global_index:
                    raise ValueError(f"V4 need row {row_index} global index mismatch")
                if not isinstance(row.get("need_recovery"), bool):
                    raise ValueError(f"V4 need row {row_index} lacks a boolean target")
            else:
                if row.get("schema_version") != V4_REASONING_SCHEMA:
                    raise ValueError(f"V4 {task} row {row_index} has wrong schema")
                expected_type = "failure_reason" if task == "failure" else "recovery_plan"
                if row.get("sample_type") != expected_type:
                    raise ValueError(f"V4 {task} row {row_index} has wrong sample_type")
                observation = row.get("current_observation")
                if not isinstance(observation, Mapping) or observation.get("source_type") != "real":
                    raise ValueError(f"V4 {task} row {row_index} lacks real observation identity")
                if int(row.get("frame_index", -1)) != int(observation.get("frame_index", -2)):
                    raise ValueError(f"V4 {task} row {row_index} has inconsistent frame identity")
                offset = int(row.get("frame_offset", -1))
                if (
                    int(observation.get("frame_offset", -2)) != offset
                    or int(row.get("window_start", -1)) + offset != int(row["frame_index"])
                ):
                    raise ValueError(f"V4 {task} row {row_index} has inconsistent window identity")
                if task == "failure":
                    target = str(row.get("target_failure_reason", ""))
                    if target not in legal_failure_reasons():
                        raise ValueError(f"V4 failure row {row_index} target is outside full V3 grammar")
                    if target != str(observation.get("failure_reason", "")):
                        raise ValueError(f"V4 failure row {row_index} target differs from real observation")
                else:
                    target = str(row.get("target_recovery_plan", ""))
                    if target not in legal_recovery_plans():
                        raise ValueError(f"V4 plan row {row_index} target is outside full V3 grammar")
                    target_match = _V4_PLAN_RE.fullmatch(target)
                    if target_match is None:
                        raise ValueError(f"V4 plan row {row_index} target is outside the rotation subset")
                    memory = row.get("failure_recovery_memory")
                    if not isinstance(memory, list) or len(memory) != int(row.get("memory_length", -1)):
                        raise ValueError(f"V4 plan row {row_index} has invalid memory length")
                    if not 1 <= len(memory) <= 4:
                        raise ValueError(f"V4 plan row {row_index} memory length is outside [1,4]")
                    if (target_match.group(2) == "moderately" and len(memory) != 1) or (
                        target_match.group(2) == "slightly" and len(memory) not in {2, 3, 4}
                    ):
                        raise ValueError(f"V4 plan row {row_index} target/memory length mismatch")
                    variant_id = str(row.get("variant_id", ""))
                    rule_version = str(row.get("rule_version", ""))
                    seed = int(row.get("seed", -1))
                    if not variant_id or not rule_version or seed < 0:
                        raise ValueError(f"V4 plan row {row_index} lacks synthetic provenance")
                    failure_directions: list[str] = []
                    for pair_index, pair in enumerate(memory):
                        if not isinstance(pair, Mapping) or pair.get("source_type") != "synthetic":
                            raise ValueError(f"V4 plan row {row_index} contains non-synthetic memory")
                        if {"donor_episode_id", "donor_attempt_id"} & set(pair):
                            raise ValueError(f"V4 plan row {row_index} contains fake donor provenance")
                        if (
                            pair.get("variant_id") != variant_id
                            or pair.get("rule_version") != rule_version
                            or int(pair.get("seed", -1)) != seed
                            or int(pair.get("pair_index", -1)) != pair_index
                        ):
                            raise ValueError(f"V4 plan row {row_index} pair provenance mismatch")
                        failure_match = _V4_FAILURE_RE.fullmatch(str(pair.get("failure_reason", "")))
                        if failure_match is None:
                            raise ValueError(f"V4 plan row {row_index} pair failure is outside V4 grammar")
                        failure_directions.append(failure_match.group(1))
                        plan_text = str(pair.get("recovery_plan", ""))
                        if pair_index == 0:
                            if plan_text != "initial plan":
                                raise ValueError(f"V4 plan row {row_index} first pair is not initial plan")
                        else:
                            pair_plan = _V4_PLAN_RE.fullmatch(plan_text)
                            expected_magnitude = "moderately" if pair_index == 1 else "slightly"
                            if (
                                pair_plan is None
                                or pair_plan.group(1) != failure_directions[pair_index - 1]
                                or pair_plan.group(2) != expected_magnitude
                            ):
                                raise ValueError(f"V4 plan row {row_index} memory chain is incompatible")
                    real_failure = str(observation.get("failure_reason", ""))
                    if str(memory[-1]["failure_reason"]) != real_failure:
                        raise ValueError(f"V4 plan row {row_index} terminal failure differs from observation")
                    if target_match.group(1) != failure_directions[-1]:
                        raise ValueError(f"V4 plan row {row_index} target direction differs from terminal failure")
                    target_source = row.get("target_source")
                    if not isinstance(target_source, Mapping) or target_source.get("source_type") != "real":
                        raise ValueError(f"V4 plan row {row_index} target is not real")
                    if (
                        int(target_source.get("episode_id", -1)) != int(observation["episode_id"])
                        or int(target_source.get("failed_attempt_id", -1)) != int(observation["attempt_id"])
                        or int(target_source.get("plan_attempt_id", -1)) != int(observation["attempt_id"]) + 1
                    ):
                        raise ValueError(f"V4 plan row {row_index} target is not from the adjacent real attempt")
            samples.append((row, global_index, row_index))

        self.samples = samples
        self.task = task
        self.prompt_profile = prompt_profile
        if lerobot_dataset is None:
            _ensure_openpi_imports()
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

            lerobot_dataset = LeRobotDataset(
                dataset_repo_id,
                root=Path(dataset_dir),
                download_videos=False,
                video_backend=video_backend,
            )
        self._dataset = lerobot_dataset

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _identity(item: Mapping[str, Any]) -> tuple[int, int, int, int]:
        return (
            int(_scalar(item["index"])),
            int(_scalar(item["episode_id"])),
            int(_scalar(item["attempt_id"])),
            int(_scalar(item["frame_index"])),
        )

    def __getitem__(self, dataset_index: int) -> dict:
        row, global_index, row_index = self.samples[dataset_index]
        item = self._dataset[global_index]
        item_identity = self._identity(item)
        if item_identity[0] != global_index:
            raise ValueError(
                f"V4 {self.task} row {row_index} retrieved LeRobot index {item_identity[0]}, expected {global_index}"
            )
        if self.task == "need":
            row_identity = (
                int(row["global_index"]),
                int(row["episode_id"]),
                int(row["attempt_id"]),
                int(row["frame_index"]),
            )
        else:
            observation = row["current_observation"]
            row_identity = (
                global_index,
                int(observation["episode_id"]),
                int(observation["attempt_id"]),
                int(observation["frame_index"]),
            )
            if _text(item["tactile_caption"]) != str(observation["tactile_caption"]):
                raise ValueError(f"V4 {self.task} row {row_index} tactile caption mismatch")
        if row_identity != item_identity:
            raise ValueError(
                f"V4 {self.task} row {row_index} identity mismatch: manifest={row_identity}, LeRobot={item_identity}"
            )

        instruction = _text(item["instruction"])
        tactile_caption = _text(item["tactile_caption"])
        input_recovery_plan = _text(item["input_recovery_plan"])
        result = {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": item["observation.state"],
            "global_index": item_identity[0],
            "episode_id": item_identity[1],
            "attempt_id": item_identity[2],
            "frame_index": item_identity[3],
        }
        if self.task == "need":
            target = bool(row["need_recovery"])
            stored = bool(_scalar(item["need_recovery"]))
            if target != stored:
                raise ValueError(
                    f"V4 need row {row_index} target={target} differs from LeRobot label={stored}"
                )
            result.update(
                {
                    "prompt": build_monitor_prompt(
                        instruction=instruction,
                        tactile_caption=tactile_caption,
                        input_recovery_plan=input_recovery_plan,
                        prompt_profile=self.prompt_profile,
                    ),
                    "need_recovery_label": int(target),
                }
            )
            return result

        if not bool(_scalar(item["need_recovery"])):
            raise ValueError(f"V4 {self.task} row {row_index} maps to a non-failure-active frame")
        if self.task == "failure":
            target = str(row["target_failure_reason"])
            stored_target = _text(item["failure_reason"])
            if target != stored_target or not bool(_scalar(item["failure_reason_mask"])):
                raise ValueError(f"V4 failure row {row_index} target/mask differs from LeRobot")
            result.update(
                {
                    "prompt": build_failure_prompt(
                        instruction=instruction,
                        tactile_caption=tactile_caption,
                        input_recovery_plan=input_recovery_plan,
                        prompt_profile=self.prompt_profile,
                    ),
                    "target_text": target,
                }
            )
            return result

        target = str(row["target_recovery_plan"])
        stored_target = _text(item["recovery_plan"])
        if target != stored_target or not bool(_scalar(item["recovery_plan_mask"])):
            raise ValueError(f"V4 plan row {row_index} target/mask differs from LeRobot")
        result.update(
            {
                "prompt": build_recovery_prompt(
                    instruction=instruction,
                    failed_tactile_caption=tactile_caption,
                    failure_recovery_memory=row["failure_recovery_memory"],
                    prompt_profile=self.prompt_profile,
                ),
                "target_text": target,
            }
        )
        return result


def build_transform(
    model_config: Any,
    *,
    norm_stats: dict | None = None,
    use_quantile_norm: bool = True,
    use_delta_actions: bool = True,
    delta_action_dims: int = 7,
):
    _ensure_openpi_imports()
    import openpi.transforms as openpi_transforms
    from openpi.models import tokenizer as openpi_tokenizer

    use_state_history = bool(getattr(model_config, "use_state_history", False))
    transforms = [
        TactileVLAOpenPIInputs(
            use_state_history=use_state_history,
            state_history_len=int(getattr(model_config, "state_history_len", 60)),
            state_history_dim=int(getattr(model_config, "state_history_dim", 7)),
        )
    ]
    if use_delta_actions:
        transforms.append(openpi_transforms.DeltaActions(openpi_transforms.make_bool_mask(delta_action_dims)))
    transforms.extend(
        [
            NormalizeStateHistory(norm_stats, use_quantiles=use_quantile_norm),
            openpi_transforms.Normalize(norm_stats, use_quantiles=use_quantile_norm),
            CastStateActionFloat32(),
            openpi_transforms.ResizeImages(224, 224),
            openpi_transforms.TokenizePrompt(
                openpi_tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=getattr(model_config, "discrete_state_input", False),
            ),
            openpi_transforms.PadStatesAndActions(model_config.action_dim),
        ]
    )
    return openpi_transforms.compose(
        transforms
    )


def build_structured_text_transform(
    model_config: Any,
    *,
    tokenizer: Any,
    grammar: ConstrainedTokenGrammar,
    max_len: int | None = None,
    norm_stats: dict | None = None,
    use_quantile_norm: bool = True,
):
    """Build the frozen-backbone input transform plus causal response labels."""

    _ensure_openpi_imports()
    import openpi.transforms as openpi_transforms

    transforms = [
        TactileVLAOpenPIInputs(use_state_history=False),
        NormalizeStateHistory(norm_stats, use_quantiles=use_quantile_norm),
        openpi_transforms.Normalize(norm_stats, use_quantiles=use_quantile_norm),
        CastStateActionFloat32(),
        openpi_transforms.ResizeImages(224, 224),
        TokenizeStructuredResponse(
            tokenizer=tokenizer,
            grammar=grammar,
            max_len=int(max_len or model_config.max_token_len),
        ),
        openpi_transforms.PadStatesAndActions(model_config.action_dim),
    ]
    return openpi_transforms.compose(transforms)


def build_structured_inference_transform(
    model_config: Any,
    *,
    tokenizer: Any,
    max_len: int | None = None,
    norm_stats: dict | None = None,
    use_quantile_norm: bool = True,
):
    _ensure_openpi_imports()
    import openpi.transforms as openpi_transforms

    return openpi_transforms.compose(
        [
            TactileVLAOpenPIInputs(use_state_history=False),
            openpi_transforms.Normalize(norm_stats, use_quantiles=use_quantile_norm),
            CastStateActionFloat32(),
            openpi_transforms.ResizeImages(224, 224),
            TokenizeStructuredPrompt(
                tokenizer=tokenizer,
                max_len=int(max_len or model_config.max_token_len),
            ),
            openpi_transforms.PadStatesAndActions(model_config.action_dim),
        ]
    )


def build_action_output_transform(
    *,
    norm_stats: dict | None = None,
    use_quantile_norm: bool = True,
    use_delta_actions: bool = True,
    delta_action_dims: int = 7,
):
    """Convert model action outputs back to the raw robot action space."""

    _ensure_openpi_imports()
    import openpi.transforms as openpi_transforms

    transforms = [openpi_transforms.Unnormalize(norm_stats, use_quantiles=use_quantile_norm)]
    if use_delta_actions:
        transforms.append(openpi_transforms.AbsoluteActions(openpi_transforms.make_bool_mask(delta_action_dims)))
    return openpi_transforms.compose(transforms)


class TransformedTactileVLADataset(torch.utils.data.Dataset):
    def __init__(self, dataset: TactileVLAFrameDataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        return self.transform(self.dataset[index])


def collate_numpy(items: Sequence[dict]) -> dict:
    import jax

    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)
