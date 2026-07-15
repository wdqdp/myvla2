"""OpenPI data bridge for the local tactile VLA LeRobot dataset."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from tactile_vla.vla import labels as vla_labels
from tactile_vla.vla.prompts import build_execution_prompt
from tactile_vla.vla.prompts import build_reasoning_prompt


StageName = Literal["execution", "status", "reasoning"]


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
        fps: int = 30,
        video_backend: str = "pyav",
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
        self.fps = fps
        delta_timestamps = {}
        if self.state_history_len > 0:
            delta_timestamps["observation.state"] = [
                step / fps for step in range(-(self.state_history_len - 1), 1)
            ]
        if stage == "execution":
            delta_timestamps["action"] = [step / fps for step in range(action_horizon)]
        self._dataset = LeRobotDataset(
            "tactile_vla",
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
            )
        return build_execution_prompt(
            instruction=str(item["instruction"]),
            tactile_caption=str(item["tactile_caption"]),
            input_recovery_plan=str(item["input_recovery_plan"]),
            case_id=str(item["case_id"]),
            attempt_id=int(_scalar(item["attempt_id"])),
        )

    def __getitem__(self, dataset_index: int) -> dict:
        item = self._dataset[self.indices[dataset_index]]
        reasoning_item = item
        if self.stage == "reasoning" and self.reasoning_source_indices is not None:
            reasoning_item = self._dataset[self.reasoning_source_indices[dataset_index]]
        need_recovery = bool(_scalar(item["need_recovery"]))

        failure_reason_label = -100
        failure_reason_mask = False
        failure_reason = str(item["failure_reason"]).strip()
        if failure_reason:
            failure_reason_label = vla_labels.failure_reason_to_id(failure_reason)
            failure_reason_mask = True

        recovery_plan_label = -100
        recovery_plan_mask = False
        recovery_plan = str(reasoning_item["reasoning_recovery_plan"]).strip() if self.stage == "reasoning" else ""
        if self.stage == "reasoning" and recovery_plan:
            recovery_plan_label = vla_labels.recovery_plan_to_id(recovery_plan)
            recovery_plan_mask = True
        prompt_item = reasoning_item if self.stage == "reasoning" else item

        state = item["observation.state"]
        result = {
            "observation/image": item["observation.images.front"],
            "observation/wrist_image": item["observation.images.left"],
            "observation/state": state[-1] if self.state_history_len > 0 else state,
            "prompt": self._prompt(prompt_item),
            "need_recovery_label": int(need_recovery),
            "failure_reason_label": int(failure_reason_label),
            "failure_reason_mask": bool(failure_reason_mask),
            "recovery_plan_label": int(recovery_plan_label),
            "recovery_plan_mask": bool(recovery_plan_mask),
            "global_index": int(_scalar(item["index"])),
            "episode_id": int(_scalar(item["episode_id"])),
            "attempt_id": int(_scalar(item["attempt_id"])),
            "frame_index": int(_scalar(item["frame_index"])),
        }
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
