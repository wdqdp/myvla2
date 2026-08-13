from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest

from tactile_vla.vla.recovery_action_finetuning import FineTuneFrame
from tactile_vla.vla.recovery_action_finetuning import FineTuneSelectionConfig
from tactile_vla.vla.recovery_action_finetuning import HierarchicalGroupSampler
from tactile_vla.vla.recovery_action_finetuning import NormalStratumConfig
from tactile_vla.vla.recovery_action_finetuning import NormalSuccessGroupConfig
from tactile_vla.vla.recovery_action_finetuning import RecoveryGroupConfig
from tactile_vla.vla.recovery_action_finetuning import SelectedFrame
from tactile_vla.vla.recovery_action_finetuning import load_selection_config
from tactile_vla.vla.recovery_action_finetuning import select_finetune_frames


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/recovery_action_finetune/moderately_k15.json"


def _attempt_frames(
    episode_id: int,
    attempt_id: int,
    *,
    result: str,
    direction: str = "none",
    magnitude: str = "moderately",
    frame_count: int = 10,
    action_horizon: int = 3,
) -> list[FineTuneFrame]:
    prompt = "" if attempt_id == 1 else f"recovery_plan=move horizontally {direction} {magnitude}, rest"
    base_index = episode_id * 1000 + attempt_id * 100
    return [
        FineTuneFrame(
            global_index=base_index + frame_index,
            lerobot_episode_index=episode_id * 2 + attempt_id,
            original_episode_id=episode_id,
            attempt_id=attempt_id,
            frame_index=frame_index,
            ros_timestamp=1000.0 + frame_index * 0.1,
            result=result,
            horizontal_direction=direction,
            horizontal_magnitude=magnitude,
            input_recovery_plan=prompt,
            execution_eligible=frame_index + action_horizon <= frame_count,
        )
        for frame_index in range(frame_count)
    ]


def _selection_config() -> FineTuneSelectionConfig:
    return FineTuneSelectionConfig(
        selection_seed=42,
        forbidden_recovery_magnitudes=frozenset({"slightly"}),
        normal_success_group=NormalSuccessGroupConfig(
            name="normal_success",
            attempt_id=1,
            result="success",
            sampling_weight=1.0,
            strata=(
                NormalStratumConfig(name="a", episode_ids=(1, 2), select_count=1),
                NormalStratumConfig(name="b", episode_ids=(3, 4), select_count=1),
            ),
        ),
        recovery_groups=(
            RecoveryGroupConfig(
                name="recovery_moderately",
                enabled=True,
                attempt_id=2,
                result="success",
                horizontal_magnitude="moderately",
                clip_seconds=0.6,
                sampling_weight=3.0,
                episode_ids_by_direction={"right": (10,), "left": (11,)},
            ),
        ),
    )


def test_default_config_is_moderately_only_and_has_expected_episodes() -> None:
    config = load_selection_config(DEFAULT_CONFIG)
    enabled = [group for group in config.recovery_groups if group.enabled]
    assert len(enabled) == 1
    assert enabled[0].horizontal_magnitude == "moderately"
    assert "slightly" in config.forbidden_recovery_magnitudes
    assert enabled[0].episode_ids_by_direction == {
        "right": tuple(range(41, 53)),
        "left": tuple(range(61, 73)),
        "front": tuple(range(81, 93)),
        "back": tuple(range(101, 113)),
    }
    assert sum(stratum.select_count for stratum in config.normal_success_group.strata) == 16
    assert enabled[0].clip_seconds == 15.0
    assert enabled[0].sampling_weight / config.normal_success_group.sampling_weight == 3.0


def test_config_rejects_enabled_slightly_group(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text())
    payload["recovery_groups"][0]["horizontal_magnitude"] = "slightly"
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="forbidden magnitude"):
        load_selection_config(path)


def test_config_can_add_slightly_group_without_code_changes(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text())
    payload["forbidden_recovery_magnitudes"] = []
    group = dict(payload["recovery_groups"][0])
    group.update(
        name="recovery_slightly",
        horizontal_magnitude="slightly",
        episode_ranges_by_direction={"right": [[121, 132]]},
    )
    payload["recovery_groups"].append(group)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload))
    config = load_selection_config(path)
    assert [group.horizontal_magnitude for group in config.recovery_groups if group.enabled] == [
        "moderately",
        "slightly",
    ]


def test_selection_respects_split_and_requires_full_horizon_inside_clip() -> None:
    frames: list[FineTuneFrame] = []
    for episode_id in range(1, 5):
        frames.extend(_attempt_frames(episode_id, 1, result="success"))
    frames.extend(_attempt_frames(10, 2, result="success", direction="right"))
    frames.extend(_attempt_frames(11, 2, result="success", direction="left"))
    selection = select_finetune_frames(
        frames,
        split_episode_ids=(1, 2, 3, 4, 10),
        split="train",
        config=_selection_config(),
        action_horizon=3,
    )

    recovery = [frame for frame in selection.frames if frame.source_kind == "recovery"]
    normal = [frame for frame in selection.frames if frame.source_kind == "normal_success"]
    assert selection.selected_episode_ids["recovery_moderately"] == (10,)
    assert selection.excluded_by_split["recovery_moderately"] == (11,)
    assert len(recovery) == 5
    assert max(frame.frame_index for frame in recovery) == 4
    assert max(frame.relative_timestamp for frame in recovery) == pytest.approx(0.4)
    assert len(normal) == 16
    assert len({frame.original_episode_id for frame in normal}) == 2
    assert selection.summary()["recovery_magnitude_frame_counts"] == {"moderately": 5}


def test_selection_rejects_recovery_metadata_mismatch() -> None:
    frames: list[FineTuneFrame] = []
    for episode_id in range(1, 5):
        frames.extend(_attempt_frames(episode_id, 1, result="success"))
    wrong = _attempt_frames(10, 2, result="success", direction="right", magnitude="slightly")
    frames.extend(wrong)
    config = _selection_config()
    config = replace(config, recovery_groups=(replace(config.recovery_groups[0], episode_ids_by_direction={"right": (10,)}),))
    with pytest.raises(ValueError, match="metadata mismatch"):
        select_finetune_frames(
            frames,
            split_episode_ids=(1, 2, 3, 4, 10),
            split="train",
            config=config,
            action_horizon=3,
        )


def _selected(
    global_index: int,
    *,
    group: str,
    kind: str,
    balance_key: str,
    episode_id: int,
) -> SelectedFrame:
    return SelectedFrame(
        global_index=global_index,
        source_group=group,
        source_kind=kind,
        balance_key=balance_key,
        original_episode_id=episode_id,
        attempt_id=2 if kind == "recovery" else 1,
        frame_index=0,
        relative_timestamp=0.0,
        horizontal_direction=balance_key if kind == "recovery" else "none",
        horizontal_magnitude="moderately" if kind == "recovery" else "none",
    )


def test_sampler_enforces_group_ratio_and_hierarchical_balance() -> None:
    frames: list[SelectedFrame] = []
    index = 0
    for direction in ("left", "right", "front", "back"):
        for episode_id in (index + 1, index + 2):
            for _ in range(3):
                frames.append(
                    _selected(
                        index,
                        group="recovery_moderately",
                        kind="recovery",
                        balance_key=direction,
                        episode_id=episode_id,
                    )
                )
                index += 1
    for stratum, episodes in (("capture_block_1", (101, 102)), ("capture_block_2", (103, 104))):
        for episode_id in episodes:
            for _ in range(3):
                frames.append(
                    _selected(
                        index,
                        group="normal_success",
                        kind="normal_success",
                        balance_key=stratum,
                        episode_id=episode_id,
                    )
                )
                index += 1

    sampler = HierarchicalGroupSampler(
        frames,
        group_weights={"recovery_moderately": 3.0, "normal_success": 1.0},
        num_samples=40,
        seed=7,
    )
    sampled_frames = [frames[position] for position in sampler]
    group_counts = Counter(frame.source_group for frame in sampled_frames)
    assert group_counts == {"recovery_moderately": 30, "normal_success": 10}
    direction_counts = Counter(
        frame.balance_key for frame in sampled_frames if frame.source_kind == "recovery"
    )
    assert max(direction_counts.values()) - min(direction_counts.values()) <= 1
    stratum_counts = Counter(
        frame.balance_key for frame in sampled_frames if frame.source_kind == "normal_success"
    )
    assert stratum_counts == {"capture_block_1": 5, "capture_block_2": 5}

    second = HierarchicalGroupSampler(
        frames,
        group_weights={"recovery_moderately": 3.0, "normal_success": 1.0},
        num_samples=40,
        seed=7,
    )
    assert list(HierarchicalGroupSampler(
        frames,
        group_weights={"recovery_moderately": 3.0, "normal_success": 1.0},
        num_samples=40,
        seed=7,
    )) == list(second)
