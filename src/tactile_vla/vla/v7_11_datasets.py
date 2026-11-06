"""Dataset wrappers for V7.11 direct tactile recovery."""

from tactile_vla.vla.openpi_bridge import TactileVLAFrameDataset
from tactile_vla.vla.v7_7_multitask_data import V77ManifestDataset
from tactile_vla.vla.v7_11_data import direct_recovery_prompt, phase_prompt_without_plan

class DirectActionDataset(TactileVLAFrameDataset):
    """Preserve execution action prompts; use the current tactile caption for adjustment."""

    def _prompt(self, item: dict) -> str:
        # Parent validates phase/frame identity and preserves the execution prompt byte for byte.
        original = super()._prompt(item)
        phase = self.action_phase_by_global_index[int(item["index"])]["phase"]
        if phase == "execution":
            return original
        if phase != "adjustment":
            raise ValueError(f"Unexpected V7.7.2 action phase: {phase!r}")
        return direct_recovery_prompt(str(item["instruction"]), str(item["tactile_caption"]))


class DirectPhaseDataset(V77ManifestDataset):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        result["prompt"] = phase_prompt_without_plan(result["prompt"])
        return result


