"""Deterministic phase-balanced action replay and trajectory-stratified evaluation."""

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

PHASES = ("adjustment", "execution")
ACTION_SAMPLING_POLICY = {
    "schema_version": "book_v9_4_5_action_sampling_v1",
    "phases": list(PHASES),
    "ratio": "1:1_per_batch",
    "batch_size": 8,
    "samples_per_phase_per_batch": 4,
    "seed": 42,
    "pool_selection": "shuffle_without_replacement_per_phase_pass_drop_incomplete_half_batch",
    "shorter_pool": "reshuffle_and_repeat_independently",
    "action_candidates": "unchanged_stage_a_phase_pure_H30",
}
ACTION_EVAL_POLICY = {
    "schema_version": "book_v9_4_5_action_eval_v1",
    "split": "val",
    "phases": list(PHASES),
    "selection": "balanced_episode_attempt_quotas_uniform_across_each_trajectory",
    "default_samples_per_phase": 256,
    "baseline": "same_Stage_A_initialization_same_selected_frames_same_rng",
    "overall_loss": "mean_of_two_phase_losses",
    "gate": "overall_and_each_phase_degradation_le_existing_limit",
}


def phase_positions(indices, phase_lookup):
    pools = {phase: [] for phase in PHASES}
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate action candidate indices")
    for position, global_index in enumerate(indices):
        row = phase_lookup[global_index]
        phase = row["phase"]
        if phase not in pools or not row.get("trainable") or not row.get("chunk_phase_pure"):
            raise ValueError("V9.4.5 action replay requires eligible phase-pure Stage A chunks")
        pools[phase].append(position)
    if any(not values for values in pools.values()):
        raise ValueError("V9.4.5 action stream must contain both phases")
    return pools


class PhaseBalancedActionBatchSampler:
    """Every batch has equal phase support; no duplicate position inside a batch."""

    def __init__(self, pools, *, batch_size=8, seed=42):
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("Phase-balanced batch_size must be positive and even")
        self.half = batch_size // 2
        self.pools = {phase: tuple(pools[phase]) for phase in PHASES}
        if any(len(p) < self.half or len(set(p)) != len(p) for p in self.pools.values()):
            raise ValueError("Each phase pool must contain enough distinct positions")
        if set(self.pools[PHASES[0]]) & set(self.pools[PHASES[1]]):
            raise ValueError("Action phase pools overlap")
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return max(len(pool) // self.half for pool in self.pools.values())

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        rng = random.Random(self.seed + epoch)

        def chunks(phase):
            # Give each phase its own RNG; the larger pool never changes the smaller pool's draws.
            local = random.Random((self.seed + epoch) * 2 + PHASES.index(phase))
            while True:
                positions = list(self.pools[phase])
                local.shuffle(positions)
                for start in range(0, len(positions) - self.half + 1, self.half):
                    yield positions[start : start + self.half]

        streams = {phase: chunks(phase) for phase in PHASES}
        for _ in range(len(self)):
            batch = [position for phase in PHASES for position in next(streams[phase])]
            rng.shuffle(batch)
            yield batch


def phase_eval_positions(indices, phase_lookup, *, samples_per_phase=256, seed=42):
    """Cover every episode/attempt in each phase, rather than a chronological prefix."""
    if samples_per_phase <= 0:
        raise ValueError("Action eval samples per phase must be positive")
    pools = phase_positions(indices, phase_lookup)
    selected = {}
    for phase, positions in pools.items():
        groups = defaultdict(list)
        for position in positions:
            row = phase_lookup[indices[position]]
            groups[row["episode_id"], row["attempt_id"]].append(position)
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        count = min(samples_per_phase, len(positions))
        if count < len(keys):
            raise ValueError("Action evaluation cap must cover all episode/attempt groups")
        quotas = dict.fromkeys(keys, 0)
        remaining = count
        while remaining:
            for key in keys:
                if quotas[key] < len(groups[key]):
                    quotas[key] += 1
                    remaining -= 1
                    if not remaining:
                        break
        chosen = []
        for key in keys:
            ordered = sorted(groups[key], key=lambda p: phase_lookup[indices[p]]["frame_index"])
            offsets = np.linspace(0, len(ordered) - 1, quotas[key], dtype=int)
            chosen.extend(ordered[offset] for offset in offsets)
        selected[phase] = chosen
    return selected
