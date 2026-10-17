"""Two-domain loaders and sampling, shared by joint Stage A and multitask."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from tactile_vla.vla.v12_1_data import DOMAINS, TASKS, load_jsonl


class BalancedGroupsBatchSampler:
    """Equal quota per disjoint pool, with independent no-replacement passes."""

    def __init__(self, pools, *, batch_size=8, seed=42):
        if batch_size <= 0 or not pools or batch_size % len(pools):
            raise ValueError("Batch size must be divisible by the number of balanced pools (4 for action)")
        self.pools = {k: tuple(v) for k, v in sorted(pools.items())}
        self.quota, self.seed, self.epoch = batch_size // len(pools), seed, 0
        seen = set()
        for values in self.pools.values():
            if len(values) < self.quota or len(set(values)) != len(values) or seen.intersection(values):
                raise ValueError("Pools must be distinct, disjoint and large enough")
            seen.update(values)

    def __len__(self):
        return max(len(v) // self.quota for v in self.pools.values())

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1

        def chunks(values, slot):
            rng = random.Random((self.seed + epoch) * len(self.pools) + slot)
            while True:
                shuffled = list(values)
                rng.shuffle(shuffled)
                for i in range(0, len(shuffled) - self.quota + 1, self.quota):
                    yield shuffled[i : i + self.quota]

        streams = [chunks(v, i) for i, v in enumerate(self.pools.values())]
        rng = random.Random(self.seed + epoch)
        for _ in range(len(self)):
            batch = [p for stream in streams for p in next(stream)]
            rng.shuffle(batch)
            yield batch


def check_args(args):
    if args.batch_size <= 0 or args.batch_size % 4:
        raise ValueError("Joint action batches require positive --batch-size divisible by 4")
    if (
        args.action_horizon != 30
        or args.action_dim != 32
        or args.use_state_history
        or args.state_history_len != 0
        or args.history_hidden_dim != 0
    ):
        raise ValueError("V12.1 requires H30, action_dim32 and no dense state history")
    for key in ("num_steps", "eval_interval", "save_interval", "keep_period", "log_interval", "fsdp_devices"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.no_norm:
        raise ValueError("V12.1 requires shared train-only normalization")
    if args.overwrite:
        raise ValueError("V12.1 refuses overwrite; select a new run name")


def build_loaders(args, config, index, *, tokenizer=None, failure_codec=None, plan_codec=None, action_only=False):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.shared import normalize
    from tactile_vla.vla.openpi_bridge import (
        TactileVLAFrameDataset,
        TransformedTactileVLADataset,
        build_transform,
        build_structured_inference_transform,
        build_structured_text_transform,
        collate_numpy,
    )
    from tactile_vla.vla.v7_7_multitask_data import V77ManifestDataset, TransformedV77Dataset
    from tactile_vla.vla.book_v9_4_5_action_replay import phase_positions, phase_eval_positions

    norm = normalize.load(args.norm_stats_dir)
    transforms = {"action": build_transform(config, norm_stats=norm)}
    tasks = ("action",) if action_only else TASKS
    if not action_only:
        transforms.update(
            {
                "adjustment": build_structured_inference_transform(config, tokenizer=tokenizer, norm_stats=norm),
                "need": build_structured_inference_transform(config, tokenizer=tokenizer, norm_stats=norm),
                "failure": build_structured_text_transform(
                    config, tokenizer=tokenizer, grammar=failure_codec, norm_stats=norm
                ),
                "plan": build_structured_text_transform(
                    config,
                    tokenizer=tokenizer,
                    grammar=plan_codec,
                    norm_stats=norm,
                    max_len=args.reasoning_max_token_len,
                ),
            }
        )
    manifests = {t: load_jsonl(Path(index[f"{t}_manifest_file"])) for t in tasks if t != "action"}
    output = {s: {} for s in ("train", "val")}
    parts = {s: {t: [] for t in tasks} for s in output}
    pools = {t: {} for t in tasks}
    audit = {}
    workers = {"multiprocessing_context": "spawn", "persistent_workers": True} if args.num_workers else {}

    def loader(dataset, *, sampler=None):
        options = (
            {"batch_sampler": sampler} if sampler is not None else {"batch_size": args.batch_size, "shuffle": False}
        )
        return DataLoader(dataset, num_workers=args.num_workers, collate_fn=collate_numpy, **options, **workers)

    for domain in DOMAINS:
        source = index["domains"][domain]
        dataset_dir = Path(source["dataset_dir"])
        lookup = {int(r["global_index"]): r for r in load_jsonl(Path(source["phase_manifest_file"]))}
        shared = LeRobotDataset(
            dataset_dir.name,
            root=dataset_dir,
            delta_timestamps={"action": [i / 30.0 for i in range(30)]},
            download_videos=False,
            video_backend=args.video_backend,
        )
        plain = (
            None
            if action_only
            else LeRobotDataset(
                dataset_dir.name, root=dataset_dir, download_videos=False, video_backend=args.video_backend
            )
        )
        for split in output:
            for task in tasks:
                stream = index["splits"][split][domain][task]
                if task == "action":
                    raw = TactileVLAFrameDataset(
                        dataset_dir=dataset_dir,
                        indices=stream["indices"],
                        stage="execution",
                        action_horizon=30,
                        state_history_len=0,
                        prompt_profile="phase_v2",
                        action_phase_by_global_index=lookup,
                        dataset_repo_id=dataset_dir.name,
                        lerobot_dataset=shared,
                        video_backend=args.video_backend,
                    )
                    transformed = TransformedTactileVLADataset(raw, transforms[task])
                else:
                    raw = V77ManifestDataset(
                        rows=manifests[task],
                        row_indices=stream["manifest_row_indices"],
                        global_indices=stream["global_indices"],
                        task=task,
                        lerobot_dataset=plain,
                    )
                    transformed = TransformedV77Dataset(raw, transforms[task])
                offset = sum(len(v) for v in parts[split][task])
                parts[split][task].append(transformed)
                if split == "train":
                    if task == "action":
                        for phase, values in phase_positions(stream["indices"], lookup).items():
                            pools[task][domain + "/" + phase] = [offset + p for p in values]
                    else:
                        pools[task][domain] = list(range(offset, offset + len(transformed)))
                else:
                    output["val"][task + "_" + domain] = loader(transformed)
                    if task == "action":
                        selected = phase_eval_positions(
                            stream["indices"],
                            lookup,
                            samples_per_phase=args.action_eval_samples_per_phase,
                            seed=args.seed,
                        )
                        for phase, positions in selected.items():
                            name = domain + "/" + phase
                            output["val"]["action_" + name] = loader(Subset(transformed, positions))
                            audit[name] = {
                                "count": len(positions),
                                "global_indices": [stream["indices"][p] for p in positions],
                            }
    for task in tasks:
        concat = ConcatDataset(parts["train"][task])
        output["train"][task] = loader(
            concat, sampler=BalancedGroupsBatchSampler(pools[task], batch_size=args.batch_size, seed=args.seed)
        )
        output["val"][task] = loader(ConcatDataset(parts["val"][task]))
        output["val"][task].domain_loaders = {d: output["val"][task + "_" + d] for d in DOMAINS}
    output["action_eval_selection"] = audit
    if not action_only:
        output["token_validation"] = validate_token_lengths(args, index, manifests, tokenizer, norm)
    return output


def validate_token_lengths(args, index, manifests, tokenizer, norm):
    """Validate every selected prompt/target, not just the first dry-run batch."""
    from openpi.transforms import Normalize
    from tactile_vla.vla.v5_3_adjustment_end_data import scan_selected_qpos

    normalizer = Normalize(norm, use_quantiles=True)
    summary = {}
    for domain in DOMAINS:
        source = index["domains"][domain]
        episodes = set().union(*(set(v) for v in source["episode_splits"].values()))
        states = scan_selected_qpos(dataset_dir=Path(source["dataset_dir"]), selected_episode_ids=episodes)
        for task, rows in manifests.items():
            maximum, count = 0, 0
            limit = args.reasoning_max_token_len if task == "plan" else args.max_token_len
            for row in rows:
                if row["domain"] != domain:
                    continue
                state = normalizer({"state": np.asarray(states[int(row["global_index"])], dtype=np.float32)})[
                    "state"
                ].astype(np.float32)
                target = (
                    row.get("target_failure_reason")
                    if task == "failure"
                    else row.get("target_recovery_plan")
                    if task == "plan"
                    else None
                )
                target_tokens = tokenizer.encode_text(target, add_eos=True) if target else None
                _, mask, _, _, _ = tokenizer.tokenize_structured_response(
                    row["prompt"], state, target_tokens, max_len=limit
                )
                maximum = max(maximum, int(mask.sum()))
                count += 1
            summary[domain + "/" + task] = {"count": count, "max_tokens": maximum, "limit": limit}
    return summary


def evaluate_action_report(state, loaders, data_sharding, *, seed):
    """Works on Pi0 Stage A as well as V77MultitaskModel; evaluate all four groups."""
    from flax import nnx
    import jax
    from openpi.shared import nnx_utils
    from scripts import train_vla_stage_b_v3 as base

    model = nnx.merge(state.model_def, state.params)
    model.eval()
    compute = nnx_utils.module_jit(getattr(model, "backbone", model).compute_loss)
    losses, support = {}, {}
    for domain in DOMAINS:
        for phase in ("adjustment", "execution"):
            name = domain + "/" + phase
            values, counts = [], []
            rng = jax.random.key(seed)
            for batch in loaders["action_" + name]:
                batch, count = base.pad_eval_batch(batch, multiple=len(data_sharding.device_set))
                obs, actions = base.batch_to_jax(batch, "action", data_sharding)
                rng, key = jax.random.split(rng)
                loss = compute(key, obs, actions)[:count].mean()
                values.append(float(jax.device_get(loss)))
                counts.append(count)
            if not values:
                raise ValueError("Empty action evaluation group")
            losses[name] = float(np.average(values, weights=counts))
            support[name] = sum(counts)
    return {"loss": float(np.mean(list(losses.values()))), "by_phase": losses, "support_by_phase": support}


def print_dry_run(loaders):
    """Inspect both domains, rather than mistaking a concatenation index for a frame ID."""
    for task, loader in loaders.items():
        batch = next(iter(loader))
        concat = loader.dataset
        print(
            json.dumps(
                {
                    "task": task,
                    "batch_shapes": {
                        k: list(np.shape(v))
                        for k, v in batch.items()
                        if k in ("actions", "state", "tokenized_prompt", "adjustment_end_label", "need_recovery_label")
                    },
                }
            )
        )
        for domain, dataset in zip(DOMAINS, concat.datasets, strict=True):
            raw = dataset.dataset[0]
            print(
                json.dumps(
                    {
                        "domain": domain,
                        "task": task,
                        "prompt": raw["prompt"],
                        "global_index": int(raw["global_index"]),
                        "target": raw.get(
                            "target_text", int(raw.get("need_recovery_label", raw.get("adjustment_end_label", -1)))
                        ),
                    },
                    ensure_ascii=False,
                )
            )
