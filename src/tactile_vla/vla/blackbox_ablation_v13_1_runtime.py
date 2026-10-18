"""HDF5 inputs, deterministic batches and plan-only LoRA model for V13.1."""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from tactile_vla.vla.blackbox_ablation_v13_1_data import GROUPS, metrics, prediction_record


@lru_cache(maxsize=4)
def lerobot_dataset(root, video_backend):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root)
    return LeRobotDataset(root.name, root=root, download_videos=False, video_backend=video_backend)


class PlanDataset:
    def __init__(self, rows, *, group, transform, inference=False, video_backend='pyav'):
        if group not in GROUPS:
            raise ValueError(group)
        self.rows, self.group, self.transform, self.inference = rows, group, transform, inference
        self.video_backend = video_backend

    def __len__(self):
        return len(self.rows)

    def raw(self, index):
        row = self.rows[index]
        if row.get('lerobot_dataset_dir'):
            item = lerobot_dataset(row['lerobot_dataset_dir'], self.video_backend)[row['lerobot_global_index']]
            actual = tuple(int(item[k]) for k in ('index', 'episode_index', 'episode_id', 'attempt_id', 'frame_index'))
            expected = (row['lerobot_global_index'], row['lerobot_episode_index'], row['episode_id'], 1, row['frame_index'])
            if actual != expected or item['tactile_caption'] != row['current_touch']:
                raise ValueError('LeRobot image/label identity mismatch')
            state = np.asarray(item['observation.state'], dtype=np.float32)
            if not np.array_equal(state, np.asarray(row['qpos'], dtype=np.float32)):
                raise ValueError('LeRobot current qpos mismatch')
            result = {'observation/image': item['observation.images.front'],
                      'observation/wrist_image': item['observation.images.left'],
                      'observation/state': state, 'prompt': row['prompts'][self.group],
                      'global_index': np.int64(row['sample_id']), 'episode_id': np.int64(row['episode_id']),
                      'attempt_id': np.int64(1), 'frame_index': np.int64(row['frame_index'])}
            if not self.inference:
                result['target_text'] = row['target_recovery_plan']
            return result
        frame = row['frame_index']
        with h5py.File(row['hdf5_path']) as h:
            state = np.asarray(h['arm/jointStatePosition/puppetRight'][frame], dtype=np.float32)
            if not np.array_equal(state, np.asarray(row['qpos'], dtype=np.float32)):
                raise ValueError('HDF5 state changed after index construction')
            def decode(camera):
                path = row.get('image_sources', {}).get(camera)
                if path is not None:
                    with Image.open(path) as image:
                        return np.asarray(image.convert('RGB')).copy()
                value = h[f'camera/color/{camera}'][frame]
                data = value if isinstance(value, bytes) else np.asarray(value, dtype=np.uint8).tobytes()
                with Image.open(io.BytesIO(data)) as image:
                    return np.asarray(image.convert('RGB')).copy()
            result = {'observation/image': decode('front'), 'observation/wrist_image': decode('left'),
                      'observation/state': state, 'prompt': row['prompts'][self.group],
                      'global_index': np.int64(row['sample_id']), 'episode_id': np.int64(row['episode_id']),
                      'attempt_id': np.int64(1), 'frame_index': np.int64(frame)}
        # Inference never receives target_text; no teacher-forced answer can leak.
        if not self.inference:
            result['target_text'] = row['target_recovery_plan']
        return result

    def __getitem__(self, index):
        return self.transform(self.raw(index))


class StepBatchSampler:
    """Order is a pure function of seed and optimizer step, shared by all groups."""
    def __init__(self, length, batch_size, seed, num_steps, start_step=0):
        if batch_size <= 0 or length < batch_size or not 0 <= start_step <= num_steps:
            raise ValueError('Invalid deterministic batch sampler')
        self.length, self.batch_size, self.seed = length, batch_size, seed
        self.num_steps, self.start_step = num_steps, start_step

    def __len__(self):
        return self.num_steps - self.start_step

    def __iter__(self):
        batches = self.length // self.batch_size
        epoch, permutation = None, None
        for step in range(self.start_step, self.num_steps):
            current_epoch, position = divmod(step, batches)
            if current_epoch != epoch:
                epoch = current_epoch
                permutation = np.random.default_rng(self.seed + epoch).permutation(self.length)
            start = position * self.batch_size
            yield permutation[start:start + self.batch_size].tolist()


def make_loader(rows, *, group, model_config, tokenizer, grammar, norm, batch_size,
                num_workers=0, sampler=None, inference=False):
    from tactile_vla.vla.openpi_bridge import (
        build_structured_text_transform, build_structured_inference_transform, collate_numpy,
    )
    common = dict(tokenizer=tokenizer, norm_stats=norm, max_len=model_config.max_token_len)
    transform = (build_structured_inference_transform(model_config, **common) if inference else
                 build_structured_text_transform(model_config, grammar=grammar, **common))
    dataset = PlanDataset(rows, group=group, transform=transform, inference=inference)
    options = {'batch_sampler': sampler} if sampler is not None else {'batch_size': batch_size, 'shuffle': False}
    workers = {'multiprocessing_context': 'spawn', 'persistent_workers': True} if num_workers else {}
    return DataLoader(dataset, num_workers=num_workers, collate_fn=collate_numpy, **options, **workers)


def validate_tokens(rows, *, tokenizer, grammar, norm, max_len):
    from openpi.transforms import Normalize
    normalizer = Normalize(norm, use_quantiles=True)
    report = {g: {'count': 0, 'max_tokens': 0, 'limit': max_len} for g in GROUPS}
    for row in rows:
        state = normalizer({'state': np.asarray(row['qpos'], dtype=np.float32)})['state'].astype(np.float32)
        target = grammar.sequence_for_text(row['target_recovery_plan'])
        for group in GROUPS:
            _, mask, _, _, _ = tokenizer.tokenize_structured_response(
                row['prompts'][group], state, target, max_len=max_len)
            report[group]['count'] += 1
            report[group]['max_tokens'] = max(report[group]['max_tokens'], int(mask.sum()))
    return report


def evaluate(state, loader, grammar, *, record_fn=prediction_record, metrics_fn=metrics):
    import jax
    from flax import nnx
    from openpi.models.model import Observation
    from openpi.shared import nnx_utils
    from tactile_vla.vla.structured_generation import constrained_greedy_generate
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    prefill = nnx_utils.module_jit(model.generation_prefill)
    step = nnx_utils.module_jit(model.generation_step)
    records = []
    lookup = {r['sample_id']: r for r in loader.dataset.rows}
    for batch in loader:
        observation = Observation.from_dict(batch)
        if np.any(np.asarray(observation.token_loss_mask)):
            raise ValueError('Evaluation received teacher-forced target tokens')
        for i, identity in enumerate(batch['global_index']):
            single = jax.tree.map(lambda x: None if x is None else x[i:i + 1], observation)
            text = constrained_greedy_generate(
                None, single, grammar,
                prefill_fn=lambda _, obs, ids: prefill(obs, ids),
                step_fn=lambda _, *args: step(*args),
            )
            records.append(record_fn(lookup[int(identity)], text))
    return metrics_fn(records), records


def model_class():
    from flax import nnx
    from tactile_vla.vla import stage_b_v3_jax as language

    class PlanOnlyModel(nnx.Module):
        def __init__(self, backbone):
            self.backbone = backbone

        def structured_token_logits(self, observation, compact_token_ids, *, rng=None, train=False):
            return language.structured_token_logits(self.backbone, observation, compact_token_ids,
                                                   preprocess_rng=rng, train=train)

        def generation_prefill(self, observation, compact_token_ids):
            return language.generation_prefill(self.backbone, observation, compact_token_ids)

        def generation_step(self, token, compact_token_ids, kv_cache, prefix_mask, semantic_position):
            return language.generation_step(self.backbone, token, compact_token_ids, kv_cache,
                                            prefix_mask, semantic_position)

    return PlanOnlyModel


def trainable_filter():
    from flax import nnx
    from openpi.shared import nnx_utils
    return nnx.All(nnx.Param, nnx_utils.PathRegex('.*backbone.*llm.*lora.*'),
                   nnx.Not(nnx_utils.PathRegex('.*_1.*')))


def init_state(args, config, mesh):
    """Reuse mature weight/optimizer helpers, without creating any classifier heads."""
    import jax
    from flax import nnx
    from openpi.shared import array_typing as at
    from openpi.training import sharding, utils, weight_loaders
    from scripts import train_vla_stage_b_v3 as base
    from tactile_vla.vla.stage_b_v3_checkpoint import cast_frozen_params

    cls, filter_ = model_class(), trainable_filter()
    tx = base.make_optimizer(args)
    def init(rng, backbone_params=None):
        backbone = config.create(rng)
        if backbone_params is not None:
            graph, values = nnx.split(backbone)
            values.replace_by_pure_dict(backbone_params)
            backbone = nnx.merge(graph, values)
        model = cls(backbone)
        params = cast_frozen_params(nnx.state(model), nnx.All(nnx.Param, nnx.Not(filter_)))
        return utils.TrainState(step=0, params=params, model_def=nnx.graphdef(model),
                               opt_state=tx.init(params.filter(filter_)), tx=tx,
                               ema_decay=None, ema_params=None)
    rng = jax.random.key(args.seed)
    shape = jax.eval_shape(init, rng)
    partition = sharding.fsdp_sharding(shape, mesh, log=True)
    loader = weight_loaders.CheckpointWeightLoader(str(Path(args.stage_a_checkpoint) / 'params'))
    params = base.load_weights_and_validate(loader, shape.params['backbone'].to_pure_dict())
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with at.disable_typechecking():
        state = jax.jit(init, donate_argnums=(1,), in_shardings=(replicated, replicated),
                        out_shardings=partition)(rng, params)
    return state, partition, filter_


def restore_overlay(params, path, *, include_rng=False):
    """Restore onto the CURRENT device topology, not training's saved device IDs."""
    import jax
    import orbax.checkpoint as ocp
    from flax import nnx
    from tactile_vla.vla.stage_b_v3_checkpoint import merge_delta_params
    filter_ = nnx.Any(trainable_filter(), nnx.RngState) if include_rng else trainable_filter()
    reference = {'params': params.filter(filter_)}
    restore_args = jax.tree.map(lambda a: ocp.ArrayRestoreArgs(restore_type=jax.Array, sharding=a.sharding), reference)
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(Path(path), item=reference, restore_args=restore_args)
    return merge_delta_params(params, restored['params'])
