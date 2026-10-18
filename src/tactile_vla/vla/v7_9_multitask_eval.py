"""Keep aggregate V7.7.2 adjustment metrics; additionally report factual/CF subsets."""
from tactile_vla.common.metrics import classification_report


def report_by_variant(true, pred, variants):
    if not len(true) == len(pred) == len(variants):
        raise ValueError('Adjustment prediction/variant lengths differ')
    report = classification_report(true, pred, num_classes=2, class_names=['false', 'true'])
    report['by_sample_variant'] = {}
    for variant in sorted(set(variants)):
        selected = [i for i, value in enumerate(variants) if value == variant]
        report['by_sample_variant'][variant] = classification_report(
            [true[i] for i in selected], [pred[i] for i in selected],
            num_classes=2, class_names=['false', 'true'])
    return report


def evaluate_adjustment(state, loader, data_sharding, *, max_samples):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from openpi.shared import nnx_utils
    from scripts import train_vla_stage_b_v3 as base

    model = nnx.merge(state.model_def, state.params)
    model.eval()
    logits_fn = nnx_utils.module_jit(model.adjustment_end_logits)
    manifest = loader.dataset.dataset.rows
    true, pred, variants = [], [], []
    for batch in loader:
        batch, valid_count = base.pad_eval_batch(batch, multiple=len(data_sharding.device_set))
        observation, targets = base.batch_to_jax(batch, 'adjustment', data_sharding)
        logits = logits_fn(observation)
        usable = min(valid_count, max_samples - len(true))
        true.extend(np.asarray(jax.device_get(targets['adjustment'])).reshape(-1)[:usable].tolist())
        pred.extend(np.asarray(jax.device_get(jnp.argmax(logits, axis=-1))).reshape(-1)[:usable].tolist())
        variants.extend(manifest[int(i)]['sample_variant']
                        for i in np.asarray(batch['manifest_row_index']).reshape(-1)[:usable])
        if len(true) >= max_samples:
            break
    return report_by_variant(true, pred, variants)
