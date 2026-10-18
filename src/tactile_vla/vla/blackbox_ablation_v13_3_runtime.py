"""V13.2 observations/labels with qpos masked identically in train/val/test."""
import numpy as np

from tactile_vla.vla.blackbox_ablation_v13_1_data import GROUPS
from tactile_vla.vla.blackbox_ablation_v13_1_runtime import make_loader as original_loader
from tactile_vla.vla.blackbox_ablation_v13_2_runtime import evaluate  # noqa: F401
from tactile_vla.vla.qpos_masking import mask_structured_transform

SCHEMA = 'blackbox_plan_ablation_v13_3'
QPOS_PROTOCOL = 'normalized_zero_before_tokenization_train_val_test_v1'


def audit_inputs(loader, *, inference):
    row = loader.dataset.raw(0)
    first = loader.dataset.transform(dict(row))
    changed = {**row, 'observation/state': np.asarray(row['observation/state'], dtype=np.float32) + .123}
    second = loader.dataset.transform(changed)
    for key in ('state', 'tokenized_prompt', 'tokenized_prompt_mask', 'token_ar_mask', 'token_loss_mask'):
        if not np.array_equal(first[key], second[key]):
            raise AssertionError(f'Qpos survived masking in {key}')
    if np.any(first['state']) or bool(np.any(first['token_loss_mask'])) == inference:
        raise AssertionError('Unexpected state or answer mask in V13.3')
    if inference and 'target_text' in first:
        raise AssertionError('Teacher-forced answer in inference')
    return {'state_all_zero': True, 'qpos_token_invariant': True, 'inference': inference,
            'state_shape': list(first['state'].shape)}


def make_loader(rows, **kwargs):
    inference = kwargs.get('inference', False)
    loader = original_loader(rows, **kwargs)
    loader.dataset.transform = mask_structured_transform(loader.dataset.transform, inference=inference)
    # Runs on real data before training/evaluation, with no extra model computation.
    loader.qpos_audit = audit_inputs(loader, inference=inference)
    return loader


def validate_tokens(rows, *, tokenizer, grammar, norm, max_len):
    del norm  # The tokenizer sees normalized zeros, never raw/normalized sample-specific qpos.
    state = np.zeros(7, dtype=np.float32)
    report = {g: {'count': 0, 'max_tokens': 0, 'limit': max_len, 'qpos_input_protocol': QPOS_PROTOCOL} for g in GROUPS}
    for row in rows:
        target = grammar.sequence_for_text(row['target_recovery_plan'])
        for group in GROUPS:
            _, mask, _, _, _ = tokenizer.tokenize_structured_response(
                row['prompts'][group], state, target, max_len=max_len)
            report[group]['count'] += 1
            report[group]['max_tokens'] = max(report[group]['max_tokens'], int(mask.sum()))
    return report
