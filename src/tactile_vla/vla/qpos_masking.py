"""Hold normalized qpos AND its text representation constant, before tokenization."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskNormalizedQpos:
    def __call__(self, data):
        if 'state_history' in data:
            raise ValueError('Qpos masking expects no qpos history')
        state = np.asarray(data['state'])
        if state.shape != (7,):
            raise ValueError(f'Expected 7 normalized qpos values before padding, got {state.shape}')
        return {**data, 'state': np.zeros_like(state, dtype=np.float32)}


def mask_structured_transform(transform, *, inference):
    from openpi import transforms
    from tactile_vla.vla.openpi_bridge import TokenizeStructuredPrompt, TokenizeStructuredResponse

    steps = list(transform.transforms)
    tokenizer_type = TokenizeStructuredPrompt if inference else TokenizeStructuredResponse
    tokenizers = [i for i, step in enumerate(steps) if isinstance(step, tokenizer_type)]
    normalizers = [i for i, step in enumerate(steps) if isinstance(step, transforms.Normalize)]
    if len(tokenizers) != 1 or len(normalizers) != 1 or normalizers[0] >= tokenizers[0]:
        raise ValueError('Expected one matching tokenizer after one normalizer; refusing unsafe masking')
    if any(isinstance(step, MaskNormalizedQpos) for step in steps):
        raise ValueError('Qpos masking has already been inserted')
    steps.insert(tokenizers[0], MaskNormalizedQpos())
    return transforms.compose(steps)
