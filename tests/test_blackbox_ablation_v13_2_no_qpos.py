"""No GPU/model weights: prove normalized qpos and discrete prompt state are both masked."""
# ruff: noqa: E402
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'openpi/src')]

from scripts import eval_blackbox_ablation_v13_2_no_qpos as script
from openpi import transforms
from openpi.shared.normalize import NormStats
from tactile_vla.vla.openpi_bridge import TokenizeStructuredPrompt, build_structured_inference_transform


class SpyTokenizer:
    def __init__(self):
        self.calls = []

    def tokenize_structured_response(self, prompt, state, target, max_len):
        self.calls.append((prompt, state.copy(), target))
        ids = np.rint(state * 100).astype(np.int32)
        return ids, np.ones_like(ids, dtype=bool), np.zeros_like(ids), np.zeros_like(ids, dtype=bool), 0


def make_transform():
    tokenizer = SpyTokenizer()
    config = SimpleNamespace(action_dim=32, max_token_len=384)
    norm = {'state': NormStats(mean=np.full(7, 2.), std=np.full(7, .5),
                               q01=np.full(7, 1.), q99=np.full(7, 3.))}
    transform = build_structured_inference_transform(config, tokenizer=tokenizer, norm_stats=norm)
    return transform, tokenizer


def raw(state):
    return {'observation/state': np.full(7, state, dtype=np.float32),
            'observation/image': np.full((16, 24, 3), 127, dtype=np.uint8),
            'observation/wrist_image': np.full((16, 24, 3), 93, dtype=np.uint8),
            'prompt': 'Task: transfer. Touch[rotation=clockwise] Failure-recovery memory: none',
            'global_index': 0, 'episode_id': 1, 'attempt_id': 1, 'frame_index': 9}


def test_mask_before_tokenization_but_after_normalization_preserves_other_inputs():
    transform, tokenizer = make_transform()
    baseline = transform(raw(2.5))
    assert np.any(baseline['state'])
    assert np.any(tokenizer.calls[-1][1])
    masked = script.mask_inference_transform(transform)
    source = raw(2.5)
    original = copy.deepcopy(source)
    result = masked(source)
    changed = masked(raw(1.25))
    assert np.all(source['observation/state'] == original['observation/state'])
    assert result['state'].shape == (32,) and not result['state'].any()
    assert not tokenizer.calls[-1][1].any()
    assert tokenizer.calls[-1][0] == original['prompt']
    assert tokenizer.calls[-1][2] is None
    assert np.array_equal(result['tokenized_prompt'], changed['tokenized_prompt'])
    assert not np.array_equal(baseline['tokenized_prompt'], result['tokenized_prompt'])
    for camera in baseline['image']:
        assert np.array_equal(baseline['image'][camera], result['image'][camera])
        assert baseline['image_mask'][camera] == result['image_mask'][camera]
    assert result['global_index'] == baseline['global_index']
    assert not result['token_loss_mask'].any() and 'target_text' not in result
    # Existing transform remains unaffected, including raw-zero != normalized-zero.
    assert np.any(transform(raw(0.))['state'])


def test_audit_checks_real_transform_and_rejects_unmasked_qpos():
    transform, _ = make_transform()
    dataset = SimpleNamespace(raw=lambda i: raw(2.5), transform=script.mask_inference_transform(transform))
    report = script.audit_batch(SimpleNamespace(dataset=dataset))
    assert report['qpos_token_invariant'] and report['state_all_zero']
    dataset.transform = transform
    with pytest.raises(AssertionError, match='survived'):
        script.audit_batch(SimpleNamespace(dataset=dataset))


def test_guard_against_unsafe_or_double_insertion():
    transform, tokenizer = make_transform()
    with pytest.raises(ValueError, match='already'):
        script.mask_inference_transform(script.mask_inference_transform(transform))
    unsafe = transforms.compose([TokenizeStructuredPrompt(tokenizer, 384), transforms.Normalize(None)])
    with pytest.raises(ValueError, match='unsafe'):
        script.mask_inference_transform(unsafe)
    with pytest.raises(ValueError, match='history'):
        script.MaskNormalizedQpos()({'state': np.zeros(7), 'state_history': np.zeros((11, 7))})


def test_output_names_never_collide_with_original_eval(tmp_path):
    paths = script.output_paths(tmp_path, 'test', 'best')
    assert [p.name for p in paths] == ['test_best_qpos_masked.metrics.json', 'test_best_qpos_masked.predictions.json']
    args = script.parse_args(['--run-dir', str(tmp_path), '--dry-run'])
    assert args.dry_run and args.checkpoint == 'best' and not args.overwrite
    with pytest.raises(SystemExit):
        script.parse_args(['--run-dir', str(tmp_path), '--checkpoint', '../best'])
