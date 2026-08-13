import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_tokenize_structured_response_has_causal_answer_masks():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    target = tokenizer.encode_text("failure_reason=rotate left,grasp appropriate.", add_eos=True)
    tokens, valid, ar_mask, loss_mask, prefix_length = tokenizer.tokenize_structured_response(
        "Mode: failure diagnosis. Task: test.",
        np.zeros((4,), dtype=np.float32),
        target,
    )

    assert tokens.shape == valid.shape == ar_mask.shape == loss_mask.shape == (64,)
    assert ar_mask.dtype == np.int32
    assert not ar_mask[:prefix_length].any()
    assert ar_mask[prefix_length : prefix_length + len(target)].all()
    assert loss_mask.sum() == len(target)


def test_tokenize_structured_response_rejects_truncation():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=4)
    with pytest.raises(ValueError, match="exceeds max token length"):
        tokenizer.tokenize_structured_response(
            "This prompt cannot fit.",
            np.zeros((4,), dtype=np.float32),
            [1, 2],
        )


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
