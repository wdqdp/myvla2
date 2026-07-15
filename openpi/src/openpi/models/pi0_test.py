import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_pi05_state_history_params_are_trainable():
    config = _pi0_config.Pi0Config(
        pi05=True,
        use_state_history=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    all_params = nnx.state(abstract_model, nnx.Param).flat_state()
    frozen_params = _get_frozen_state(config)

    assert any(any(str(part).startswith("history_") for part in path) for path in all_params)
    assert not any(any(str(part).startswith("history_") for part in path) for path in frozen_params)


def test_pi05_state_history_graphdef_is_stable():
    config = _pi0_config.Pi0Config(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=30,
        max_token_len=32,
        pi05=True,
        use_state_history=True,
        state_history_len=60,
        state_history_dim=7,
        history_hidden_dim=16,
        pytorch_compile_mode=None,
    )
    graphdef_a = nnx.graphdef(nnx.eval_shape(config.create, jax.random.key(0)))
    graphdef_b = nnx.graphdef(nnx.eval_shape(config.create, jax.random.key(1)))

    assert jax.tree_util.tree_structure(graphdef_a) == jax.tree_util.tree_structure(graphdef_b)
