import pytest
import torch

from experiments.advfd_cleanroom.generators import (
    checkpoint_state_dict,
    pmf_state_dict_for_advfd,
    pmf_state_dict_from_advfd,
)


def test_checkpoint_state_dict_accepts_plain_and_unique_online_payload() -> None:
    state = {"weight": torch.ones(2)}
    assert checkpoint_state_dict(state) is state
    assert checkpoint_state_dict({"model": state}) is state


def test_checkpoint_state_dict_refuses_ambiguous_or_ema_only_payload() -> None:
    state = {"weight": torch.ones(2)}
    with pytest.raises(ValueError, match="multiple"):
        checkpoint_state_dict({"model": state, "state_dict": state})
    with pytest.raises(ValueError, match="EMA"):
        checkpoint_state_dict({"ema": state})


def test_pmf_state_dict_for_advfd_converts_only_layout_differences() -> None:
    state = {
        "net.time_tokens": torch.arange(6).reshape(1, 2, 3),
        "net.block.proj._flax_linear.weight": torch.ones(3, 4),
        "net.labels._flax_embedding.weight": torch.full((5, 3), 2.0),
        "net.rope_freqs": torch.ones(4, dtype=torch.complex64),
        "net.norm.weight": torch.full((3,), 4.0),
    }

    converted = pmf_state_dict_for_advfd(state)

    assert set(converted) == {
        "net.time_tokens",
        "net.block.proj.linear.weight",
        "net.labels.embedding.weight",
        "net.norm.weight",
    }
    assert torch.equal(converted["net.time_tokens"], state["net.time_tokens"][0])
    assert converted["net.block.proj.linear.weight"] is state[
        "net.block.proj._flax_linear.weight"
    ]
    assert converted["net.labels.embedding.weight"] is state[
        "net.labels._flax_embedding.weight"
    ]


def test_pmf_state_dict_for_advfd_refuses_key_collisions() -> None:
    state = {
        "net.proj._flax_linear.weight": torch.ones(2, 2),
        "net.proj.linear.weight": torch.zeros(2, 2),
    }
    with pytest.raises(ValueError, match="duplicate"):
        pmf_state_dict_for_advfd(state)


def test_pmf_advfd_layout_conversion_is_reversible() -> None:
    upstream = {
        "net.time_tokens": torch.randn(1, 4, 8),
        "net.block.proj._flax_linear.weight": torch.randn(8, 8),
        "net.labels._flax_embedding.weight": torch.randn(3, 8),
        "net.rope_freqs": torch.randn(2, 2),
    }
    public = pmf_state_dict_for_advfd(upstream)
    assert "net.rope_freqs" not in public
    assert public["net.time_tokens"].shape == (4, 8)
    restored = pmf_state_dict_from_advfd(public)
    for key in upstream.keys() - {"net.rope_freqs"}:
        torch.testing.assert_close(restored[key], upstream[key])
