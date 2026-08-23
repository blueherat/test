import pytest
import torch

from experiments.advfd_cleanroom.generators import checkpoint_state_dict


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
