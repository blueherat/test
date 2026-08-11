from __future__ import annotations

import torch

from experiments.run_prediction_target_bayes_oracle_v5 import build_model
from experiments.train_prediction_target_internal_guidance import (
    InternalResidualDenoiseMLP,
)


def test_internal_model_output_shapes() -> None:
    model = InternalResidualDenoiseMLP(
        D=8, hidden=16, depth=5, time_dim=8, intermediate_after=1
    )
    state = torch.randn(7, 8)
    time = torch.rand(7)
    intermediate, final = model(state, time)
    assert intermediate.shape == state.shape
    assert final.shape == state.shape


def test_final_branch_initialization_matches_original_residual_model() -> None:
    seed = 1234
    torch.manual_seed(seed)
    baseline = build_model(
        "residual", D=8, hidden=16, depth=5, time_dim=8
    )
    torch.manual_seed(seed)
    internal = InternalResidualDenoiseMLP(
        D=8, hidden=16, depth=5, time_dim=8, intermediate_after=1
    )
    internal_state = internal.state_dict()
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, internal_state[name]), name
