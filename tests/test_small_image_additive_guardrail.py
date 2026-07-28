import torch

from experiments.small_image_additive_guardrail import (
    additive_guardrail_weights,
    time_scale_control_weights,
)
from experiments.small_image_basis_transport import OrthogonalDirectionLoss


def _analyzer() -> OrthogonalDirectionLoss:
    dimension = 16
    groups = torch.arange(dimension) // 4
    return OrthogonalDirectionLoss(
        torch.eye(dimension),
        torch.tensor([4.0, 1.0, 0.4, 0.1]).repeat_interleave(4),
        groups,
        gamma=0.5,
        min_weight=0.2,
        max_weight=2.0,
    )


def test_additive_guardrail_preserves_detail_weights_exactly():
    analyzer = _analyzer()
    time = torch.tensor([0.1, 0.5, 0.9])
    original = analyzer.weights(time)
    guardrail = additive_guardrail_weights(analyzer, time)
    coarse = analyzer.group_index.eq(0)

    assert torch.equal(guardrail[:, coarse], torch.ones_like(guardrail[:, coarse]))
    assert torch.equal(guardrail[:, ~coarse], original[:, ~coarse])


def test_time_scale_control_matches_total_weight_but_not_direction():
    analyzer = _analyzer()
    time = torch.tensor([0.5, 0.9])
    original = analyzer.weights(time)
    guardrail = additive_guardrail_weights(analyzer, time)
    control = time_scale_control_weights(analyzer, time)

    assert torch.allclose(control.sum(dim=1), guardrail.sum(dim=1), atol=1e-6)
    assert torch.allclose(
        control[:, 1:] / control[:, :-1],
        original[:, 1:] / original[:, :-1],
        atol=1e-6,
    )
    assert not torch.allclose(control, guardrail)
