import copy

import torch

from experiments.small_image_asymmetric_guardrail import (
    AsymmetricVelocityField,
    asymmetric_losses,
    asymmetric_parameter_counts,
)
from experiments.small_image_basis_transport import OrthogonalDirectionLoss


def _analyzer() -> OrthogonalDirectionLoss:
    dimension = 16
    return OrthogonalDirectionLoss(
        torch.eye(dimension),
        torch.tensor([4.0, 1.0, 0.4, 0.1]).repeat_interleave(4),
        torch.arange(dimension) // 4,
        gamma=0.5,
        min_weight=0.2,
        max_weight=2.0,
    )


def test_asymmetric_and_wide_controls_are_parameter_matched():
    counts = asymmetric_parameter_counts(24, 12, 27, 2)
    assert counts == {"asymmetric": 287_966, "wide": 290_629}
    assert abs(counts["asymmetric"] / counts["wide"] - 1.0) < 0.01


def test_asymmetric_weighting_keeps_coarse_gradient_identical():
    analyzer = _analyzer()
    baseline = AsymmetricVelocityField(
        analyzer, detail_width=5, coarse_width=3, depth=1
    )
    weighted = copy.deepcopy(baseline)
    state = torch.randn((4, 1, 4, 4), generator=torch.Generator().manual_seed(2))
    target = torch.randn((4, 1, 4, 4), generator=torch.Generator().manual_seed(3))
    time = torch.tensor([0.3, 0.5, 0.7, 0.9])
    baseline_coarse, _, _ = asymmetric_losses(
        baseline, state, target, time, analyzer, weighted=False
    )
    weighted_coarse, _, _ = asymmetric_losses(
        weighted, state, target, time, analyzer, weighted=True
    )
    baseline_coarse.backward()
    weighted_coarse.backward()

    assert torch.equal(baseline_coarse, weighted_coarse)
    for first, second in zip(
        baseline.coarse.parameters(), weighted.coarse.parameters()
    ):
        assert torch.equal(first.grad, second.grad)
