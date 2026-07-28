import copy

import torch

from experiments.small_image_basis_transport import OrthogonalDirectionLoss
from experiments.small_image_split_guardrail import branchwise_losses
from experiments.small_image_training_rescue import SplitVelocityField


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


def test_split_baseline_and_guardrail_have_identical_coarse_gradients():
    analyzer = _analyzer()
    baseline = SplitVelocityField(analyzer, width=4, depth=1)
    guardrail = copy.deepcopy(baseline)
    state = torch.randn((5, 1, 4, 4), generator=torch.Generator().manual_seed(3))
    target = torch.randn((5, 1, 4, 4), generator=torch.Generator().manual_seed(4))
    time = torch.linspace(0.2, 0.9, len(state))

    baseline_coarse, _, _ = branchwise_losses(
        baseline, state, target, time, analyzer, "split_baseline"
    )
    guard_coarse, _, _ = branchwise_losses(
        guardrail, state, target, time, analyzer, "split_additive_guardrail"
    )
    baseline_coarse.backward()
    guard_coarse.backward()

    assert torch.equal(baseline_coarse, guard_coarse)
    for first, second in zip(
        baseline.coarse.parameters(), guardrail.coarse.parameters()
    ):
        assert torch.equal(first.grad, second.grad)


def test_split_guardrail_changes_detail_but_not_coarse_objective():
    analyzer = _analyzer()
    model = SplitVelocityField(analyzer, width=4, depth=1)
    state = torch.randn((4, 1, 4, 4), generator=torch.Generator().manual_seed(5))
    target = torch.randn((4, 1, 4, 4), generator=torch.Generator().manual_seed(6))
    time = torch.tensor([0.5, 0.7, 0.8, 0.9])
    baseline_coarse, baseline_detail, _ = branchwise_losses(
        model, state, target, time, analyzer, "split_baseline"
    )
    guard_coarse, guard_detail, _ = branchwise_losses(
        model, state, target, time, analyzer, "split_additive_guardrail"
    )

    assert torch.equal(baseline_coarse, guard_coarse)
    assert not torch.equal(baseline_detail, guard_detail)
