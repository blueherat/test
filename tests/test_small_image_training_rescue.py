import copy

import torch

from experiments.mnist_spectral_rollout_toy import TinyVelocityUNet
from experiments.small_image_basis_transport import OrthogonalDirectionLoss
from experiments.small_image_training_rescue import (
    SplitVelocityField,
    coarse_protected_weights,
    coefficient_loss,
    parameter_counts,
)


def _analyzer(size: int = 4) -> OrthogonalDirectionLoss:
    dimension = size * size
    basis = torch.eye(dimension)
    groups = torch.arange(dimension) // (dimension // 4)
    moments = torch.tensor([4.0, 1.0, 0.4, 0.1]).repeat_interleave(
        dimension // 4
    )
    return OrthogonalDirectionLoss(
        basis,
        moments,
        groups,
        gamma=0.5,
        min_weight=0.2,
        max_weight=2.0,
    )


def test_coarse_protection_has_unit_coarse_weight_and_mean_one():
    analyzer = _analyzer()
    time = torch.tensor([0.1, 0.5, 0.9])
    original = analyzer.weights(time)
    protected = coarse_protected_weights(analyzer, time)
    coarse = analyzer.group_index.eq(0)
    detail = ~coarse

    assert torch.equal(protected[:, coarse], torch.ones_like(protected[:, coarse]))
    assert torch.allclose(protected.mean(dim=1), torch.ones(len(time)), atol=1e-6)
    original_detail_ratios = original[:, detail][:, 1:] / original[:, detail][:, :-1]
    protected_detail_ratios = protected[:, detail][:, 1:] / protected[:, detail][:, :-1]
    assert torch.allclose(original_detail_ratios, protected_detail_ratios, atol=1e-6)


def test_split_field_outputs_are_confined_to_their_subspaces():
    analyzer = _analyzer()
    model = SplitVelocityField(analyzer, width=4, depth=1)
    value = torch.randn((3, 1, 4, 4), generator=torch.Generator().manual_seed(4))
    time = torch.tensor([0.2, 0.5, 0.8])
    coarse_output = model.project(model.coarse(value, time), coarse=True)
    detail_output = model.project(model.detail(value, time), coarse=False)
    coarse_coefficients = analyzer.transform(coarse_output)
    detail_coefficients = analyzer.transform(detail_output)
    coarse = analyzer.group_index.eq(0)

    assert torch.allclose(coarse_coefficients[:, ~coarse], torch.zeros_like(coarse_coefficients[:, ~coarse]))
    assert torch.allclose(detail_coefficients[:, coarse], torch.zeros_like(detail_coefficients[:, coarse]))
    assert torch.allclose(model(value, time), coarse_output + detail_output)


def test_split_variants_can_start_exactly_paired():
    analyzer = _analyzer()
    baseline = SplitVelocityField(analyzer, width=4, depth=1)
    weighted = copy.deepcopy(baseline)
    value = torch.randn((2, 1, 4, 4), generator=torch.Generator().manual_seed(5))
    time = torch.tensor([0.3, 0.7])
    assert torch.equal(baseline(value, time), weighted(value, time))


def test_protected_loss_reduces_to_mse_for_uniform_weights():
    analyzer = _analyzer()
    analyzer.gamma = 0.0
    prediction = torch.randn((5, 1, 4, 4), generator=torch.Generator().manual_seed(1))
    target = torch.randn((5, 1, 4, 4), generator=torch.Generator().manual_seed(2))
    time = torch.linspace(0.1, 0.9, 5)
    actual = coefficient_loss(
        prediction, target, analyzer, time, protected=True
    )
    assert torch.allclose(actual, torch.mean((prediction - target).square()), atol=1e-6)


def test_width_17_split_is_parameter_matched_to_width_24_raw():
    counts = parameter_counts(24, 17, 2)
    assert counts == {"raw": 229_897, "split": 231_678}
    assert abs(counts["split"] / counts["raw"] - 1.0) < 0.01
    assert sum(parameter.numel() for parameter in TinyVelocityUNet(24, 2).parameters()) == counts["raw"]
