import copy

import torch

from experiments.mnist_spectral_rollout_toy import TinyVelocityUNet
from experiments.small_image_basis_transport import (
    OrthogonalDirectionLoss,
    dct_pixel_basis,
)
from experiments.small_image_residual_adapter import (
    ProjectedResidualField,
    detail_weighted_mse,
    normalized_band_drift_loss,
    protection_error,
    residual_variant_terms,
    self_generated_state,
)


def _analyzer(size: int = 4) -> OrthogonalDirectionLoss:
    dimension = size * size
    return OrthogonalDirectionLoss(
        dct_pixel_basis(size),
        torch.linspace(2.0, 0.5, dimension),
        torch.arange(dimension) // (dimension // 4),
        gamma=0.5,
    )


def test_projected_residual_preserves_group_zero_exactly():
    analyzer = _analyzer()
    baseline = TinyVelocityUNet(width=4, depth=1)
    model = ProjectedResidualField(
        baseline, analyzer, adapter_width=4, depth=1
    )
    with torch.no_grad():
        model.adapter.output.weight.normal_()
        model.adapter.output.bias.normal_()
    value = torch.randn(3, 1, 4, 4)
    time = torch.tensor([0.2, 0.6, 0.9])
    assert protection_error(model, value, time, analyzer) < 1e-5
    delta_coefficients = analyzer.transform(model.field_delta(value, time))
    assert torch.count_nonzero(delta_coefficients[:, analyzer.group_index != 0]) > 0


def test_residual_scale_interpolates_from_frozen_baseline():
    analyzer = _analyzer()
    baseline = TinyVelocityUNet(width=4, depth=1)
    full = ProjectedResidualField(
        copy.deepcopy(baseline), analyzer, adapter_width=4, depth=1, residual_scale=1.0
    )
    half = ProjectedResidualField(
        copy.deepcopy(baseline), analyzer, adapter_width=4, depth=1, residual_scale=0.5
    )
    half.adapter.load_state_dict(full.adapter.state_dict())
    with torch.no_grad():
        full.adapter.output.weight.normal_()
        full.adapter.output.bias.normal_()
        half.adapter.load_state_dict(full.adapter.state_dict())
    value = torch.randn(2, 1, 4, 4)
    time = torch.tensor([0.4, 0.8])
    baseline_value = baseline(value, time)
    torch.testing.assert_close(
        half(value, time) - baseline_value,
        0.5 * (full(value, time) - baseline_value),
        atol=2e-6,
        rtol=2e-6,
    )


def test_detail_loss_ignores_protected_error():
    analyzer = _analyzer()
    target = torch.randn(2, 1, 4, 4)
    coefficients = torch.zeros(2, 16)
    coefficients[:, analyzer.group_index == 0] = 3.0
    coarse_error = (coefficients @ analyzer.basis.T).reshape_as(target)
    time = torch.tensor([0.4, 0.8])
    loss = detail_weighted_mse(target + coarse_error, target, analyzer, time)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0.0)


def test_normalized_drift_loss_is_zero_for_exact_target_and_positive_otherwise():
    analyzer = _analyzer()
    state = torch.randn(8, 1, 4, 4)
    target = state.clone()
    exact = normalized_band_drift_loss(target, target, state, analyzer)
    wrong = normalized_band_drift_loss(torch.zeros_like(target), target, state, analyzer)
    torch.testing.assert_close(exact, torch.zeros_like(exact), atol=1e-8, rtol=0.0)
    assert float(wrong) > 0.1


def test_self_generated_state_uses_bounded_sampling_direction_step():
    class Ones(torch.nn.Module):
        def forward(self, value, time):
            del time
            return torch.ones_like(value)

    state = torch.zeros(2, 1, 2, 2)
    time = torch.tensor([0.05, 0.8])
    generated, generated_time = self_generated_state(
        Ones(), state, time, max_step=0.1
    )
    torch.testing.assert_close(generated_time, torch.tensor([0.025, 0.7]))
    torch.testing.assert_close(
        generated[:, 0, 0, 0], torch.tensor([-0.025, -0.1])
    )


def test_residual_variant_terms_separate_offpath_mse_and_drift():
    assert residual_variant_terms("teacher_residual") == (False, False)
    assert residual_variant_terms("rollout_drift_residual") == (True, True)
    assert residual_variant_terms("offpath_mse_residual") == (True, False)
    assert residual_variant_terms("drift_only_residual") == (False, True)
