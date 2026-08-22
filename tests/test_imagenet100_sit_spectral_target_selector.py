from __future__ import annotations

import torch

from experiments.imagenet100_sit_spectral_target_selector import (
    SpectralTargetSelector,
    bernstein_basis,
    orthonormal_dct_matrix,
)


def _selector() -> SpectralTargetSelector:
    return SpectralTargetSelector(channels=2, side=4, time_terms=4)


def test_dct_and_bernstein_are_normalized() -> None:
    dct = orthonormal_dct_matrix(7)
    torch.testing.assert_close(dct @ dct.T, torch.eye(7), atol=2e-6, rtol=2e-6)
    weights = bernstein_basis(torch.tensor([0.0, 0.2, 0.7, 1.0]), 5)
    assert torch.all(weights >= 0)
    torch.testing.assert_close(weights.sum(1), torch.ones(4))


def test_spectral_transform_round_trip_with_channel_rotation() -> None:
    selector = _selector()
    with torch.no_grad():
        selector.channel_raw.copy_(torch.tensor([[1.0, 0.4], [-0.2, 0.9]]))
    value = torch.randn(3, 2, 4, 4)
    reconstructed = selector.from_spectral(selector.to_spectral(value))
    torch.testing.assert_close(reconstructed, value, atol=3e-5, rtol=3e-5)


def test_operator_endpoints_recover_native_v_and_x_parameterizations() -> None:
    selector = _selector()
    data = torch.randn(5, 2, 4, 4)
    noise = torch.randn_like(data)
    time = torch.linspace(0.1, 0.9, len(data))
    broadcast_time = time[:, None, None, None]
    state = (1.0 - broadcast_time) * noise + broadcast_time * data
    velocity = data - noise

    all_v = torch.ones_like(data)
    v_target = selector.native_target(
        data=data, velocity=velocity, eigenvalues=all_v
    )
    torch.testing.assert_close(v_target, velocity, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        selector.output_to_velocity(
            v_target,
            state=state,
            time_value=time,
            eigenvalues=all_v,
            denominator_floor=1e-4,
        ),
        velocity,
        atol=5e-5,
        rtol=5e-5,
    )

    all_x = torch.zeros_like(data)
    x_target = selector.native_target(
        data=data, velocity=velocity, eigenvalues=all_x
    )
    torch.testing.assert_close(x_target, data, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        selector.output_to_velocity(
            x_target,
            state=state,
            time_value=time,
            eigenvalues=all_x,
            denominator_floor=1e-4,
        ),
        velocity,
        atol=5e-5,
        rtol=5e-5,
    )


def test_soft_operator_target_has_exact_velocity_inverse_and_gradients() -> None:
    selector = _selector()
    data = torch.randn(6, 2, 4, 4)
    noise = torch.randn_like(data)
    time = torch.linspace(0.05, 0.95, len(data))
    state = (1.0 - time[:, None, None, None]) * noise + time[:, None, None, None] * data
    velocity = data - noise
    eigenvalues = torch.rand_like(data) * 0.8 + 0.1
    target = selector.native_target(
        data=data,
        velocity=velocity,
        eigenvalues=eigenvalues,
    )
    recovered = selector.output_to_velocity(
        target,
        state=state,
        time_value=time,
        eigenvalues=eigenvalues,
        denominator_floor=1e-4,
    )
    torch.testing.assert_close(recovered, velocity, atol=7e-5, rtol=7e-5)

    predicted = (target + 0.01 * torch.randn_like(target)).detach().requires_grad_(True)
    loss = selector.output_to_velocity(
        predicted,
        state=state,
        time_value=time,
        eigenvalues=selector.eigenvalues(time),
        denominator_floor=0.05,
    ).square().mean()
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert selector.gate_raw.grad is not None


def test_initial_selector_is_close_to_velocity_and_projection_is_bounded() -> None:
    selector = _selector()
    time = torch.linspace(0.0, 1.0, 9)
    initial = selector.eigenvalues(time)
    torch.testing.assert_close(
        initial,
        torch.full_like(initial, 0.999),
        atol=2e-6,
        rtol=2e-6,
    )
    with torch.no_grad():
        selector.gate_raw.fill_(-3.0)
        selector.project_parameters_()
    bounded = selector.eigenvalues(time)
    assert torch.all((0.0 <= bounded) & (bounded <= 1.0))
