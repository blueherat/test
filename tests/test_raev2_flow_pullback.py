from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_flow_pullback import (  # noqa: E402
    flow_pullback_direction,
    frozen_covector_vjp,
    full_euler_flow,
    normalize_like,
)


class TimeVaryingAffineHeads(nn.Module):
    """A nonnormal velocity with noncommuting early/late Jacobians."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.8, dtype=torch.float64))
        self.seen_times: list[torch.Tensor] = []

    def matrix(self, time):
        zero = torch.zeros_like(time)
        return torch.stack(
            (0.2 + zero, self.weight * time, 1.0 - time, -0.1 + zero), dim=-1
        ).reshape(-1, 2, 2)

    def forward(self, state, time, *, context, attn_mask=None):
        del attn_mask
        self.seen_times.append(time.detach().clone())
        times = time.to(state.dtype)
        velocity = torch.einsum("bij,bj->bi", self.matrix(times), state)
        velocity = velocity + torch.stack((context, -context), dim=1).to(state) * 0.03
        full = state - times[:, None] * velocity
        gap = torch.stack((1.0 + 0.2 * state[:, 1], -0.3 + 0.4 * state[:, 0]), dim=1)
        return full, full - gap


def _analytic_jacobian(model, start, end, substeps):
    result = torch.eye(2, dtype=torch.float64)
    factors = []
    dt = (end - start) / substeps
    for index in range(substeps):
        time = torch.tensor([start + index * dt], dtype=torch.float32).double()
        factor = torch.eye(2, dtype=torch.float64) + dt * model.matrix(time)[0]
        result = factor @ result
        factors.append(factor)
    return result, factors


def test_vjp_uses_ordered_time_varying_jacobians_and_frozen_covector():
    model = TimeVaryingAffineHeads().eval()
    source = torch.tensor([[0.2, -0.5], [0.7, 0.3]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([2, 7])
    endpoint = full_euler_flow(model, source, labels, 0.9, 0.3, 3, checkpoint_forward=False)
    covector = endpoint.square() + torch.tensor([0.6, -0.7], dtype=torch.float64)
    frozen = covector.detach().clone()
    actual = frozen_covector_vjp(endpoint, source, covector)
    jacobian, factors = _analytic_jacobian(model, 0.9, 0.3, 3)
    torch.testing.assert_close(actual, frozen @ jacobian, rtol=1e-12, atol=1e-12)
    reversed_order = factors[0] @ factors[1] @ factors[2]
    assert torch.max(torch.abs(jacobian - reversed_order)) > 1e-3
    assert torch.max(torch.abs(actual - frozen @ reversed_order)) > 1e-3
    assert model.weight.grad is None
    assert source.grad is None
    assert not actual.requires_grad


@pytest.mark.parametrize("checkpoint_forward", [False, True])
def test_vjp_matches_central_finite_difference_of_future_linear_work(checkpoint_forward):
    model = TimeVaryingAffineHeads().eval()
    source = torch.tensor([[0.2, -0.5]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([3])
    covector = torch.tensor([[0.4, -0.8]], dtype=torch.float64)
    direction = torch.tensor([[-0.6, 0.3]], dtype=torch.float64)
    endpoint = full_euler_flow(
        model, source, labels, 0.9, 0.3, 4, checkpoint_forward=checkpoint_forward
    )
    pullback = frozen_covector_vjp(endpoint, source, covector)
    epsilon = 1e-6
    plus = full_euler_flow(
        model, source.detach() + epsilon * direction, labels, 0.9, 0.3, 4,
        checkpoint_forward=False,
    )
    minus = full_euler_flow(
        model, source.detach() - epsilon * direction, labels, 0.9, 0.3, 4,
        checkpoint_forward=False,
    )
    finite_difference = ((plus - minus) * covector).sum() / (2 * epsilon)
    torch.testing.assert_close((pullback * direction).sum(), finite_difference, rtol=1e-8, atol=1e-10)


def test_checkpoint_recomputation_retains_each_steps_original_time():
    model = TimeVaryingAffineHeads().eval()
    source = torch.tensor([[0.2, -0.5]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([4])
    endpoint = full_euler_flow(model, source, labels, 0.9, 0.3, 3)
    covector = torch.tensor([[0.5, -0.4]], dtype=torch.float64)
    actual = frozen_covector_vjp(endpoint, source, covector)
    jacobian, _ = _analytic_jacobian(model, 0.9, 0.3, 3)
    torch.testing.assert_close(actual, covector @ jacobian, rtol=1e-12, atol=1e-12)
    assert len(model.seen_times) == 6
    for forward, recomputed in zip(model.seen_times[:3], reversed(model.seen_times[3:])):
        torch.testing.assert_close(forward, recomputed, rtol=0, atol=0)


def test_normalization_matches_individual_norms_and_has_safe_degenerate_fallback():
    reference = torch.tensor([[3.0, 4.0], [2.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    direction = torch.tensor([[2.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    actual = normalize_like(direction, reference)
    torch.testing.assert_close(actual.norm(dim=1), reference.norm(dim=1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual[1], reference[1])
    torch.testing.assert_close(actual[2], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(actual[0] / actual[0].norm(), direction[0] / direction[0].norm())


@pytest.mark.parametrize("outer_mode", [torch.no_grad, torch.inference_mode])
def test_wrapper_is_input_only_detached_and_norm_matched_inside_outer_modes(outer_mode):
    model = TimeVaryingAffineHeads().eval()
    model.weight.grad = torch.tensor(7.0, dtype=torch.float64)
    with outer_mode():
        state = torch.tensor([[0.2, -0.5], [0.7, 0.3]], dtype=torch.float64)
        labels = torch.tensor([2, 7])
        gap = torch.tensor([[0.6, -0.4], [0.9, 0.2]], dtype=torch.float64)
        result = flow_pullback_direction(
            model, state, labels, gap, 0.9, 0.3, 3,
        )
        pulled, raw, metrics = result.direction, result.raw_future_direction, result.telemetry
    jacobian, _ = _analytic_jacobian(model, 0.9, 0.3, 3)
    with torch.inference_mode(False):
        source = state.clone()
        endpoint = full_euler_flow(model, source, labels.clone(), 0.9, 0.3, 3, checkpoint_forward=False)
        full, base = model(endpoint, torch.full((2,), 0.3), context=labels.clone())
        future_gap = (full - base).detach()
        expected = normalize_like(future_gap @ jacobian, gap.clone())
        torch.testing.assert_close(pulled, expected)
        torch.testing.assert_close(raw, normalize_like(future_gap, gap.clone()))
        torch.testing.assert_close(pulled.norm(dim=1), gap.clone().norm(dim=1))
    assert not pulled.requires_grad and not raw.requires_grad
    assert not torch.is_inference(pulled) and not torch.is_inference(raw)
    assert model.weight.grad.item() == 7.0
    assert model.weight.requires_grad
    assert not metrics["pullback_rms"].requires_grad
    assert not metrics["pullback_fallback"].any()


def test_zero_horizon_is_identity_pullback():
    model = TimeVaryingAffineHeads().eval()
    state = torch.tensor([[0.2, -0.5]], dtype=torch.float64)
    gap = torch.tensor([[0.6, -0.4]], dtype=torch.float64)
    result = flow_pullback_direction(
        model, state, torch.tensor([3]), gap, 0.7, 0.7, 2,
    )
    pulled, raw, metrics = result.direction, result.raw_future_direction, result.telemetry
    torch.testing.assert_close(pulled, raw)
    torch.testing.assert_close(metrics["pullback_gain"], torch.ones(1, dtype=torch.float64))


def test_floor_and_reverse_time_convention_are_respected():
    class ZeroHeads(nn.Module):
        def forward(self, state, time, *, context, attn_mask=None):
            return torch.zeros_like(state), torch.zeros_like(state)

    source = torch.tensor([[2.0, 3.0]], requires_grad=True)
    actual = full_euler_flow(ZeroHeads(), source, torch.tensor([0]), 0.04, 0.02, 1)
    torch.testing.assert_close(actual, 0.6 * source)


def test_invalid_flow_protocol_and_inference_tensor_fail_early():
    model = TimeVaryingAffineHeads()
    state = torch.zeros(2, 2)
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="end_time"):
        full_euler_flow(model, state, labels, 0.3, 0.9, 4)
    with pytest.raises(ValueError, match="substeps"):
        full_euler_flow(model, state, labels, 0.9, 0.3, 0)
    with pytest.raises(ValueError, match="labels"):
        full_euler_flow(model, state, labels[:1], 0.9, 0.3, 4)
    with torch.inference_mode():
        inference_state = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="inference"):
        full_euler_flow(model, inference_state, labels, 0.9, 0.3, 4)
