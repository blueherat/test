"""Semigroup-consistent corrections for finite-strength score guidance."""

from __future__ import annotations

import torch


def local_jensen_velocity_coefficient(
    beta: float | torch.Tensor,
    time_value: torch.Tensor,
) -> torch.Tensor:
    """Return the coefficient of ``J_gap.T @ gap`` for a linear flow path.

    The linear interpolation is ``z_t = t*x + (1-t)*epsilon``.  If standard
    weak-to-strong guidance has exponent ``beta = 1 + gamma``, the first local
    term of the exact conditional-Jensen correction is

        beta * (beta - 1) * t * (1 - t) * J_gap.T @ gap.

    This coefficient is fixed by the endpoint power tilt; it is not an
    additional guidance scale.
    """

    beta_tensor = torch.as_tensor(beta, dtype=time_value.dtype, device=time_value.device)
    if torch.any(beta_tensor < 1.0):
        raise ValueError("power-guidance exponent beta must be at least one")
    return beta_tensor * (beta_tensor - 1.0) * time_value * (1.0 - time_value)


def local_jensen_velocity_correction(
    velocity_gap: torch.Tensor,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    beta: float | torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute the lowest-order semigroup correction in velocity space.

    ``velocity_gap`` must be evaluated from ``state`` without detaching its
    autograd graph.  The function uses one reverse-mode VJP and never builds a
    dense Jacobian.
    """

    if not state.requires_grad:
        raise ValueError("state must require gradients")
    if velocity_gap.shape != state.shape:
        raise ValueError("velocity gap and state must have identical shapes")
    coefficient = local_jensen_velocity_coefficient(beta, time_value)
    gap_energy = 0.5 * velocity_gap.float().flatten(1).square().sum(dim=1)
    gap_vjp = torch.autograd.grad(
        gap_energy.sum(),
        state,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    while coefficient.ndim < gap_vjp.ndim:
        coefficient = coefficient.unsqueeze(-1)
    return coefficient * gap_vjp
