from __future__ import annotations

import torch

from experiments.posterior_response_projector import (
    posterior_response_action,
    posterior_response_basis,
    posterior_response_blend,
)


def test_response_blend_has_correct_projector_limits() -> None:
    anchor = torch.tensor([[1.0, 2.0, 3.0]])
    other = torch.tensor([[4.0, 5.0, 6.0]])
    zero_action = torch.zeros_like(anchor)
    identity_action = anchor - other

    torch.testing.assert_close(
        posterior_response_blend(anchor, other, zero_action), other
    )
    torch.testing.assert_close(
        posterior_response_blend(anchor, other, identity_action), anchor
    )
    torch.testing.assert_close(
        posterior_response_blend(
            anchor, other, identity_action, strength=0.0
        ),
        anchor,
    )


def test_response_action_recovers_linear_projector() -> None:
    generator = torch.Generator().manual_seed(3)
    D, rank, batch = 17, 3, 11
    raw = torch.randn(D, rank, generator=generator)
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    projector = basis @ basis.T
    alpha = torch.linspace(0.2, 0.9, batch)

    def clean_estimator(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return (state @ projector) / alpha.repeat_interleave(
            len(state) // batch
        )[:, None]

    state = torch.randn(batch, D, generator=generator)
    direction = torch.randn(batch, D, generator=generator)
    time = torch.linspace(0.1, 0.8, batch)
    estimated = posterior_response_action(
        clean_estimator,
        state=state,
        time=time,
        direction=direction,
        alpha=alpha,
        relative_step=1e-2,
    )
    expected = direction @ projector
    torch.testing.assert_close(estimated, expected, atol=2e-5, rtol=2e-5)


def test_randomized_response_basis_finds_projector_range() -> None:
    generator = torch.Generator().manual_seed(7)
    D, rank, batch = 23, 2, 8
    raw = torch.randn(D, rank, generator=generator)
    true_basis, _ = torch.linalg.qr(raw, mode="reduced")
    projector = true_basis @ true_basis.T
    alpha = torch.full((batch,), 0.7)

    def clean_estimator(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return (state @ projector) / 0.7

    state = torch.randn(batch, D, generator=generator)
    time = torch.full((batch,), 0.3)
    estimated_basis, singular_values = posterior_response_basis(
        clean_estimator,
        state=state,
        time=time,
        alpha=alpha,
        probes=6,
        rank=rank,
        relative_step=1e-2,
        generator=generator,
    )
    overlap = torch.einsum("di,bdj->bij", true_basis, estimated_basis)
    principal_cosines = torch.linalg.svdvals(overlap)
    assert float(principal_cosines.min()) > 0.9999
    assert torch.all(singular_values[:, 1] > 100 * singular_values[:, 2])
