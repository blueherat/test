from __future__ import annotations

import pytest
import torch

from experiments.run_rae_latent_trust_rollout import (
    evaluate_rollout_batch,
    perturbation_conditions,
)


class ZeroVelocity(torch.nn.Module):
    def forward(
        self, state: torch.Tensor, time: torch.Tensor, *, y: torch.Tensor
    ) -> torch.Tensor:
        del time, y
        return torch.zeros_like(state)


def test_perturbations_have_matched_per_sample_norms() -> None:
    generator = torch.Generator().manual_seed(5)
    clean = torch.randn((3, 4, 3, 3), generator=generator)
    reference = torch.randn(clean.shape, generator=generator)
    bases = {
        "basis_a": torch.eye(4)[:, :1],
        "basis_b": torch.eye(4)[:, 1:2],
    }
    conditions, deltas = perturbation_conditions(
        clean, reference, bases, amplitudes=(1.0,)
    )
    assert len(conditions) == 2
    expected = reference.flatten(1).norm(dim=1)
    for delta in deltas:
        torch.testing.assert_close(
            delta.flatten(1).norm(dim=1), expected, atol=2e-6, rtol=2e-6
        )


def test_zero_velocity_preserves_unit_endpoint_shift_gain() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn((2, 4, 3, 3), generator=generator)
    noise = torch.randn(clean.shape, generator=generator)
    labels = torch.arange(2)
    reference_basis = torch.eye(4)[:, :1]
    rows = evaluate_rollout_batch(
        ZeroVelocity(),
        clean,
        labels,
        noise,
        reference_basis,
        {
            "basis_a": torch.eye(4)[:, :1],
            "basis_b": torch.eye(4)[:, 1:2],
        },
        {
            "static": {"mode": "static"},
            "spc": {"mode": "annealed", "power": 2.0, "floor": 0.2},
        },
        torch.tensor([0.85, 0.5, 0.0]),
        target_time=0.85,
        amplitudes=(1.0,),
        model_batch_size=2,
    )
    assert len(rows) == 4
    for row in rows:
        assert row["endpoint_shift_gain"] == pytest.approx(1.0, rel=2e-5)
