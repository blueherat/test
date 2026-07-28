from __future__ import annotations

import torch

from experiments.run_rae_predictability_basis_sensitivity import (
    evaluate_basis_sensitivity,
    summarize,
)


class IdentityVelocity(torch.nn.Module):
    def forward(
        self, state: torch.Tensor, time: torch.Tensor, *, y: torch.Tensor
    ) -> torch.Tensor:
        del time, y
        return state


def test_identity_model_has_unit_gain_in_every_basis() -> None:
    generator = torch.Generator().manual_seed(17)
    clean = torch.randn((4, 4, 3, 3), generator=generator)
    noise = torch.randn((4, 4, 3, 3), generator=generator)
    labels = torch.arange(4)
    reference = torch.eye(4)[:, :1]
    bases = {
        "reference_guided": reference,
        "fractional": torch.eye(4)[:, 1:2],
        "top_pca": torch.eye(4)[:, 2:3],
        "random_0": torch.eye(4)[:, 3:4],
    }
    frame = evaluate_basis_sensitivity(
        IdentityVelocity(),
        clean,
        labels,
        noise,
        reference,
        bases,
        {
            "static": {"mode": "static"},
            "spc": {"mode": "annealed", "power": 2.0, "floor": 0.2},
        },
        seed=3,
        step=10,
        times=(0.85, 0.3),
        batch_size=2,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(
        torch.tensor(frame["total_gain"].to_numpy()),
        torch.ones(len(frame), dtype=torch.float64),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        torch.tensor(frame["within_basis_gain"].to_numpy()),
        torch.ones(len(frame), dtype=torch.float64),
        atol=2e-5,
        rtol=2e-5,
    )
    assert float(frame["complement_gain"].max()) < 2e-10

    per_seed, aggregate = summarize(frame)
    assert set(per_seed["basis"]) == {
        "reference_guided",
        "fractional",
        "top_pca",
        "random_mean",
    }
    assert float((per_seed["gain_over_random"] - 1.0).abs().max()) < 2e-5
    assert len(aggregate) == 8
