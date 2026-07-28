from __future__ import annotations

import pandas as pd
import pytest
import torch

from experiments.small_image_signed_leverage import (
    canonical_band_energies,
    endpoint_moment_loss,
    finite_difference_directional_derivative,
    summarize_signed_leverage,
    update_alignment,
)


def test_canonical_band_energy_and_loss_are_exact_for_matching_samples():
    images = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    basis = torch.eye(4)
    groups = torch.tensor([0, 0, 1, 1])
    energy = canonical_band_energies(images, basis, groups, 2)
    loss, gap = endpoint_moment_loss(images, energy, basis, groups)
    assert loss == pytest.approx(0.0)
    assert torch.allclose(gap, torch.zeros_like(gap))


def test_update_alignment_preserves_signed_direction():
    endpoint = (torch.tensor([1.0, 0.0]),)
    baseline = (torch.tensor([0.0, 0.0]),)
    harmful_weighted_gradient = (torch.tensor([-1.0, 0.0]),)
    helpful_weighted_gradient = (torch.tensor([1.0, 0.0]),)
    harmful = update_alignment(endpoint, baseline, harmful_weighted_gradient)
    helpful = update_alignment(endpoint, baseline, helpful_weighted_gradient)
    assert harmful["raw_directional_derivative"] > 0
    assert helpful["raw_directional_derivative"] < 0


def test_finite_difference_matches_analytic_directional_derivative():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    direction = (torch.ones_like(model.weight),)

    def objective():
        return model.weight.square().sum()

    finite = finite_difference_directional_derivative(
        model, direction, objective, relative_step=1e-3
    )
    assert finite == pytest.approx(4.0, rel=1e-3)
    assert model.weight.item() == pytest.approx(2.0)


def test_summary_checks_prediction_sign_against_endpoint_damage():
    rows = []
    for probe_seed in (1, 2):
        rows.append(
            {
                "dataset": "mnist",
                "basis": "dct",
                "training_seed": 4,
                "probe_seed": probe_seed,
                "endpoint_moment_loss": 1.0,
                "endpoint_band0_log_gap": -0.2,
                "raw_directional_derivative": -0.5,
                "raw_cosine": -0.1,
                "block_directional_derivative": -1.0,
                "block_cosine": -0.2,
                "raw_finite_difference_derivative": -0.5,
                "observed_endpoint_fid_ratio": 0.8,
                "observed_endpoint_log_damage": -0.2,
            }
        )
    summary = summarize_signed_leverage(pd.DataFrame(rows)).iloc[0]
    assert summary["probe_seeds"] == 2
    assert bool(summary["raw_sign_correct"])
    assert bool(summary["block_sign_correct"])
