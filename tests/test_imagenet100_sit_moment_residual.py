from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.estimate_imagenet100_sit_diagonal_moments import (
    estimate_diagonal_moments,
)
from experiments.imagenet100_sit_moment_residual import (
    DiagonalMomentStats,
    diagonal_lmmse_terms,
    load_diagonal_moment_stats,
    moment_residual_losses,
)
from experiments.imagenet100_sit_prediction_targets import prediction_losses
from experiments.train_imagenet100_sit_flow import sit_training_losses
from experiments.sample_imagenet100_sit_fid import conditional_velocity


def make_stats(mean: torch.Tensor, variance: torch.Tensor) -> DiagonalMomentStats:
    return DiagonalMomentStats(
        mean=mean,
        variance=variance,
        count=100,
        cache_manifest_sha256="manifest",
        scaling_factor=0.18215,
        source_path="train_moments.npy",
        source_sha256="moments",
    )


def test_diagonal_lmmse_matches_scalar_closed_form() -> None:
    mean = torch.tensor([[[1.5]]])
    variance = torch.tensor([[[4.0]]])
    stats = make_stats(mean, variance)
    state = torch.tensor([[[[2.0]]], [[[3.0]]]])
    time_value = torch.tensor([0.25, 0.75])
    actual, residual_std = diagonal_lmmse_terms(
        state, time_value, stats, variance_floor=1e-8
    )
    t = time_value.reshape(-1, 1, 1, 1)
    covariance = (1 - t).square() + t.square() * 4.0
    expected = 1.5 + ((t * 4.0 - (1 - t)) / covariance) * (state - t * 1.5)
    assert torch.allclose(actual, expected)
    assert torch.allclose(residual_std.square(), 4.0 / covariance)


def test_recovered_velocity_loss_is_literal_velocity_mse() -> None:
    torch.manual_seed(4)
    shape = (5, 2, 3, 3)
    stats = make_stats(torch.randn(shape[1:]), torch.rand(shape[1:]) + 0.2)
    state = torch.randn(shape)
    velocity_target = torch.randn(shape)
    prediction = torch.randn(shape)
    time_value = torch.rand(shape[0])
    analytic, scale = diagonal_lmmse_terms(
        state, time_value, stats, variance_floor=1e-6
    )
    expected = torch.mean((analytic + scale * prediction - velocity_target).square())
    losses = moment_residual_losses(
        prediction,
        state=state,
        velocity_target=velocity_target,
        time_value=time_value,
        stats=stats,
        variance_floor=1e-6,
    )
    assert torch.allclose(losses["optimized"], expected)
    assert torch.equal(losses["optimized"], losses["velocity"])


def test_native_training_branch_is_exactly_unchanged() -> None:
    torch.manual_seed(8)
    state = torch.randn(3, 4, 2, 2)
    data = torch.randn_like(state)
    noise = torch.randn_like(state)
    prediction = torch.randn_like(state)
    time_value = torch.rand(3)
    expected = prediction_losses(
        prediction,
        state=state,
        data=data,
        noise=noise,
        time_value=time_value,
        prediction_target="velocity",
        loss_space="velocity",
        denominator_floor=1e-3,
    )
    actual = sit_training_losses(
        prediction,
        state=state,
        data=data,
        noise=noise,
        time_value=time_value,
        prediction_target="velocity",
        loss_space="velocity",
        denominator_floor=1e-3,
        velocity_decomposition="native",
        moment_stats=None,
        moment_variance_floor=1e-6,
    )
    for key in expected:
        assert torch.equal(actual[key], expected[key])


def test_estimator_includes_posterior_variance(tmp_path: Path) -> None:
    moments = np.zeros((2, 8, 32, 32), dtype=np.float32)
    moments[0, :4] = 1.0
    moments[1, :4] = 3.0
    moments[:, 4:] = 2.0
    path = tmp_path / "train_moments.npy"
    np.save(path, moments, allow_pickle=False)
    mean, variance, count = estimate_diagonal_moments(
        path, batch_size=1, scaling_factor=0.5
    )
    assert count == 2
    assert np.allclose(mean, 1.0)
    # Var of posterior means after scaling is 0.25; posterior variance is 1.0.
    assert np.allclose(variance, 1.25)


def test_stats_loader_rejects_different_cache_manifest(tmp_path: Path) -> None:
    stats = make_stats(torch.zeros(1, 2, 2), torch.ones(1, 2, 2))
    path = tmp_path / "stats.pt"
    torch.save(stats.checkpoint_payload(), path)
    with pytest.raises(ValueError, match="different cache manifest"):
        load_diagonal_moment_stats(
            path, expected_cache_manifest_sha256="validation-or-other-cache"
        )


def test_sampler_recovers_velocity_from_residual_output() -> None:
    class ZeroModel(torch.nn.Module):
        def forward(self, state, time_value, labels):
            return torch.zeros_like(state)

    stats = make_stats(torch.full((1, 2, 2), 0.2), torch.full((1, 2, 2), 0.7))
    state = torch.randn(3, 1, 2, 2)
    labels = torch.zeros(3, dtype=torch.long)
    time_value = torch.tensor(0.4)
    velocity, counter = conditional_velocity(
        ZeroModel(),
        labels,
        autocast_dtype=None,
        velocity_decomposition="diagonal_lmmse",
        moment_stats=stats,
        moment_variance_floor=1e-6,
    )
    actual = velocity(time_value, state)
    expected, _ = diagonal_lmmse_terms(
        state, time_value.expand(3), stats, variance_floor=1e-6
    )
    assert torch.allclose(actual, expected)
    assert counter["nfe"] == 1
