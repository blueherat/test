from __future__ import annotations

import math

import pytest
import torch

from experiments.run_dual_target_closed_loop_spiral_toy import (
    ContinuousSpiralDistribution,
    v4,
)


def make_distribution(
    *,
    ambient_dim: int = 16,
    bayes_batch_chunk: int = 64,
    data_jitter: float = 0.015,
    quadrature_points: int = 512,
) -> ContinuousSpiralDistribution:
    return ContinuousSpiralDistribution(
        ambient_dim,
        data_jitter=data_jitter,
        quadrature_points=quadrature_points,
        locator_points=1024,
        frequency_scale=6.0,
        embedding_seed=17,
        device=torch.device("cpu"),
        scale_mode="unit_rms",
        curvature=0.0,
        bayes_batch_chunk=bayes_batch_chunk,
    )


def test_sampling_is_exactly_the_v4_continuous_spiral_protocol() -> None:
    distribution = make_distribution()
    first = torch.Generator().manual_seed(23)
    second = torch.Generator().manual_seed(23)

    ambient, intrinsic, _ = distribution.sample(128, generator=first)
    expected_intrinsic = v4.sample_spiral_2d(
        128,
        device=torch.device("cpu"),
        jitter=0.015,
        generator=second,
    )

    assert torch.equal(intrinsic, expected_intrinsic)
    assert torch.allclose(ambient, distribution.embedding.embed(expected_intrinsic))


def test_distribution_accounts_for_v4_post_jitter_scale() -> None:
    distribution = make_distribution(data_jitter=0.015)

    assert distribution.intrinsic_jitter_std == pytest.approx(0.024)


def test_intrinsic_nll_uses_scaled_jitter_variance() -> None:
    distribution = make_distribution(
        ambient_dim=2,
        data_jitter=0.015,
        quadrature_points=256,
    )
    point = distribution.quadrature_u[73:74] + torch.tensor([[0.012, -0.006]])
    variance = (1.6 * 0.015) ** 2
    residual = point[:, None, :] - distribution.quadrature_u[None, :, :]
    expected = -(
        torch.logsumexp(-0.5 * residual.square().sum(dim=2) / variance, dim=1)
        - math.log(2.0 * math.pi * variance)
        - math.log(len(distribution.quadrature_u))
    )

    assert torch.allclose(distribution.intrinsic_nll(point), expected, atol=1e-6)


def test_bayes_clean_equals_prior_mean_at_noise_endpoint() -> None:
    distribution = make_distribution()
    generator = torch.Generator().manual_seed(29)
    state = torch.randn(31, distribution.ambient_dim, generator=generator)
    time_value = torch.zeros(len(state))

    actual = distribution.bayes_clean(state, time_value)
    expected_u = distribution.quadrature_u.mean(dim=0, keepdim=True).expand(len(state), -1)
    expected = distribution.embedding.embed(expected_u)

    assert torch.allclose(actual, expected, atol=2e-5)


def test_bayes_clean_recovers_on_distribution_state_at_data_endpoint() -> None:
    distribution = make_distribution()
    generator = torch.Generator().manual_seed(31)
    clean, _, _ = distribution.sample(96, generator=generator)
    time_value = torch.ones(len(clean))

    actual = distribution.bayes_clean(clean, time_value)

    assert torch.allclose(actual, clean, atol=3e-5)


def test_bayes_clean_is_invariant_to_batch_chunking() -> None:
    first = make_distribution(bayes_batch_chunk=7)
    second = make_distribution(bayes_batch_chunk=128)
    generator = torch.Generator().manual_seed(37)
    state = torch.randn(53, first.ambient_dim, generator=generator)
    time_value = torch.linspace(0.01, 0.99, len(state))

    a = first.bayes_clean(state, time_value)
    b = second.bayes_clean(state, time_value)

    assert torch.allclose(a, b, atol=1e-6)


def test_formal_quadrature_matches_doubled_resolution() -> None:
    formal = make_distribution(quadrature_points=1024)
    doubled = make_distribution(quadrature_points=2048)
    generator = torch.Generator().manual_seed(39)
    state = torch.randn(27, formal.ambient_dim, generator=generator)
    time_value = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]).repeat(6)[: len(state)]

    a = formal.bayes_velocity(state, time_value, denominator_floor=1e-3)
    b = doubled.bayes_velocity(state, time_value, denominator_floor=1e-3)
    relative_rms = (a - b).square().mean().sqrt() / b.square().mean().sqrt()

    assert relative_rms < 2e-4


def test_intrinsic_nll_prefers_spiral_samples_to_far_points() -> None:
    distribution = make_distribution()
    generator = torch.Generator().manual_seed(41)
    _, intrinsic, _ = distribution.sample(256, generator=generator)
    far = intrinsic + torch.tensor([4.0, -4.0])

    assert distribution.intrinsic_nll(intrinsic).mean() < distribution.intrinsic_nll(far).mean()


def test_nonzero_curvature_is_rejected_for_exact_oracle_protocol() -> None:
    try:
        ContinuousSpiralDistribution(
            8,
            data_jitter=0.015,
            quadrature_points=128,
            locator_points=256,
            frequency_scale=6.0,
            embedding_seed=43,
            device=torch.device("cpu"),
            curvature=0.5,
        )
    except ValueError as error:
        assert "curvature=0" in str(error)
    else:
        raise AssertionError("nonzero curvature must not silently use the linear Bayes formula")
