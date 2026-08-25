from __future__ import annotations

import torch

from experiments.advfd_score_counterexample import (
    build_advfd_witness,
    frechet_distance_1d,
    moment_matched_disjoint_pair,
    noised_reverse_kl_and_score_metrics,
    official_loaded_real_whitened_fd_1d,
    paper_regularized_real_whitened_fd_1d,
    shared_support_pearson_control,
    witness_generator_gradient,
    witness_statistics,
)


def test_disjoint_pair_matches_identity_moments_but_not_support() -> None:
    real, fake = moment_matched_disjoint_pair()
    torch.testing.assert_close(real.mean, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(fake.mean, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(real.variance, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(fake.variance, torch.tensor(1.0, dtype=torch.float64))
    assert set(real.atoms.tolist()).isdisjoint(set(fake.atoms.tolist()))
    distance = frechet_distance_1d(
        real.mean, real.variance, fake.mean, fake.variance
    )
    torch.testing.assert_close(distance, torch.tensor(0.0, dtype=torch.float64))


def test_shared_support_mean_only_control_equals_pearson_chi_square() -> None:
    values = shared_support_pearson_control()
    torch.testing.assert_close(values["objective"], values["pearson_chi_square"])
    torch.testing.assert_close(
        values["objective"], torch.tensor(0.36, dtype=torch.float64)
    )


def test_hermite_witness_prescribes_values_and_fake_derivatives() -> None:
    coefficients = build_advfd_witness(
        7.0, fake_derivatives=(1.25, -2.5)
    )
    stats = witness_statistics(coefficients)
    torch.testing.assert_close(
        stats["real_features"], torch.tensor([-1.0, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        stats["fake_features"], torch.tensor([7.0, 7.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        stats["fake_derivatives"],
        torch.tensor([1.25, -2.5], dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_paper_and_released_regularization_match_their_closed_forms() -> None:
    epsilon = 0.1
    amplitude = 4.0
    scalars = tuple(torch.tensor(value, dtype=torch.float64) for value in (0, 1, 4, 0))
    paper = paper_regularized_real_whitened_fd_1d(
        *scalars, epsilon=epsilon
    )
    released = official_loaded_real_whitened_fd_1d(
        *scalars, epsilon=epsilon
    )
    expected_paper = amplitude**2 / (1 + epsilon) + 1 / (1 + epsilon)
    expected_released = amplitude**2 / (1 + epsilon) + (
        1 - (epsilon / (1 + epsilon)) ** 0.5
    ) ** 2
    torch.testing.assert_close(paper, paper.new_tensor(expected_paper))
    torch.testing.assert_close(released, released.new_tensor(expected_released))


def test_same_advfd_value_allows_zero_or_arbitrary_generator_gradient() -> None:
    flat = build_advfd_witness(4.0, fake_derivatives=(0.0, 0.0))
    plus = build_advfd_witness(4.0, fake_derivatives=(1.0, 1.0))
    minus = build_advfd_witness(4.0, fake_derivatives=(-1.0, -1.0))
    flat_fd, flat_gradient = witness_generator_gradient(flat, epsilon=1e-3)
    plus_fd, plus_gradient = witness_generator_gradient(plus, epsilon=1e-3)
    minus_fd, minus_gradient = witness_generator_gradient(minus, epsilon=1e-3)

    torch.testing.assert_close(flat_fd, plus_fd, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(flat_fd, minus_fd, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(
        flat_gradient, torch.zeros_like(flat_gradient), atol=1e-8, rtol=0
    )
    torch.testing.assert_close(plus_gradient, -minus_gradient, atol=1e-8, rtol=1e-8)
    assert float(plus_gradient.norm()) > 1.0


def test_flat_witness_fd_grows_while_official_normalized_gradient_stays_zero() -> None:
    low = build_advfd_witness(2.0)
    high = build_advfd_witness(20.0)
    low_fd, low_gradient = witness_generator_gradient(
        low, epsilon=0.1, normalization_epsilon=0.01
    )
    high_fd, high_gradient = witness_generator_gradient(
        high, epsilon=0.1, normalization_epsilon=0.01
    )
    # The mean term grows as M^2; the loaded covariance contributes the same
    # positive constant to both values, so the finite ratio is below 100.
    assert float(high_fd) > 80.0 * float(low_fd)
    torch.testing.assert_close(
        low_gradient, torch.zeros_like(low_gradient), atol=1e-8, rtol=0
    )
    torch.testing.assert_close(
        high_gradient, torch.zeros_like(high_gradient), atol=1e-7, rtol=0
    )


def test_gaussian_noising_makes_score_flow_finite_and_descending() -> None:
    real, fake = moment_matched_disjoint_pair()
    metrics = noised_reverse_kl_and_score_metrics(
        real,
        fake,
        sigma=0.4,
        grid_points=20_001,
        step_factor=0.02,
    )
    assert torch.isfinite(metrics["reverse_kl"])
    assert float(metrics["fisher_divergence"]) > 0
    assert float(metrics["continuity_kl_derivative"]) < 0
    assert float(metrics["parameterized_kl_derivative"]) < 0
    assert float(metrics["reverse_kl_change"]) < 0
