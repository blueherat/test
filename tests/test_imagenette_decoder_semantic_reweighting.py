import numpy as np
import torch
import torch.nn as nn

from experiments.analyze_imagenette_decoder_semantic_reweighting import (
    crossfit_class_weights,
    frechet_components,
    frechet_from_moments,
    normalized_weighted_moments,
    restricted_predictions,
    weighted_histogram,
)


def test_uniform_weighted_moments_match_numpy() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(32, 5, generator=generator, dtype=torch.float64)
    mean, covariance, effective_count = normalized_weighted_moments(
        features, torch.ones(32)
    )
    np.testing.assert_allclose(mean, features.mean(dim=0).numpy(), atol=1e-12)
    np.testing.assert_allclose(
        covariance, np.cov(features.numpy(), rowvar=False), atol=1e-12
    )
    assert abs(effective_count - 32.0) < 1e-12


def test_frechet_moments_is_zero_for_identical_distribution() -> None:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(64, 7, generator=generator, dtype=torch.float64)
    mean, covariance, _ = normalized_weighted_moments(features, torch.ones(64))
    assert frechet_from_moments(mean, covariance, mean, covariance) < 1e-10


def test_frechet_components_sum_to_total() -> None:
    identity = np.eye(3)
    mean_component, covariance_component = frechet_components(
        np.zeros(3), identity, np.ones(3), 4.0 * identity
    )
    assert abs(mean_component - 3.0) < 1e-12
    assert abs(covariance_component - 3.0) < 1e-12
    assert abs(
        frechet_from_moments(np.zeros(3), identity, np.ones(3), 4.0 * identity)
        - mean_component
        - covariance_component
    ) < 1e-12


def test_crossfit_weights_reduce_class_mass_shift() -> None:
    empirical = torch.tensor([0] * 60 + [1] * 40)
    prior = torch.tensor([0] * 20 + [1] * 80)
    weights = crossfit_class_weights(empirical, prior, seed=17, smoothing=0.1)
    empirical_histogram = weighted_histogram(empirical, torch.ones(len(empirical)))
    prior_histogram = weighted_histogram(prior, torch.ones(len(prior)))
    weighted = weighted_histogram(prior, weights)
    before = 0.5 * (prior_histogram - empirical_histogram).abs().sum()
    after = 0.5 * (weighted - empirical_histogram).abs().sum()
    assert float(after) < float(before) * 0.1


def test_shifted_application_is_a_wrong_semantic_control() -> None:
    empirical = torch.tensor([0] * 60 + [1] * 40)
    prior = torch.tensor([0] * 20 + [1] * 80)
    correct = crossfit_class_weights(
        empirical, prior, seed=23, smoothing=0.1, application_shift=0
    )
    shifted = crossfit_class_weights(
        empirical, prior, seed=23, smoothing=0.1, application_shift=1
    )
    target = weighted_histogram(empirical, torch.ones(len(empirical)))
    correct_error = (weighted_histogram(prior, correct) - target).abs().sum()
    shifted_error = (weighted_histogram(prior, shifted) - target).abs().sum()
    assert float(correct_error) < float(shifted_error)


def test_restricted_predictions_accept_float64_features() -> None:
    evaluator = nn.Module()
    evaluator.classifier = nn.Linear(4, 1_000)
    predictions = restricted_predictions(
        torch.randn(6, 4, dtype=torch.float64), evaluator
    )
    assert predictions.shape == (6,)
    assert predictions.dtype == torch.long
