from __future__ import annotations

import torch

from experiments.residual_score_toy import (
    NearEquilibriumGaussianMixture,
    FactorizedDomainNoiseEstimator,
    RatioEstimator,
    field_metrics,
    factorized_dsm_score_difference,
    pushforward_kl,
    ratio_score_difference,
)


def _mixture(dtype: torch.dtype = torch.float64) -> NearEquilibriumGaussianMixture:
    return NearEquilibriumGaussianMixture.ring(
        components=8,
        radius=2.0,
        component_std=0.3,
        perturbation_amplitude=0.7,
        dtype=dtype,
    )


def test_mixture_weights_are_valid_and_near_equilibrium_is_linear() -> None:
    mixture = _mixture()
    real = mixture.weights(0.3, "real")
    fake_small = mixture.weights(0.1, "fake")
    fake_large = mixture.weights(0.3, "fake")

    torch.testing.assert_close(real.sum(), torch.tensor(1.0, dtype=real.dtype))
    torch.testing.assert_close(fake_small.sum(), torch.tensor(1.0, dtype=real.dtype))
    assert torch.all(fake_small > 0)
    assert torch.all(fake_large > 0)
    torch.testing.assert_close(fake_large - real, 3.0 * (fake_small - real))


def test_analytic_score_matches_log_density_gradient_at_multiple_noise_scales() -> None:
    mixture = _mixture()
    states = torch.tensor(
        [[-1.2, 0.4], [0.1, -0.7], [1.5, 1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    sigma = torch.tensor([0.0, 0.2, 0.9], dtype=torch.float64)

    for domain in ("real", "fake"):
        log_probability, analytic_score = mixture.log_prob_and_score(
            states,
            sigma=sigma,
            epsilon=0.2,
            domain=domain,
        )
        autodiff_score = torch.autograd.grad(
            log_probability.sum(), states, retain_graph=True
        )[0]
        torch.testing.assert_close(
            analytic_score,
            autodiff_score,
            atol=2e-10,
            rtol=2e-10,
        )


def test_bayes_real_logit_gradient_is_real_minus_fake_score() -> None:
    mixture = _mixture()
    states = torch.tensor(
        [[-0.8, 0.2], [0.4, -1.0], [1.3, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    sigma = torch.tensor([0.05, 0.3, 1.1], dtype=torch.float64)
    logit = mixture.bayes_real_logit(states, sigma=sigma, epsilon=0.15)
    logit_gradient = torch.autograd.grad(logit.sum(), states)[0]
    target = mixture.residual_score(states, sigma=sigma, epsilon=0.15)
    torch.testing.assert_close(logit_gradient, target, atol=2e-10, rtol=2e-10)


def test_ratio_estimator_score_difference_has_expected_shape_and_graph() -> None:
    torch.manual_seed(3)
    model = RatioEstimator(2, hidden_dim=16, depth=2, frequencies=3)
    states = torch.randn(7, 2, requires_grad=True)
    sigma = torch.full((7,), 0.4)
    field = ratio_score_difference(model, states, sigma, create_graph=True)
    assert field.shape == states.shape
    loss = field.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_field_metrics_are_exact_for_identity_and_sign_flip() -> None:
    target = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    identity = field_metrics(target, target)
    opposite = field_metrics(-target, target)
    assert identity["relative_l2"] == 0.0
    assert identity["global_cosine"] == 1.0
    assert identity["norm_ratio"] == 1.0
    assert opposite["global_cosine"] == -1.0
    assert opposite["relative_l2"] == 2.0


def test_true_residual_score_small_pushforward_reduces_reverse_kl() -> None:
    mixture = _mixture()
    generator = torch.Generator().manual_seed(17)
    states = mixture.sample_clean(
        4096,
        epsilon=0.2,
        domain="fake",
        generator=generator,
    ).requires_grad_(True)
    field = mixture.residual_score(states, sigma=0.0, epsilon=0.2)
    result = pushforward_kl(
        mixture,
        states,
        field,
        sigma=0.0,
        epsilon=0.2,
        step_size=0.01,
    )
    assert result["positive_jacobian_fraction"] == 1.0
    assert result["kl_change"] < 0.0


def test_factorized_dsm_residual_cancels_common_output_exactly() -> None:
    torch.manual_seed(23)
    model = FactorizedDomainNoiseEstimator(
        2, hidden_dim=8, depth=1, frequencies=2
    )
    states = torch.randn(5, 2)
    sigma = torch.linspace(0.1, 0.5, 5)
    real_domain = torch.ones(5, dtype=torch.long)
    fake_domain = torch.zeros(5, dtype=torch.long)
    real_noise = model(states, sigma, real_domain)
    fake_noise = model(states, sigma, fake_domain)
    explicit = (fake_noise - real_noise) / sigma[:, None]
    factored = factorized_dsm_score_difference(model, states, sigma)
    torch.testing.assert_close(explicit, factored)
