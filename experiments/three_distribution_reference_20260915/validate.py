"""CPU probability/optimization checks; these are NOT image-generation experiments."""

import json

import numpy as np
import torch
from scipy.optimize import minimize

from .objectives import (
    endpoint_responsibility,
    mixture_discriminator_logit,
    noisy_observation_log_weights,
    normalize_endpoint_log_weights,
    normalize_endpoint_weights,
    reference_fraction,
    weighted_endpoint_mse,
)


def kl(g, m):
    mask = g > 0
    return float(np.sum(g[mask] * np.log(g[mask] / m[mask])))


def projection(p, g, kappa):
    """Discrete form of the known AdaGAN optimal mixture component."""
    low, high = kappa, 1.0
    for _ in range(80):
        c = (low + high) / 2
        mass = np.maximum(c * g - (1 - kappa) * p, 0).sum()
        if mass < kappa:
            low = c
        else:
            high = c
    c = (low + high) / 2
    return np.maximum(c * g - (1 - kappa) * p, 0) / kappa, c


def em_step(p, g, q, kappa):
    tau = kappa * q / ((1 - kappa) * p + kappa * q)
    next_q = g * tau
    return next_q / next_q.sum()


def run():
    rng = np.random.default_rng(20260915)
    kappa = reference_fraction(0.8)
    p, g, q = rng.dirichlet(np.ones(9), size=3)
    priors = np.array([0.2, 0.5, 0.3])
    logits = torch.tensor(np.log(np.stack([p, g, q], axis=1) * priors), dtype=torch.float64)
    tau = endpoint_responsibility(logits, kappa, priors).numpy()
    expected_tau = kappa * q / ((1 - kappa) * p + kappa * q)
    np.testing.assert_allclose(tau, expected_tau, rtol=1e-12, atol=1e-12)
    shifted = endpoint_responsibility(logits + torch.arange(9)[:, None] * 100, kappa, priors)
    np.testing.assert_allclose(shifted, tau, rtol=1e-12, atol=1e-12)
    logit = mixture_discriminator_logit(logits, kappa, priors).numpy()
    np.testing.assert_allclose(logit, np.log(g / ((1 - kappa) * p + kappa * q)), atol=1e-12)
    changed = logits.clone()
    changed[:, 2] = torch.tensor(np.log(g * priors[2]))
    assert not torch.allclose(endpoint_responsibility(changed, kappa, priors), torch.from_numpy(tau))

    labels = np.arange(9) % 3
    weights, _ = normalize_endpoint_weights(tau, labels)
    for label in range(3):
        np.testing.assert_allclose(weights[labels == label].mean(), 1.0, atol=1e-12)
    prediction = torch.arange(18, dtype=torch.float32).reshape(9, 2).requires_grad_()
    target = torch.ones_like(prediction, requires_grad=True)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, requires_grad=True)
    loss = weighted_endpoint_mse(prediction, target, weight_tensor)
    loss.backward()
    assert prediction.grad is not None and target.grad is None and weight_tensor.grad is None
    torch.testing.assert_close(prediction.grad, 2 * (prediction.detach() - 1) * weight_tensor.detach()[:, None] / 18)

    max_opt_error, max_em_gap, largest_em_increase = 0.0, 0.0, 0.0
    for _ in range(12):
        p, g = rng.dirichlet(np.ones(7) * 2, size=2)
        qstar, c = projection(p, g, kappa)
        np.testing.assert_allclose(qstar.sum(), 1, atol=1e-12)
        mixture = (1 - kappa) * p + kappa * qstar
        np.testing.assert_allclose(mixture, np.maximum((1 - kappa) * p, c * g), atol=1e-12)
        objective = lambda candidate: kl(g, (1 - kappa) * p + kappa * candidate)
        jacobian = lambda candidate: -kappa * g / ((1 - kappa) * p + kappa * candidate)
        result = minimize(objective, g, jac=jacobian, method="SLSQP", bounds=[(0, 1)] * 7,
                          constraints={"type": "eq", "fun": lambda v: v.sum() - 1,
                                       "jac": lambda v: np.ones_like(v)},
                          options={"ftol": 1e-12, "maxiter": 2000})
        assert result.success, result.message
        max_opt_error = max(max_opt_error, float(np.max(np.abs(result.x - qstar))))
        current, previous = g.copy(), objective(g)
        for _ in range(6000):
            current = em_step(p, g, current, kappa)
            value = objective(current)
            largest_em_increase = max(largest_em_increase, value - previous)
            previous = value
        max_em_gap = max(max_em_gap, objective(current) - objective(qstar))
    assert max_opt_error < 2e-5, max_opt_error
    assert max_em_gap < 2e-6, max_em_gap
    assert largest_em_increase < 1e-12, largest_em_increase

    # The mixture commutes with a common noising kernel; projection generally does not.
    p, residual = rng.dirichlet(np.ones(5), size=2)
    g = (1 - kappa) * p + kappa * residual
    channel = rng.dirichlet(np.ones(4), size=5).T
    np.testing.assert_allclose(channel @ g, (1 - kappa) * (channel @ p) + kappa * (channel @ residual), atol=1e-14)
    qstar, c = projection(p, g, kappa)
    np.testing.assert_allclose(qstar, residual, atol=1e-12)
    p = np.array([0.75, 0.20, 0.05])
    g = np.array([0.08, 0.42, 0.50])
    channel = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    endpoint_target, _ = projection(p, g, kappa)
    noisy_target, _ = projection(channel @ p, channel @ g, kappa)
    noncommutation = float(np.max(np.abs(channel @ endpoint_target - noisy_target)))
    assert noncommutation > 1e-3

    # For an observed/noised mixture, EM reweights latent Q endpoints using
    # K.T @ (g_t/m_t). Validate its likelihood monotonicity independently.
    observed_g, observed_p = channel @ g, channel @ p
    current = g.copy()
    noisy_previous = kl(observed_g, (1 - kappa) * observed_p + kappa * (channel @ current))
    noisy_largest_increase = 0.0
    for _ in range(1000):
        observed_q = channel @ current
        ratios = observed_g / ((1 - kappa) * observed_p + kappa * observed_q)
        current *= channel.T @ ratios
        current /= current.sum()
        value = kl(observed_g, (1 - kappa) * observed_p + kappa * (channel @ current))
        noisy_largest_increase = max(noisy_largest_increase, value - noisy_previous)
        noisy_previous = value
    assert noisy_largest_increase < 1e-12
    optimal_noisy = minimize(
        lambda candidate: kl(observed_g, (1 - kappa) * observed_p + kappa * (channel @ candidate)),
        g, method="SLSQP", bounds=[(0, 1)] * 3,
        constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    assert optimal_noisy.success
    noisy_gap = noisy_previous - optimal_noisy.fun
    assert abs(noisy_gap) < 1e-8
    channel_logits = torch.tensor(np.log(np.stack([observed_p, observed_g, channel @ g], axis=1)), dtype=torch.float64)
    example_draws = torch.stack([channel_logits, channel_logits.flip(0)], dim=1)
    sampled_log_weights = noisy_observation_log_weights(example_draws, kappa)
    direct_ratio = observed_g / ((1 - kappa) * observed_p + kappa * (channel @ g))
    np.testing.assert_allclose(sampled_log_weights.exp(), (direct_ratio + direct_ratio[::-1]) / 2, atol=1e-12)
    stable_weights, _ = normalize_endpoint_log_weights(np.array([1000., 1001., -1000., -1001.]), np.array([0, 0, 1, 1]))
    assert np.isfinite(stable_weights).all()
    np.testing.assert_allclose(stable_weights.reshape(2, 2).mean(1), np.ones(2), atol=1e-12)

    # Ideal first EM step from Q=P is G. Small-error amplification is gradual.
    p = np.array([0.2, 0.3, 0.5])
    direction = np.array([0.1, -0.2, 0.1])
    epsilon = 1e-5
    g = p + epsilon * direction
    np.testing.assert_allclose(em_step(p, g, p, kappa), g, atol=1e-14)
    current, amplification = p.copy(), 0.0
    recurrence_error = 0.0
    for _ in range(5):
        current = em_step(p, g, current, kappa)
        amplification = 1 + (1 - kappa) * amplification
        recurrence_error = max(recurrence_error, float(np.max(np.abs(current - p - epsilon * amplification * direction))))
    assert recurrence_error < 5e-10

    return dict(
        passed=True, kind="CPU identities and convex probability optimization only",
        generated_images=0, gpu_training_started=False,
        unequal_source_priors_corrected=True, endpoint_feedback_changes_weights=True,
        classwise_full_bank_normalization=True, weights_and_targets_detached=True,
        projection_vs_independent_optimizer_max_error=max_opt_error,
        ideal_em_max_objective_increase=largest_em_increase,
        ideal_em_max_suboptimality=max_em_gap,
        exact_mixture_commutes_with_common_channel=True,
        projection_noising_noncommutation=noncommutation,
        noisy_observation_em_max_objective_increase=noisy_largest_increase,
        noisy_observation_em_suboptimality=noisy_gap,
        noisy_observation_uses_weak_endpoints=True,
        log_weight_normalization_handles_large_logits=True,
        first_order_em_recurrence_max_error=recurrence_error,
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
