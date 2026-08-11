from __future__ import annotations

import torch

from experiments.run_prediction_target_bayes_oracle_v5 import (
    ResidualDenoiseMLP,
    TangentGaussianMixture,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import clean_from_output


def make_mixture(
    *, D: int = 8, sigma_tangent: float = 0.25, sigma_normal: float = 0.05
) -> TangentGaussianMixture:
    return TangentGaussianMixture(
        D=D,
        components=4,
        curvature=0.4,
        frequency_scale=3.0,
        center_rms=0.7,
        sigma_tangent=sigma_tangent,
        sigma_normal=sigma_normal,
        seed=17,
        device=torch.device("cpu"),
    )


def test_bayes_posterior_at_noise_endpoint_is_prior_mean() -> None:
    mixture = make_mixture()
    generator = torch.Generator().manual_seed(19)
    x_t = torch.randn(11, mixture.D, generator=generator)
    t = torch.ones(11)
    posterior, weights = mixture.posterior_clean(x_t, t, return_weights=True)
    expected = mixture.means.mean(dim=0, keepdim=True).expand_as(posterior)
    torch.testing.assert_close(posterior, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        weights,
        torch.full_like(weights, 1.0 / mixture.components),
        atol=2e-6,
        rtol=2e-6,
    )


def test_isotropic_mixture_bayes_formula_matches_manual_expression() -> None:
    sigma = 0.2
    mixture = make_mixture(D=6, sigma_tangent=sigma, sigma_normal=sigma)
    generator = torch.Generator().manual_seed(23)
    x_t = torch.randn(7, mixture.D, generator=generator)
    t = torch.linspace(0.1, 0.9, len(x_t))
    actual = mixture.posterior_clean(x_t, t)

    a = 1.0 - t
    variance = a.square() * sigma**2 + t.square()
    residual = x_t[:, None, :] - a[:, None, None] * mixture.means[None]
    logits = -0.5 * (
        residual.square().sum(dim=2) / variance[:, None]
        + mixture.D * variance.log()[:, None]
    )
    weights = torch.softmax(logits, dim=1)
    coefficient = (a * sigma**2 / variance)[:, None, None]
    component_mean = mixture.means[None] + coefficient * residual
    expected = torch.einsum("bk,bkd->bd", weights, component_mean)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


def test_all_exact_parameterizations_convert_to_same_bayes_clean_mean() -> None:
    mixture = make_mixture()
    generator = torch.Generator().manual_seed(29)
    x, eps, t, x_t, _ = mixture.noised_batch(
        17, t_min=0.05, t_max=0.95, generator=generator
    )
    del x, eps
    bayes = mixture.posterior_clean(x_t, t)
    outputs = {
        "x": bayes,
        "v": (x_t - bayes) / t[:, None],
        "eps": (x_t - (1.0 - t[:, None]) * bayes) / t[:, None],
    }
    for target, output in outputs.items():
        recovered = clean_from_output(output, x_t, t, target, 1e-6)
        torch.testing.assert_close(recovered, bayes, atol=3e-6, rtol=3e-6)


def test_bayes_risk_decomposes_into_irreducible_and_excess_risk() -> None:
    mixture = make_mixture(D=6)
    generator = torch.Generator().manual_seed(31)
    x, _, t, x_t, _ = mixture.noised_batch(
        40000, t_min=0.1, t_max=0.9, generator=generator
    )
    bayes = mixture.posterior_clean(x_t, t)
    candidate = bayes + 0.08 * torch.tanh(x_t)
    paired = (candidate - x).double().square().mean()
    irreducible = (bayes - x).double().square().mean()
    excess = (candidate - bayes).double().square().mean()
    assert abs(float(paired - irreducible - excess)) < 3e-4


def test_logit_normal_time_sampler_uses_noise_time_coordinate() -> None:
    mixture = make_mixture(D=6)
    generator = torch.Generator().manual_seed(37)
    _, _, time, _, _ = mixture.noised_batch(
        20000,
        t_min=1e-4,
        t_max=1.0 - 1e-4,
        time_sampler="logit_normal",
        time_logit_mean=0.8,
        time_logit_std=0.8,
        generator=generator,
    )
    assert 0.66 < float(time.mean()) < 0.70
    assert bool((time >= 1e-4).all())
    assert bool((time <= 1.0 - 1e-4).all())


def test_full_rank_state_skip_survives_hidden_bottleneck() -> None:
    model = ResidualDenoiseMLP(
        D=12, hidden=4, depth=3, time_dim=4, state_skip=True
    )
    with torch.no_grad():
        model.out_proj.weight.zero_()
        model.out_proj.bias.zero_()
    generator = torch.Generator().manual_seed(37)
    x = torch.randn(5, 12, generator=generator)
    t = torch.rand(5, generator=generator)
    output = model(x, t)
    torch.testing.assert_close(output, x, atol=1e-6, rtol=1e-6)


def test_tangent_normal_split_is_orthogonal_for_selected_components() -> None:
    mixture = make_mixture(D=10)
    generator = torch.Generator().manual_seed(41)
    vector = torch.randn(9, mixture.D, generator=generator)
    component = torch.arange(9) % mixture.components
    tangent, normal = mixture.split_by_component(vector, component)
    torch.testing.assert_close(tangent + normal, vector, atol=2e-6, rtol=2e-6)
    inner = (tangent.double() * normal.double()).sum(dim=1)
    torch.testing.assert_close(inner, torch.zeros_like(inner), atol=2e-6, rtol=0)
