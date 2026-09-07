"""Analytic independent quadrature checks; these do not assert FID improvement."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
import torch

from experiments.raev2_guidance_quadrature import (
    MODES, extrapolation_weight, log_snr, native_clean, selected_component, step,
)


@pytest.mark.parametrize("previous,current,following", [(0.95, 0.9, 0.8), (0.7, 0.6, 0.4), (0.5002, 0.5001, 0.5)])
def test_matches_direct_integral_for_linear_clean(previous, current, following):
    # A manufactured nonautonomous denoiser with a closed variation-of-
    # constants solution. The numerical integration below does not use the
    # implemented phi function or extrapolation formula.
    slope, intercept, initial = 0.7, -0.2, 1.3
    clean_fn = lambda t: intercept + slope * log_snr(t)
    expected = following / current * initial + following * quad(
        lambda u: clean_fn(u) / u ** 2, following, current, epsabs=1e-12
    )[0]
    scalar = lambda v: torch.tensor([v], dtype=torch.float64)
    actual, _ = step(scalar(initial), scalar(clean_fn(current)), scalar(clean_fn(current)),
                     scalar(clean_fn(previous)), previous, current, following, mode="exponential_2m")
    assert actual.item() == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("triplet", [(None, 1., .99), (1., .99, .98), (.3, .2, 0.), (.2, .12, .08)])
def test_singular_endpoints_and_hard_event_bootstrap(triplet):
    assert extrapolation_weight(*triplet) == 0


def test_constant_prediction_is_exact_and_split_is_additive():
    rng = torch.Generator().manual_seed(81)
    state, full, base, old_full, old_base = [torch.randn(2, 5, generator=rng, dtype=torch.float64) for _ in range(5)]
    g, old_g = native_clean(full, base, .7).double(), native_clean(old_full, old_base, .8).double()
    outputs = {}
    for mode in MODES:
        comp, old = selected_component(full, g, mode), selected_component(old_full, old_g, mode)
        outputs[mode], _ = step(state, g, comp, old, .8, .7, .6, mode=mode)
        stationary, _ = step(state, g, comp, comp, .8, .7, .6, mode=mode)
        assert torch.allclose(stationary, .6 / .7 * state + (1 - .6 / .7) * g, atol=1e-7)
    assert torch.allclose(outputs['guidance_2m'] + outputs['full_2m'] - outputs['official'], outputs['exponential_2m'], atol=1e-7)


def test_official_keeps_native_bfloat16_arithmetic():
    rng = torch.Generator().manual_seed(82)
    f, b = [torch.randn(3, 17, generator=rng).bfloat16() for _ in range(2)]
    assert torch.equal(native_clean(f, b, .8), (b + 1.78 * (f - b)).float())
    assert torch.equal(native_clean(f, b, .08), f.float())


def test_manufactured_quadratic_prediction_second_order_convergence():
    # Interior interval avoids endpoint singularities. Exact previous
    # prediction is supplied for startup, and clean(lambda)=lambda^2.
    errors = []
    initial, lo, hi = 0.31, log_snr(.9), log_snr(.2)
    for count in (20, 40, 80):
        lambdas = np.linspace(lo, hi, count + 1)
        times = 1 / (1 + np.exp(lambdas))
        delta = lambdas[1] - lambdas[0]
        old_time = float(1 / (1 + math.exp(lo - delta)))
        old_clean = torch.tensor([(lo - delta) ** 2], dtype=torch.float64)
        z = torch.tensor([initial], dtype=torch.float64)
        for i, (t, s) in enumerate(zip(times[:-1], times[1:])):
            g = torch.tensor([lambdas[i] ** 2], dtype=torch.float64)
            z, _ = step(z, g, g, old_clean, old_time, float(t), float(s), mode="exponential_2m")
            old_time, old_clean = float(t), g
        expected = .2 / .9 * initial + .2 * quad(lambda u: log_snr(u) ** 2 / u ** 2, .2, .9, epsabs=1e-12)[0]
        errors.append(abs(z.item() - expected))
    assert errors[0] / errors[1] > 3.8 and errors[1] / errors[2] > 3.8
