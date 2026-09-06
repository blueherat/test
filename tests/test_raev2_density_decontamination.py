import math

import numpy as np
import pytest
import torch

from experiments.raev2_density_decontamination import (
    current_beta, frozen_coefficients_from_fields, frozen_step_coefficients,
    implicit_density_step, log1mexp,
)


def test_batch_roots_preserve_positivity_and_solve_original_equation():
    ell = torch.tensor([-8., -1., 0., .5], dtype=torch.float64)
    A = torch.tensor([-20., 0., 1., 50.], dtype=torch.float64)
    B = torch.tensor([.001, .5, 20., 5.], dtype=torch.float64)
    result = implicit_density_step(ell, A, B)
    L = math.log(1.78/.78)
    assert torch.all(result.ell_new < L)
    assert torch.all(1.78-.78*result.ell_new.exp() > 0)
    torch.testing.assert_close(result.beta, current_beta(result.ell_new), rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(result.ell_new+B*result.beta, ell+A, rtol=2e-14, atol=2e-14)
    assert 0 < result.iterations <= 128
    assert result.ell_new.dtype == torch.float64


def test_zero_B_is_exact_and_inadmissible_rhs_fails_instead_of_clipping():
    old = torch.tensor([-.4, .2], dtype=torch.float64)
    A = torch.tensor([.1, -.7], dtype=torch.float64)
    result = implicit_density_step(old, A, 0.)
    assert torch.equal(result.ell_new, old+A)
    assert torch.equal(result.residual, torch.zeros_like(old))
    assert result.iterations == 0
    for rhs in (math.log1p(1/.78), 100.):
        with pytest.raises(ValueError, match="no admissible root"):
            implicit_density_step(0., rhs, 0.)


def test_w1_is_exact_zero_guidance_without_a_barrier():
    old = np.array([-100., 0., 100.])
    A = np.array([3., 10000., -2.])
    result = implicit_density_step(old, A, np.array([0., 1., 1e20]), w=1.)
    assert torch.equal(result.ell_new, torch.from_numpy(old+A))
    assert torch.equal(result.beta, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(current_beta(result.ell_new, w=1.), result.beta)
    full = torch.tensor([2., -3., 4.], dtype=torch.float64)
    assert torch.equal(full+result.beta*torch.tensor([10., -20., 30.]), full)


def test_large_rhs_has_finite_huge_uncapped_beta_with_stable_barrier_distance():
    result = implicit_density_step(0., 1e10, .3)
    assert torch.isfinite(result.beta) and result.beta > 1e10
    assert result.ell_new < math.log1p(1/.78)
    assert result.barrier_distance > 0
    torch.testing.assert_close(result.beta, 1/torch.expm1(result.barrier_distance), rtol=1e-14, atol=0)
    assert abs(float(result.residual)) < 1e-4
    # Beyond float64's ability to represent ell<L, fail explicitly; no cap or
    # nextafter substitution can produce the requested mathematical root.
    with pytest.raises(FloatingPointError, match="not representable"):
        implicit_density_step(0., 1e30, 1.)


def test_mixed_zero_positive_batch_and_scalar_broadcasting():
    result = implicit_density_step(-1., torch.tensor([0., 4.]), torch.tensor([0., 2.]))
    assert result.ell_new.shape == (2,)
    assert result.ell_new[0] == -1
    torch.testing.assert_close(result.ell_new[1]+2*result.beta[1], torch.tensor(3., dtype=torch.float64))


def test_frozen_integrals_match_independent_quadrature_and_near_equal_times():
    t, s = .83, .27
    divergence, dz, db, norm2 = 4., -2., 3., 5.
    coefficients = frozen_step_coefficients(divergence, dz, db, norm2, t, s)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    u = s+(nodes+1)*(t-s)/2
    integrate = lambda values: float(np.dot(weights, values)*(t-s)/2)
    assert math.isclose(float(coefficients.A), integrate(divergence/u-dz/u**3+db*(1-u)/u**3), rel_tol=2e-14)
    assert math.isclose(float(coefficients.Bcoef), integrate(norm2*(1-u)/u**3), rel_tol=2e-14)
    near = frozen_step_coefficients(0., 0., 0., 1., 1., 1.-1e-10)
    assert near.K > 0
    assert math.isclose(float(near.K), (1-(1-1e-10))**2/(2*(1-1e-10)**2), rel_tol=1e-14)
    with pytest.raises(ValueError, match="s=0"):
        frozen_step_coefficients(divergence, dz, db, norm2, t, 0.)


def test_field_coefficients_use_full_coordinate_sums_and_promote_precision():
    state = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)/7
    full, base = .5*state, -.25*state
    gap = full.double()-base.double()
    expected = frozen_step_coefficients(torch.tensor([2., 3.]),
        (gap*state.double()).flatten(1).sum(1), (gap*base.double()).flatten(1).sum(1),
        gap.square().flatten(1).sum(1), .8, .6)
    actual = frozen_coefficients_from_fields(state, full, base, torch.tensor([2., 3.]), .8, .6)
    assert torch.equal(actual.A, expected.A)
    assert torch.equal(actual.Bcoef, expected.Bcoef)


def test_domain_validation_and_stable_log1mexp():
    near = torch.tensor(-1e-20, dtype=torch.float64)
    torch.testing.assert_close(log1mexp(near), torch.tensor(math.log(1e-20), dtype=torch.float64))
    for function in (lambda: current_beta(math.log1p(1/.78)),
                     lambda: implicit_density_step(0., 0., -1.),
                     lambda: current_beta(0., w=.99),
                     lambda: current_beta(float("nan"))):
        with pytest.raises(ValueError):
            function()
