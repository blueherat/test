import pytest
import torch

from experiments.raev2_full_read_after_write import euler, prewritten_state, reread_clean


@pytest.mark.parametrize('beta', [1., 1.78, 7., 12.])
def test_cached_identity_and_affine_finite_feedback_at_large_strength(beta):
    generator = torch.Generator().manual_seed(972)
    z, b = torch.randn(2, 5, generator=generator, dtype=torch.float64)
    matrix = torch.randn(5, 5, generator=generator, dtype=torch.float64)
    bias = torch.randn(5, generator=generator, dtype=torch.float64)
    f = matrix@z+bias
    guided = b+beta*(f-b)
    t, s = .83, .27
    alpha, q = s/t, 1-s/t
    written = prewritten_state(z, f, guided, t, s)
    torch.testing.assert_close(alpha*written+q*f, euler(z, guided, t, s), atol=2e-14, rtol=2e-14)
    actual = euler(z, reread_clean(guided, matrix@written+bias, f), t, s)
    expected = euler(z, guided, t, s)+q*((t-s)/s)*(matrix@(guided-f))
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)


def test_finite_quadratic_read_retains_the_large_extrapolation_term():
    z = torch.zeros(2, dtype=torch.float64)
    full = torch.zeros_like(z)
    guided = torch.tensor([0., 6.], dtype=torch.float64)  # (beta-1)*gap, beta=7
    written = prewritten_state(z, full, guided, .8, .4)
    def reader(x):
        return torch.stack([x[0]+2*x[1]-x[1]**2, x[1]])
    actual = euler(z, reread_clean(guided, reader(written), full), .8, .4)
    torch.testing.assert_close(actual, torch.tensor([-12., 6.], dtype=torch.float64))
    # A single Jacobian at z would instead predict a positive first component.
    linear = euler(z, guided+torch.tensor([12., 6.], dtype=torch.float64), .8, .4)
    assert linear[0] > 0 and actual[0] < 0


def test_zero_read_response_preserves_native_fp32_arithmetic_exactly():
    generator = torch.Generator().manual_seed(973)
    z, full, guided = torch.randn(3, 64, generator=generator)
    effective = reread_clean(guided, full, full)
    assert torch.equal(effective, guided)
    assert torch.equal(euler(z, effective, .25, .1), euler(z, guided, .25, .1))
    assert torch.equal(prewritten_state(z, full, full, .25, .1), z)


def test_terminal_state_has_no_finite_prewrite_representation():
    z = torch.ones(3)
    with pytest.raises(ValueError, match='following'):
        prewritten_state(z, z, 2*z, .1, 0.)
