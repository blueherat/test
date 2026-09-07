import pytest
import torch

from experiments.raev2_causal_reference import calibrated_clean, euler


@pytest.mark.parametrize('beta', [1., 1.78, 7., 12.])
def test_exact_finite_affine_reader(beta):
    # This identity retains all powers of beta present in the finite queries.
    dtype = torch.float64
    state = torch.tensor([[.4, -.7]], dtype=dtype)
    full = torch.tensor([[1., .2]], dtype=dtype)
    base = torch.tensor([[.1, -.3]], dtype=dtype)
    matrix = torch.tensor([[.5, .2], [-.3, 1.1]], dtype=dtype)
    t, s = .8, .3
    native = base+beta*(full-base)
    z0, z1 = euler(state, full, t, s), euler(state, native, t, s)
    b0, b1 = z0@matrix.T+.4, z1@matrix.T+.4
    actual = calibrated_clean(native, b1, b0, beta)
    expected = native-beta*((t-s)/t)*(native-full)@matrix.T
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)


def test_no_extra_write_and_constant_reader_are_exact_nulls():
    x = torch.tensor([[-2., .4]], dtype=torch.float32)
    f = torch.tensor([[.1, .8]], dtype=torch.float32)
    z0, z1 = euler(x, f, .5, .2), euler(x, f, .5, .2)
    assert torch.equal(calibrated_clean(f, z1, z0), f)
    constant = torch.ones_like(x)
    assert torch.equal(calibrated_clean(f, constant, constant), f)


def test_finite_nonlinear_response_is_not_replaced_by_a_derivative():
    x, f, g = (torch.tensor([[v]], dtype=torch.float64) for v in [0., 1., 7.])
    z0, z1 = euler(x, f, 1., .5), euler(x, g, 1., .5)
    actual = calibrated_clean(g, z1.square(), z0.square(), beta=7.)
    exact = g-7.*(z1.square()-z0.square())
    linearized = g-7.*(2*z0)*(z1-z0)
    assert torch.equal(actual, exact)
    assert not torch.allclose(actual, linearized)
