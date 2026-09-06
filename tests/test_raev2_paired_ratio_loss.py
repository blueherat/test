import math
import numpy as np
import torch
import torch.nn.functional as F
from experiments.raev2_paired_ratio_loss import paired_scaled_logistic


def normal_quadrature():
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    return torch.from_numpy(nodes * math.sqrt(2)), torch.from_numpy(weights / math.sqrt(math.pi))


def test_exact_centered_bce_at_positive_signal():
    p = torch.tensor([-.2, 3., -4.], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([.8, -1., 2.], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([.0013, .3, .925], dtype=torch.float64)
    direct = (.5 * (F.softplus(-a * p) + F.softplus(a * q)) - math.log(2)) / a.square()
    torch.testing.assert_close(paired_scaled_logistic(p, q, a), direct, atol=2e-10, rtol=1e-10)


def test_known_gaussian_density_ratio_is_stationary():
    eps, weights = normal_quadrature()
    a = torch.full_like(eps, .4)
    t = 1 - a
    mean_p, mean_q = .4, 0.
    zp, zq = t * eps + a * mean_p, t * eps + a * mean_q
    # Exact log p_t/q_t divided by a for two Gaussian marginals.
    w = torch.tensor((mean_p - mean_q) / .6**2, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-.4 * (mean_p**2 - mean_q**2) / (2 * .6**2), dtype=torch.float64, requires_grad=True)
    objective = (weights * paired_scaled_logistic(w * zp + b, w * zq + b, a)).sum()
    gradients = torch.autograd.grad(objective, (w, b))
    assert max(float(g.abs()) for g in gradients) < 1e-12


def test_high_noise_limit_and_common_noise_cancellation():
    eps, weights = normal_quadrature()
    errors = []
    for signal in (.1, .01, .0001):
        a = torch.full_like(eps, signal)
        zp, zq = (1 - a) * eps + .4 * a, (1 - a) * eps
        w = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        objective = (weights * paired_scaled_logistic(w * zp, w * zq, a)).sum()
        derivative, = torch.autograd.grad(objective, w)
        errors.append(abs(float(derivative) - (.7 - .4) / 4))
        # At w=0, the per-pair gradient is (zq-zp)/(4a). Shared noise
        # removes its divergent random term exactly; independent noise does not.
        common_gradient = (zq - zp) / (4 * a)
        torch.testing.assert_close(common_gradient, torch.full_like(eps, -.1), atol=3e-12, rtol=0)
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 5e-5
