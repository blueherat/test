"""Independent Gaussian density and weak-loss check for the actual-law limit."""
import numpy as np
import torch


def test_actual_initial_velocity_boundary_matches_density_and_weak_loss():
    # Initial native clean G(z)=m+gamma*z is deliberately nonconstant.
    m, gamma, real_mean = .3, .4, -.2
    z = np.array([-2., -.4, .7, 2.1])
    expected = gamma + (real_mean-m)*z - gamma*z*z
    a = 1e-6
    real_variance = (1-a)**2 + a*a*1.7
    fake_variance = (1+a*(gamma-1))**2
    log_ratio = (-.5*np.log(real_variance) - (z-a*real_mean)**2/(2*real_variance)
                 + .5*np.log(fake_variance) + (z-a*m)**2/(2*fake_variance))
    np.testing.assert_allclose(log_ratio/a, expected, atol=1e-5, rtol=1e-5)
    # The same function is stationary for the paired high-noise weak loss.
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    x = torch.tensor(nodes*np.sqrt(2), dtype=torch.float64)
    w = torch.tensor(weights/np.sqrt(np.pi), dtype=torch.float64)
    b = torch.tensor([0., real_mean-m, -gamma], dtype=torch.float64, requires_grad=True)
    f = b[0] + b[1]*x + b[2]*(x*x-1)
    grad_f = b[1] + 2*b[2]*x
    risk = (w*(.25*grad_f*(m+gamma*x-real_mean) + .125*f*f)).sum()
    gradient, = torch.autograd.grad(risk, b)
    assert gradient.abs().max().item() < 1e-12
