"""Exact finite regression identities and loss gradients; no proxy selection."""
import numpy as np
import torch


def analytic_checks():
    rng = np.random.default_rng(642026)
    mass = rng.uniform(.1, 1., 60); mass /= mass.sum()
    h = np.arange(60) % 5
    y, strong = rng.normal(size=(2, 60, 3))
    a = .8
    conditional = lambda v: np.array([np.sum(mass[h == k, None] * v[h == k], axis=0) / mass[h == k].sum() for k in range(5)])[h]
    w0 = conditional(y)
    bias = conditional(strong-y)
    wstar = conditional(strong) + bias/a
    risk = lambda w: np.sum(mass[:,None] * ((1+a)*strong-a*w-y)**2)
    expected = (1+a)**2 * np.sum(mass[:,None]*bias**2)
    np.testing.assert_allclose(risk(w0)-risk(wstar), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wstar-w0, (1+1/a)*bias, rtol=1e-12, atol=1e-12)
    # Perfect conditional mean on a finer partition: the tower property leaves
    # the old and new shallow optima equal, including for a nontrivial target.
    finer = np.arange(60) % 15
    exact = np.array([np.sum(mass[finer == k,None]*y[finer == k],axis=0)/mass[finer == k].sum() for k in range(15)])[finer]
    np.testing.assert_allclose(conditional(exact-y), 0., atol=1e-15)
    s = torch.tensor(strong); truth = torch.tensor(y)
    weak = torch.tensor(w0, requires_grad=True)
    direct = ((weak - (s+(s-truth)/a))**2).mean()
    composite = (((s+a*(s-weak)-truth)/a)**2).mean()
    dg, = torch.autograd.grad(direct, weak)
    cg, = torch.autograd.grad(composite, weak)
    torch.testing.assert_close(direct, composite, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(dg, cg, rtol=1e-14, atol=1e-14)
    return dict(passed=True, risk_identity_absolute_error=float(abs(risk(w0)-risk(wstar)-expected)),
        tower_property_exact=True, loss_and_gradient_equivalent=True,
        note='These identities concern teacher-state quadratic risk, not FID or rollout density.')
