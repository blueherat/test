"""Direct truncated Feynman--Kac correction; no learned value or stop-gradient drift."""
import math
import torch
from torch.utils.checkpoint import checkpoint


def path_logweight(clean_fn, z, noise_time, epsilon, *, weight=1.78):
    """Two left quadrature nodes over [0.75 tau, tau], tau=(t/(1-t))**2.

    clean_fn(z,t) returns calibrated strong and weak clean estimates. The
    auxiliary law is ordinary IG reverse *heat SDE*, never the corrected ODE.
    The second, final SDE innovation integrates out of left quadrature and
    therefore is not sampled. epsilon drives the one relevant transition.
    """
    tau = (noise_time / (1.0 - noise_time)) ** 2
    delta = tau / 8.0
    y = z / (1.0 - noise_time)
    logweight = torch.zeros(len(z), device=z.device, dtype=z.dtype)
    for j in range(2):
        u = tau - j * delta
        root = math.sqrt(u)
        t = root / (1.0 + root)
        a = 1.0 - t
        f, b = clean_fn(a * y, t)
        f, b = f.float(), b.float()
        gap = (f - b) / u
        potential = 0.5 * weight * (weight - 1.0) * gap.flatten(1).square().sum(1)
        logweight = logweight + delta * potential
        if j == 0:
            clean_ig = b + weight * (f - b)
            score = (clean_ig - y) / u
            y = y + delta * score + math.sqrt(delta) * epsilon
    return logweight


def correction(clean_fn, z, t, epsilon, *, weight=1.78, checkpoint_model=True):
    """Pathwise gradient of log[(exp(L+) + exp(L-))/2].

    Sequential gradients keep only one path alive. Detached softmax weights
    implement the exact first derivative of this finite-particle log moment;
    no path drift/head output is detached. No temperature or dimension scaling.
    """
    logs, gradients = [], []
    with torch.enable_grad():
        for sign in (1.0, -1.0):
            leaf = z.detach().requires_grad_(True)
            def forward(q, time):
                if checkpoint_model:
                    return checkpoint(lambda x: clean_fn(x, time), q, use_reentrant=False,
                                      preserve_rng_state=False)
                return clean_fn(q, time)
            value = path_logweight(forward, leaf, float(t), sign * epsilon, weight=weight)
            gradient, = torch.autograd.grad(value.sum(), leaf)
            logs.append(value.detach())
            gradients.append(gradient.detach())
    logs = torch.stack(logs)
    weights = torch.softmax(logs, dim=0)
    shape = (len(z),) + (1,) * (z.ndim - 1)
    gradient = sum(weights[k].reshape(shape) * gradients[k] for k in range(2))
    # Linear bridge probability-flow drift correction - beta(t)/2 * grad_z log C.
    drift = -float(t) / (1.0 - float(t)) * gradient
    info = dict(logweight_mean=float(logs.mean()), logweight_max=float(logs.max()),
                particle_ess=float(weights.square().sum(0).reciprocal().mean()),
                logweight_difference=float((logs[0]-logs[1]).abs().mean()),
                gradient_rms=float(gradient.square().mean().sqrt()),
                correction_rms=float(drift.square().mean().sqrt()))
    if not torch.isfinite(drift).all() or not torch.isfinite(logs).all():
        raise FloatingPointError('Nonfinite FK weights/gradient; no clipping fallback.')
    return drift, info
