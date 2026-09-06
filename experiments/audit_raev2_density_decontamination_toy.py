#!/usr/bin/env python3
"""Exact Gaussian-mixture checks for transported density decontamination.

The mixture premise is constructed, not inferred from RAEv2. These checks
verify equations and discretization against known densities and terminal
quantiles. They do not constitute image-quality evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

W = 1.78


def moments(z, t, means, variances, weights):
    """Exact density, posterior mean, and trace of posterior covariance."""
    alpha = 1-t
    variance = alpha**2*variances+t**2
    residual = z[:, None, :]-alpha*means[None]
    logits = (weights.log()[None]-.5*(residual.square()/variance[None]
              + torch.log(2*torch.pi*variance)[None]).sum(-1))
    log_density = torch.logsumexp(logits, dim=1)
    responsibility = logits.softmax(1)
    component_mean = means[None]+alpha*variances[None]/variance[None]*residual
    component_variance = variances*t**2/variance
    mean = (responsibility[:, :, None]*component_mean).sum(1)
    trace = (responsibility*(component_variance.sum(-1)[None]
             +(component_mean-mean[:, None]).square().sum(-1))).sum(1)
    return log_density, mean, trace


def oracle(z, t, *, dimension=1):
    if dimension == 1:
        means = torch.tensor([[1.], [-.5]], dtype=z.dtype)
        variances = torch.tensor([[.16], [1.96]], dtype=z.dtype)
    else:
        means = torch.tensor([[1., -1.], [-.5, .7]], dtype=z.dtype)
        variances = torch.tensor([[.16, .49], [1.96, 1.21]], dtype=z.dtype)
    weights = torch.tensor([1/W, 1-1/W], dtype=z.dtype)
    lf, f, cf = moments(z, t, means, variances, weights)
    lb, b, cb = moments(z, t, means[1:], variances[1:], torch.ones(1, dtype=z.dtype))
    _, target, _ = moments(z, t, means[:1], variances[:1], torch.ones(1, dtype=z.dtype))
    return lf, lb, f, b, (1-t)/t**2*(cf-cb), target


def identity_audit():
    generator = torch.Generator().manual_seed(202609073)
    max_gain_error, max_derivative_error = 0., 0.
    for current in (1., .93, .71, .39, .14):
        center = (1-current)*torch.tensor([1., -1.], dtype=torch.float64)
        variance = (1-current)**2*torch.tensor([.16, .49], dtype=torch.float64)+current**2
        z = (center+variance.sqrt()*torch.randn((32, 2), generator=generator,
                                              dtype=torch.float64)).requires_grad_(True)
        t = torch.tensor(current, dtype=torch.float64, requires_grad=True)
        lf, lb, f, b, divergence, target = oracle(z, t, dimension=2)
        ell = lb-lf
        denominator = W-(W-1)*ell.exp()
        if not bool((denominator > 0).all()):
            raise AssertionError("constructed mixture lost positivity")
        beta = (W-1)*ell.exp()/denominator
        reconstructed = f+beta[:, None]*(f-b)
        # Some tails have tiny target mass, hence cancellation amplifies FP64
        # roundoff. Record the actual absolute error without changing weights.
        max_gain_error = max(max_gain_error, float((reconstructed-target).abs().max().detach()))
        guided = f+.37*(f-b)+.1*z.sin()  # deliberately arbitrary path
        dot_z = (z-guided)/t
        gradient = torch.autograd.grad(ell.sum(), z, retain_graph=True)[0]
        time_derivatives = torch.stack([
            torch.autograd.grad(ell[index], t, retain_graph=True)[0] for index in range(len(z))])
        actual = time_derivatives+(gradient*dot_z).sum(1)
        gap = f-b
        formula = (gap*(z-(1-t)*(b+f-guided))).sum(1)/t**3-divergence/t
        max_derivative_error = max(max_derivative_error, float((actual-formula).abs().max().detach()))
    if max_gain_error > 1e-7 or max_derivative_error > 1e-9:
        raise AssertionError((max_gain_error, max_derivative_error))
    return {"maximum_guided_posterior_mean_error": max_gain_error,
            "maximum_log_ratio_total_derivative_error": max_derivative_error}


def integrate(steps, mode):
    from experiments.raev2_density_decontamination import (
        current_beta, frozen_coefficients_from_fields, implicit_density_step,
    )
    # Deterministic Gaussian quantiles avoid an empirical moment/FID estimate.
    quantiles = (torch.arange(128, dtype=torch.float64)+.5)/128
    initial = (2.**.5*torch.erfinv(2*quantiles-1))[:, None]
    z = initial.clone()
    ell = torch.zeros(128, dtype=torch.float64)
    grid = torch.linspace(1, 0, steps+1, dtype=torch.float64)
    grid = 8*grid/(1+7*grid)
    maximum_ratio_error = 0.
    maximum_beta = 0.
    for t, s in zip(grid[:-1].tolist(), grid[1:].tolist()):
        lf, lb, f, b, divergence, target = oracle(z, t)
        if mode == "target_oracle":
            guided = target
        elif mode == "ordinary":
            guided = f+(W-1)*(f-b) if .1 <= t <= 1 else f
        elif mode == "density_transport":
            maximum_ratio_error = max(maximum_ratio_error, float((ell-(lb-lf)).abs().max()))
            if s > 0:
                coefficients = frozen_coefficients_from_fields(z, f, b, divergence, t, s)
                result = implicit_density_step(ell, coefficients.A, coefficients.Bcoef, W)
                ell, beta = result.ell_new, result.beta
            else:
                beta = current_beta(ell, W)
            guided = f+beta[:, None]*(f-b)
            maximum_beta = max(maximum_beta, float(beta.max()))
        else:
            raise ValueError(mode)
        z = z-(t-s)*(z-guided)/t
    target_quantiles = 1+.4*initial
    return {"steps": steps, "mode": mode,
            "terminal_quantile_RMSE": float((z-target_quantiles).square().mean().sqrt()),
            "terminal_mean": float(z.mean()), "terminal_std": float(z.std(unbiased=False)),
            "maximum_log_ratio_error_at_query_states": maximum_ratio_error,
            "maximum_guidance_beta": maximum_beta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identities = identity_audit()
    rows = [integrate(steps, mode) for steps in (100, 400, 1600)
            for mode in ("target_oracle", "ordinary", "density_transport")]
    transport = [row for row in rows if row["mode"] == "density_transport"]
    if not transport[-1]["terminal_quantile_RMSE"] < transport[0]["terminal_quantile_RMSE"]:
        raise AssertionError("transported ratio discretization did not improve on refinement")
    payload = {"complete": True, "w": W, "mixture_is_constructed": True,
               "dimension_identity_audit": 2, "dimension_transport_audit": 1,
               "identities": identities, "terminal_quantile_experiments": rows,
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__), ROOT / "experiments/raev2_density_decontamination.py")},
               "claim_boundary": "exact constructed density model and numerical consistency only; no RAEv2 mixture or FID claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
