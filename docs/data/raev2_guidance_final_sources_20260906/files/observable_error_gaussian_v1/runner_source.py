#!/usr/bin/env python3
"""Exact Gaussian audit of a density-weighted minimum guidance correction.

This is a mathematical test, not a RAEv2 model or FID experiment. The oracle
Gaussian density is known. The purpose is to distinguish weighted divergence
from ordinary Jacobian symmetry, and continuous marginal correctness from
finite-Euler accuracy. No method strength or schedule is selected by the test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_sylvester


MU = np.array([.7, -.4])
COV = np.array([[1.5, .4], [.4, .6]])
GRAD_ERROR = np.array([[.3, .12], [.12, -.15]])
BIAS = np.array([.15, -.1])
OMEGA = np.array([[0., -1.1], [1.1, 0.]])


def target(s):
    return s*MU, (1-s)**2*np.eye(2)+s*s*COV


def coefficients(s, method):
    mean, covariance = target(s)
    inverse = np.linalg.inv(covariance)
    oracle = (s*COV-(1-s)*np.eye(2))@inverse
    invisible = OMEGA@inverse
    error = GRAD_ERROR+invisible
    weighted_gradient = solve_sylvester(covariance, covariance,
                                         error@covariance+covariance@error.T)
    if method == "oracle":
        slope, bias = oracle, np.zeros(2)
    elif method == "baseline":
        slope, bias = oracle+error, BIAS
    elif method == "weighted_correction":
        slope, bias = oracle+error-weighted_gradient, np.zeros(2)
    elif method == "euclidean_symmetric_correction":
        slope, bias = oracle+error-(error+error.T)/2, np.zeros(2)
    elif method == "pure_invisible_error":
        slope, bias = oracle+invisible, np.zeros(2)
    else:
        raise ValueError(method)
    return slope, MU+bias-slope@mean


def covariance_sqrt(value):
    vals, vecs = np.linalg.eigh((value+value.T)/2)
    if vals.min() < -1e-10:
        raise ValueError("covariance is not positive semidefinite")
    return (vecs*np.sqrt(np.maximum(vals, 0)))@vecs.T


def distance(mean, covariance):
    root = covariance_sqrt(COV)
    raw = (mean-MU)@(mean-MU)+np.trace(covariance+COV-2*covariance_sqrt(root@covariance@root))
    if raw < -1e-9:
        raise ValueError("negative squared Wasserstein distance beyond roundoff")
    return {"mean_squared_error": float((mean-MU)@(mean-MU)),
            "covariance_frobenius_error": float(np.linalg.norm(covariance-COV)),
            "w2_squared": float(max(raw, 0)), "unclipped_w2_squared": float(raw)}


def continuous(method):
    def rhs(s, state):
        mean, covariance = state[:2], state[2:].reshape(2, 2)
        slope, intercept = coefficients(s, method)
        return np.r_[slope@mean+intercept, (slope@covariance+covariance@slope.T).ravel()]
    solution = solve_ivp(rhs, (0., 1.), np.r_[np.zeros(2), np.eye(2).ravel()],
                         method="DOP853", rtol=1e-11, atol=1e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    last = solution.y[:, -1]
    return {**distance(last[:2], last[2:].reshape(2, 2)), "rhs_calls": solution.nfev}


def euler(method, steps):
    # Same mathematical shift=8 family as RAEv2; NumPy FP64, not its BF16
    # production arithmetic. Grid refinement is a numerical comparison only.
    t = np.linspace(1., 0., steps+1)
    t = 8*t/(1+7*t)
    s = 1-t
    mean, covariance = np.zeros(2), np.eye(2)
    for current, following in zip(s[:-1], s[1:]):
        h = following-current
        slope, intercept = coefficients(current, method)
        matrix = np.eye(2)+h*slope
        mean = matrix@mean+h*intercept
        covariance = matrix@covariance@matrix.T
    return {"steps": steps, **distance(mean, covariance)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks = []
    for s in (0., .2, .5, .8, 1.):
        mean, covariance = target(s)
        invisible = OMEGA@np.linalg.inv(covariance)
        error = GRAD_ERROR+invisible
        gradient = solve_sylvester(covariance, covariance,
                                   error@covariance+covariance@error.T)
        weighted = error-gradient
        euclidean = error-(error+error.T)/2
        item = {"s": s,
                "weighted_gradient_matrix_error": float(np.linalg.norm(gradient-GRAD_ERROR)),
                "weighted_divergence_matrix_residual": float(np.linalg.norm(weighted@covariance+covariance@weighted.T)),
                "euclidean_divergence_matrix_residual": float(np.linalg.norm(euclidean@covariance+covariance@euclidean.T)),
                "gradient_invisible_inner_product": float(np.trace(GRAD_ERROR@covariance@invisible.T)),
                "full_error_energy": float(np.trace(error@covariance@error.T)+BIAS@BIAS),
                "observable_error_energy": float(np.trace(gradient@covariance@gradient.T)+BIAS@BIAS),
                "invisible_error_energy": float(np.trace(invisible@covariance@invisible.T))}
        item["orthogonal_energy_residual"] = item["full_error_energy"]-item["observable_error_energy"]-item["invisible_error_energy"]
        for name in ("weighted_gradient_matrix_error", "weighted_divergence_matrix_residual",
                     "gradient_invisible_inner_product", "orthogonal_energy_residual"):
            if abs(item[name]) > 1e-10:
                raise AssertionError((name, item))
        checks.append(item)
    methods = ("oracle", "baseline", "weighted_correction", "euclidean_symmetric_correction", "pure_invisible_error")
    results = {method: {"continuous": continuous(method),
                        "euler": [euler(method, steps) for steps in (100, 200, 400)]}
               for method in methods}
    for method in ("oracle", "weighted_correction", "pure_invisible_error"):
        if results[method]["continuous"]["covariance_frobenius_error"] > 1e-8:
            raise AssertionError("exact continuous marginal identity failed")
    if results["euclidean_symmetric_correction"]["continuous"]["w2_squared"] <= 1e-5:
        raise AssertionError("counterexample did not distinguish weighted and Euclidean symmetry")
    result = {"protocol": "weighted_poisson_gaussian_guidance_identity_v1", "complete": True,
              "target_mean": MU.tolist(), "target_covariance": COV.tolist(),
              "gradient_error": GRAD_ERROR.tolist(), "bias": BIAS.tolist(), "omega": OMEGA.tolist(),
              "checks": checks, "results": results,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "boundary": "Known 2D Gaussian oracle test only. The correction preserves density-weighted divergence-free motion. Continuous identity does not prove finite-Euler equality or any RAEv2/FID improvement.",
              "raev2_queries": 0, "training": False, "fid": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({m: results[m]["continuous"] for m in methods}), flush=True)


if __name__ == "__main__":
    main()
