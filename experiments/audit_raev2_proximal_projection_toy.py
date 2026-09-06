"""CPU finite-step audit of diagonal convex proximal error correction.

Fixed examples, official mathematical 100-step shifted grid, no guidance
scale/window sweep. Exact Gaussian moments or product quantile quadrature
give actual W2; per-step Lipschitz/teacher-defect recurrences give bounds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.special import roots_hermitenorm


def time_grid() -> np.ndarray:
    base = np.linspace(1.0, 0.0, 101)
    return 8.0 * base / (1.0 + 7.0 * base)


def projection_statistics(mean_r, variance_r, covariance_ry, variance_y):
    """Population closed-form cone fit, then exact proximal defect risk."""
    slope = np.maximum(covariance_ry / variance_y, 0.0)
    before = float(np.sum(variance_r + mean_r**2))
    removed = float(np.sum(mean_r**2 + slope**2 * variance_y))
    residual_variance = variance_r - 2*slope*covariance_ry + slope**2*variance_y
    after = float(np.sum(residual_variance / (1.0 + slope)**2))
    assert abs(float(np.sum(residual_variance)) - (before - removed)) < 1e-10
    assert after <= before - removed + 1e-10
    return slope, before, max(after, 0.0), removed


def gaussian_distance_squared(mean, covariance, target_mean, target_covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(target_covariance)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    cross = np.linalg.eigvalsh(root @ covariance @ root)
    return max(float(
        np.sum((mean-target_mean)**2) + np.trace(covariance)
        + np.trace(target_covariance) - 2*np.sqrt(np.maximum(cross, 0)).sum()
    ), 0.0)


def gaussian_bridge(oracle: bool = False) -> dict:
    data_mean = np.array([1.0, -1.0])
    data_variance = np.array([0.5, 2.0])
    bias = np.zeros(2) if oracle else np.array([0.3, -0.2])
    gain_error = np.zeros(2) if oracle else np.array([0.6, 0.4])
    means = [np.zeros(2), np.zeros(2)]
    variances = [np.ones(2), np.ones(2)]
    bounds = [0.0, 0.0]
    step_rows = []
    for t, s in zip(time_grid()[:-1], time_grid()[1:]):
        delta, u, v = t-s, 1-t, 1-s
        current_mean, target_mean = u*data_mean, v*data_mean
        current_variance = u*u*data_variance + t*t
        target_variance = v*v*data_variance + s*s
        covariance_zy = u*v*data_variance + t*s
        true_gain = (u*data_variance-t)/current_variance
        multiplier = 1.0 + delta*(true_gain + gain_error)
        intercept = target_mean + delta*bias - multiplier*current_mean
        mean_r = delta*bias
        variance_r = multiplier**2*current_variance + target_variance - 2*multiplier*covariance_zy
        covariance_ry = multiplier*covariance_zy - target_variance
        slope, risk_old, risk_new, removed = projection_statistics(
            mean_r, variance_r, covariance_ry, target_variance,
        )
        proximal_multiplier = multiplier/(1.0+slope)
        proximal_intercept = (intercept-mean_r+slope*target_mean)/(1.0+slope)
        # Actual single-step marginal theorem: starting from the TRUE p_t,
        # the correction caps only expansion beyond the Gaussian OT gain.
        ot_multiplier = np.sqrt(target_variance/current_variance)
        cap_multiplier = target_variance/covariance_zy
        assert np.all(multiplier >= 0)
        assert np.all(cap_multiplier >= ot_multiplier-1e-14)
        assert np.allclose(proximal_multiplier, np.minimum(multiplier, cap_multiplier),
                           atol=1e-14, rtol=1e-14)
        marginal_old = float(np.sum(mean_r**2 + (
            multiplier*np.sqrt(current_variance)-np.sqrt(target_variance))**2))
        marginal_new = float(np.sum((
            proximal_multiplier*np.sqrt(current_variance)-np.sqrt(target_variance))**2))
        assert marginal_new <= marginal_old + 1e-12
        lip_old, lip_new = float(np.max(np.abs(multiplier))), float(np.max(np.abs(proximal_multiplier)))
        assert lip_new <= lip_old + 1e-14
        for index, (factor, offset) in enumerate(((multiplier, intercept), (proximal_multiplier, proximal_intercept))):
            means[index] = factor*means[index] + offset
            variances[index] = factor**2*variances[index]
        bounds[0] = lip_old*bounds[0] + math.sqrt(max(risk_old, 0))
        bounds[1] = lip_new*bounds[1] + math.sqrt(risk_new)
        step_rows.append({"time": float(t), "next_time": float(s), "slope": slope.tolist(),
                          "teacher_risk_old": risk_old, "teacher_risk_new": risk_new,
                          "removed_energy": removed, "lip_old": lip_old, "lip_new": lip_new,
                          "true_input_marginal_w2_squared_old": marginal_old,
                          "true_input_marginal_w2_squared_new": marginal_new})
    results = []
    for index, name in enumerate(("ordinary", "proximal")):
        distance = gaussian_distance_squared(means[index], np.diag(variances[index]), data_mean, np.diag(data_variance))
        assert distance <= bounds[index]**2 + 1e-10
        results.append({"condition": name, "w2_squared": distance, "w2_bound": bounds[index],
                        "endpoint_mean": means[index].tolist(), "endpoint_variance": variances[index].tolist()})
    assert bounds[1] <= bounds[0] + 1e-12
    maximum_slope = max(max(row["slope"]) for row in step_rows)
    if oracle:
        assert maximum_slope < 1e-12
        assert abs(results[0]["w2_squared"]-results[1]["w2_squared"]) < 1e-12
    else:
        assert results[1]["w2_squared"] < results[0]["w2_squared"]
    return {"oracle": oracle, "target_mean": data_mean.tolist(), "target_variance": data_variance.tolist(),
            "maximum_slope": maximum_slope, "results": results, "steps": step_rows}


def gaussian_single_step_boundary() -> dict:
    """Cover oracle/OT/uncertainty-gap/overexpansion without a scale search.

    These four gains are algebraically defined theorem boundary cases,
    not candidate guidance coefficients or tuned experimental conditions.
    """
    t, s = time_grid()[-2:]
    data_mean = np.array([1.0, -1.0])
    data_variance = np.array([0.5, 2.0])
    mean_t, mean_s = (1-t)*data_mean, (1-s)*data_mean
    variance_t = (1-t)**2*data_variance + t*t
    variance_s = (1-s)**2*data_variance + s*s
    covariance_ts = (1-t)*(1-s)*data_variance + t*s
    oracle = covariance_ts/variance_t
    ot = np.sqrt(variance_s/variance_t)
    cap = variance_s/covariance_ts
    assert np.all(oracle < ot)
    assert np.all(ot < cap)
    rows = []
    for name, gain in (("bayes_oracle", oracle), ("optimal_transport", ot),
                       ("inside_uncertainty_gap", (ot+cap)/2),
                       ("overexpansion", 2*cap)):
        offset = mean_s-gain*mean_t
        slope = np.maximum(gain*covariance_ts/variance_s-1, 0)
        corrected_gain = gain/(1+slope)
        before = float(np.sum((gain*np.sqrt(variance_t)-np.sqrt(variance_s))**2))
        after = float(np.sum((corrected_gain*np.sqrt(variance_t)-np.sqrt(variance_s))**2))
        assert after <= before + 1e-14
        assert np.allclose(corrected_gain, np.minimum(gain, cap))
        if name != "overexpansion":
            assert np.max(slope) < 1e-14
        else:
            assert after < before
        rows.append({"case": name, "gain": gain.tolist(), "offset": offset.tolist(),
                     "corrected_gain": corrected_gain.tolist(), "slope": slope.tolist(),
                     "w2_squared_before": before, "w2_squared_after": after})
    return {"time": float(t), "next_time": float(s), "oracle_gain": oracle.tolist(),
            "ot_gain": ot.tolist(), "cap_gain": cap.tolist(), "cases": rows,
            "scope": "exact true-input one-step Gaussian marginals, not rollout marginals"}


def nonlinear_product(nodes: int) -> dict:
    # Independent coordinates; target coupling is Y=Z~N(0,I), ordinary
    # drift z+sin(z). E[Z sin Z]=exp(-1/2), so the projection is analytic.
    initial, weights = roots_hermitenorm(nodes)
    weights = weights/math.sqrt(2*math.pi)
    ordinary, proximal = initial.copy(), initial.copy()
    c = math.exp(-0.5)
    risk_drift = 2*((1-math.exp(-2))/2 + 1 + 2*c)
    risk_residual = 2*((1-math.exp(-2))/2 - c*c)
    bounds = [0.0, 0.0]
    for t, s in zip(time_grid()[:-1], time_grid()[1:]):
        delta = t-s
        slope = delta*(1+c)
        ordinary = ordinary + delta*(ordinary+np.sin(ordinary))
        proximal = (proximal + delta*(proximal+np.sin(proximal)))/(1+slope)
        lip_old = 1+2*delta
        lip_new = lip_old/(1+slope)
        bounds[0] = lip_old*bounds[0] + delta*math.sqrt(risk_drift)
        bounds[1] = lip_new*bounds[1] + delta*math.sqrt(risk_residual)/(1+slope)
    results = []
    for index, (name, endpoint) in enumerate((("ordinary", ordinary), ("proximal", proximal))):
        # Each Euler map is increasing, so quantile coupling is exact W2.
        distance = 2*float(weights @ (endpoint-initial)**2)
        assert distance <= bounds[index]**2 + 1e-8
        results.append({"condition": name, "w2_squared": distance, "w2_bound": bounds[index]})
    assert bounds[1] < bounds[0]
    assert results[1]["w2_squared"] < results[0]["w2_squared"]
    return {"quadrature_nodes": nodes, "results": results}


def cancellation_counterexample() -> dict:
    # Fixed linear rotation/stretch loop. T_k=exp(A Delta_k) is also an
    # Euler map for b_k(z)=(T_k-I)z/Delta_k. Product T_k=exp(A)=I.
    a = math.log(2.0)
    omega = math.sqrt(a*a+4*math.pi**2)
    generator = np.array([[a, -omega], [omega, -a]])
    mappings = [np.eye(2), np.eye(2)]
    bounds = [0.0, 0.0]
    removed_energy = 0.0
    for t, s in zip(time_grid()[:-1], time_grid()[1:]):
        source = expm(generator*(t-s))
        error = source-np.eye(2)
        slope, risk_old, risk_new, removed = projection_statistics(
            np.zeros(2), np.diag(error@error.T), np.diag(error), np.ones(2),
        )
        correction = source/(1+slope)[:, None]
        lips = [float(np.linalg.norm(source, 2)), float(np.linalg.norm(correction, 2))]
        assert lips[1] <= lips[0] + 1e-14
        for i, matrix in enumerate((source, correction)):
            mappings[i] = matrix@mappings[i]
            bounds[i] = lips[i]*bounds[i] + math.sqrt((risk_old, risk_new)[i])
        removed_energy += removed
    results = []
    for i, name in enumerate(("ordinary", "proximal")):
        covariance = mappings[i]@mappings[i].T
        distance = gaussian_distance_squared(np.zeros(2), covariance, np.zeros(2), np.eye(2))
        results.append({"condition": name, "w2_squared": distance, "w2_bound": bounds[i],
                        "endpoint_covariance": covariance.tolist()})
    assert results[0]["w2_squared"] < 1e-12
    assert results[1]["w2_squared"] > 1e-3
    assert bounds[1] < bounds[0]
    return {"meaning": "all step certificates improve but actual endpoint cancellation is lost",
            "sum_removed_teacher_energy": removed_energy, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    nonlinear = nonlinear_product(2048)
    coarse = nonlinear_product(1024)
    quadrature_difference = max(abs(a["w2_squared"]-b["w2_squared"])
                                for a, b in zip(nonlinear["results"], coarse["results"]))
    assert quadrature_difference < 1e-6
    report = {"status": "passed", "scope": "population discrete proximal certificates; CPU only; no strength/window search",
              "time_grid": "100 Euler steps, shift 8 (official mathematical grid)",
              "gaussian_bridge": gaussian_bridge(), "gaussian_bayes_oracle": gaussian_bridge(oracle=True),
              "gaussian_single_step_boundary": gaussian_single_step_boundary(),
              "nonlinear_product": nonlinear, "quadrature_difference": quadrature_difference,
              "cancellation_counterexample": cancellation_counterexample()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    summary = {"status": report["status"], "quadrature_difference": quadrature_difference}
    for key in ("gaussian_bridge", "gaussian_bayes_oracle", "nonlinear_product", "cancellation_counterexample"):
        summary[key] = report[key]["results"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
