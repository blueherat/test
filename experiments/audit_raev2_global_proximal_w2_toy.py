"""CPU audit of the scalar origin-anchored proximal W2 theorem.

No guidance scale/window fitting: one scalar is determined by coupling
second moments. Finite-support W2 uses an exact linear transport program;
Gaussian examples use exact moments. Counterexamples preserve claim limits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from audit_raev2_proximal_projection_toy import gaussian_distance_squared, time_grid


def fit_gain(target_energy: float, cross_moment: float) -> float:
    if target_energy <= 0 or cross_moment <= target_energy:
        return 1.0
    return target_energy/cross_moment


def discrete_w2(x, y, px, py):
    cost = ((x[:, None, :]-y[None, :, :])**2).sum(-1)
    constraints = np.vstack((np.kron(np.eye(len(x)), np.ones((1, len(y)))),
                             np.kron(np.ones((1, len(x))), np.eye(len(y)))))
    result = linprog(cost.ravel(), A_eq=constraints, b_eq=np.r_[px, py],
                     bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun)


def discrete_non_gaussian() -> dict:
    y = np.array([[-2., 1.], [0., 3.], [3., -1.], [4., 2.]])
    p = np.array([.1, .2, .3, .4])
    x = 3*y+np.array([1., -.5])
    # A fixed non-optimal coupling: diagonal plus independent mass.
    coupling = .8*np.diag(p)+.2*np.outer(p, p)
    a_energy = float(p @ (x*x).sum(1))
    b_energy = float(p @ (y*y).sum(1))
    c_pair = float(np.sum(coupling*(x@y.T)))
    gain = fit_gain(b_energy, c_pair)
    before = discrete_w2(x, y, p, p)
    after = discrete_w2(gain*x, y, p, p)
    optimal_cross = (a_energy+b_energy-before)/2
    optimal_gain = optimal_cross/a_energy
    assert c_pair < optimal_cross
    assert optimal_gain <= gain < 1
    assert after <= before
    assert abs(after-(gain*gain*a_energy+b_energy-2*gain*optimal_cross)) < 1e-8
    return {"target_mean": (p@y).tolist(), "target_energy": b_energy,
            "coupling_cross": c_pair, "optimal_cross": optimal_cross,
            "gain": gain, "optimal_ray_gain": optimal_gain,
            "w2_squared_before": before, "w2_squared_after": after,
            "coupling_is_optimal": False}


def gaussian_bridge(oracle: bool = False) -> dict:
    data_mean, data_variance = np.array([1., -1.]), np.array([.5, 2.])
    bias = np.zeros(2) if oracle else np.array([.3, -.2])
    gain_error = np.zeros(2) if oracle else np.array([.6, .4])
    means, variances = [np.zeros(2), np.zeros(2)], [np.ones(2), np.ones(2)]
    bounds, rows = [0., 0.], []
    for t, s in zip(time_grid()[:-1], time_grid()[1:]):
        delta, u, v = t-s, 1-t, 1-s
        mt, my = u*data_mean, v*data_mean
        vt = u*u*data_variance+t*t
        vy = v*v*data_variance+s*s
        cty = u*v*data_variance+t*s
        true_gain = (u*data_variance-t)/vt
        k = 1+delta*(true_gain+gain_error)
        offset = my+delta*bias-k*mt
        mx, vx, cxy = k*mt+offset, k*k*vt, k*cty
        a_energy = float(np.sum(vx+mx*mx))
        b_energy = float(np.sum(vy+my*my))
        c_pair = float(np.sum(cxy+mx*my))
        gain = fit_gain(b_energy, c_pair)
        local_old = max(a_energy+b_energy-2*c_pair, 0.)
        local_new = max(gain*gain*a_energy+b_energy-2*gain*c_pair, 0.)
        assert local_new <= local_old+1e-12
        marginal_old = gaussian_distance_squared(mx, np.diag(vx), my, np.diag(vy))
        marginal_new = gaussian_distance_squared(gain*mx, np.diag(gain*gain*vx), my, np.diag(vy))
        assert marginal_new <= marginal_old+1e-12
        for j, scale in enumerate((1., gain)):
            means[j] = scale*(k*means[j]+offset)
            variances[j] = scale*scale*k*k*variances[j]
            bounds[j] = scale*float(np.max(np.abs(k)))*bounds[j]+np.sqrt((local_old, local_new)[j])
        rows.append({"time": float(t), "next_time": float(s), "gain": gain,
                     "teacher_risk_before": local_old, "teacher_risk_after": local_new,
                     "true_input_w2_squared_before": marginal_old,
                     "true_input_w2_squared_after": marginal_new})
    results = []
    for j, name in enumerate(("ordinary", "global_proximal")):
        w2 = gaussian_distance_squared(means[j], np.diag(variances[j]), data_mean, np.diag(data_variance))
        assert w2 <= bounds[j]**2+1e-10
        results.append({"condition": name, "w2_squared": w2, "w2_bound": bounds[j],
                        "mean": means[j].tolist(), "variance": variances[j].tolist()})
    assert bounds[1] <= bounds[0]+1e-12
    if oracle:
        assert all(row["gain"] == 1. for row in rows)
        assert results[0]["w2_squared"] == results[1]["w2_squared"]
    return {"oracle": oracle, "results": results, "steps": rows}


def mean_and_rollout_boundaries() -> dict:
    # Y~N(1,1), T=2Y-1: mean already correct, but variance too large.
    # Zero anchor may change the correct mean while TOTAL W2 decreases.
    gain = fit_gain(2., 3.)
    before = (1-1)**2+(2-1)**2
    after = (gain-1)**2+(2*gain-1)**2
    assert after < before
    # Two stationary Gaussian target steps. T1=2I, T2=I/2 cancel exactly.
    # True-input local W2 decreases at step 1 and is unchanged at step 2;
    # correcting T1 destroys cancellation in the actual composed rollout.
    gains = [fit_gain(2., 4.), fit_gain(2., 1.)]
    ordinary_final, corrected_final = 1., (2*gains[0])*(.5*gains[1])
    ordinary_w2 = 2*(ordinary_final-1)**2
    corrected_w2 = 2*(corrected_final-1)**2
    old_bound = .5*np.sqrt(2)+np.sqrt(.5)
    new_bound = np.sqrt(.5)
    assert ordinary_w2 == 0 and corrected_w2 == .5
    assert new_bound < old_bound
    assert corrected_w2 <= new_bound*new_bound+1e-12
    assert fit_gain(0., 0.) == 1.
    return {"nonzero_mean": {"gain": gain, "mean_error_squared_before": 0.,
                              "mean_error_squared_after": (gain-1)**2,
                              "w2_squared_before": before, "w2_squared_after": after},
            "two_step_cancellation": {"base_maps": [2., .5], "gains": gains,
                                      "w2_squared_before": ordinary_w2,
                                      "w2_squared_after": corrected_w2,
                                      "w2_bound_before": old_bound, "w2_bound_after": new_bound},
            "q_mismatch": "For the second comparison, incoming q=N(0,I/4) makes T=2I exact; calibrated gain 1/2 then worsens W2² from 0 to 1/2.",
            "zero_target_energy_returns_identity": True}


def finite_fit_certificate_boundary() -> dict:
    # A deliberately inaccurate fixed fit, not a candidate scale search:
    # X=3Y, Y~N(0,1), gain=1/4 goes past the ray optimum 1/3.
    # The weaker secant certificate still correctly certifies improvement.
    gain, source_energy, target_energy = .25, 9., 1.
    h_stay = gain*gain*source_energy-target_energy
    h_sec = (1+gain)**2*source_energy-4*target_energy
    before, after = (3-1)**2, (3*gain-1)**2
    assert h_stay < 0 < h_sec and after < before
    return {"frozen_inaccurate_gain": gain, "optimal_ray_gain": 1/3,
            "H_stay": h_stay, "H_sec": h_sec,
            "w2_squared_before": before, "w2_squared_after": after}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"status": "passed", "scope": "CPU population scalar proximal theorem; no strength/window search",
              "non_gaussian_discrete": discrete_non_gaussian(),
              "gaussian_bridge": gaussian_bridge(), "gaussian_oracle": gaussian_bridge(True),
              "boundaries": mean_and_rollout_boundaries(),
              "finite_fit_certificate_boundary": finite_fit_certificate_boundary()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    summary = dict(report)
    for key in ("gaussian_bridge", "gaussian_oracle"):
        summary[key] = report[key]["results"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
