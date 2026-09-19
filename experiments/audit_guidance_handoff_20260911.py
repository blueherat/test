"""CPU audit of conditional information, future basins, and CFG handoff.

An exact one-dimensional Gaussian-mixture linear flow-matching model.
Time runs from Gaussian noise (0) to data (1). This is not an image benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit, ndtr, ndtri


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/data/guidance_handoff_20260911"
M, SIGMA = 2.0, 0.5
N = 8192
TIMES = np.array([0, 0.025, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5, 0.7, 0.75, 0.9, 1.0])


def variance(t):
    return (1 - t) ** 2 + t * t * SIGMA**2


def posterior(t, x):
    return expit(2 * t * M * x / variance(t))


def fields(t, x):
    q = variance(t)
    a = (-(1 - t) + t * SIGMA**2) / q
    b = M * (1 - t) / q
    p = posterior(t, x)
    vu = a * x + b * (2 * p - 1)
    vc = a * x + b
    return vu, vc


def cdf(t, x):
    sd = np.sqrt(variance(t))
    return 0.5 * (ndtr((x - t * M) / sd) + ndtr((x + t * M) / sd))


def inverse_cdf(t, p):
    lo = np.full_like(np.asarray(p), -12.0, dtype=float)
    hi = np.full_like(lo, 12.0)
    for _ in range(75):
        mid = 0.5 * (lo + hi)
        go_right = cdf(t, mid) < p
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
    return 0.5 * (lo + hi)


def conditional_tail(t, x):
    return M + SIGMA * (x - t * M) / np.sqrt(variance(t))


def null_tail(t, x):
    return inverse_cdf(1.0, cdf(t, x))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u = (np.arange(N) + 0.5) / N
    z = ndtri(u)
    target = M + SIGMA * z
    sol = solve_ivp(
        lambda t, x: fields(t, x)[0] + 4 * (fields(t, x)[1] - fields(t, x)[0]),
        (0, 1), z, t_eval=TIMES, method="DOP853", rtol=2e-10, atol=2e-11,
    )
    assert sol.success
    rows = []
    for w in (1, 4):
        for i, t in enumerate(TIMES):
            state = t * M + np.sqrt(variance(t)) * z if w == 1 else sol.y[:, i]
            for tail_name, tail in (("conditional", conditional_tail), ("null", null_tail)):
                final = tail(t, state)
                # One-dimensional monotone transports: shared input quantiles
                # give the optimal W2 coupling, approximated by midpoint quadrature.
                assert np.min(np.diff(final)) > -1e-8
                rows.append({
                    "prefix_w": w, "switch_t": float(t), "tail": tail_name,
                    "positive_endpoint_fraction": float(np.mean(final > 0)),
                    "positive_fraction_initial_z_negative": float(np.mean(final[z < 0] > 0)),
                    "positive_fraction_initial_z_positive": float(np.mean(final[z > 0] > 0)),
                    "endpoint_mean": float(np.mean(final)),
                    "endpoint_std": float(np.std(final)),
                    "w2_squared_to_target": float(np.mean((final - target) ** 2)),
                    "mean_noising_posterior_at_switch": float(np.mean(posterior(t, state))),
                    "null_tail_positive_fraction_at_switch": float(np.mean(state > 0)),
                })
    write_csv("handoff_sweep.csv", rows)

    # At pure noise, the noising-pair label is independent of z; the endpoint
    # under a deterministic null flow still has a seed-dependent basin.
    probes = []
    for t in (0, 0.1, 0.5, 0.9):
        for x in (-1.0, -0.05, 0.05, 1.0):
            xu = float(null_tail(t, np.array(x)))
            probes.append({"t": t, "x": x, "noising_posterior_cat": float(posterior(t, x)),
                           "null_endpoint": xu, "null_endpoint_positive": int(xu > 0)})
    write_csv("posterior_vs_deterministic_future.csv", probes)

    x = np.linspace(-3, 3, 101)
    mixture_identity_error = 0.0
    for t in (0, 0.1, 0.5, 0.9):
        vu, vc = fields(t, x)
        _, vd_neg = fields(t, -x)
        vd = -vd_neg
        mixture_identity_error = max(mixture_identity_error,
                                     float(np.max(np.abs(vu - (posterior(t, x) * vc + (1-posterior(t, x)) * vd)))))

    # Quantile transport independently checked against the exact null ODE.
    null_ode_errors = []
    for t in (0.1, 0.5, 0.9):
        xp = np.array([-1.2, -0.15, 0.15, 1.2])
        s = solve_ivp(lambda tt, xx: fields(tt, xx)[0], (t, 1), xp,
                      method="DOP853", rtol=2e-11, atol=2e-12)
        assert s.success
        null_ode_errors.append(float(np.max(np.abs(s.y[:, -1] - null_tail(t, xp)))))

    # Frozen desired endpoint and inverse null map implement exact FSG identity.
    inverse_checks = []
    for t in (0.1, 0.5, 0.9):
        xp = np.array([-1.0, -0.1, 0.1, 1.0])
        desired = conditional_tail(t, xp)
        written = inverse_cdf(t, cdf(1, desired))
        inverse_checks.append(float(np.max(np.abs(null_tail(t, written) - desired))))

    # Exact total-variation identity for conditioning on a terminal event.
    tv_rows = []
    for h in (0.5, 0.9, 0.99, 0.999):
        base = np.array([0.3*h, 0.7*h, 1-h])
        conditioned = np.array([0.3, 0.7, 0.0])
        tv = float(0.5*np.sum(np.abs(base-conditioned)))
        assert abs(tv - (1-h)) < 1e-15
        tv_rows.append({"h": h, "tv": tv, "one_minus_h": 1-h})
    write_csv("terminal_conditioning_tv.csv", tv_rows)

    summary = {
        "model": "Y in {-1,1} equally likely; X|Y ~ N(2Y,0.5^2); X_t=tX+(1-t)Z; Z~N(0,1) independent",
        "semantic_event": "endpoint > 0 (not the latent mixture component label)",
        "target_negative_tail_probability": float(ndtr(-M/SIGMA)),
        "n_midpoint_quantiles": N,
        "target_grid_mean": float(np.mean(target)), "target_grid_std": float(np.std(target)),
        "mixture_velocity_identity_max_error": mixture_identity_error,
        "null_ode_quantile_max_error": max(null_ode_errors),
        "inverse_null_written_endpoint_max_error": max(inverse_checks),
        "conditional_prefix_conditional_tail_w2_squared_max": max(r["w2_squared_to_target"] for r in rows if r["prefix_w"] == 1 and r["tail"] == "conditional"),
        "strong_prefix_ode_nfe": sol.nfev,
        "scope": "Exact fields, numerical CPU integration/quadrature. No learned image model, no FID, no GPU runs.",
    }
    assert mixture_identity_error < 1e-12
    assert max(null_ode_errors) < 1e-8
    assert max(inverse_checks) < 1e-8
    assert summary["conditional_prefix_conditional_tail_w2_squared_max"] < 1e-25
    summary["status"] = "PASS"
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    inventory = {}
    for p in sorted(OUT.glob("*")):
        if p.name != "inventory.json":
            inventory[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    inventory["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for row in rows:
        if row["switch_t"] in (0.1, 0.25, 0.5, 0.75, 1.0):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
