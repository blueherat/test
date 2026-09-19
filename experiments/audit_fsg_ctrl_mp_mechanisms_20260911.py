"""Source-table extraction and CPU counterexamples for FSG / CFG-Ctrl / CFG-MP.

Run from the repository root. No generative checkpoint or GPU is used. The exact
Gaussian-mixture experiment concerns distributional correctness, not image FID.
Published metrics retain their original units and evaluation protocols.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import expit, ndtr, ndtri


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "readings/fsg_ctrl_mp_comparison_20260911"
OUT = ROOT / "docs/data/fsg_ctrl_mp_comparison_20260911"
URL = {
    "fsg": "https://arxiv.org/pdf/2510.21512v1",
    "cfg_ctrl": "https://openaccess.thecvf.com/content/CVPR2026/papers/"
    "Wang_CFG-Ctrl_Control-Based_Classifier-Free_Diffusion_Guidance_CVPR_2026_paper.pdf",
    "cfg_mp": "https://arxiv.org/pdf/2601.21892v2",
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def published_tables():
    rows = []

    def add(paper, table, backbone, dataset, method, nfe, scale, metric, value):
        rows.append(dict(
            paper=paper, table=table, backbone=backbone, dataset=dataset,
            method=method, nfe_as_reported=nfe, guidance_as_reported=scale,
            metric=metric, value=float(value), higher_better=metric not in {"FID", "seconds"},
            units="original paper table; no cross-paper normalization",
            source_url=URL[paper],
        ))

    fsg = (SOURCES / "fsg.txt").read_text()
    part = fsg.split("Table 2: The quantitative results on the SDXL model", 1)[1]
    part = part.split("Practical design", 1)[0]
    pattern = r"^\s*(CFG(?:\+\+)?(?:×[23])?|Z-Sampling|Resampling|FSG \(ours\))\s+(50|100|150)\s+(.+)$"
    count = 0
    for line in part.splitlines():
        match = re.match(pattern, line)
        if not match:
            continue
        method, nfe, rest = match.groups()
        vals = [float(x) for x in rest.split()]
        assert len(vals) == 9, line
        add("fsg", "2", "SDXL / DDIM", "combined timing", method, nfe, "see Appendix D.1", "seconds", vals[0])
        for j, dataset in enumerate(["DrawBench", "Pick-a-Pic first 100 prompts"]):
            for k, metric in enumerate(["ImageReward", "HPSv2", "Aesthetic", "CLIP"]):
                add("fsg", "2", "SDXL / DDIM", dataset, method, nfe, "see Appendix D.1", metric, vals[1 + 4*j + k])
        count += 1
    assert count == 15, count

    mp = (SOURCES / "cfg_mp.txt").read_text()
    part = mp.split("Table 1. FID (", 1)[1].split("4.2. Text-to-image", 1)[0]
    count = 0
    for line in part.splitlines():
        match = re.match(r"^\s*(CFG\(DDIM\)|Z-sampling|Re-sampling|FSG|CFG\(D2F\)|CFG-MP\+?)\s+(.+)$", line)
        if not match:
            continue
        method, rest = match.groups()
        vals = [float(x) for x in rest.split()]
        assert len(vals) == 12, line
        for j, scale in enumerate([1.5, 2.0, 2.5]):
            for k, nfe in enumerate([60, 120]):
                for metric, offset in [("FID", 0), ("IS", 2)]:
                    add("cfg_mp", "1", "DiT-XL/2-256", "ImageNet-1K, 50k generated", method, nfe, scale, metric, vals[4*j + k + offset])
        count += 1
    assert count == 7, count

    ctrl = (SOURCES / "cfg_ctrl.txt").read_text()
    part = ctrl.split("Table 2. Quantitative evaluation of CFG methods.", 1)[1]
    part = part.split("We introduce a sliding-mode correction", 1)[0]
    count = 0
    backbone = None
    metrics = ["FID", "CLIP", "Aesthetic", "ImageReward", "PickScore", "HPSv2", "HPSv2.1", "MPS"]
    for line in part.splitlines():
        stripped = line.strip()
        new_backbone = next((b for b in ["SD3.5", "Flux-dev", "Qwen-Image"] if stripped.startswith(b + " [")), None)
        if new_backbone:
            backbone = new_backbone
            method = "without external CFG"
        elif stripped.startswith("w/ "):
            method = stripped[3:].split("[")[0].strip() if "[" in stripped else "SMC-CFG"
        else:
            continue
        vals = re.findall(r"\d+\.\d+", stripped)[-8:]
        assert len(vals) == 8 and backbone, line
        for metric, value in zip(metrics, vals):
            add("cfg_ctrl", "2", backbone, "MS-COCO held-out 5k pairs", method, "not specified in table", "model-specific default", metric, value)
        count += 1
    assert count == 15, count
    write_csv(OUT / "published_main_tables.csv", rows)

    comparisons = []
    mp_rows = [r for r in rows if r["paper"] == "cfg_mp"]
    for nfe in [60, 120]:
        for scale in [1.5, 2.0, 2.5]:
            for method in ["FSG", "CFG-MP", "CFG-MP+"]:
                baseline = "CFG(DDIM)" if method == "FSG" else "CFG(D2F)"
                def value(m, metric):
                    return next(r["value"] for r in mp_rows if r["method"] == m and r["metric"] == metric and r["nfe_as_reported"] == nfe and r["guidance_as_reported"] == scale)
                comparisons.append(dict(
                    method=method, baseline=baseline, nfe=nfe, guidance=scale,
                    FID=value(method, "FID"), baseline_FID=value(baseline, "FID"),
                    FID_change_percent=100*(value(method, "FID")/value(baseline, "FID") - 1),
                    IS=value(method, "IS"), baseline_IS=value(baseline, "IS"),
                    IS_change_percent=100*(value(method, "IS")/value(baseline, "IS") - 1),
                ))
    write_csv(OUT / "mp_table1_within_sampler_deltas.csv", comparisons)
    return {"extracted_metric_values": len(rows), "within_sampler_comparisons": len(comparisons)}


def exact_mixture():
    """Both fields are exact for a compatible conditional / marginal FM model.

    Y is uniform on {-1,+1}; X1|Y ~ N(2Y, .5^2); X0 ~ N(0,1),
    independent. Xt=(1-t)X0+tX1. Apply a common positive translation at
    t=.5 to correct conditional samples, then follow exact conditional flow.
    Conditional and marginal maps use exact CDF conservation in 1D.
    """
    mu, sigma, t = 2.0, .5, .5
    n = 4096
    quantiles = (np.arange(n) + .5) / n
    variance = lambda s: (1-s)**2 + (s*sigma)**2
    x = t*mu + np.sqrt(variance(t))*ndtri(quantiles)

    def marginal_cdf(z, s):
        sd = np.sqrt(variance(s))
        return .5*(ndtr((z-s*mu)/sd) + ndtr((z+s*mu)/sd))

    def unconditional_endpoint(z):
        p = marginal_cdf(z, t)
        lo, hi = np.full_like(z, -10.), np.full_like(z, 10.)
        for _ in range(72):
            mid = (lo+hi)/2
            lower = marginal_cdf(mid, 1.) < p
            lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
        endpoint = (lo+hi)/2
        assert np.max(np.abs(marginal_cdf(endpoint, 1.)-p)) < 2e-14
        return endpoint

    def gap(z, s):
        return 2*mu*(1-s)/variance(s)*expit(-2*s*mu*z/variance(s))

    # Independently check CDF conservation against integration of the exact
    # posterior-mixture velocity, rather than only inverting the same CDF.
    probe = t*mu + np.sqrt(variance(t))*ndtri(np.linspace(.01, .99, 41)) + .2
    def marginal_velocity(s, z):
        a = (s*sigma*sigma-(1-s))/variance(s)
        b = (1-s)/variance(s)
        posterior_mean_mu = mu*np.tanh(s*mu*z/variance(s))
        return a*z+b*posterior_mean_mu
    flow_check = solve_ivp(marginal_velocity, (t, 1.), probe, rtol=2e-11, atol=2e-13)
    assert flow_check.success
    endpoint_check_error = float(np.max(np.abs(flow_check.y[:, -1]-unconditional_endpoint(probe))))
    assert endpoint_check_error < 1e-8, endpoint_check_error

    gx, gw = np.polynomial.legendre.leggauss(96)
    times = t+(gx+1)*(1-t)/2
    weights = gw*(1-t)/2
    result = []
    for delta in [0., .1, .25, .5, 1.]:
        shifted = x+delta
        endpoint_c = mu + sigma/np.sqrt(variance(t))*(shifted-t*mu)
        endpoint_u = unconditional_endpoint(shifted)
        states = times[:, None]*mu + np.sqrt(variance(times[:, None])/variance(t))*(shifted[None, :]-t*mu)
        integrated_gap = np.dot(weights, np.mean(gap(states, times[:, None])**2, axis=1))
        exact_w2_squared = sigma**2/variance(t)*delta**2
        expected_c = mu+sigma*ndtri(quantiles)+sigma/np.sqrt(variance(t))*delta
        assert np.max(np.abs(endpoint_c-expected_c)) < 2e-14
        result.append(dict(
            latent_translation=delta,
            current_raw_gap_squared_mean=float(np.mean(gap(shifted, t)**2)),
            future_endpoint_residual_squared_mean=float(np.mean((endpoint_c-endpoint_u)**2)),
            cumulative_raw_gap_squared_mean=float(integrated_gap),
            true_conditional_W2_squared=exact_w2_squared,
            generated_conditional_mean=float(endpoint_c.mean()),
        ))
    for key in ["current_raw_gap_squared_mean", "future_endpoint_residual_squared_mean", "cumulative_raw_gap_squared_mean"]:
        assert all(a[key] > b[key] for a, b in zip(result, result[1:])), key
    assert result[0]["true_conditional_W2_squared"] == 0
    write_csv(OUT / "exact_compatible_mixture.csv", result)
    return dict(
        model="X1|Y ~ N(2Y, 0.5^2); P(Y=+1)=P(Y=-1)=0.5; X0 ~ N(0,1)",
        flow="Xt=(1-t)X0+tX1, exact conditional and compatible marginal velocities",
        t=t, midpoint_quantile_nodes=n, time_quadrature_nodes=96,
        CDF_map_vs_independent_ODE_max_error=endpoint_check_error,
        intervention="positive latent translation, followed by exact conditional flow",
        caveat="Exact W2 of 1D Gaussian distributions; not an image-FID experiment or a three-algorithm ranking.",
        result=result,
    )


def transport_identity():
    t0, terminal = .2, 1.
    au = np.array([[.3, -.8], [.4, -.2]])
    ac = np.array([[-.1, .5], [-.3, .25]])
    start = np.array([.2, -.4])
    def vu(t, z): return au@z + np.array([.1*np.cos(t), .2*t])
    def vc(t, z): return ac@z + np.array([.3*t, -.2*np.sin(t)])
    def rhs(t, y):
        c, u = y[:2], y[2:4]
        g = vc(t, c)-vu(t, c)
        return np.concatenate([vc(t, c), vu(t, u), expm((terminal-t)*au)@g])
    solution = solve_ivp(rhs, (t0, terminal), np.r_[start, start, [0., 0.]], rtol=2e-12, atol=2e-14)
    assert solution.success
    end = solution.y[:, -1]
    error = float(np.linalg.norm(end[:2]-end[2:4]-end[4:]))
    assert error < 1e-10
    rows = []
    g0 = vc(t0, start)-vu(t0, start)
    for h in [.1, .03, .01, .003, .001]:
        c = solve_ivp(vc, (t0, t0+h), start, rtol=2e-12, atol=2e-14).y[:, -1]
        u = solve_ivp(vu, (t0, t0+h), start, rtol=2e-12, atol=2e-14).y[:, -1]
        rows.append({"H": h, "norm_R_H_over_H_minus_gap": float(np.linalg.norm((c-u)/h-g0))})
    assert rows[-1]["norm_R_H_over_H_minus_gap"] < .011*rows[0]["norm_R_H_over_H_minus_gap"]
    write_csv(OUT / "short_horizon_limit.csv", rows)
    return dict(
        identity_residual_norm=error, endpoint_difference=(end[:2]-end[2:4]).tolist(),
        transported_gap_integral=end[4:].tolist(),
        cancellation_example={"vu": "0", "vc": "2t-1", "interval": [0, 1], "terminal_difference": 0, "integrated_gap_squared": 1/3},
        instantaneous_zero_example={"vu": "0", "vc": "t", "initial_gap": 0, "terminal_difference": .5},
    )


def operator_checks():
    # Gap identically zero, but Euler split G is not an identity projection.
    a, linear_rate, z = .1, 2., 1.
    intermediate = z-a*linear_rate*z
    mp_z = intermediate+a*linear_rate*intermediate
    assert abs(mp_z - (1-a*a*linear_rate**2)*z) < 1e-14
    # Main-paper singular-value assumption does not determine control direction.
    s, gain, gamma = 1., 1., -1.
    v_dot = s*gamma*(-gain*np.sign(s))
    assert abs(gamma) == 1 and v_dot > 0
    # A potential descent step need not decrease the norm of its gradient.
    z_new = 1.1
    # Fixed-state least-squares identity at two different states. The residual
    # orthogonal to the guidance line changes when moving the state.
    ls = []
    for gap_x, orthogonal_error in [(1., 1.), (.1, 10.)]:
        u = np.array([0., orthogonal_error]); g = np.array([gap_x, 0.])
        target = np.array([.5*gap_x, 0.]); w = .5
        w_star = np.dot(target-u, g)/np.dot(g, g)
        error2 = np.sum((u+w*g-target)**2)
        decomposition = np.sum((u+w_star*g-target)**2)+(w-w_star)**2*np.sum(g*g)
        assert abs(error2-decomposition) < 1e-14
        ls.append(dict(gap_norm=float(np.linalg.norm(g)), approximation_error_squared=float(error2)))
    changing_optimum = []
    for g in [.5, .05]:
        w, target, u = 2., 1., 0.
        w_star = (target-u)/g
        error2 = (u+w*g-target)**2
        assert abs(error2-(w-w_star)**2*g*g) < 1e-14
        changing_optimum.append(dict(gap=g, w=w, w_star=w_star, approximation_error_squared=error2))
    return dict(
        common_field_split={"vc_equals_vu": "2x", "a": a, "x": z, "G_x": mp_z, "true_Euclidean_projection": z, "raw_gap": 0},
        ctrl_main_text_counterexample={"Gamma": gamma, "sigma_min_Gamma": abs(gamma), "drift": 0, "K": gain, "s": s, "V_dot": v_dot,
            "scope": "Refutes main-paper Eq.26 from a singular-value bound alone; excluded by the supplementary nominal-direction-dominance assumption."},
        potential_descent={"F": "-x^2/2", "before_x": 1., "after_x": z_new, "F_before": -.5, "F_after": -.5*z_new**2, "gradient_norm_before": 1., "gradient_norm_after": z_new},
        least_squares_identity_at_different_states=ls,
        least_squares_changing_optimum_with_zero_orthogonal_error=changing_optimum,
    )


def official_ctrl_history():
    import torch
    file = SOURCES / "code/cfg_ctrl/pipeline/common_cfg_ctrl.py"
    spec = importlib.util.spec_from_file_location("archived_cfg_ctrl_audit", file)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    params = module.CFGCtrlParams(smc_cfg_enable=True, smc_cfg_lambda=.05, smc_cfg_K=.3)
    state, mixin = module.CFGCtrlState(), module.CFGCtrlMixin()
    values = []
    for i, raw in enumerate([1., .7, .6]):
        out = mixin._cfg_ctrl_apply(noise_pred_posi=torch.tensor([raw], dtype=torch.float64), noise_pred_nega=torch.zeros(1, dtype=torch.float64), cfg_scale=2., progress_id=i, params=params, state=state)
        values.append(dict(step=i, raw_gap=raw, stored_modified_gap=state.prev_guidance_eps.item(), guided_prediction=out.item()))
    assert np.allclose([v["guided_prediction"] for v in values], [1.4, .8, .6], atol=1e-14, rtol=0)
    return values


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(
        scope="CPU mathematical and archived-code audit; published-table extraction; no new image benchmark",
        versions={"numpy": np.__version__, "scipy": scipy.__version__},
        tables=published_tables(), mixture=exact_mixture(), transport=transport_identity(),
        operators=operator_checks(), official_ctrl_history=official_ctrl_history(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES.glob("*.pdf")},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"tables": result["tables"], "transport_error": result["transport"]["identity_residual_norm"], "mixture": result["mixture"]["result"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
