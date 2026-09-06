#!/usr/bin/env python3
"""CPU-only identities and counterexamples; no model/data loading or parameter search."""
import json
import math
import time
from pathlib import Path


def main():
    start = time.perf_counter()
    z, base, full, lam = 0.37, -0.21, 0.68, 0.6
    gap = full - base
    records = []
    errors = {k: 0.0 for k in ["flow_vs_euler", "bridge_vs_euler", "native_as_flow", "closed_form_bridge_gain"]}
    for n in range(100, 0, -1):
        t = 8 * n / (100 + 7 * n)
        s = 8 * (n - 1) / (100 + 7 * (n - 1))
        h, r = t - s, s / t
        q = h / t
        d = base + lam * gap
        flow_direct = d + (z - base) * r
        flow_gain = lam / q
        flow_euler = z - h * (z - (base + flow_gain * gap)) / t
        noise_base = (z - (1 - t) * base) / t
        bridge_direct = (1 - s) * d + s * noise_base
        bridge_gain = lam * (1 - s) / q
        bridge_euler = z - h * (z - (base + bridge_gain * gap)) / t
        native_gain = 1.78 if t >= 0.1 else 1.0
        native_euler = z - h * (z - (base + native_gain * gap)) / t
        native_as_flow = base + native_gain * q * gap + r * (z - base)
        errors["flow_vs_euler"] = max(errors["flow_vs_euler"], abs(flow_direct - flow_euler))
        errors["bridge_vs_euler"] = max(errors["bridge_vs_euler"], abs(bridge_direct - bridge_euler))
        errors["native_as_flow"] = max(errors["native_as_flow"], abs(native_as_flow - native_euler))
        errors["closed_form_bridge_gain"] = max(errors["closed_form_bridge_gain"], abs(bridge_gain - lam * n * (101 - n) / 100))
        records.append(dict(n=n, t=t, next_t=s, q=q, flow_gain=flow_gain, bridge_gain=bridge_gain,
                            native_gain=native_gain, native_flow_lambda=native_gain * q))
    assert max(errors.values()) < 1e-10

    # Derive denoised-mean changes by first generating the next noisy state,
    # then reevaluating two affine noise predictors at that state and noise level.
    def predictions(x, a, b):
        eu = 0.1 * x + 0.13 * b
        ec = eu + 0.4 * x - 0.3 * b
        du, dc = (x - b * eu) / a, (x - b * ec) / a
        return eu, ec, du, dc

    at, bt, ass, bs = 0.6, 0.8, 0.8, 0.6
    st, ss, omega = bt / at, bs / ass, 2.0
    eu, ec, du, dc = predictions(z, at, bt)
    delta_t = dc - du
    old_cfg = du + omega * delta_t
    old_plus = du + lam * delta_t
    new_z_cfg = ass * old_cfg + bs * (eu + omega * (ec - eu))
    new_z_plus = ass * old_plus + bs * eu
    recurrence = {}
    for method, new_z, w, old in [("cfg", new_z_cfg, omega, old_cfg), ("cfgpp", new_z_plus, lam, old_plus)]:
        eus, ecs, dus, dcs = predictions(new_z, ass, bs)
        delta_s = dcs - dus
        measured = dus + w * delta_s - old
        exact = ss * (eu - eus) + w * delta_s
        if method == "cfg":
            exact -= w * ss / st * delta_t
        # Literal published definition d epsilon = epsilon(new) - epsilon(old).
        printed = ss * (eus - eu) + w * delta_s
        if method == "cfg":
            printed -= w * delta_t
        recurrence[method] = dict(measured=measured, exact=exact, printed=printed,
                                  exact_abs_error=abs(measured - exact), printed_abs_error=abs(measured - printed))
        assert abs(measured - exact) < 1e-12
        assert abs(measured - printed) > 1e-4

    # Exact loss L(x)=(x-D(x))^2 with D(x)=2x increases under detached-target descent.
    x, eta = 1.0, 0.1
    loss = lambda v: (v - 2 * v) ** 2
    eps = 1e-6
    true_grad_fd = (loss(x + eps) - loss(x - eps)) / (2 * eps)
    detached_grad = 2 * (x - 2 * x)
    detach_counterexample = dict(x=x, step=eta, true_grad_fd=true_grad_fd,
                                 detached_grad=detached_grad, before=loss(x),
                                 after_detached=loss(x - eta * detached_grad),
                                 after_exact=loss(x - eta * 2 * x))
    assert detach_counterexample["after_detached"] > detach_counterexample["before"]

    # SD1.5 and SDXL source differ in the current denoised term in 2M history.
    hh, rr, old_b = 0.2, 0.8, -0.3
    d = base + lam * gap
    source_sd15 = d - math.exp(-hh) * base + (1 - math.exp(-hh)) * (d - old_b) / (2 * rr) + math.exp(-hh) * z
    source_sdxl = d - math.exp(-hh) * base + (1 - math.exp(-hh)) * (base - old_b) / (2 * rr) + math.exp(-hh) * z
    expected_difference = (1 - math.exp(-hh)) * lam * gap / (2 * rr)
    assert abs(source_sd15 - source_sdxl - expected_difference) < 1e-12

    def improve(a, b):
        return 100 * (a - b) / a

    result = dict(
        scope="Scalar CPU algebra and counterexamples only; not RAE model validation, image generation, or FID replication",
        model_forward_calls=0, gpu_seconds=0,
        equivalence_max_abs_error=errors,
        grid=dict(steps=100, shift=8, smallest_positive_t=records[-1]["t"],
                  native_flow_lambda_min=min(x["native_flow_lambda"] for x in records),
                  native_flow_lambda_max=max(x["native_flow_lambda"] for x in records),
                  fixed_lambda=lam, flow_gain_first=records[0]["flow_gain"], flow_gain_last=records[-1]["flow_gain"],
                  bridge_gain_max=max(x["bridge_gain"] for x in records), bridge_gain_first=records[0]["bridge_gain"]),
        corrected_posterior_recurrence=recurrence,
        detached_loss_counterexample=detach_counterexample,
        sd15_minus_sdxl_2m=dict(measured=source_sd15 - source_sdxl, expected=expected_difference),
        table1_relative_fid_improvement_percent=[improve(a, b) for a, b in zip([13.84,15.08,17.71,20.01,21.23],[12.75,14.95,17.47,19.34,20.88])],
        table2_relative_fid_improvement_percent=[improve(a,b) for a,b in [(59.67,59.21),(56.11,55.19),(32.72,32.58)]],
        cpu_wall_seconds=time.perf_counter()-start,
    )
    out = Path(__file__).resolve().parent
    (out / "algebra_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "coefficient_grid.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
