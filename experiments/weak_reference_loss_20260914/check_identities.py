"""Check the research note's identities; this is NOT a generation experiment.

Run from the repository root with Python + NumPy. No models, images, or GPUs.
"""

import json
from pathlib import Path

import numpy as np


def normal(x, mean, variance):
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


def mixture(x, means, variances, weights):
    components = weights[:, None] * normal(x[None], means[:, None], variances[:, None])
    density = components.sum(axis=0)
    derivative = (components * (means[:, None] - x[None]) / variances[:, None]).sum(axis=0)
    return density, derivative / density, components / density


def main():
    x = np.linspace(-8, 8, 16001)
    means = np.array([-1.3, 1.7])
    variances = np.array([0.6, 0.9])
    weights = np.array([0.4, 0.6])
    t, tau, beta = 0.43, 0.7, 0.5
    a, b = t, 1 - t
    d2 = b * b + a * a * tau * tau
    p, sp, _ = mixture(x, a * means, a * a * variances + b * b, weights)
    r, sr, posterior = mixture(x, a * means, a * a * variances + d2, weights)
    q = (1 - beta) * p + beta * r
    sq = ((1 - beta) * p * sp + beta * r * sr) / q
    omega = beta * r / q
    mix_error = np.max(np.abs((sp - sq) - omega * (sp - sr)))

    # Integrate out the extra endpoint noise, conditioning on the base clean X.
    cond_x = means[:, None] + (
        a * variances / (a * a * variances + d2)
    )[:, None] * (x[None] - a * means[:, None])
    mean_x = (posterior * cond_x).sum(axis=0)
    mean_y_rb = b * b / d2 * mean_x + a * tau * tau / d2 * x
    mean_y_from_score = (x + b * b * sr) / a
    rb_error = np.max(np.abs(mean_y_rb - mean_y_from_score))
    score_from_wrong_clean_target = (a * mean_x - x) / (b * b)
    wrong_target_identity = np.max(np.abs(score_from_wrong_clean_target - d2 / (b * b) * sr))
    shifted_t = a / (a + np.sqrt(d2))
    scale = a + np.sqrt(d2)
    p_shift, s_shift, _ = mixture(
        x / scale, shifted_t * means,
        shifted_t ** 2 * variances + (1 - shifted_t) ** 2, weights,
    )
    shift_density_error = np.max(np.abs(r - p_shift / scale))
    shift_score_error = np.max(np.abs(sr - s_shift / scale))

    # Rao-Blackwellized velocity target, using a single standard Gaussian draw.
    eps = np.linspace(-3, 3, len(x))
    z = a * x + np.sqrt(d2) * eps
    rb_clean = x + a * tau * tau / np.sqrt(d2) * eps
    rb_velocity = x + (a * tau * tau - b) / np.sqrt(d2) * eps
    rb_velocity_error = np.max(np.abs(rb_velocity - (rb_clean - z) / b))

    # Excess-mass reference: all weights are positive and bounded.
    p0, sp0, _ = mixture(x, means, variances, weights)
    g0, sg0, _ = mixture(x, means + np.array([0.2, -0.1]), variances,
                         np.array([0.25, 0.75]))
    excess = np.maximum(g0 - p0, 0)
    overlap = np.minimum(g0, p0)
    lam = 1.0
    w = 1 + lam * np.maximum(1 - p0 / g0, 0)
    mass = np.trapezoid(excess, x)
    q_num = g0 * w
    q_from_parts = overlap + (1 + lam) * excess
    parts_error = np.max(np.abs(q_num - q_from_parts))
    weight_bounds = bool(np.all((w >= 1) & (w <= 1 + lam)))
    active = g0 > p0
    q_derivative = g0 * sg0 + lam * np.where(active, g0 * sg0 - p0 * sp0, 0)
    sq0 = q_derivative / q_num
    coeff = np.where(active, lam * (p0 / g0) / w, 0)
    excess_score_error = np.max(np.abs((sg0 - sq0) - coeff * (sp0 - sg0)))
    density_normalization_error = abs(np.trapezoid(q_num / (1 + lam * mass), x) - 1)

    # Positive-part and noising do not commute. K is a column-stochastic kernel.
    pp = np.array([0.6, 0.3, 0.1])
    gg = np.array([0.3, 0.3, 0.4])
    kernel = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    ee = np.maximum(gg - pp, 0)
    ww = 1 + lam * ee / gg
    norm = np.sum(gg * ww)
    q_forward = kernel @ (gg * ww / norm)
    g_forward = kernel @ gg
    posterior_weight = (kernel @ (gg * ww)) / g_forward
    endpoint_weight_identity = np.max(np.abs(q_forward - g_forward * posterior_weight / norm))
    noncommutation = np.max(np.abs(kernel @ ee - np.maximum(kernel @ (gg - pp), 0)))

    # A signed density / signed quadratic loss has no general positivity guarantee.
    signed_candidate = 2 * normal(x, 0, 0.36) - normal(x, 0, 1)
    signed_min = float(signed_candidate.min())

    checks = {
        "kind": "algebra_checks_only_not_image_quality",
        "mixture_score_identity_max_error": float(mix_error),
        "mixture_log_ratio_max": float(np.max(np.log(p / q))),
        "mixture_log_ratio_upper_bound": float(-np.log(1 - beta)),
        "gaussian_rb_clean_identity_max_error": float(rb_error),
        "wrong_clean_target_scaled_score_identity_max_error": float(wrong_target_identity),
        "wrong_clean_target_score_multiplier": float(d2 / (b * b)),
        "fm_shift_density_max_error": float(shift_density_error),
        "fm_shift_score_max_error": float(shift_score_error),
        "gaussian_rb_velocity_identity_max_error": float(rb_velocity_error),
        "excess_overlap_decomposition_max_error": float(parts_error),
        "excess_weight_bounds_pass": weight_bounds,
        "excess_endpoint_score_identity_max_error": float(excess_score_error),
        "excess_normalization_quadrature_error": float(density_normalization_error),
        "forward_posterior_weight_identity_max_error": float(endpoint_weight_identity),
        "positive_part_noising_noncommutation_max_gap": float(noncommutation),
        "signed_density_counterexample_minimum": signed_min,
    }
    for key, value in checks.items():
        if key.endswith("max_error") or key.endswith("quadrature_error"):
            assert value < 1e-8, (key, value)
    assert checks["mixture_log_ratio_max"] <= checks["mixture_log_ratio_upper_bound"] + 1e-12
    assert weight_bounds and noncommutation > 0.01 and signed_min < -0.01
    checks["passed"] = True
    output = Path(__file__).resolve().parents[2] / "docs/data/weak_reference_loss_20260914/algebra_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
