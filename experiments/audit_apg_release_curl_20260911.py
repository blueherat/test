#!/usr/bin/env python3
"""CPU-only, analytic counterexamples accompanying the APG/release/curl review.

These checks establish algebra and numerical mechanisms, not image quality.
Run: OPENBLAS_NUM_THREADS=1 python experiments/audit_apg_release_curl_20260911.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import itertools
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/data/apg_release_curl_20260911"
FIG = ROOT / "docs/figures/apg_release_curl_20260911"
RNG = np.random.default_rng(20260911)


def dump_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vu(x):
    return np.array([0.17 * x[0], -0.23 * x[1] + 0.12 * x[0] ** 2])


def phi_u(x, dt):
    a, b, k = 0.17, -0.23, 0.12
    cross = np.expm1((2 * a - b) * dt) / (2 * a - b)
    return np.array([np.exp(a * dt) * x[0], np.exp(b * dt) * (x[1] + k * x[0] ** 2 * cross)])


def jac_u(x, dt):
    a, b, k = 0.17, -0.23, 0.12
    cross = np.expm1((2 * a - b) * dt) / (2 * a - b)
    return np.array([[np.exp(a * dt), 0.0], [np.exp(b * dt) * 2 * k * x[0] * cross, np.exp(b * dt)]])


def phi_c(x, dt):
    if dt == 0:
        return x.copy()
    sol = solve_ivp(lambda _, z: vu(z) + np.array([0.35, -0.21]), (0, dt), x,
                    method="DOP853", rtol=2e-12, atol=2e-13)
    assert sol.success
    return sol.y[:, -1]


def check_release():
    errors = dict(release_semigroup=0.0, passive_conservation=0.0, release_derivative=0.0, semantic_derivative=0.0)
    gap = np.array([0.35, -0.21])
    eps = 1e-5
    for _ in range(64):
        x = RNG.normal(size=2)
        t = RNG.uniform(0.05, 0.3)
        s = RNG.uniform(0.4, 0.8)
        z = phi_c(x, s - t)
        end = phi_c(x, 1 - t)
        released = phi_u(z, 1 - s)
        e1 = np.linalg.norm(released - end)
        e2 = np.linalg.norm(released - phi_c(z, 1 - s))
        errors["release_semigroup"] = max(errors["release_semigroup"], abs(e1 - e2))
        errors["passive_conservation"] = max(errors["passive_conservation"], float(np.linalg.norm(phi_u(phi_u(x, s - t), 1 - s) - phi_u(x, 1 - t))))
        plus = phi_u(phi_c(x, s + eps - t), 1 - s - eps)
        minus = phi_u(phi_c(x, s - eps - t), 1 - s + eps)
        exact = jac_u(z, 1 - s) @ gap
        errors["release_derivative"] = max(errors["release_derivative"], float(np.linalg.norm((plus - minus) / (2 * eps) - exact)))
        direction = np.array([1.0, -0.3])
        finite = (np.tanh(direction @ plus) - np.tanh(direction @ minus)) / (2 * eps)
        predicted = (1 - np.tanh(direction @ released) ** 2) * (direction @ exact)
        errors["semantic_derivative"] = max(errors["semantic_derivative"], abs(finite - predicted))
    assert max(errors.values()) < 2e-8, errors
    return dict(cases=64, maximum_errors=errors)


def projected_gap(x):
    g = np.array([1.0, 0.0])
    return g - x * (x @ g) / (x @ x)


def jac_fd(fun, x, eps=1e-6):
    eye = np.eye(len(x))
    return np.column_stack([(fun(x + eps * z) - fun(x - eps * z)) / (2 * eps) for z in eye])


def check_geometry():
    curl_error = norm_error = sym_error = 0.0
    for _ in range(128):
        x = RNG.normal(size=2)
        x *= (0.5 + np.linalg.norm(x)) / np.linalg.norm(x)
        j = jac_fd(projected_gap, x)
        curl_error = max(curl_error, abs(j[1, 0] - j[0, 1] + x[1] / (x @ x)))
        u = projected_gap(x)
        alpha = RNG.uniform(-2, 2)
        norm_error = max(norm_error, abs(np.linalg.norm(x + alpha * u) ** 2 - np.linalg.norm(x) ** 2 - alpha ** 2 * (u @ u)))
        matrix = RNG.normal(size=(5, 5))
        d = RNG.normal(size=5)
        sym = (matrix + matrix.T) / 2
        sym_error = max(sym_error, abs(2 * d @ matrix @ d - 2 * d @ sym @ d))
    assert curl_error < 2e-8
    assert norm_error < 1e-12
    assert sym_error < 1e-12
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    x = np.ones(2)
    isotropic = float(-x @ rot @ x)
    anisotropic = float(-x @ np.diag([1.0, 0.25]) @ rot @ x)
    assert isotropic == 0 and anisotropic == 0.75
    return dict(cases=128, apg_curl_error=curl_error, finite_norm_identity_error=norm_error,
                symmetric_growth_identity_error=sym_error, apg_curl_at_1_1=-0.5,
                gaussian_weighted_divergence_over_density=dict(isotropic=isotropic, anisotropic=anisotropic),
                zero_curl_error_example="d(x) = M e_1 for any M",
                high_curl_small_error=[dict(epsilon=e, uniform_field_bound=e, curl_at_origin=-1 / e) for e in (0.1, 0.01, 0.001)])


def check_momentum():
    beta = -0.5
    clean = corrected = naive = np.zeros(4)
    max_error = 0.0
    gap_rows = []
    previous_b = None
    for k, b in enumerate(np.linspace(1.0, 0.1, 32)):
        gap = RNG.normal(size=4)
        clean = b * gap + beta * clean
        corrected = gap + (0 if previous_b is None else beta * previous_b / b) * corrected
        naive = gap + beta * naive
        err = np.linalg.norm(b * corrected - clean)
        max_error = max(max_error, err)
        gap_rows.append(dict(step=k, b=b, corrected_error=err, constant_velocity_beta_error=np.linalg.norm(b * naive - clean)))
        previous_b = b
    assert max_error < 1e-12
    assert gap_rows[-1]["constant_velocity_beta_error"] > 1e-3
    dump_csv("momentum_parameterization.csv", gap_rows)
    constant = alternating = 0.0
    response = []
    for k in range(24):
        constant = 1 + beta * constant
        sign = (-1) ** k
        alternating = sign + beta * alternating
        response.append(dict(step=k + 1, constant_input=1, constant_response=constant,
                             alternating_input=sign, alternating_response=alternating))
    dump_csv("momentum_response.csv", response)
    assert abs(constant - 2 / 3) < 1e-6
    assert abs(abs(alternating) - 2) < 1e-6
    return dict(beta=beta, clean_memory_equivalence_max_error=max_error,
                naive_velocity_memory_final_error=gap_rows[-1]["constant_velocity_beta_error"],
                constant_signal_limit=2 / 3, alternating_signal_amplitude_limit=2.0)


def rotation_data():
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    eye = np.eye(2)
    rows = []
    for omega in (1.0, 4.0, 8.0, 16.0):
        for steps in (8, 16, 32, 64):
            z = omega / steps
            matrices = dict(Exact=expm(z * rot), Euler=eye + z * rot,
                            Heun=eye + z * rot - z * z * eye / 2,
                            Cayley=np.linalg.solve(eye - z * rot / 2, eye + z * rot / 2))
            for method, matrix in matrices.items():
                end = np.linalg.matrix_power(matrix, steps)
                variance = float(np.trace(end @ end.T) / 2)
                theoretical = (1 + z * z) ** steps if method == "Euler" else (1 + z ** 4 / 4) ** steps if method == "Heun" else 1.0
                assert np.isclose(variance, theoretical, rtol=3e-13, atol=3e-13)
                rows.append(dict(angular_speed=omega, steps=steps, dimensionless_step=z,
                                 method=method, final_variance=variance, theory=theoretical))
    dump_csv("rotation_variance.csv", rows)
    return rows


def check_horizon_units():
    # Scalar output, equal query costs, identical zero-set before/after rescaling.
    e, m, c = 0.2, 0.5, 10.0
    original = abs(m * e)
    rescaled = abs((c * m) * (c * e))
    assert np.isclose(rescaled / original, c * c)
    # A spatially constant guidance scale cancels in a curl/Jacobian norm ratio.
    j = RNG.normal(size=(4, 4))
    ratio = np.linalg.norm(j - j.T) / np.linalg.norm(j)
    errors = [abs(np.linalg.norm(w * j - (w * j).T) / np.linalg.norm(w * j) - ratio) for w in (0.2, 1, 5, 20)]
    assert max(errors) < 1e-12
    return dict(output_rescaling=c, unchanged_zero_set=True, criterion_multiplier=rescaled / original,
                fixed_state_cfg_ratio_invariance_max_error=max(errors))


def check_random_curl_estimator():
    j = RNG.normal(size=(4, 4))
    probes = np.array(list(itertools.product((-1.0, 1.0), repeat=4)))
    samples = [(p @ j @ q - q @ j @ p) ** 2 for p in probes for q in probes]
    estimate = float(np.mean(samples))
    exact = float(np.linalg.norm(j - j.T) ** 2)
    assert np.isclose(estimate, exact, rtol=1e-14, atol=1e-14)
    return dict(exhaustive_probe_pairs=len(samples), expectation=estimate, exact_frobenius_squared=exact)


def check_gauge_mixing():
    # A compatible Gaussian joint: C~N(0,I/2), X|C=c~N(c,I/2).
    # p_u=N(0,I), p_c=N(m,I/2), so the canonical conditional score is 2(m-x).
    # The conditional rotation is p_c-divergence-free. Its CFG combination
    # need not be divergence-free under pi_w=N(2w*m/(1+w),I/(1+w)).
    eye = np.eye(2)
    unit_rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    m = np.array([1.0, 0.0])
    rows = []
    max_identity_error = 0.0
    for omega in (0.0, 0.5, 1.0):
        a = omega * unit_rot
        for w in (0.0, 1.0, 2.0, 4.0):
            precision = 1 + w
            drift_jac = -precision * eye + w * a
            intercept = w * (2 * eye - a) @ m
            stationary = np.linalg.solve(-drift_jac, intercept)
            ideal = 2 * w / precision * m
            assert np.linalg.norm((drift_jac + drift_jac.T) / precision + 2 * eye) < 1e-13
            assert np.linalg.norm(drift_jac @ stationary + intercept) < 1e-13
            for _ in range(32):
                x = RNG.normal(size=2)
                rc = a @ (x - m)
                ru = np.zeros(2)
                residual = w * rc
                # div(pi_w residual)/pi_w; div(Ax)=0.
                direct = -precision * (x - ideal) @ residual
                score_gap = 2 * (m - x) + x
                identity = w * (1 - w) * ((ru - rc) @ score_gap)
                max_identity_error = max(max_identity_error, abs(direct - identity))
            rows.append(dict(omega=omega, scale=w, ideal_mean_x=ideal[0], ideal_mean_y=ideal[1],
                             stationary_mean_x=stationary[0], stationary_mean_y=stationary[1],
                             mean_error=float(np.linalg.norm(stationary - ideal)),
                             stationary_variance_x=1 / precision, stationary_variance_y=1 / precision))
    assert max_identity_error < 1e-12
    example = next(r for r in rows if r["omega"] == 1.0 and r["scale"] == 2.0)
    assert np.allclose([example["stationary_mean_x"], example["stationary_mean_y"]], [16 / 13, 2 / 13])
    # A genuinely categorical example: p_u = (N(m,I)+N(-m,I))/2, p_c=N(m,I).
    x, w = np.array([0.0, 1.0]), 2.0
    sc, su = m - x, -x + m * np.tanh(m @ x)
    rc = unit_rot @ (x - m)
    discrete_direct = ((1 - w) * su + w * sc) @ (w * rc)
    discrete_identity = w * (1 - w) * (-rc @ (sc - su))
    assert discrete_direct == discrete_identity == -2.0
    dump_csv("gauge_mixing_stationary.csv", rows)
    return dict(cases=len(rows), weighted_divergence_identity_max_error=max_identity_error, example=example,
                compatible_joint="C~N(0,I/2), X|C=c~N(c,I/2)",
                discrete_mixture_weighted_divergence_over_density=discrete_direct,
                scope="Exact stationary Langevin counterexample, not a claim that general CFG has tempered marginals")


def figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    ax = axes[0]
    points = {"Conditional":(2, 0), "CFG":(4, 2), "Projected update":(2, 2)}
    geometry_rows = []
    for label, p in points.items():
        start = (0, 0) if label == "Conditional" else (2, 0)
        color = "#222222" if label == "Conditional" else "#999999" if label == "CFG" else "#265a8a"
        ax.annotate("", xy=p, xytext=start, arrowprops=dict(arrowstyle="->", lw=2, color=color))
        ax.plot(*p, "o", color=color)
        ax.text(p[0] + 0.08, p[1] + 0.08, label, color=color)
        geometry_rows.append(dict(label=label, start_x=start[0], start_y=start[1], end_x=p[0], end_y=p[1]))
    ax.plot([2, 4], [0, 0], "--", color="#bbbbbb")
    ax.set(xlim=(-0.2, 5.8), ylim=(-0.4, 3.1), xlabel="Coordinate 1", ylabel="Coordinate 2", title="Projection-only illustrative geometry")
    ax.set_aspect("equal", adjustable="box")
    ax = axes[1]
    for method, color in [("Euler", "#999999"), ("Heun", "#265a8a"), ("Cayley", "#222222")]:
        subset = [r for r in rows if r["method"] == method and r["steps"] == 16]
        ax.plot([r["angular_speed"] for r in subset], [r["final_variance"] for r in subset], "o-", label=method, color=color)
    ax.axhline(1, color="#777777", ls=":", lw=1, label="Exact variance")
    ax.set(yscale="log", xlabel="Angular speed", ylabel="Final variance / initial variance", title="Pure rotation, 16 steps, total time = 1")
    ax.legend(frameon=False)
    fig.savefig(FIG / "projection_and_rotation.svg")
    fig.savefig(FIG / "projection_and_rotation.png", dpi=180)
    plt.close(fig)
    dump_csv("projection_geometry.csv", geometry_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    checks = dict(release=check_release(), geometry=check_geometry(), momentum=check_momentum(),
                  units=check_horizon_units(), random_curl_estimator=check_random_curl_estimator(),
                  gauge_mixing=check_gauge_mixing())
    rotations = rotation_data()
    figures(rotations)
    checks["rotation_cases"] = len(rotations)
    checks["passed"] = True
    checks["scope"] = "CPU analytic/numerical identities only; no neural model or image-quality experiment"
    checks["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checks["numpy_version"] = np.__version__
    (OUT / "audit.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
