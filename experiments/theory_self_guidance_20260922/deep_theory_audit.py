"""CPU audits of gauge invariants, future-response activation, and anchor bias.

All covariances below use exact matrix exponentials for continuous linear flows.
These are mathematical examples, not GAN training or image-quality experiments.
"""
from pathlib import Path
import json
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eqvae_self_guidance_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
from scipy.stats import binom
from numpy.polynomial.hermite import hermgauss


OUT = Path(__file__).resolve().parents[2] / "docs/research/self_guidance_breakthrough_20260922"
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def covariance(flow):
    return flow @ flow.T


def kl_to_standard(cov):
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    return float((np.trace(cov) - 2 - logdet) / 2)


def two_stage_response(k=.8, angle=np.pi / 6):
    first = np.diag([k, -k])
    rotation = expm(angle * J)
    second = rotation @ first @ rotation.T

    def prefix(t):
        if t <= .5:
            return expm(t * first)
        return expm((t - .5) * second) @ expm(.5 * first)

    def suffix(t):
        if t <= .5:
            return expm(.5 * second) @ expm((.5 - t) * first)
        return expm((1 - t) * second)

    terminal = covariance(prefix(1))
    terminal_costate = np.eye(2) - np.linalg.inv(terminal)

    def response(t):
        ct = covariance(prefix(t))
        future = suffix(t)
        kt = future.T @ terminal_costate @ future
        # Field perturbation is +omega(t)*J*x. For v=v0-a*u, absorb -a
        # into omega; this does not change the weak marginal preservation.
        return float(np.trace(kt @ J @ ct))

    finite_checks = []
    for center in [.15, .35, .65, .85]:
        left, right = center - .01, center + .01
        local = first if center < .5 else second

        def loss(epsilon):
            flow = suffix(right) @ expm((right - left) * (local + epsilon * J)) @ prefix(left)
            return kl_to_standard(covariance(flow))

        eps = 1e-4
        finite = (loss(eps) - loss(-eps)) / (2 * eps)
        predicted = quad(response, left, right, epsabs=1e-12)[0]
        finite_checks.append(dict(time=center, predicted=predicted, finite_difference=finite,
                                  absolute_error=abs(predicted - finite)))
    times = np.linspace(0, 1, 201)
    values = np.array([response(t) for t in times])
    assert abs(values[0]) < 1e-12 and abs(values[-1]) < 1e-12
    assert np.max(np.abs(values)) > .1
    assert max(row['absolute_error'] for row in finite_checks) < 1e-8
    return times, values, dict(
        baseline_terminal_KL=kl_to_standard(terminal),
        endpoint_response=[float(values[0]), float(values[-1])],
        max_absolute_interior_response=float(np.max(np.abs(values))),
        finite_difference_checks=finite_checks,
    )


def torus_density_steering():
    """Two divergence-free directions with smooth gates steer a fixed density.

    The analytic target path is q_t/p = 1 + amplitude*t*sin(z) on T^3.
    Verify the actual characteristic density using a separate Liouville ODE.
    """
    amplitude = .5
    starts = np.random.default_rng(20260923).uniform(-np.pi, np.pi, (32, 3))

    def velocity(t, points):
        x, _, z = points.T
        ratio = 1 + amplitude * t * np.sin(z)
        b1 = amplitude * .5 * np.sin(2 * x) * np.sin(z)
        b2 = -2 * amplitude * np.sin(x) * np.cos(z)
        return -np.column_stack([b1, b2 * np.cos(x), b2 * np.sin(x)]) / ratio[:, None]

    def divergence(t, points):
        x, _, z = points.T
        ratio = 1 + amplitude * t * np.sin(z)
        b2 = -2 * amplitude * np.sin(x) * np.cos(z)
        return (-amplitude * np.sin(z) / ratio
                + b2 * np.sin(x) * amplitude * t * np.cos(z) / ratio**2)

    finite_div = np.zeros(len(starts))
    for coordinate in range(3):
        shift = np.zeros(3)
        shift[coordinate] = 1e-5
        finite_div += (velocity(.4, starts + shift)[:, coordinate]
                       - velocity(.4, starts - shift)[:, coordinate]) / 2e-5
    divergence_error = float(np.max(np.abs(finite_div - divergence(.4, starts))))
    assert divergence_error < 1e-8

    def rhs(t, state):
        rows = state.reshape(-1, 4)
        return np.column_stack([velocity(t, rows[:, :3]),
                                -divergence(t, rows[:, :3])]).ravel()

    initial = np.column_stack([starts, np.zeros(len(starts))]).ravel()
    solution = solve_ivp(rhs, (0, 1), initial, method='DOP853', rtol=1e-10, atol=1e-11)
    assert solution.success
    terminal = solution.y[:, -1].reshape(-1, 4)
    expected_log_density = np.log1p(amplitude * np.sin(terminal[:, 2]))
    density_error = float(np.max(np.abs(terminal[:, 3] - expected_log_density)))
    assert density_error < 1e-8
    return dict(
        reference="Uniform probability on the three-dimensional torus",
        target_ratio="1 + 0.5*sin(z)",
        directions=["(1,0,0)", "(0,cos(x),sin(x))"],
        bracket="(0,-sin(x),cos(x)); spans all three coordinates with the directions",
        number_of_characteristics=len(starts),
        pointwise_divergence_finite_difference_error=divergence_error,
        characteristic_log_density_error=density_error,
        claim="Exact smooth density path from analytic continuity equation; numerical characteristic check",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # A pure p-preserving rotation may rearrange q but cannot reduce KL(q||p).
    initial_cov = np.diag([4.0, .25])
    pure_angles = np.linspace(0, np.pi, 101)
    pure_kls = [kl_to_standard(covariance(expm(angle * J) @ np.diag([2., .5])))
                for angle in pure_angles]
    assert np.ptp(pure_kls) < 1e-12

    # Gauge response can be zero to first order, while a finite control succeeds.
    k = .8
    strain = np.diag([k, -k])

    def strain_loss(omega):
        return kl_to_standard(covariance(expm(strain + omega * J)))

    exact_omega = math.sqrt(k * k + np.pi * np.pi)
    eps = 1e-3
    derivative = (strain_loss(eps) - strain_loss(-eps)) / (2 * eps)
    second_derivative = (strain_loss(eps) - 2 * strain_loss(0) + strain_loss(-eps)) / eps**2
    exact_flow = expm(strain + exact_omega * J)
    assert abs(derivative) < 1e-12
    assert second_derivative < 0
    assert np.linalg.norm(exact_flow + np.eye(2)) < 1e-12

    # A nonlinear Gaussian-preserving gauge escapes the linear rotation's
    # first-order symmetry trap: u=(x*(y*y-1), y*(1-x*x)).
    nonlinear_times = np.linspace(0, 1, 101)
    nonlinear_responses = 2 * (np.sinh(2 * k * (1 - nonlinear_times))
                               + np.sinh(2 * k * nonlinear_times) - np.sinh(2 * k))
    assert np.max(nonlinear_responses[1:-1]) < 0

    # A state gate activates a direction invisible to every time-only gate.
    # p=N(0,I), S=W=0, augmented weak=Jx; a(x)=beta*x*y.
    # The exact map keeps radius and changes tan(theta) by exp(-beta*r^2*t).
    nodes, weights = hermgauss(50)
    gx, gy = np.meshgrid(np.sqrt(2) * nodes, np.sqrt(2) * nodes, indexing='ij')
    gw = weights[:, None] * weights[None, :] / np.pi
    radius_sq = gx * gx + gy * gy

    def gated_cov(epsilon):
        attenuation = np.exp(-epsilon * radius_sq)
        denominator = np.sqrt(gx * gx + attenuation * attenuation * gy * gy)
        tx = np.sqrt(radius_sq) * gx / denominator
        ty = np.sqrt(radius_sq) * attenuation * gy / denominator
        return np.array([[np.sum(gw * tx * tx), np.sum(gw * tx * ty)],
                         [np.sum(gw * tx * ty), np.sum(gw * ty * ty)]])

    gate_eps = 1e-5
    gate_derivative = (gated_cov(gate_eps) - gated_cov(-gate_eps)) / (2 * gate_eps)
    assert np.max(np.abs(gate_derivative - np.diag([2., -2.]))) < 1e-7
    assert abs(np.trace(gated_cov(.2)) - 2) < 1e-12

    times, response, response_info = two_stage_response()

    # Exact minibatch penalty after optimizing an absorbable global scale.
    # e0=1; e_phi takes 1 or 9 with equal probability. The nonlinear penalty
    # retains Var(log(mean(e_phi))) even after its mean is centered by scale.
    anchor_rows = []
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        successes = np.arange(n + 1)
        probs = binom.pmf(successes, n, .5)
        logs = np.log(1 + 8 * successes / n)
        mean = float(probs @ logs)
        variance = float(probs @ (logs - mean)**2)
        anchor_rows.append(dict(probe_batch=n, minimum_penalty=variance,
                                optimal_gap_scale=float(np.exp(-mean / 2)),
                                large_batch_leading_term=.64 / n))
    assert abs(anchor_rows[0]['minimum_penalty'] - np.log(3.)**2) < 1e-12
    assert all(row['minimum_penalty'] > 0 for row in anchor_rows)

    # Same output direction can be much cheaper through the weak block.
    allocation_rows = [dict(epsilon=e, projected_budget_descent=e,
                            joint_budget_descent=float(np.sqrt(1 + e * e)),
                            efficiency_ratio=float(e / np.sqrt(1 + e * e)))
                       for e in [1., .1, .01, .001]]

    result = dict(
        kind="CPU continuous Gaussian and exact binomial theoretical audits; not image FID",
        pure_gauge=dict(initial_covariance=initial_cov.tolist(), KL_value=pure_kls[0],
                        KL_max_minus_min=float(np.ptp(pure_kls))),
        first_order_blind_case=dict(k=k, baseline_KL=strain_loss(0),
                                    derivative_at_zero=derivative,
                                    second_derivative_at_zero=second_derivative,
                                    exact_finite_control=exact_omega,
                                    finite_control_KL_roundoff=strain_loss(exact_omega),
                                    flow_error_to_minus_identity=float(np.linalg.norm(exact_flow + np.eye(2)))),
        nonlinear_gauge=dict(field="(x*(y^2-1), y*(1-x^2))",
                             weighted_divergence="identically zero under N(0,I)",
                             midpoint_KL_first_variation=float(nonlinear_responses[50]),
                             all_interior_responses_negative=bool(np.all(nonlinear_responses[1:-1] < 0))),
        state_gate=dict(field="-beta*x*y*J*x; beta*time is the map parameter",
                        covariance_derivative=gate_derivative.tolist(),
                        expected_covariance_derivative=[[2., 0.], [0., -2.]],
                        exact_map_covariance_at_beta_time_point2=gated_cov(.2).tolist(),
                        invariant="Every particle radius is constant, so radial density cannot be corrected."),
        future_response=response_info,
        multi_direction_torus=torus_density_steering(),
        minibatch_anchor=anchor_rows,
        scale_first_budget_counterexample=allocation_rows,
        assumptions=["Gaussian examples hold all non-rotation controls fixed.",
                     "Anchor calculation ignores epsilon and assumes the global scale is absorbable.",
                     "No claim that Gaussian reference, critic, or perfect score assumptions hold for SiT/JiT."],
    )
    (OUT / "deep_theory_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savetxt(OUT / "future_gauge_response.csv", np.column_stack([times, response]),
               delimiter=",", header="time,terminal_KL_derivative_per_unit_rotation", comments="")
    np.savetxt(OUT / "nonlinear_gauge_response.csv", np.column_stack([nonlinear_times, nonlinear_responses]),
               delimiter=",", header="time,terminal_KL_derivative_per_unit_nonlinear_gauge", comments="")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    axes[0].plot(times, response, color="#4477aa", linewidth=2)
    axes[0].axhline(0, color="black", linewidth=.6)
    axes[0].set(title="Future dynamics activate gauge feedback", xlabel="Time", ylabel="Terminal KL response")
    omegas = np.linspace(-4, 4, 241)
    axes[1].plot(omegas, [strain_loss(w) for w in omegas], color="#cc6677", linewidth=2)
    axes[1].scatter([0, exact_omega], [strain_loss(0), strain_loss(exact_omega)], color="black", s=22)
    axes[1].set(title="Zero gradient can hide a useful control", xlabel="Constant rotation rate", ylabel="Terminal KL")
    axes[2].loglog([r['probe_batch'] for r in anchor_rows],
                   [r['minimum_penalty'] for r in anchor_rows], 'o-', color="#228833", label="Exact remaining penalty")
    axes[2].loglog([r['probe_batch'] for r in anchor_rows],
                   [r['large_batch_leading_term'] for r in anchor_rows], '--', color="#777777", label="0.64 / batch")
    axes[2].set(title="Batch size changes the anchor objective", xlabel="Probe batch size", ylabel="Penalty after best scale")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT / "deep_theory_audit.png", dpi=180)
    fig.savefig(OUT / "deep_theory_audit.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
