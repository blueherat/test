"""CPU illustration of stable second-order dynamics and terminal-dual input.

This is an analytic control example, not a fit to JiT or SiT schedules.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eqvae_self_guidance_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, quad_vec, solve_ivp
from scipy.linalg import expm

OUT = Path(__file__).resolve().parents[2] / "docs/research/self_guidance_schedule_shape_20260923"


def main():
    horizon = 1.
    damping = 1.5
    damped_frequency = 3*np.pi
    frequency_squared = damping*damping+damped_frequency*damped_frequency
    matrix = np.array([[0., 1.], [-frequency_squared, -2*damping]])
    direction = np.array([0., 1.])
    target = np.array([.02, 0.])

    def impulse(tau):
        return np.exp(-damping*tau)*np.array([
            np.sin(damped_frequency*tau)/damped_frequency,
            np.cos(damped_frequency*tau)-damping/damped_frequency*np.sin(damped_frequency*tau),
        ])

    gramian = quad_vec(lambda tau: np.outer(impulse(tau), impulse(tau)), 0, horizon,
                       epsabs=1e-13, epsrel=1e-13)[0]
    dual = np.linalg.solve(gramian, target)
    control = lambda t: float(impulse(horizon-t)@dual)
    times = np.linspace(0, horizon, 1601)
    solution = solve_ivp(lambda t, state: matrix@state+direction*control(t),
                         [0, horizon], np.zeros(2), t_eval=times,
                         rtol=2e-11, atol=2e-13)
    assert solution.success
    endpoint_error = np.linalg.norm(solution.y[:, -1]-target)
    assert endpoint_error < 1e-10
    assert abs(gramian[0, 1]) < 1e-13
    inputs = np.array([control(t) for t in times])
    tau = horizon-times
    coefficient = dual[0]/damped_frequency
    analytic_inputs = coefficient*np.exp(-damping*tau)*np.sin(damped_frequency*tau)
    analytic_error = float(np.max(abs(inputs-analytic_inputs)))
    assert analytic_error < 1e-10
    first_derivative = coefficient*np.exp(-damping*tau)*(
        damping*np.sin(damped_frequency*tau)-damped_frequency*np.cos(damped_frequency*tau))
    second_derivative = coefficient*np.exp(-damping*tau)*(
        (damping*damping-damped_frequency*damped_frequency)*np.sin(damped_frequency*tau)
        -2*damping*damped_frequency*np.cos(damped_frequency*tau))
    adjoint_ode_residual = float(np.max(abs(second_derivative-2*damping*first_derivative
                                           +frequency_squared*analytic_inputs)))
    assert adjoint_ode_residual < 1e-10
    energy = quad(lambda t: control(t)**2, 0, horizon, epsabs=1e-12)[0]
    energy_formula = float(target@np.linalg.solve(gramian, target))
    assert abs(energy-energy_formula) < 1e-11
    lobe_peaks = []
    for left, right in [(0, 1/3), (1/3, 2/3), (2/3, 1)]:
        mask = (times >= left)&(times <= right)
        indices = np.flatnonzero(mask)
        index = indices[np.argmax(abs(inputs[mask]))]
        lobe_peaks.append(dict(time=float(times[index]), value=float(inputs[index])))
    assert lobe_peaks[0]["value"] > 0 > lobe_peaks[1]["value"]
    assert lobe_peaks[2]["value"] > 0
    assert abs(lobe_peaks[2]["value"]) > abs(lobe_peaks[1]["value"]) > abs(lobe_peaks[0]["value"])

    # Heun's discrete local dual signal follows the inverse-map recurrence.
    step = 1/60
    mapping = np.eye(2)+step*matrix+step*step/2*(matrix@matrix)
    local_input = step*(np.eye(2)+step/2*matrix)@direction
    inverse_mapping = np.linalg.inv(mapping)
    signal = np.array([local_input@np.linalg.matrix_power(mapping.T, 59-i)@dual for i in range(60)])
    recurrence = signal[2:]-np.trace(inverse_mapping)*signal[1:-1]+np.linalg.det(inverse_mapping)*signal[:-2]
    recurrence_error = float(np.max(abs(recurrence)))
    assert recurrence_error < 1e-10

    # A +,-,+ signal can also come from THREE real, stable modes.
    roots = np.exp(-np.array([.25, .75]))
    real_dual = np.array([roots.prod(), -roots.sum(), 1.])
    real_matrix = np.diag([-1., -2., -3.])
    real_direction = np.ones(3)
    real_gramian = quad_vec(lambda tau: np.outer(expm(real_matrix*tau)@real_direction,
                                                 expm(real_matrix*tau)@real_direction), 0, 1)[0]
    real_target = real_gramian@real_dual
    real_input = lambda t: float(real_direction@expm(real_matrix.T*(1-t))@real_dual)
    real_values = [real_input(t) for t in [0., .5, 1.]]
    assert real_values[0] > 0 > real_values[1] and real_values[2] > 0

    result = dict(
        kind="Analytic stable-oscillator CPU example; illustration, not fit to learned schedules",
        horizon=horizon, damping_gamma=damping, damped_frequency=damped_frequency,
        natural_frequency=float(np.sqrt(frequency_squared)),
        plant_eigenvalues=[dict(real=float(e.real), imag=float(e.imag)) for e in np.linalg.eigvals(matrix)],
        plant_matrix=matrix.tolist(), input_direction=direction.tolist(),
        target=target.tolist(), endpoint=solution.y[:, -1].tolist(), endpoint_error=float(endpoint_error),
        controllability_gramian=gramian.tolist(), gramian_eigenvalues=np.linalg.eigvalsh(gramian).tolist(),
        dual_vector=dual.tolist(), energy=float(energy), energy_from_gramian=energy_formula,
        input_formula="exp(-gamma*(T-t))*[C*sin(wd*(T-t))+D*cos(wd*(T-t))]",
        input_second_order_ode="a''-2*gamma*a'+omega0^2*a=0",
        analytic_input_error=analytic_error, input_ode_residual=adjoint_ode_residual,
        internal_zero_crossings=[1/3, 2/3], lobe_peaks=lobe_peaks,
        input_at_endpoints=[control(0.), control(1.)],
        heun_dual_recurrence=dict(step=step, trace_inverse=float(np.trace(inverse_mapping)),
                                  determinant_inverse=float(np.linalg.det(inverse_mapping)),
                                  maximum_residual=recurrence_error),
        nonoscillatory_counterexample=dict(
            eigenvalues=[-1., -2., -3.], target=real_target.tolist(),
            input_at_0_half_1=real_values, internal_zero_crossings=[.25, .75],
            caveat="Three real stable modes already permit positive-negative-positive; the shape does not identify a complex pair."),
        interpretation="Late-amplified input is a terminal-dual boundary-value signal. In remaining time T-t its envelope decays; this is not a proof of exploding backpropagation.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"second_order_control_audit.json").write_text(json.dumps(result, indent=2)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 3.8), constrained_layout=True)
    impulse_position = np.array([impulse(t)[0] for t in times])
    axes[0].plot(times, impulse_position, lw=2, color="#2864a0", label="Position impulse response")
    impulse_envelope = np.exp(-damping*times)/damped_frequency
    axes[0].plot(times, impulse_envelope, "--", lw=1, color="#8b9ba8")
    axes[0].plot(times, -impulse_envelope, "--", lw=1, color="#8b9ba8")
    axes[0].set(title="Stable plant: decaying response", xlabel="Elapsed time after impulse", ylabel="Position per unit impulse")
    axes[1].plot(times, inputs, lw=2, color="#b34a43", label="Minimum-energy input")
    envelope = abs(coefficient)*np.exp(-damping*(horizon-times))
    axes[1].plot(times, envelope, "--", lw=1, color="#ba9591")
    axes[1].plot(times, -envelope, "--", lw=1, color="#ba9591")
    axes[1].set(title="Terminal dual: later lobes grow", xlabel="Generation/control time t", ylabel="Control a(t)")
    axes[2].plot(times, solution.y[0]/target[0], lw=2, color="#2864a0", label="Position / target")
    axes[2].plot(times, solution.y[1]/(damped_frequency*target[0]), lw=1.8, color="#c78a29", label="Velocity / (wd * target)")
    axes[2].scatter([horizon], [1.], color="#2864a0", s=28)
    axes[2].set(title="Driven state is a different signal", xlabel="Control time t", ylabel="Normalized state")
    for axis in axes:
        axis.axhline(0., color="#7b7b7b", lw=.7, alpha=.6)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("Stable damped oscillator: analytic illustration, not a fit to SiT/JiT", fontsize=12)
    fig.savefig(OUT/"second_order_control.png", dpi=190)
    fig.savefig(OUT/"second_order_control.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
