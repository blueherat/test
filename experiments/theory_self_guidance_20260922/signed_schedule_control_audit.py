"""CPU analytic audit: signed schedules from geometry versus Heun error.

No model loading and no GPU. The minimum-energy example is an explanatory
control problem, not the unregularized current GAN schedule objective.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

OUT = Path(__file__).resolve().parents[2] / "docs/research/self_guidance_schedule_shape_20260923"


def heun_factor(z):
    return 1 + z + z*z/2


def main():
    omega = 2*np.pi
    target_mean = 1.0
    energy_weight = .25
    amplitude = target_mean / (energy_weight + .5)
    control = lambda t: amplitude*np.cos(omega*t)
    mean = np.array([
        quad(lambda t: np.cos(omega*(1-t))*control(t), 0, 1, epsabs=1e-13)[0],
        quad(lambda t: np.sin(omega*(1-t))*control(t), 0, 1, epsabs=1e-13)[0],
    ])
    terminal_kl = .5*np.sum((mean-np.array([target_mean, 0.]))**2)
    energy = quad(lambda t: control(t)**2, 0, 1, epsabs=1e-13)[0]
    objective = terminal_kl + energy_weight*energy/2
    assert np.max(np.abs(mean-np.array([amplitude/2, 0.]))) < 1e-12
    constant_control_response = np.array([
        quad(lambda t: np.cos(omega*(1-t)), 0, 1)[0],
        quad(lambda t: np.sin(omega*(1-t)), 0, 1)[0],
    ])
    assert np.max(abs(constant_control_response)) < 1e-12
    # The first-order condition is a(t)=k(t).(target-mu)/lambda.
    times = np.linspace(0, 1, 501)
    kernel = np.stack([np.cos(omega*(1-times)), np.sin(omega*(1-times))], axis=1)
    optimality_error = np.max(np.abs(control(times)-kernel@(np.array([target_mean, 0.])-mean)/energy_weight))
    assert optimality_error < 1e-12
    gramian = np.array([[quad(lambda t: np.cos(omega*t)**2, 0, 1)[0],
                         quad(lambda t: -np.cos(omega*t)*np.sin(omega*t), 0, 1)[0]],
                        [quad(lambda t: -np.cos(omega*t)*np.sin(omega*t), 0, 1)[0],
                         quad(lambda t: np.sin(omega*t)**2, 0, 1)[0]]])
    assert np.max(abs(gramian-.5*np.eye(2))) < 1e-12

    # x'=a(t)x, physical thirds have a=(A,-2A,A). Exact endpoint is x0.
    # Repeating Heun substeps within each third isolates solver error.
    scale = 1.5
    refinements = []
    for substeps in [1, 2, 4, 8, 16, 32, 64, 128]:
        step = 1/(3*substeps)
        z = step*scale
        multiplier = (heun_factor(z)**2 * heun_factor(-2*z))**substeps
        refinements.append(dict(steps=3*substeps, multiplier=float(multiplier),
                                error=float(abs(multiplier-1))))
    z = .5
    exact_polynomial = 1+z**3+9*z**4/4+3*z**5/2+z**6/2
    assert abs(heun_factor(z)**2*heun_factor(-2*z)-exact_polynomial) < 1e-14
    assert abs(refinements[0]["multiplier"]-1.3203125) < 1e-14
    assert refinements[-1]["error"] < refinements[0]["error"]/10000

    # Three pulses remove the first two temporal moments. In the rotation
    # example, the response is exactly 2(cos(omega*h)-1)*k(t0).
    center = .5
    kernel_at = lambda t: np.array([np.cos(omega*(1-t)), np.sin(omega*(1-t))])
    pulse_results = []
    for spacing in [.1, .05, .025, .0125]:
        response = kernel_at(center-spacing)-2*kernel_at(center)+kernel_at(center+spacing)
        curvature_prediction = spacing**2*(-omega**2)*kernel_at(center)
        relative_error = np.linalg.norm(response-curvature_prediction)/np.linalg.norm(curvature_prediction)
        pulse_results.append(dict(spacing=spacing, response=response.tolist(),
                                  second_derivative_prediction=curvature_prediction.tolist(),
                                  relative_error=float(relative_error)))
    assert pulse_results[-1]["relative_error"] < pulse_results[0]["relative_error"]/50

    # Separate two positive endpoint-statistic kernels by a signed control.
    # This checks the constrained analytic problem, not the image GAN objective.
    orthogonalization_results = []
    for epsilon in [.5, .2, .1]:
        gain_kernel = lambda t: 1 + epsilon*np.cos(omega*t)
        signed_control = lambda t: 2*target_mean/epsilon*np.cos(omega*t)
        preserved_change = quad(signed_control, 0, 1, epsabs=1e-12)[0]
        desired_change = quad(lambda t: gain_kernel(t)*signed_control(t),
                              0, 1, epsabs=1e-12)[0]
        min_energy = .5*quad(lambda t: signed_control(t)**2,
                            0, 1, epsabs=1e-12)[0]
        rho_squared = 1/(1+epsilon**2/2)
        energy_without_neutrality = target_mean**2/(2*(1+epsilon**2/2))
        assert abs(preserved_change) < 1e-12
        assert abs(desired_change-target_mean) < 1e-12
        assert abs(min_energy-target_mean**2/epsilon**2) < 1e-10
        assert abs(min_energy/energy_without_neutrality -
                   1/(1-rho_squared)) < 1e-9
        orthogonalization_results.append(dict(
            epsilon=epsilon, control_amplitude=2*target_mean/epsilon,
            preserved_statistic_change=float(preserved_change),
            target_statistic_change=float(desired_change),
            minimum_energy=float(min_energy),
            energy_inflation=float(min_energy/energy_without_neutrality),
        ))

    result = dict(
        kind="Analytic continuous control and CPU Heun audit; not image training",
        rotating_drift=dict(
            drift="S(x)=2*pi*J*x", control_direction="B=e1", weak="W=S-e1",
            initial="N(0,I)", strong_endpoint="N(0,I)", weak_endpoint="N(0,I)",
            target="N(e1,I)", energy_weight=energy_weight,
            optimal_control="a(t)=cos(2*pi*t)/(lambda+1/2)",
            amplitude=float(amplitude), values_at_0_half_1=control(np.array([0., .5, 1.])).tolist(),
            endpoint_mean=mean.tolist(), endpoint_kl=float(terminal_kl),
            control_energy=float(energy), total_objective=float(objective),
            constant_control_endpoint_shift=constant_control_response.tolist(),
            controllability_gramian=gramian.tolist(), stationarity_error=float(optimality_error),
            caveat="Actual GAN schedule training has no a^2 penalty; this is a constructive existence example."),
        heun_only_effect=dict(
            ode="x'=a(t)x", physical_control=[scale, -2*scale, scale],
            physical_intervals=[1/3]*3, continuous_multiplier=1.,
            three_step_multiplier_polynomial="1+z^3+9*z^4/4+3*z^5/2+z^6/2",
            refinements=refinements),
        three_pulse_curvature=pulse_results,
        statistic_orthogonalization=dict(
            gain_kernel="f(t)=1+epsilon*cos(2*pi*t)",
            preserved_kernel="g(t)=1",
            objective="min integral(a^2)/2; integral(f*a)=1; integral(g*a)=0",
            solution="a(t)=2*cos(2*pi*t)/epsilon",
            caveat="Local response interpretation needs a valid small-perturbation regime.",
            cases=orthogonalization_results),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"control_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
