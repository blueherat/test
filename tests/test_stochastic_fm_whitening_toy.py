import numpy as np

from experiments.stochastic_fm_whitening_toy import (
    analytic_operators,
    analytic_trajectory,
    make_toy,
    mechanism_checks,
    monte_carlo_validation,
    optimal_gamma,
    run_sweep,
)


def test_full_whitening_isotropic_curvature_and_full_batch_converges():
    toy = make_toy(parameter_dim=8, seed=0)
    operators = analytic_operators(toy, gamma=1.0, batch_size=np.inf)
    trajectory = analytic_trajectory(toy, gamma=1.0, batch_size=np.inf, steps=40)

    assert np.isclose(operators.condition_number, 1.0, atol=1e-10)
    assert np.allclose(operators.hessian, np.eye(8) / 8.0, atol=1e-12)
    assert np.isclose(trajectory["risk"].iloc[0], 1.0, atol=1e-12)
    assert trajectory["risk"].iloc[-1] < 1e-12
    assert np.allclose(trajectory["misadjustment"], 0.0)


def test_finite_batch_has_fractional_optimum_that_moves_with_batch_size():
    toy = make_toy(parameter_dim=8, seed=0)
    _, summary = run_sweep(
        toy,
        gammas=np.linspace(0.0, 1.0, 21),
        batch_sizes=(4, 16, 64, 256, np.inf),
        steps=500,
    )
    optimum = optimal_gamma(summary)
    finite = optimum[np.isfinite(optimum["batch_size"])]

    assert np.all(finite["gamma"].to_numpy() < 1.0)
    assert np.all(np.diff(finite["gamma"].to_numpy()) >= 0.0)
    assert optimum[np.isinf(optimum["batch_size"])]["gamma"].item() == 1.0


def test_monte_carlo_matches_closed_form_within_sampling_error():
    toy = make_toy(parameter_dim=4, seed=0)
    validation = monte_carlo_validation(
        toy,
        gamma=0.5,
        batch_size=16,
        steps=80,
        runs=4096,
        seed=123,
    )
    selected = validation.iloc[[10, 20, 40, 80]]
    tolerance = 4.0 * selected["monte_carlo_se"] + 2e-4
    assert np.all(selected["absolute_error"] <= tolerance)


def test_default_mechanism_checks_pass():
    checks = mechanism_checks(
        make_toy(parameter_dim=8, seed=0),
        make_toy(parameter_dim=4, seed=0),
        steps=500,
    )
    assert checks["passed"].all(), checks.to_string(index=False)
