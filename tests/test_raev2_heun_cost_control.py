import torch
from experiments.raev2_heun_cost_control import predictor, corrected


def test_linear_dataward_ode_has_heun_polynomial():
    state = torch.tensor([[1., -2.]], dtype=torch.float64)
    matrix = torch.tensor([[.3, .2], [-.1, .4]], dtype=torch.float64)
    t, s = .7, .4
    clean = state + t * (state @ matrix.T)
    predicted = predictor(state, clean, t, s)
    next_clean = predicted + s * (predicted @ matrix.T)
    result = corrected(state, predicted, clean, next_clean, t, s)
    expected = state + (t-s) * (state @ matrix.T) + .5 * (t-s)**2 * (state @ matrix.T @ matrix.T)
    torch.testing.assert_close(result, expected, atol=1e-15, rtol=1e-15)


def test_final_euler_step_does_not_require_zero_time_denoiser():
    state = torch.tensor([2., 3.])
    clean = torch.tensor([1., -1.])
    torch.testing.assert_close(predictor(state, clean, .075, 0.), clean)
