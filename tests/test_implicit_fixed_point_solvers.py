from __future__ import annotations

import torch

from experiments.implicit_fixed_point_solvers import integrate_fixed_grid


class LinearField:
    def __init__(self, rate: float):
        self.rate = float(rate)
        self.nfe = 0

    def __call__(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del time
        self.nfe += 1
        return self.rate * state


def test_trapezoid_one_correction_is_heun() -> None:
    initial = torch.tensor([[1.0, -2.0]])
    times = torch.linspace(0.0, 1.0, 9)
    heun = integrate_fixed_grid(LinearField(-0.7), initial, times, method="heun")
    trapezoid = integrate_fixed_grid(
        LinearField(-0.7),
        initial,
        times,
        method="implicit_trapezoid",
        corrections=1,
    )
    torch.testing.assert_close(trapezoid.endpoint, heun.endpoint, rtol=0, atol=0)
    assert trapezoid.nfe == heun.nfe == 16


def test_midpoint_one_correction_is_explicit_midpoint() -> None:
    initial = torch.tensor([1.0])
    times = torch.tensor([0.0, 0.2])
    result = integrate_fixed_grid(
        LinearField(-2.0),
        initial,
        times,
        method="implicit_midpoint",
        corrections=1,
    )
    expected = initial + 0.2 * (-2.0 * (initial + 0.1 * (-2.0 * initial)))
    torch.testing.assert_close(result.endpoint, expected)
    assert result.nfe == 2


def test_backward_euler_picard_converges_to_closed_form() -> None:
    initial = torch.tensor([1.0])
    times = torch.tensor([0.0, 0.2])
    result = integrate_fixed_grid(
        LinearField(-2.0),
        initial,
        times,
        method="backward_euler",
        corrections=12,
    )
    expected = initial / (1.0 - 0.2 * -2.0)
    torch.testing.assert_close(result.endpoint, expected, rtol=2e-5, atol=2e-5)
    assert result.nfe == 13
    assert result.max_last_update_rms < 5e-5


def test_evaluation_counts() -> None:
    initial = torch.tensor([1.0])
    times = torch.linspace(0.0, 1.0, 6)
    cases = {
        "euler": (1, 5),
        "heun": (1, 10),
        "backward_euler": (2, 15),
        "implicit_midpoint": (3, 20),
        "implicit_trapezoid": (4, 25),
    }
    for method, (corrections, expected_nfe) in cases.items():
        field = LinearField(-0.5)
        result = integrate_fixed_grid(
            field,
            initial,
            times,
            method=method,
            corrections=corrections,
        )
        assert result.nfe == expected_nfe
        assert field.nfe == expected_nfe
