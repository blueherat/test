import torch

from experiments.rae_vector_field_switch_probe import (
    SCHEDULES,
    switched_endpoint,
    uses_partial,
)


class ConstantVelocity(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, state, time, y):
        del time, y
        return torch.full_like(state, self.value)


def test_switch_schedule_partition():
    assert not uses_partial("partial_high_ge_085", 0.849)
    assert uses_partial("partial_high_ge_085", 0.85)
    assert uses_partial("partial_mid_030_085", 0.30)
    assert not uses_partial("partial_mid_030_085", 0.85)
    assert uses_partial("partial_low_lt_030", 0.299)
    assert not uses_partial("partial_low_lt_030", 0.30)
    assert set(SCHEDULES) == {
        "baseline",
        "partial",
        "partial_high_ge_085",
        "partial_mid_030_085",
        "partial_low_lt_030",
        "baseline_high_partial_below_085",
    }


def test_switched_endpoint_uses_requested_vector_field():
    baseline = ConstantVelocity(1.0)
    partial = ConstantVelocity(3.0)
    noise = torch.zeros((2, 1, 1, 1))
    labels = torch.zeros(2, dtype=torch.long)
    times = torch.tensor((1.0, 0.8, 0.2, 0.0))
    baseline_result = switched_endpoint(baseline, partial, noise, labels, times, "baseline")
    partial_result = switched_endpoint(baseline, partial, noise, labels, times, "partial")
    high_result = switched_endpoint(
        baseline, partial, noise, labels, times, "partial_high_ge_085"
    )
    torch.testing.assert_close(baseline_result, torch.full_like(noise, -1.0))
    torch.testing.assert_close(partial_result, torch.full_like(noise, -3.0))
    torch.testing.assert_close(high_result, torch.full_like(noise, -1.4))
