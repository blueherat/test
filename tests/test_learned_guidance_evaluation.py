"""Sampling protocol checks: signed intervals, Heun stage sharing, final Euler."""
import pytest
import torch

from classifier_guidance.evaluate_learned_guidance import integrate
from classifier_guidance.sampler import Sampler
from classifier_guidance.schedules import GuidanceSchedule


@pytest.mark.parametrize('final_euler', [False, True])
def test_evaluation_matches_training_integrator_with_signed_schedule(final_euler):
    schedule = GuidanceSchedule(4).eval()
    schedule.coefficients.data.copy_(torch.tensor([.8, 0., -.7, 1.2]))
    labels = torch.tensor([0, 3])
    noise = torch.tensor([[.2, -.3], [.7, -.5]])
    seen = []
    def field(x, t, y, index, active):
        assert active
        seen.append((float(t), int(index)))
        return torch.sin(x+t+y[:, None]*.1)+schedule(index)*torch.cos(x-t)
    grid = torch.linspace(0, 1, 5)
    engine = Sampler(field, schedule, noise, labels, grid, torch.arange(4), [True]*4,
                     heun=[True]*3+[not final_euler], graphs=False)
    with torch.no_grad():
        expected = engine(noise, labels)
        seen.clear()
        actual = integrate(field, noise, labels, 4, final_euler)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert seen == [
        pair for i in range(4)
        for pair in ([(i/4, i)] if final_euler and i == 3 else [(i/4, i), ((i+1)/4, i)])]


def test_negative_and_zero_scales_remain_active_in_late_intervals():
    coefficients = torch.tensor([2., 0., -4., -2.])
    def field(x, t, y, index, active):
        assert active
        return torch.ones_like(x)*coefficients[index.long()]
    actual = integrate(field, torch.zeros(2, 3), torch.tensor([0, 1]), 4)
    torch.testing.assert_close(actual, torch.full((2, 3), -1.), rtol=0, atol=0)
