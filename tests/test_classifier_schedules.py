import pytest
import torch

from classifier_guidance.sampler import Sampler
from classifier_guidance.schedules import GuidanceSchedule


@pytest.mark.parametrize('heun', [False, True, (True, True, True, False)])
def test_signed_schedule_has_exact_full_trajectory_gradients_at_zero_and_negative(heun):
    schedule = GuidanceSchedule(4).double().eval()
    with torch.no_grad():
        schedule.coefficients[1] = -.2
    noise = torch.tensor([[.2, -.3], [.7, .1]], dtype=torch.double, requires_grad=True)
    labels = torch.tensor([0, 1])
    grid = torch.tensor([0., .1, .5, .8, 1.], dtype=torch.double)
    indices = torch.arange(4, dtype=torch.double)
    frozen = torch.nn.Linear(2, 2).double().eval().requires_grad_(False)

    def field(x, time, y, index, active):
        strong = torch.tanh(frozen(x)+time+y[:, None]*.2)
        weak = torch.sin(frozen(x)*.3-time)
        return strong + schedule(index)*(strong-weak)

    engine = Sampler(field, schedule, noise, labels, grid, indices, [True]*4,
                     heun=heun, graphs=False)
    def reference(x):
        for i in range(4):
            h = grid[i+1]-grid[i]
            v = field(x, grid[i], labels, indices[i], True)
            p = x+h*v
            use_heun = heun if isinstance(heun, bool) else heun[i]
            x = x+h/2*(v+field(p,grid[i+1],labels,indices[i],True)) if use_heun else p
        return x

    expected, actual = reference(noise), engine(noise, labels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    variables = (noise, schedule.coefficients)
    reference_grad = torch.autograd.grad(expected.square().mean(), variables)
    actual_grad = torch.autograd.grad(actual.square().mean(), variables)
    for a,b in zip(actual_grad,reference_grad):
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    assert torch.all(actual_grad[1][2:].abs()>1e-8), 'Initially zero tail must receive gradients'
    assert torch.autograd.gradcheck(lambda x,*p:engine(x,labels),variables,
                                   eps=1e-6,atol=1e-6,rtol=1e-4)
    assert all(p.grad is None for p in frozen.parameters())
