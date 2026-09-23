"""Signed, per-solver-step IG coefficients with frozen native SiT heads."""
import torch
from torch import nn

from .sampler import Sampler, prepare_adapter


class GuidanceSchedule(nn.Module):
    def __init__(self, steps=64, initial=.6):
        super().__init__()
        if steps < 2 or steps % 2:
            raise ValueError('Use a positive even step count')
        values = torch.zeros(steps)
        values[:steps//2] = initial
        # No exp, softplus, clipping or zero-dependent gate: negative values
        # and gradients at initially zero late-time coefficients are allowed.
        self.coefficients = nn.Parameter(values)

    def forward(self, index):
        # A scalar CUDA tensor used as a Python index would synchronize to the
        # host; gather keeps the changing step index entirely on the device.
        return torch.gather(self.coefficients,0,index.to(dtype=torch.long).reshape(1)).squeeze(0)


class NativeIGField:
    def __init__(self, adapter, schedule):
        self.adapter, self.schedule = adapter, schedule

    def __call__(self, state, time, labels, index, active):
        a = self.adapter
        with a.autocast():
            strong, _ = a.full(state, time, labels)
            if not active:
                return strong  # Unused inactive capture; rollout enables every step.
            # Frozen parameters retain all input Jacobians; do not detach.
            weak = a.unpatch(a.native(a.values['context'], a.values['condition'])).float()
            return strong + self.schedule(index) * (strong-weak)


def for_native_sit(adapter, schedule, noise, labels, *, graphs=True):
    if adapter.name != 'sit_small':
        raise ValueError('This controlled experiment uses the original SiT depth4 head')
    if any(p.requires_grad for m in (adapter.model, adapter.native) for p in m.parameters()):
        raise ValueError('Both strong model and native weak head must be frozen')
    if any(m.training for root in (adapter.model, adapter.native, schedule) for m in root.modules()):
        raise ValueError('All sampling modules must use eval mode')
    prepare_adapter(adapter, schedule, noise.device)
    steps = schedule.coefficients.numel()
    grid = torch.linspace(0, 1, steps+1, device=noise.device)
    # Fixed indices are merely field inputs. Learnable coefficients live inside
    # schedule.parameters(), so the existing exact discrete VJP accumulates them.
    indices = torch.arange(steps, device=noise.device, dtype=noise.dtype)
    return Sampler(NativeIGField(adapter, schedule), schedule, noise, labels,
                   grid, indices, [True]*steps, heun=True, graphs=graphs)


def native_reference(adapter, schedule, noise, labels):
    """Independent native readout with exact checkpointed ordinary autograd."""
    from torch.utils.checkpoint import checkpoint
    grid = torch.linspace(0, 1, schedule.coefficients.numel()+1, device=noise.device)
    state = noise
    def step(state, coefficient, t, u):
        def field(x, time):
            strong, weak = adapter.full(x, time, labels, native=True)
            return strong + coefficient*(strong-weak)
        first = field(state, t)
        second = field(state+(u-t)*first, u)
        return state+(u-t)/2*(first+second)
    for i, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
        inputs = (state, schedule.coefficients[i], t, u)
        state = checkpoint(step, *inputs, use_reentrant=False) if torch.is_grad_enabled() else step(*inputs)
    return state
