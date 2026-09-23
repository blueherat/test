"""Frozen JiT 1-block SSG with trainable signed, full-trajectory guidance scales."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from experiments.adversarial_weak_training_20260915 import common as c
from .jit_ssg import Runtime, ROOT as HEAD_ROOT
from .evaluate_jit_ssg import optimize_constants
from .adapters import materialize_autocast_weights
from .features import enable_backbone_checkpointing
from .schedules import GuidanceSchedule
from .sampler import Sampler

ROOT = HEAD_ROOT.parent/'jit_block1_gan_schedule_20260922'
HEAD = HEAD_ROOT/'blocks1/training_50k/checkpoint_050000.pt'


def signature(module):
    """Byte fingerprint also supporting frozen BF16 parameters."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_runtime(path=HEAD):
    runtime = Runtime(1)
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['step'] == 50000 and state['config']['blocks'] == 1
    assert state['config']['attachment_layer'] == 6 and state['frozen'] == runtime.frozen_hash()
    runtime.head.load_state_dict(state['ema'], strict=True)
    runtime.net.eval().requires_grad_(False); runtime.head.eval().requires_grad_(False)
    provenance = dict(weak_checkpoint=str(path), weak_sha256=c.sha(path), weak_weights='ema',
                      weak_steps=50000, weak_blocks=1, weak_layer=6,
                      strong_checkpoint=state['config']['pretrained_checkpoint'],
                      strong_sha256=state['config']['checkpoint_sha256'], strong_weights='model_ema1')
    return runtime, provenance


def constant_schedule(initial=.5):
    schedule = GuidanceSchedule(50, initial).cuda().eval()
    with torch.no_grad():
        schedule.coefficients.fill_(initial)
    return schedule


class Field:
    def __init__(self, runtime, schedule):
        self.runtime, self.schedule = runtime, schedule

    def at_coefficient(self, state, time, labels, coefficient):
        times = time.expand(len(state)); tv = times[:, None, None, None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            strong, weak = self.runtime.net.forward_with_intermediate(state, times, labels)
            # Use the published clean-prediction mixture and FP32 scale shape.
            # A scalar FP32 * BF16 would instead promote to BF16 in PyTorch.
            total = (1+coefficient).reshape(1, 1, 1, 1).expand(len(state), 1, 1, 1)
            clean = weak + total*(strong-weak)
            return (clean-state)/(1-tv).clamp_min(.05)

    def __call__(self, state, time, labels, index, active):
        assert active
        return self.at_coefficient(state, time, labels, self.schedule(index))


def optimize(runtime, *, precast=True, checkpoint_backbone=False):
    optimize_constants(runtime)
    view = SimpleNamespace(name='jit', model=runtime.net)
    saved = materialize_autocast_weights(view) if precast else 0
    if checkpoint_backbone:
        # The weak adapter also needs input derivatives, including its attention.
        modules = [*runtime.net.blocks, *runtime.net.ssg_adapter_blocks]
        enable_backbone_checkpointing(SimpleNamespace(model=SimpleNamespace(blocks=modules)))
    return saved


def sampler(runtime, schedule, noise, labels, *, graphs=True):
    if any(p.requires_grad for p in runtime.net.parameters()):
        raise ValueError('Strong backbone and 1-block weak head must both be frozen')
    grid = torch.linspace(0, 1, 51, device=noise.device)
    indices = torch.arange(50, device=noise.device, dtype=noise.dtype)
    return Sampler(Field(runtime, schedule), schedule, noise, labels, grid, indices,
                   [True]*50, heun=[True]*49+[False], graphs=graphs)


def reference(runtime, schedule, noise, labels):
    """Ordinary autograd through all 49 Heun updates and the last Euler update."""
    field = Field(runtime, schedule)
    grid = torch.linspace(0, 1, 51, device=noise.device)

    def update(state, coefficient, t, u, labels, heun):
        first = field.at_coefficient(state, t, labels, coefficient)
        if heun:
            second = field.at_coefficient(state+(u-t)*first, u, labels, coefficient)
            return state+(u-t)*(.5*(first+second))
        return state+(u-t)*first

    state = noise
    for i in range(50):
        values = (state, schedule.coefficients[i], grid[i], grid[i+1], labels, i<49)
        state = (checkpoint(update, *values, use_reentrant=False, preserve_rng_state=False)
                 if torch.is_grad_enabled() else update(*values))
    return state


def decoder(state):
    return ((state.float()+1)/2).clamp(0, 1)
