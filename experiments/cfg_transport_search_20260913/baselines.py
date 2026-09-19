"""Fixed SiT-S/2 CFG/APG/CTRL baselines for paired transport experiments.

``alpha`` is the EXTRA coefficient: CFG = c + alpha * (c - u).
The cutoff is tested at the accepted left endpoint and held fixed for both
Heun stages. APG and CTRL likewise share the previously accepted history in
both stages, then commit only the first-stage proposal. This reproduces the
Sep10 portfolio APG and Sep11 strong CTRL conventions.

This module uses only the full conditional/null fields; the historical weak
readout evaluated alongside those fields does not enter any of these rules.
Counts are single-branch full/prefix evaluations per output, not multiplied
by batch size. No RNG, image decoder, classifier or auxiliary path is used.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.guidance_pasted_20260912 import common

EPS = 1e-12
TRACE_COLUMNS = ('step', 'time', 'step_size', 'active_alpha', 'state_rms', 'first_velocity_rms')


def make_runtime():
    return common.runtime('sit_small')


def squared(x):
    return x.square().flatten(1).sum(1).reshape(-1, 1, 1, 1)


def norm(x):
    return squared(x).sqrt()


def projection(x, onto):
    inner = (x * onto).flatten(1).sum(1).reshape(-1, 1, 1, 1)
    return onto * (inner / squared(onto).clamp_min(EPS))


def cap(x, radius):
    return x * (radius / norm(x).clamp_min(EPS)).clamp_max(1.)


def field(rt, z, time_value, labels, config, history=None, *, active=True):
    """One field query and a proposed history; does not mutate the old history."""
    kind = config['kind']
    alpha = float(config.get('alpha', 1.25)) if active else 0.
    saved_labels = rt.labels
    try:
        rt.labels = labels
        t = z.new_tensor(time_value)
        conditional = rt.field(z, t, 'full')
        if alpha == 0.:
            return conditional, history
        rt.labels = torch.full_like(labels, 100)
        unconditional = rt.field(z, t, 'full')
    finally:
        rt.labels = saved_labels
    gap = conditional - unconditional
    if kind == 'cfg':
        return conditional + alpha * gap, history
    if kind == 'apg':
        previous = torch.zeros_like(gap) if history is None else history
        momentum = gap + float(config.get('beta', -.5)) * previous
        bounded = cap(momentum, 2 * norm(gap))
        clean = z + (1 - t) * conditional
        direction = bounded - projection(bounded, clean)
        return conditional + alpha * direction, momentum.detach()
    if kind == 'ctrl':
        previous = gap if history is None else history
        decay = float(config.get('lambda_ctrl', config.get('ctrl_lambda', config.get('lambda', 5.))))
        gain = float(config.get('K', .2))
        sliding = gap - previous + decay * previous
        modified = gap - gain * sliding.sign()
        # Historical CTRL corrects the WHOLE conditional/null gap. Replacing
        # this with conditional + alpha * modified would change the baseline.
        return unconditional + (1 + alpha) * modified, modified.detach()
    raise ValueError(f'Unsupported kind: {kind}')


@torch.inference_mode()
def sample(rt, noise, labels, config, snapshots=False):
    """Return latents, counted cost, and optional five-state snapshots/trace.

    Config keys: kind, alpha, steps (64 or 96), cutoff (.75), beta (-.5),
    lambda_ctrl (5), K (.2). All guidance disappears after cutoff, leaving the
    conditional field. ``snapshots`` records initial/quarter/.../final latent
    states, not decoded images or repeated independent generations.
    """
    kind = config['kind']
    steps = int(config.get('steps', 64))
    cutoff = float(config.get('cutoff', .75))
    alpha = float(config.get('alpha', 1.25))
    if getattr(rt, 'name', 'sit_small') != 'sit_small':
        raise ValueError('This sampler is specific to SiT-S/2 ImageNet100.')
    if kind not in ('cfg', 'apg', 'ctrl') or steps not in (64, 96):
        raise ValueError((kind, steps))
    if not 0. <= cutoff <= 1. or not math.isfinite(alpha):
        raise ValueError((cutoff, alpha))
    if noise.ndim != 4 or len(noise) != len(labels):
        raise ValueError('Expected BCHW latents paired with labels.')
    before = rt.counts.copy()
    rt.labels = labels
    z = noise.clone()
    history = None
    saved = {'step_000': z.detach().clone()} if snapshots else None
    save_steps = {round(steps * fraction) for fraction in (.25, .5, .75, 1.)}
    trace = []
    active_steps = 0
    with rt.context():
        for k in range(steps):
            t, h = k / steps, 1 / steps
            active = t < cutoff
            first, proposal = field(rt, z, t, labels, config, history, active=active)
            second, _ = field(rt, z + h * first, t + h, labels, config, history, active=active)
            z = z + (h / 2) * (first + second)
            if active and alpha != 0.:
                history = proposal
                active_steps += 1
            if not torch.isfinite(z).all() or z.abs().max() > 1e6:
                raise FloatingPointError(f'{config.get("arm", kind)}: invalid state at step {k}')
            if snapshots:
                trace.append(torch.stack((
                    z.new_full((len(z),), k), z.new_full((len(z),), t),
                    z.new_full((len(z),), h), z.new_full((len(z),), alpha if active else 0.),
                    z.square().flatten(1).mean(1).sqrt(),
                    first.square().flatten(1).mean(1).sqrt()), -1))
                if k + 1 in save_steps:
                    saved[f'step_{k+1:03d}'] = z.detach().clone()
    counts = {key: rt.counts[key] - before[key] for key in ('full', 'prefix')}
    expected = {'full': 2 * steps + 2 * active_steps, 'prefix': 0}
    if counts != expected:
        raise RuntimeError(f'Unaccounted model calls: {counts}, expected {expected}')
    result = dict(latents=z, counts=counts)
    if snapshots:
        result.update(snapshots=saved, trace=torch.stack(trace).cpu().numpy())
    return result
