"""Paired mean/AC decomposition of the frozen discrete CTRL recurrence.

At each two-step block, compute two virtual CTRL updates using the FIRST
stage's gap. Freeze their mean correction and half-difference for that block.
The states, times, conditional/null fields still advance normally. Both Heun
stages within a step use the same added correction. All rho arms commit the
same virtual second update as controller history, never a rho-dependent
actual output. Paths can still change future gaps indirectly.

rho=0: mean only; rho=+1: original virtual order; rho=-1: reversed AC phase.
This is a new discrete algorithm on changing gaps, not exact legacy CTRL.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from experiments.cfg_transport_search_20260913 import baselines

SOURCE_FILES = [str(Path(baselines.__file__).resolve()),
                str(Path(baselines.common.__file__).resolve()),
                str(baselines.common.WORK / 'experiments/lifting_scale_sweep_20260909.py')]
TRACE_COLUMNS = ('step', 'time', 'active_alpha', 'rho', 'state_rms',
                 'mean_correction_rms', 'ac_correction_rms',
                 'actual_correction_rms', 'gap_change_from_block_start_rms')


def make_runtime():
    return baselines.make_runtime()


def ctrl_update(gap, previous, gain, decay):
    # Keep the frozen baseline's floating-point association for the relay.
    sliding = (gap - previous) + decay * previous
    return gap - gain * sliding.sign()


def packet(gap, previous, gain, decay):
    """No model call. Return mean/AC corrections and virtual second history."""
    previous = gap if previous is None else previous
    first = ctrl_update(gap, previous, gain, decay)
    second = ctrl_update(gap, first, gain, decay)
    mean = (first + second) / 2 - gap
    ac = (first - second) / 2
    return mean, ac, second.detach()


def conditional(rt, z, t, labels):
    previous_labels = rt.labels
    rt.labels = labels
    try:
        return rt.field(z, z.new_tensor(t), 'full')
    finally:
        rt.labels = previous_labels


def pair(rt, z, t, labels):
    previous_labels = rt.labels
    try:
        rt.labels = labels
        c = rt.field(z, z.new_tensor(t), 'full')
        rt.labels = torch.full_like(labels, 100)
        u = rt.field(z, z.new_tensor(t), 'full')
        return c, u
    finally:
        rt.labels = previous_labels


def rms(x):
    return x.square().flatten(1).mean(1).sqrt()


@torch.inference_mode()
def sample(rt, noise, labels, config, snapshots=False):
    """Runner API; one trajectory and 224/0 calls at alpha>0, 64 steps, .75.

    kind='ctrl_modulation', rho in {0,+1,-1}, alpha=2.75,
    lambda_ctrl=5, K=.2, steps=64 (or 96), cutoff=.75.
    The active prefix must contain an even number of full Heun intervals.
    """
    kind = config.get('kind', 'ctrl_modulation')
    rho = float(config['rho'])
    alpha = float(config.get('alpha', 2.75))
    gain = float(config.get('K', .2))
    decay = float(config.get('lambda_ctrl', 5.))
    steps = int(config.get('steps', 64))
    cutoff = float(config.get('cutoff', .75))
    if kind != 'ctrl_modulation' or rho not in (0., 1., -1.) or steps not in (64, 96):
        raise ValueError((kind, rho, steps))
    if not all(math.isfinite(x) for x in (alpha, gain, decay, cutoff)):
        raise ValueError('Nonfinite configuration')
    if not 0 <= cutoff <= 1 or gain < 0 or decay <= 1:
        raise ValueError((cutoff, gain, decay))
    active_count = sum(k / steps < cutoff for k in range(steps)) if alpha != 0 else 0
    if active_count % 2:
        raise ValueError('The cutoff leaves an incomplete two-step control block.')
    if getattr(rt, 'name', 'sit_small') != 'sit_small' or noise.ndim != 4 or len(noise) != len(labels):
        raise ValueError('Expected SiT-S/2 ImageNet100 paired BCHW latents/labels.')
    z = noise.clone()
    rt.labels = labels
    before = rt.counts.copy()
    history = None
    mean = ac = next_history = block_gap = None
    saved = {'step_000': z.detach().clone()} if snapshots else None
    save_steps = {round(steps * fraction) for fraction in (.25, .5, .75, 1.)}
    trace = []
    w = 1 + alpha
    with rt.context():
        for k in range(steps):
            t, h = k / steps, 1 / steps
            active = k < active_count
            if active:
                c, u = pair(rt, z, t, labels)
                gap = c - u
                if k % 2 == 0:
                    block_gap = gap.detach()
                    mean, ac, next_history = packet(gap, history, gain, decay)
                phase = 1 if k % 2 == 0 else -1
                correction = w * (mean + phase * rho * ac)
                first = c + alpha * gap + correction
                c_end, u_end = pair(rt, z + h * first, t + h, labels)
                second = c_end + alpha * (c_end - u_end) + correction
            else:
                first = conditional(rt, z, t, labels)
                second = conditional(rt, z + h * first, t + h, labels)
            z = z + (h / 2) * (first + second)
            if active and k % 2 == 1:
                history = next_history
            if not torch.isfinite(z).all() or z.abs().max() > 1e6:
                raise FloatingPointError(f'{config.get("arm", kind)}: invalid state at step {k}')
            if snapshots:
                zeros = z.new_zeros((len(z),))
                trace.append(torch.stack((
                    z.new_full((len(z),), k), z.new_full((len(z),), t),
                    z.new_full((len(z),), alpha if active else 0.),
                    z.new_full((len(z),), rho), rms(z),
                    rms(w * mean) if active else zeros,
                    rms(w * ac) if active else zeros,
                    rms(correction) if active else zeros,
                    rms(gap - block_gap) if active else zeros), -1))
                if k + 1 in save_steps:
                    saved[f'step_{k+1:03d}'] = z.detach().clone()
    counts = {key: rt.counts[key] - before[key] for key in ('full', 'prefix')}
    expected = dict(full=2 * steps + 2 * active_count, prefix=0)
    if counts != expected:
        raise RuntimeError(f'Unaccounted calls: {counts}, expected {expected}')
    result = dict(latents=z, counts=counts)
    if snapshots:
        result.update(snapshots=saved, trace=torch.stack(trace).cpu().numpy())
    return result
