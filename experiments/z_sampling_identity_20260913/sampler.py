"""A bounded FM/Euler Z-Sampling comparison at 224 branch calls per output.

alpha is the EFFECTIVE extra CFG coefficient. At each of the first 42 of 56
Euler intervals, strong w_s=(1+alpha+w_b)/2, with w_b in {0,1}. Thus the
leading guided field is c+alpha*(c-u). All event queries use the physical
accepted-left time, including the inverse leg. The two-Picard variant solves
the inverse of that left-time Euler map approximately, not an exact ODE flow.

This is an existing Z/implicit-inverse construction, not a quality guarantee.
No randomness, new source conditioning, learned parameters, or image I/O.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from experiments.cfg_transport_search_20260913 import baselines as b

SOURCE_FILES = [Path(__file__).resolve(), Path(__file__).with_name('cpu_check.py'),
                Path(__file__).with_name('configs.py'), Path(__file__).with_name('configs.json'),
                Path(__file__).with_name('run.py'), Path(b.__file__).resolve()]
TRACE_COLUMNS = ('step', 'time', 'step_size', 'effective_extra_alpha',
                 'state_rms', 'first_velocity_rms')
Z_KINDS = ('z_release_euler', 'z_anchored_euler')
EULER_KINDS = ('cfg_euler', 'cfg_euler128')


def _field(rt, state, time_value, labels, branch):
    """One real conditional or null model branch; restore runtime labels."""
    if branch not in ('c', 'u'):
        raise ValueError(branch)
    saved_labels = rt.labels
    try:
        rt.labels = labels if branch == 'c' else torch.full_like(labels, 100)
        return rt.field(state, state.new_tensor(time_value), 'full')
    finally:
        rt.labels = saved_labels


def _guided(rt, state, time_value, labels, weight, cache=None):
    """Exact branch endpoints, with per-state caching for the initial B(x)."""
    if not math.isfinite(weight):
        raise ValueError(weight)
    cache = {} if cache is None else cache

    def get(branch):
        if branch not in cache:
            cache[branch] = _field(rt, state, time_value, labels, branch)
        return cache[branch]

    if weight == 0.:
        return get('u')
    if weight == 1.:
        return get('c')
    conditional, unconditional = get('c'), get('u')
    # Same operation order as the current native CFG baseline.
    return conditional + (weight - 1.) * (conditional - unconditional)


def z_event(rt, state, time_value, step_size, labels, *, strong_w, backward_w, kind):
    """One actual Z event, also allowing endpoint fields for CPU invariants.

    Return the advanced state and small diagnostic tensor references. The
    caller owns inference/context handling. Endpoint calls are optimized:
    strong=backward=0 or 1 costs three real branch queries, not five.
    """
    if kind not in Z_KINDS or backward_w not in (0., 1.):
        raise ValueError((kind, backward_w))
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError(step_size)
    initial_cache = {}
    first = _guided(rt, state, time_value, labels, strong_w, initial_cache)
    future = state + step_size * first
    if kind == 'z_release_euler':
        backward = _guided(rt, future, time_value, labels, backward_w)
        reflected = future - step_size * backward
        predictor = future
    else:
        # Reuse C(x)/U(x) from the initial strong query. In a generic event
        # this requires no extra model evaluation. All differences retain x
        # as anchor, preserving exact same-field cancellation in FP32.
        at_anchor = _guided(rt, state, time_value, labels, backward_w, initial_cache)
        predictor = state + step_size * (first - at_anchor)
        backward = _guided(rt, predictor, time_value, labels, backward_w)
        reflected = state + step_size * (first - backward)
    final = _guided(rt, reflected, time_value, labels, strong_w)
    advanced = reflected + step_size * final
    return advanced, dict(first=first, future=future, predictor=predictor,
                          reflected=reflected)


def _precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


@torch.inference_mode()
def sample(rt, noise, labels, config, snapshots=False):
    """Interface compatible with the existing paired generation runner."""
    _precision()
    if getattr(rt, 'name', 'sit_small') != 'sit_small':
        raise ValueError('Only the frozen SiT-small experiment is supported.')
    if noise.dtype != torch.float32 or noise.ndim != 4 or tuple(noise.shape[1:]) != (4, 32, 32):
        raise ValueError('Expected FP32 BCHW noise with 4x32x32 latents.')
    if labels.dtype != torch.long or labels.shape != (len(noise),) or labels.device != noise.device:
        raise ValueError('Expected one int64 label on the noise device per source.')
    if not torch.all((labels >= 0) & (labels < 100)) or not torch.isfinite(noise).all():
        raise ValueError('Invalid class or noise input.')
    kind = config['kind']
    alpha = float(config['alpha'])
    cutoff = float(config.get('cutoff', .75))
    if not math.isfinite(alpha) or alpha <= 0. or cutoff != .75:
        raise ValueError('This bounded screen fixes positive extra alpha and cutoff=.75.')

    if kind in ('cfg', 'apg', 'ctrl'):
        if int(config.get('steps', 64)) != 64:
            raise ValueError('Delegated baselines use exactly 64 Heun steps.')
        return b.sample(rt, noise, labels, config, snapshots=snapshots)
    if kind not in (*Z_KINDS, *EULER_KINDS):
        raise ValueError(kind)
    steps = 128 if kind in EULER_KINDS else 56
    if int(config.get('steps', steps)) != steps:
        raise ValueError((kind, config.get('steps')))
    if kind in Z_KINDS:
        backward_w = float(config['backward_w'])
        strong_w = (1. + alpha + backward_w) / 2.
        if backward_w not in (0., 1.) or strong_w in (0., 1.):
            raise ValueError('Search requires w_b=0/1 and generic strong w for five real calls/event.')
        if 'forward_w' in config and float(config['forward_w']) != strong_w:
            raise ValueError('forward_w is inconsistent with effective alpha and backward_w.')
    else:
        backward_w, strong_w = None, 1. + alpha

    before = rt.counts.copy()
    state = noise.clone()
    saved = {'step_000': state.detach().clone()} if snapshots else None
    save_steps = {steps // 4, steps // 2, 3 * steps // 4, steps}
    trace = []
    previous_labels = rt.labels
    try:
        rt.labels = labels
        with rt.context():
            for k in range(steps):
                time_value, h = k / steps, 1. / steps
                active = time_value < cutoff
                if kind in Z_KINDS and active:
                    state, detail = z_event(rt, state, time_value, h, labels,
                        strong_w=strong_w, backward_w=backward_w, kind=kind)
                    first = detail['first']
                else:
                    first = _guided(rt, state, time_value, labels,
                                    1. + alpha if active else 1.)
                    state = state + h * first
                if not torch.isfinite(state).all() or state.abs().max() > 1e6:
                    raise FloatingPointError(f'{kind}: invalid state at step {k}')
                if snapshots:
                    trace.append(torch.stack((
                        state.new_full((len(state),), k),
                        state.new_full((len(state),), time_value),
                        state.new_full((len(state),), h),
                        state.new_full((len(state),), alpha if active else 0.),
                        state.square().flatten(1).mean(1).sqrt(),
                        first.square().flatten(1).mean(1).sqrt()), -1))
                    if k + 1 in save_steps:
                        saved[f'step_{k+1:03d}'] = state.detach().clone()
    finally:
        rt.labels = previous_labels
    counts = {key: rt.counts[key] - before[key] for key in ('full', 'prefix')}
    if counts != {'full': 224, 'prefix': 0}:
        raise RuntimeError(f'Unaccounted branch queries: {counts}')
    result = dict(latents=state, counts=counts)
    if snapshots:
        result.update(snapshots=saved, trace=torch.stack(trace).cpu().numpy())
    return result
