"""Same-time antithetic score unsharp masking in SiT velocity coordinates.

For z_t=t*x+(1-t)*epsilon the same-time velocity is affine in z and
score. A symmetric input pair cancels the affine term exactly. Hence the
velocity residual implements the score residual with the correct coefficient,
without converting predictions at different times. Each image has a separate
probe RNG derived from its initial noise, independent of batch and arm.
"""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from experiments.self_guidance_20260913 import sampler as sg

SOURCE_FILES = [Path(__file__), Path(__file__).with_name('run.py'),
                Path(__file__).with_name('check.py'), *sg.SOURCE_FILES]
TRACE_COLUMNS = ('step', 'time', 'probe_sigma', 'conditional_rms', 'residual_rms',
                 'correction_rms', 'state_rms')


def pair_residual(center, plus, minus):
    # FP32 subtraction also when the native network is evaluated in BF16.
    return center.float() - (plus.float() + minus.float()) * .5


def probe_generators(noise, seed):
    generators = []
    for row in noise.detach().float().cpu().numpy():
        digest = hashlib.sha256(row.tobytes() + str(seed).encode()).digest()
        generators.append(torch.Generator(device=noise.device).manual_seed(
            int.from_bytes(digest[:8], 'little') % (2**63-1)))
    return generators


@torch.inference_mode()
def sample(rt, noise, labels, config, snapshots=False):
    if config['kind'] in ('baseline', 'sg', 'sg_prev'):
        return sg.sample(rt, noise, labels, config, snapshots)
    if config['kind'] != 'log' or config.get('solver', 'euler') != 'euler':
        raise ValueError(config)
    omega = float(config.get('omega', 1))
    kappa = float(config.get('kappa', .2))
    if any(not math.isfinite(x) or x < 0 for x in (omega, kappa)):
        raise ValueError((omega, kappa))
    if omega == 0 or kappa == 0:
        return sg.sample(rt, noise, labels, dict(config, kind='baseline', omega=0), snapshots)
    steps = int(config.get('steps', 64))
    if steps < 2 or noise.dtype != torch.float32 or noise.ndim != 4:
        raise ValueError('Need >=2 steps and FP32 BCHW initial states')
    lo, hi = config.get('start', 0.), config.get('stop', 1.)
    if not 0 <= lo < hi <= 1:
        raise ValueError('Invalid time gate')
    generators = probe_generators(noise, config.get('probe_seed', 2026091401))
    z = noise.clone()
    before = rt.counts.copy()
    trace = []
    saved = {'step_000': z.clone()} if snapshots else None
    active_steps = 0
    with rt.context():
        for step in range(steps):
            t = step / steps
            base, conditional, _, _ = sg.velocity(rt, z, t, labels,
                dict(config, kind='baseline', omega=0))
            # Draw every step so gates cannot shift the probe stream.
            probe = torch.stack([torch.randn(z.shape[1:], generator=g,
                device=z.device, dtype=z.dtype) for g in generators])
            sigma = kappa * (1-t)
            correction = torch.zeros_like(z)
            residual = torch.zeros_like(z)
            if lo <= t < hi:
                offset = sigma * probe
                plus = sg.query(rt, z + offset, t, labels)
                minus = sg.query(rt, z - offset, t, labels)
                residual = pair_residual(conditional, plus, minus)
                correction = omega * residual
                active_steps += 1
            z = z + (base + correction) / steps
            if not torch.isfinite(z).all() or z.abs().max() > 1e6:
                raise FloatingPointError((config['arm'], step))
            if snapshots:
                rms = lambda x: x.square().flatten(1).mean(1).sqrt()
                constant = lambda x: z.new_full((len(z),), x)
                trace.append(torch.stack((constant(step), constant(t), constant(sigma),
                    rms(conditional), rms(residual), rms(correction), rms(z)), -1))
                if step+1 in {steps//4, steps//2, 3*steps//4, steps}:
                    saved[f'step_{step+1:03d}'] = z.clone()
    counts = {k: rt.counts[k]-before[k] for k in ('full', 'prefix')}
    expected = steps + bool(config.get('alpha', 0)) * sum(
        k/steps < config.get('cutoff', .75) for k in range(steps)) + 2*active_steps
    assert counts == dict(full=expected, prefix=0), (counts, expected)
    result = dict(latents=z, counts=counts)
    if snapshots:
        result.update(snapshots=saved, trace=torch.stack(trace).cpu().numpy())
    return result
