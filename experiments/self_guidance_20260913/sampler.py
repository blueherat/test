"""SiT port of MAPLE Self-Guidance, upstream 843bda799bb531ccf8d98164d8ccf1a3b8ce3f6d.

Upstream models/flux/pipeline_flux.py:310-340 and SD3:351-403 directly
extrapolate model predictions. This implements that released velocity rule,
not an exact score conversion between different flow times. SiT uses
z(t)=t*x+(1-t)*noise, so the noisier query is max(0,t-delta).
SG-prev stores the RAW conditional prediction before either CFG or SG;
the upstream t_noise < 500 gate maps to SiT t > .5. History resets per call.
Euler is used for SG-prev to preserve the one previous accepted step meaning.
"""
from pathlib import Path
import math
import torch

UPSTREAM = Path('/home/zhoushunyu/data/eqvae/baselines/Self-Guidance')
SOURCE_FILES = [Path(__file__), Path(__file__).with_name('run.py'),
                Path(__file__).with_name('check.py'),
                UPSTREAM/'models/flux/pipeline_flux.py',
                UPSTREAM/'models/stable_diffusion_3/pipeline_sd_3.py']
TRACE_COLUMNS = ('step', 'time', 'cfg_alpha', 'sg_active', 'state_rms', 'correction_rms')


def query(rt, z, t, labels):
    saved = rt.labels
    try:
        rt.labels = labels
        return rt.field(z, z.new_tensor(t), 'full')
    finally:
        rt.labels = saved


def velocity(rt, z, t, labels, config, previous=None):
    conditional = query(rt, z, t, labels)
    alpha = float(config.get('alpha', 0)) if t < config.get('cutoff', .75) else 0.
    result = conditional
    if alpha:
        unconditional = query(rt, z, t, torch.full_like(labels, 100))
        result = conditional + alpha * (conditional - unconditional)
    omega = float(config.get('omega', 0))
    kind = config['kind']
    correction = torch.zeros_like(conditional)
    active = False
    if omega and kind == 'sg':
        reference_time = max(0., t - float(config.get('shift', .01)))
        # Clipping makes the first SG query identical; reuse it exactly.
        if reference_time != t:
            reference = query(rt, z, reference_time, labels)
            correction = omega * (conditional - reference)
            active = True
    elif omega and kind == 'sg_prev' and previous is not None and t > config.get('prev_start', .5):
        correction = omega * (conditional - previous)
        active = True
    if active:
        result = result + correction
    return result, conditional.detach().clone(), correction, active


@torch.inference_mode()
def sample(rt, noise, labels, config, snapshots=False):
    kind = config['kind']
    steps = int(config.get('steps', 64))
    solver = config.get('solver', 'euler')
    if kind not in ('baseline', 'sg', 'sg_prev') or solver not in ('euler', 'heun'):
        raise ValueError((kind, solver))
    if steps < 2 or (solver == 'heun' and kind != 'baseline'):
        raise ValueError('Heun is reserved for the existing baseline comparison.')
    for key, default in [('omega', 0), ('alpha', 0), ('shift', .01)]:
        value = float(config.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError((key, value))
    if kind == 'baseline' and config.get('omega', 0) != 0:
        raise ValueError('Baseline requires omega=0.')
    if not 0 <= config.get('cutoff', .75) <= 1 or not 0 <= config.get('prev_start', .5) <= 1:
        raise ValueError('Invalid guidance time window.')
    if noise.ndim != 4 or noise.dtype != torch.float32 or labels.shape != (len(noise),):
        raise ValueError('Expected FP32 BCHW noise and one label per image.')
    if labels.dtype != torch.long or not torch.all((labels >= 0) & (labels < 100)):
        raise ValueError('Expected int64 ImageNet100 class labels.')
    before = rt.counts.copy()
    z = noise.clone()
    previous = None
    saved = {'step_000': z.clone()} if snapshots else None
    trace = []
    with rt.context():
        for k in range(steps):
            t, h = k / steps, 1 / steps
            first, proposal, correction, active = velocity(rt, z, t, labels, config, previous)
            if solver == 'heun':
                # The existing SiT baseline freezes CFG gating at left time.
                second_config = dict(config, cutoff=2. if t < config.get('cutoff', .75) else -1.)
                second, _, _, _ = velocity(rt, z + h * first, t + h, labels, second_config)
                z = z + (h / 2) * (first + second)
            else:
                z = z + h * first
            previous = proposal
            if not torch.isfinite(z).all() or z.abs().max() > 1e6:
                raise FloatingPointError(f'{config.get("arm", kind)} invalid at step {k}')
            if snapshots:
                rms = lambda x: x.square().flatten(1).mean(1).sqrt()
                trace.append(torch.stack((z.new_full((len(z),), k), z.new_full((len(z),), t),
                    z.new_full((len(z),), config.get('alpha', 0) if t < config.get('cutoff', .75) else 0),
                    z.new_full((len(z),), float(active)), rms(z), rms(correction)), -1))
                if k + 1 in {steps // 4, steps // 2, 3 * steps // 4, steps}:
                    saved[f'step_{k+1:03d}'] = z.clone()
    counts = {key: rt.counts[key] - before[key] for key in ('full', 'prefix')}
    stages = 2 if solver == 'heun' else 1
    expected = stages * (steps + sum(k / steps < config.get('cutoff', .75)
                       for k in range(steps)) * bool(config.get('alpha', 0)))
    if kind == 'sg' and config.get('omega', 0) and config.get('shift', .01) > 0:
        expected += steps - 1
    if counts != dict(full=expected, prefix=0):
        raise RuntimeError((counts, expected))
    result = dict(latents=z, counts=counts)
    if snapshots:
        result.update(snapshots=saved, trace=torch.stack(trace).cpu().numpy())
    return result
