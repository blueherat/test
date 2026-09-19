"""Guidance from trained probability measures and a global invariant kernel."""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from . import catalog
from .data import ROOT
from experiments import sit_guidance_fusion_20260910 as previous
from experiments.lifting_scale_sweep_20260909 import array_sha, read, sha

FAMILIES = catalog.FAMILIES
configurations = catalog.configurations
limiting_checks = previous.limiting_checks
install = previous.install


def weak_model(rt, method):
    if getattr(rt, 'measure_method', None) != method:
        if not hasattr(rt, 'measure_model'):
            rt.measure_model = copy.deepcopy(rt.model).eval().requires_grad_(False)
        folder = ROOT/'training'/method
        receipt = read(folder/'complete.json')
        assert receipt['passed'] and sha(folder/'model.pt') == receipt['checkpoint_sha256']
        checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
        rt.measure_model.load_state_dict(checkpoint['ema'], strict=True)
        rt.measure_method = method
    return rt.measure_model


def conditional(rt, x, t, labels, null=False):
    before = rt.labels
    rt.labels = torch.full_like(labels, 100) if null else labels
    try:
        return rt.field(x, x.new_tensor(t), 'full')
    finally:
        rt.labels = before


def rk4(rt, x, start, end, labels, steps):
    z = x
    h = (end-start)/steps
    for i in range(steps):
        t = start+i*h
        k1 = conditional(rt, z, t, labels)
        k2 = conditional(rt, z+h*k1/2, t+h/2, labels)
        k3 = conditional(rt, z+h*k2/2, t+h/2, labels)
        k4 = conditional(rt, z+h*k3, t+h, labels)
        z = z+h*(k1+2*k2+2*k3+k4)/6
    return z


def gaussian_assets(rt):
    if not hasattr(rt, 'measure_gaussian'):
        rt.measure_gaussian = torch.load(ROOT/'gaussian_reference.pt',
            map_location='cuda', weights_only=True)
    return rt.measure_gaussian


def gaussian_noise(rt, labels, t, generator):
    assets = gaussian_assets(rt)
    factors = assets['factors'][labels]
    weights = torch.randn(factors.shape[:2], device=labels.device, generator=generator)
    # Full empirical covariance, including correlations between all latent coordinates.
    with previous.old.exact_matmul():
        low_rank = torch.einsum('bn,bnd->bd', weights, factors)/math.sqrt(factors.shape[1])
    low_rank = low_rank.reshape(len(labels), 4, 32, 32)
    diagonal = torch.randn(low_rank.shape, device=labels.device, generator=generator)
    data_noise = low_rank+assets['posterior_std'][labels]*diagonal
    diffusion_noise = torch.randn(low_rank.shape, device=labels.device, generator=generator)
    return t*data_noise+(1-t)*diffusion_noise


def refresh(rt, x, t, labels, config, generator):
    rho = config['theta']
    if config['key'] == 'gaussian_refresh':
        mean = t*gaussian_assets(rt)['means'][labels]
        return mean+rho*(x-mean)+math.sqrt(1-rho*rho)*gaussian_noise(rt, labels, t, generator)
    n = config['parameters']['inverse_steps']
    source = rk4(rt, x, t, 0., labels, n)
    if rho != 1.:
        innovation = torch.randn(source.shape, device=source.device, generator=generator)
        source = rho*source+math.sqrt(1-rho*rho)*innovation
    return rk4(rt, source, 0., t, labels, n)


@torch.inference_mode()
def sample(rt, noise, labels, config, *, zero=False):
    if config['parameters'].get('inherited_exact'):
        return previous.sample(rt, noise, labels, config, zero=zero)
    if zero:
        anchor = next(c for c in configurations() if c['family'] == 'strong')
        return previous.sample(rt, noise, labels, anchor)
    rt.labels = labels
    z = noise.clone()
    before = rt.counts.copy()
    weak_calls, refresh_calls, refresh_events = 0, 0, 0
    seed = (int(array_sha(noise.detach().cpu().numpy())[:15], 16)+20260912) % (2**63-1)
    generator = torch.Generator(device=noise.device).manual_seed(seed)
    method = config['parameters'].get('method')
    model = weak_model(rt, method) if config['key'] == 'measure_ag' else None
    partners = (torch.from_numpy(np.load(ROOT/'local_partners.npy')).to(labels.device)
                if config['key'] == 'hard_pair' else None)

    def field(x, t, amount):
        nonlocal weak_calls
        strong = conditional(rt, x, t, labels)
        if not amount:
            return strong
        if model is not None:
            rt.counts['full'] += 1
            weak_calls += 1
            weak = model(x, x.new_full((len(x),), t), labels)
        elif partners is not None:
            weak = conditional(rt, x, t, partners[labels])
        else:
            weak = conditional(rt, x, t, labels, null=True)
        return strong+amount*(strong-weak)

    for k in range(64):
        t, h = k/64, 1/64
        if config['key'] in ('conditional_refresh', 'gaussian_refresh') and k == config['parameters']['event']:
            saved = rt.counts['full']
            z = refresh(rt, z, t, labels, config, generator)
            refresh_calls += rt.counts['full']-saved
            refresh_events += 1
        amount = config['strength'] if t < config['cutoff'] else 0.
        # Guidance amount is fixed across both evaluations, matching the old Heun convention.
        first = field(z, t, amount)
        second = field(z+h*first, t+h, amount)
        z = z+h*(first+second)/2
        if not torch.isfinite(z).all() or z.abs().max() > 1e6:
            raise FloatingPointError(f'{config["arm"]}: invalid state at step {k}')
    full, prefix = [rt.counts[key]-before[key] for key in ('full', 'prefix')]
    expected = 128+(2*round(config['cutoff']*64) if config['strength'] else 0)+refresh_calls
    assert full == expected and prefix == 0, (config['arm'], full, expected, prefix)
    return z, dict(full_calls=full, prefix_calls=prefix, auxiliary_full_calls=refresh_calls,
        diagnostic_queries=0, strang_active_steps=0, diagnostics=np.zeros((len(z), 5)),
        measure_weak_calls=weak_calls, measure_refresh_full_calls=refresh_calls,
        measure_refresh_events=refresh_events, guidance_uses_external_semantics=False)
