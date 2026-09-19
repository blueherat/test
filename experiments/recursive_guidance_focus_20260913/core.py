"""Local coefficient refinements; reuse the already tested Heun feedback sampler.

Hooks are installed only during this process's single-threaded sample call and
restored in finally. No files or state in the paused experiment are changed.
"""
from copy import deepcopy
from pathlib import Path
import torch
from experiments.recursive_guidance_20260913 import core as prior
from . import catalog

_probe, _transform = prior.probe, prior.transform


def __getattr__(name):
    return getattr(prior, name)


def probe(rt, z, t, strong, weak, config, xi):
    state = _probe(rt, z, t, strong, weak, config, xi)
    coefficient = float(config['parameters'].get('probe_coefficient', state['coefficient']))
    assert 0 <= coefficient <= 4
    state['coefficient'] = coefficient
    if config['key'] == 'cfg_cycle_gain':
        state['gain'] = 1/(1+coefficient*(state['response_gain']-1).clamp_min(0))
    return state


def transform(d, state, config):
    if state is None or config['parameters'].get('correction_multiplier', 1.) == 0:
        return d
    value = _transform(d, state, config)
    if config['parameters'].get('preserve_norm'):
        value = torch.where(prior.norm(value)>prior.EPS,
                            prior.norm(d)*prior.unit(value), d)
    return value


def sample(rt, noise, labels, config, **kwargs):
    config = deepcopy(config)
    if config['parameters'].get('probe_coefficient') == 0:
        config['parameters']['correction_multiplier'] = 0.
    before_probe, before_transform = prior.probe, prior.transform
    try:
        prior.probe, prior.transform = probe, transform
        return prior.sample(rt, noise, labels, config, **kwargs)
    finally:
        prior.probe, prior.transform = before_probe, before_transform


def source_paths():
    return sorted(set(prior.source_paths()+[Path(__file__).resolve(), Path(catalog.__file__).resolve()]))


def cpu_checks():
    configs = catalog.configurations()
    assert len(configs) == 8 and len({x['arm'] for x in configs}) == 8
    assert sum(x['role'] == 'candidate' for x in configs) == 6
    generator = torch.Generator().manual_seed(137)
    d = torch.randn((3, 4, 4, 4), generator=generator)
    axis = prior.unit(torch.randn(d.shape, generator=generator))
    state = dict(key='ig_cycle_extrapolate', coefficient=.5, axis=axis,
                 valid=torch.ones_like(prior.norm(d), dtype=torch.bool))
    preserved = next(c for c in configs if c['parameters'].get('preserve_norm'))
    amplitude = next(c for c in configs if c['parameters'].get('norm_only'))
    torch.testing.assert_close(prior.norm(transform(d, state, preserved)), prior.norm(d))
    torch.testing.assert_close(prior.cosine(d, transform(d, state, amplitude)), torch.ones_like(prior.norm(d)))
    torch.testing.assert_close(transform(torch.zeros_like(d), state, preserved), torch.zeros_like(d))
    for coefficient in (.5, 2., 4.):
        response = torch.tensor([0., .8, 1., 2., 100.])
        gain = 1/(1+coefficient*(response-1).clamp_min(0))
        assert (gain > 0).all() and (gain <= 1).all() and torch.equal(gain[:3], torch.ones(3))
    return dict(passed=True, configurations=8, candidates=6, ideas=2,
                preserved_norm=True, amplitude_direction_exact=True, zero_gap_finite=True)


@torch.inference_mode()
def limiting_checks(rt, noise, labels):
    """Two short baseline parity checks and both mechanism no-op limits."""
    saved_head, saved_labels = rt.head, rt.labels
    records = []
    for source in ('cfg', 'ig'):
        cfg = next(c for c in catalog.configurations() if c['source'] == source and c['role'] == 'candidate')
        native = dict(cfg, key='native', parameters={})
        expected, _ = prior.sample(rt, noise, labels, native)
        with prior.selected_head(rt, cfg['reference']), rt.context():
            rt.grid = torch.linspace(0, 1, 65, device=noise.device)
            def field(z, t, left, step, substage):
                amount = prior.amount_at(native, left)
                if not amount:
                    return prior.branch(rt, z, t)
                strong, weak = prior.pair(rt, z, t, source)
                return strong+amount*(strong-weak)
            independent, _ = prior.common.integrate(rt, noise, labels, field)
        rt.labels = saved_labels
        assert torch.equal(independent, expected), source
        disabled = deepcopy(cfg)
        disabled['parameters'] = dict(disabled['parameters'], probe_coefficient=0.)
        actual, stats = sample(rt, noise, labels, disabled)
        assert torch.equal(actual, expected) and stats['auxiliary_full_calls'] == 0, source
        assert rt.head is saved_head and rt.labels is saved_labels
        records.append(dict(source=source, native_exact=True, zero_correction_exact=True,
                            zero_auxiliary_calls=True))
    return dict(passed=True, checks=records)
