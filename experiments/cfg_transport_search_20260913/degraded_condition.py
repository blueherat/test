"""Two-query degraded-condition guidance, an empirical CDG-family baseline.

The negative branch uses e_null + r * (e_class - e_null).  Unlike the old
three-query, norm-matched secant, the direction is v_class - v_degraded.
No probability interpretation is assumed for interpolated class embeddings.
Scale zero executes native CFG/APG operations; scale one queries the native
conditional model twice and has zero guidance. No null norm query is hidden.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import torch

from . import baselines as b

SOURCE_FILES = [Path(b.__file__).resolve()]
TRACE_COLUMNS = b.TRACE_COLUMNS


@contextlib.contextmanager
def embedding_override(rt, value):
    """Same y_embedder output hook as the frozen portfolio, always removed."""
    handle = rt.model.y_embedder.register_forward_hook(
        lambda module, args, output: value)
    try:
        yield
    finally:
        handle.remove()


def scale_from(config):
    value = float(config.get('condition_scale', config.get('scale', .5)))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'condition_scale must be in [0,1], got {value}')
    if 'condition_scale' in config and 'scale' in config:
        if value != float(config['scale']):
            raise ValueError('Conflicting condition_scale and scale')
    return value


def query_pair(rt, z, time_value, labels, condition_scale):
    """Return native conditional and degraded reference; exactly two calls.

    This helper also supports same-state probes. It does not alter history,
    perform norm matching, query any third branch, or retain an embedding hook.
    """
    scale = scale_from(dict(condition_scale=condition_scale))
    saved_labels = rt.labels
    t = z.new_tensor(time_value)
    try:
        rt.labels = labels
        conditional = rt.field(z, t, 'full')
        if scale == 0.:
            rt.labels = torch.full_like(labels, 100)
            degraded = rt.field(z, t, 'full')
        elif scale == 1.:
            # Deliberately preserve the two-query budget at this limit.
            degraded = rt.field(z, t, 'full')
        else:
            table = rt.model.y_embedder.embedding_table
            class_embedding = table(labels)
            null_embedding = table(torch.full_like(labels, 100))
            embedding = null_embedding + scale * (class_embedding - null_embedding)
            with embedding_override(rt, embedding):
                degraded = rt.field(z, t, 'full')
    finally:
        rt.labels = saved_labels
    return conditional, degraded


def field(rt, z, time_value, labels, config, history=None, *, active=True):
    """Field and proposed APG history; commit only the accepted left proposal."""
    alpha = float(config.get('alpha', 1.25)) if active else 0.
    if alpha == 0.:
        saved_labels = rt.labels
        try:
            rt.labels = labels
            return rt.field(z, z.new_tensor(time_value), 'full'), history
        finally:
            rt.labels = saved_labels
    conditional, degraded = query_pair(
        rt, z, time_value, labels, scale_from(config))
    gap = conditional - degraded
    kind = config['kind']
    if kind == 'degraded_cfg':
        return conditional + alpha * gap, history
    if kind == 'degraded_apg':
        previous = torch.zeros_like(gap) if history is None else history
        momentum = gap + float(config.get('beta', -.5)) * previous
        bounded = b.cap(momentum, 2 * b.norm(gap))
        t = z.new_tensor(time_value)
        clean = z + (1 - t) * conditional
        direction = bounded - b.projection(bounded, clean)
        return conditional + alpha * direction, momentum.detach()
    raise ValueError(f'Unsupported kind: {kind}')


@torch.inference_mode()
def sample(rt, noise, labels, config, snapshots=False):
    """Runner API; 64 Heun steps and cutoff .75 use 224/0 calls if alpha != 0."""
    kind = config['kind']
    steps = int(config.get('steps', 64))
    cutoff = float(config.get('cutoff', .75))
    alpha = float(config.get('alpha', 1.25))
    beta = float(config.get('beta', -.5))
    scale_from(config)
    if kind not in ('degraded_cfg', 'degraded_apg') or steps not in (64, 96):
        raise ValueError((kind, steps))
    if not 0 <= cutoff <= 1 or not all(map(math.isfinite, (alpha, beta, cutoff))):
        raise ValueError((alpha, beta, cutoff))
    if getattr(rt, 'name', 'sit_small') != 'sit_small':
        raise ValueError('This sampler is specific to SiT-S/2 ImageNet100.')
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
            second, _ = field(
                rt, z + h * first, t + h, labels, config, history, active=active)
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
    expected = dict(full=2 * steps + 2 * active_steps, prefix=0)
    if counts != expected:
        raise RuntimeError(f'Unaccounted model calls: {counts}, expected {expected}')
    result = dict(latents=z, counts=counts)
    if snapshots:
        result.update(snapshots=saved, trace=torch.stack(trace).cpu().numpy())
    return result


def check_cpu():
    """Bounded fake-runtime audits, with no checkpoint or GPU allocation."""
    from types import SimpleNamespace

    class Embedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_table = torch.nn.Embedding(101, 4)

        def forward(self, labels):
            return self.embedding_table(labels)

    class FakeRuntime:
        name = 'sit_small'

        def __init__(self):
            self.model = SimpleNamespace(y_embedder=Embedder())
            self.labels = torch.tensor([99])
            self.counts = dict(full=0, prefix=0)
            self.fail_hooked = False

        def context(self):
            return contextlib.nullcontext()

        def field(self, z, t, kind):
            assert kind == 'full'
            self.counts['full'] += 1
            if self.fail_hooked and self.model.y_embedder._forward_hooks:
                raise RuntimeError('deliberate hooked-branch failure')
            e = self.model.y_embedder(self.labels)[..., None, None]
            # Nonlinear condition response ensures the half secants can differ.
            return .1 * z + .02 * e + .01 * e.square() * z.sin() + .01 * t

    torch.manual_seed(2026091307)
    rt = FakeRuntime()
    noise = torch.randn(3, 4, 8, 8)
    labels = torch.tensor([0, 13, 99])
    checks = []
    conditional = b.sample(rt, noise, labels, dict(kind='cfg', alpha=0.))
    for base_kind in ('cfg', 'apg'):
        cfg = dict(kind=f'degraded_{base_kind}', alpha=2., condition_scale=0.)
        base = b.sample(rt, noise, labels, dict(kind=base_kind, alpha=2.), True)
        ours = sample(rt, noise, labels, cfg, True)
        assert torch.equal(base['latents'], ours['latents'])
        assert all(torch.equal(base['snapshots'][k], ours['snapshots'][k])
                   for k in base['snapshots'])
        assert base['counts'] == ours['counts'] == dict(full=224, prefix=0)
        full = sample(rt, noise, labels, dict(cfg, condition_scale=1.))
        assert torch.equal(full['latents'], conditional['latents'])
        assert full['counts'] == dict(full=224, prefix=0)
        inactive = sample(rt, noise, labels, dict(cfg, condition_scale=.5, alpha=0.))
        assert torch.equal(inactive['latents'], conditional['latents'])
        assert inactive['counts'] == dict(full=128, prefix=0)
        half = sample(rt, noise, labels, dict(cfg, condition_scale=.5), True)
        assert not torch.equal(half['latents'], base['latents'])
        assert half['counts'] == dict(full=224, prefix=0)
        assert half['trace'].shape == (64, 3, len(TRACE_COLUMNS))
        assert len(half['snapshots']) == 5
        checks.append(dict(kind=base_kind, scale0_baseline_bitwise=True,
                           scale1_zero_guidance_bitwise=True,
                           alpha0_bitwise=True, interior_changes_trajectory=True))
    original_labels = rt.labels
    vc, vh = query_pair(rt, noise, .25, labels, .5)
    _, vu = query_pair(rt, noise, .25, labels, 0.)
    assert not torch.equal(vc + vu, 2 * vh)
    assert rt.labels is original_labels and not rt.model.y_embedder._forward_hooks
    rt.fail_hooked = True
    try:
        query_pair(rt, noise, .25, labels, .5)
        raise AssertionError('Expected branch failure')
    except RuntimeError as error:
        assert str(error) == 'deliberate hooked-branch failure'
    assert rt.labels is original_labels and not rt.model.y_embedder._forward_hooks
    for invalid in (-.1, 1.1, float('nan')):
        try:
            scale_from(dict(condition_scale=invalid))
            raise AssertionError('Expected invalid scale rejection')
        except ValueError:
            pass
    return dict(cpu_only=True, checks=checks, query_cleanup=True,
                nonlinear_half_response=True, invalid_scale_rejected=True)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check_cpu()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
