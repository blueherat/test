"""Counted full-future probes and bounded latent updates for seven hypotheses."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments import run_sit_fsg_pasted_20260910 as catalog
from experiments.sit_fsg_followup_20260910 import core as prior
from experiments.sit_guidance_portfolio_20260910 import operators as old
from experiments.lifting_scale_sweep_20260909 import array_sha

MANIFEST = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc/manifest.json')
CLASSIFIER_WEIGHTS = Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth')
INDEPENDENT_WEIGHTS = Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
FAMILIES = {method['key']: method for method in catalog.METHODS}
SPECIAL = set(FAMILIES) | {'full_root', 'semantic_contract', 'native_sde', 'smc_control'}
SEMANTIC_KEYS = {'semantic_agreement', 'semantic_contract'}
configurations = catalog.configurations
norm, inner, cap, flat = old.norm, old.inner, old.cap, prior.flat


def install():
    pass


def source_assets():
    return [MANIFEST, CLASSIFIER_WEIGHTS, INDEPENDENT_WEIGHTS]


def event_context(noise, step):
    seed = int(array_sha(noise.detach().cpu().numpy())[:15], 16)
    generator = torch.Generator(device=noise.device).manual_seed((seed + 7919*step + 104729) % (2**63-1))
    context = old.StepContext()
    context.xi = torch.randn(noise.shape, generator=generator, device=noise.device, dtype=noise.dtype)
    context.future_noise = torch.randn((2, catalog.FUTURE_STEPS, *noise.shape),
                                      generator=generator, device=noise.device, dtype=noise.dtype)
    return context


class SemanticReadout:
    """Frozen image classifier; explicitly additional information and compute."""
    def __init__(self, rt):
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        self.rt = rt
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.model = convnext_tiny(weights=None).to('cuda').eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location='cpu', weights_only=True))
        self.transform = weights.transforms()
        records = sorted(json.loads(MANIFEST.read_text())['classes'], key=lambda row: row['label'])
        assert [row['label'] for row in records] == list(range(100))
        self.original_labels = torch.tensor([row['original_imagenet_label'] for row in records], device='cuda')
        assert len(self.original_labels.unique()) == 100
        self.decoded_images = 0
        self.classified_images = 0

    def probabilities(self, z):
        images = self.rt.small.decode_latents_in_chunks(
            self.rt.vae, z, scaling_factor=self.rt.small.SD_VAE_SCALING_FACTOR, chunk_size=2)
        self.decoded_images += len(z)
        # The continuous counterpart of the native clamp(127.5*x+128) decoder.
        images = ((127.5*images+128)/255).clamp(0, 1)
        with old.exact_matmul():
            logits = self.model(self.transform(images))
        self.classified_images += len(z)
        return logits.softmax(-1), logits[:, self.original_labels].softmax(-1)

    def features(self, z, labels, kind):
        full, subset = self.probabilities(z)
        if int(kind) == 2:
            return subset.clamp_min(1e-12).sqrt()
        target = full.gather(1, self.original_labels[labels, None]).squeeze(1)
        return torch.stack((target.clamp_min(1e-12).sqrt(),
                            (1-target).clamp_min(1e-12).sqrt()), 1)

    def target_probability(self, z, labels):
        full, _ = self.probabilities(z)
        return full.gather(1, self.original_labels[labels, None]).squeeze(1)


class FutureProbe:
    def __init__(self, rt, z, t, context, *, steps=None):
        self.rt, self.z, self.t, self.context = rt, z, float(t), context
        self.labels = rt.labels
        self.steps = catalog.FUTURE_STEPS if steps is None else steps
        self.grid = [self.t + (1-self.t)*index/self.steps for index in range(self.steps+1)]
        self.c = self.field(z, t, 'c')
        self.u = self.field(z, t, 'u')
        self.d = self.c-self.u
        self.eps = .001*norm(z).clamp_min(math.sqrt(z[0].numel()))
        self.cache = {}

    def field(self, z, t, condition):
        previous = self.rt.labels
        self.rt.labels = (torch.full_like(self.labels, 100) if condition == 'u'
                          else self.labels) if isinstance(condition, str) else condition
        try:
            with old.exact_matmul():
                return self.rt.field(z, z.new_tensor(t), 'full')
        finally:
            self.rt.labels = previous

    def rollout(self, z, condition, *, start=0, noise=None):
        state = z
        states = [state]
        for index in range(start, self.steps):
            t, end = self.grid[index:index+2]
            h = end-t
            first = self.field(state, t, condition)
            if noise is None:
                second = self.field(state+h*first, end, condition)
                state = state+(h/2)*(first+second)
            else:
                assert t > 0, 'The reverse SDE is singular at the initial t=0 endpoint'
                state = state+h*(2*first-state/t)+math.sqrt(2*(1-t)/t*h)*noise[index]
            states.append(state)
        return states

    def pair_paths(self, z):
        if z is self.z and 'paths' in self.cache:
            return self.cache['paths']
        value = self.rollout(z, 'c'), self.rollout(z, 'u')
        if z is self.z:
            self.cache['paths'] = value
        return value

    def rival(self):
        if not hasattr(self.rt, 'pasted_rivals'):
            table = F.normalize(self.rt.model.y_embedder.embedding_table.weight[:100], dim=-1)
            with old.exact_matmul():
                similarities = table @ table.T
            similarities.fill_diagonal_(-torch.inf)
            self.rt.pasted_rivals = similarities.argmax(-1)
        return self.rt.pasted_rivals[self.labels]

    def semantic(self):
        if not hasattr(self.rt, 'pasted_semantic'):
            self.rt.pasted_semantic = SemanticReadout(self.rt)
        return self.rt.pasted_semantic


def residual_function(probe, key, theta):
    z, t = probe.z, probe.t
    if key == 'stochastic_coupling':
        count = int(theta)
        assert count in (1, 2)

        def residual(x):
            pieces = []
            for index in range(count):
                noise = probe.context.future_noise[index]
                c = probe.rollout(x, 'c', noise=noise)[-1]
                u = probe.rollout(x, 'u', noise=noise)[-1]
                pieces.append(flat(c-u)/math.sqrt(count))
            return torch.cat(pieces, 1)
        return residual
    if key == 'robust_future':
        shift = theta*(1-t)*probe.context.xi

        def residual(x):
            pieces = []
            for sign in (-1, 1):
                c, u = probe.pair_paths(x+sign*shift)
                pieces.append(flat(c[-1]-u[-1])/math.sqrt(2))
            return torch.cat(pieces, 1)
        return residual
    if key == 'terminal_inverse':
        target = probe.pair_paths(z)[0][-1].detach()
        return lambda x: flat(probe.rollout(x, 'u')[-1]-target)
    if key == 'semantic_contract':
        # A simple terminal classifier objective is a necessary information-
        # matched control for semantic consistency, not a new FSG method.
        return lambda x: (theta-probe.semantic().target_probability(
            probe.rollout(x, 'u')[-1], probe.labels)).clamp_min(0)[:, None]

    c0, u0 = probe.pair_paths(z)
    scale = (c0[-1]-u0[-1]).flatten(1).square().mean(1).sqrt().clamp_min(1e-5)[:, None]

    def residual(x):
        c, u = probe.pair_paths(x)
        terminal = flat(c[-1]-u[-1])/scale
        if key == 'full_root':
            return terminal
        if key == 'golden_tube':
            pieces = [terminal]
            for index in (probe.steps//4, probe.steps//2):
                restarted = probe.rollout(u[index], 'c', start=index)[-1]
                pieces.append(math.sqrt(theta)*flat(restarted-u[-1])/scale)
            return torch.cat(pieces, 1)
        if key == 'full_path':
            pieces = [terminal]
            for index in (probe.steps//4, probe.steps//2, 3*probe.steps//4):
                pieces.append(math.sqrt(theta)*flat(c[index]-u[index])/scale)
            return torch.cat(pieces, 1)
        if key == 'semantic_agreement':
            semantics = probe.semantic()
            return (semantics.features(c[-1], probe.labels, theta)
                    - semantics.features(u[-1], probe.labels, theta))
        if key == 'contrastive_hinge':
            rival = probe.rollout(x, probe.rival())[-1]
            positive = terminal.square().mean(1)
            negative = (flat(u[-1]-rival)/scale).square().mean(1)
            hinge = (.25+positive-negative).clamp_min(0)
            return torch.cat((terminal, math.sqrt(theta*terminal.shape[1])*hinge[:, None]), 1)
        raise KeyError(key)
    return residual


def calibrate(rt, z, t, config, context, amount, *, disabled=False):
    empty = dict(accepted=0., before=0., after=0., shift=0.)
    if disabled or amount == 0:
        return z, empty
    probe = FutureProbe(rt, z, t, context)
    fn = residual_function(probe, config['key'], config['theta'])
    residual = fn(z)
    q = prior.basis(probe.d, context.xi)
    columns = []
    for index in range(2):
        changed = fn(z+probe.eps*q[:, :, index].reshape_as(z))
        columns.append((changed-residual)/probe.eps.flatten(1))
    coefficient = prior.ridge_step(residual, torch.stack(columns, -1))
    radius = (4/64)*amount*norm(probe.d)
    if config['key'] in ('full_root', 'terminal_inverse'):
        radius = radius*config['theta']
    candidate = z+cap(prior.combine(q, coefficient, z), radius)
    before, after = prior.energy(residual), prior.energy(fn(candidate))
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise FloatingPointError(f'Nonfinite full-future objective: {config["arm"]}')
    accepted = after < before
    result = prior.choose(accepted, candidate, z)
    actual = torch.where(accepted, after, before)
    assert (actual <= before).all()
    return result, dict(accepted=float(accepted.sum()), before=float(before.sum()),
                        after=float(actual.sum()), shift=float(norm(result-z).sum()))


def smc_update(gap, previous, gain, decay):
    previous = gap if previous is None else previous
    sliding = gap-previous+decay*previous
    return gap-gain*sliding.sign()


def base_field(rt, z, t, config, amount, context, *, smc=False):
    native = dict(config, key='native_cfg')
    if not smc or amount == 0:
        return old.evaluate(rt, z, t, native, amount, context)
    q = old.Queries(rt, z, t, 'native_cfg', context)
    gap = q.s-q.null()[0]
    corrected = smc_update(gap, context.history.get('smc_gap'), config['theta'],
                           config['parameters']['lambda'])
    return q.null()[0]+(1+amount)*corrected, {'smc_gap': corrected.detach()}


@torch.inference_mode()
def sample(rt, noise, labels, config, *, zero=False, disable_calibration=False):
    seed = int(array_sha(noise.detach().cpu().numpy())[:15], 16)
    if config['key'] not in SPECIAL:
        return prior.sample(rt, noise, labels, config, zero=zero)
    is_sde = config['solver'] == 'sde_tail64'
    if zero and not is_sde:
        return old.sample(rt, noise, labels, dict(config, solver='heun64'), batch_seed=seed, zero=True)
    rt.labels = labels
    z = noise.clone()
    begin = rt.counts.copy()
    semantic = getattr(rt, 'pasted_semantic', None)
    semantic_begin = (semantic.decoded_images, semantic.classified_images) if semantic else (0, 0)
    context = old.StepContext()
    generator = torch.Generator(device=z.device).manual_seed((seed+130363) % (2**63-1))
    brownian = torch.randn((64, *z.shape), generator=generator, device=z.device) if is_sde else None
    extra = dict(calibration_events=0, accepted_calibrations=0., calibration_loss_before_sum=0.,
                 calibration_loss_after_sum=0., calibration_shift_norm_sum=0., calibration_full_calls=0)
    for k in range(64):
        t, h = k/64, 1/64
        amount = 0. if zero else old.amount_at(config, t)
        if (amount and k in catalog.EVENT_STEPS
                and config['key'] not in ('native_sde', 'smc_control') and not disable_calibration):
            event = event_context(noise, k)
            before = rt.counts['full']
            z, record = calibrate(rt, z, t, config, event, amount)
            extra['calibration_events'] += 1
            extra['calibration_full_calls'] += rt.counts['full']-before
            for dest, src in [('accepted_calibrations', 'accepted'), ('calibration_loss_before_sum', 'before'),
                              ('calibration_loss_after_sum', 'after'), ('calibration_shift_norm_sum', 'shift')]:
                extra[dest] += record[src]
        smc = config['key'] == 'smc_control'
        first, info = base_field(rt, z, t, config, amount, context, smc=smc)
        if is_sde and k >= 8:
            z = z+h*(2*first-z/t)+math.sqrt(2*(1-t)/t*h)*brownian[k]
        else:
            second, _ = base_field(rt, z+h*first, t+h, config, amount, context, smc=smc)
            z = z+(h/2)*(first+second)
        if smc and amount:
            context.history['smc_gap'] = info['smc_gap']
        if not torch.isfinite(z).all() or z.abs().max() > 1e6:
            raise FloatingPointError(f'{config["arm"]}: invalid state at step {k}')
    full, prefix = rt.counts['full']-begin['full'], rt.counts['prefix']-begin['prefix']
    semantic = getattr(rt, 'pasted_semantic', None)
    extra['auxiliary_decoder_images'] = semantic.decoded_images-semantic_begin[0] if semantic else 0
    extra['auxiliary_classifier_images'] = semantic.classified_images-semantic_begin[1] if semantic else 0
    base_calls = 128 if is_sde else 224
    return z, dict(full_calls=full, prefix_calls=prefix, auxiliary_full_calls=max(0, full-base_calls),
                   diagnostic_queries=0, strang_active_steps=0,
                   diagnostics=np.zeros((len(z), 5), dtype=np.float64), **extra)


@torch.inference_mode()
def limiting_checks(rt, noise, labels):
    rt.labels = labels
    original = rt.field(noise, noise.new_tensor(.25), 'full')
    context = event_context(noise, 8)
    for method in catalog.METHODS:
        config = next(row for row in configurations() if row['idea_id'] == method['id'])
        value, _ = base_field(rt, noise, .25, config, 0., context)
        assert torch.equal(value, original), method['id']
        value, _ = calibrate(rt, noise, .25, config, context, config['strength'], disabled=True)
        assert torch.equal(value, noise), method['id']
    assert rt.labels is labels
    return dict(zero_field_exact=7, disabled_state_write_exact=7)


def cpu_checks():
    rows = configurations()
    assert sum(row['role'] == 'candidate' for row in rows) == 70
    assert len(rows) == 140
    for path in source_assets():
        assert path.exists(), path
    x = torch.tensor([[1., -2., 0.], [-1., 3., 4.]], dtype=torch.float64)
    assert torch.equal(smc_update(x, None, 0., 5.), x)
    assert torch.equal(smc_update(x, None, .2, 5.), x-.2*x.sign())
    assert torch.equal(smc_update(x, x*.5, .1, 5.), x-.1*(x-x*.5+5*x*.5).sign())
    return dict(configurations=140, candidates=70, controls=70, asset_paths_exist=True, smc_algebra_passed=True)
