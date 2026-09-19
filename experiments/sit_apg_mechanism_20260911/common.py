"""Counted, deterministic field and terminal-readout operations."""
from __future__ import annotations

import math
import numpy as np
import torch
from experiments.sit_guidance_portfolio_20260910 import operators as old
from experiments.sit_fsg_pasted_20260910 import core as semantic

norm, inner, projection, cap = old.norm, old.inner, old.projection, old.cap


def field(rt, x, t, labels, *, null=False):
    previous = rt.labels
    rt.labels = torch.full_like(labels, 100) if null else labels
    try:
        with old.exact_matmul():
            return rt.field(x, x.new_tensor(t), 'full')
    finally:
        rt.labels = previous


def pair(rt, x, t, labels):
    return field(rt, x, t, labels), field(rt, x, t, labels, null=True)


def bundle(rt, x, t, labels, amount):
    c, u = pair(rt, x, t, labels)
    gap = c-u
    clean = x+(1-t)*c
    parallel = projection(gap, clean)
    orthogonal = gap-parallel
    return dict(conditional=c, null=u, gap=gap, cfg=c+amount*gap,
                projected=c+amount*orthogonal, clean=clean,
                parallel=parallel, orthogonal=orthogonal,
                fixed_cfg_a0=c, fixed_cfg_a125=c+1.25*gap, fixed_cfg_a275=c+2.75*gap)


def heun(rt, x, t, h, labels, amount=0., *, kind='cfg', substeps=1):
    state = x
    for j in range(substeps):
        left, step = t+j*h/substeps, h/substeps
        if kind == 'null':
            first = field(rt, state, left, labels, null=True)
            second = field(rt, state+step*first, left+step, labels, null=True)
        elif amount == 0:
            first = field(rt, state, left, labels)
            second = field(rt, state+step*first, left+step, labels)
        else:
            first = bundle(rt, state, left, labels, amount)[kind]
            second = bundle(rt, state+step*first, left+step, labels, amount)[kind]
        state = state+(step/2)*(first+second)
    return state


def future(rt, x, t, labels, steps=16):
    return heun(rt, x, t, 1-t, labels, kind='null', substeps=steps)


def readout(rt, terminal, labels):
    assert torch.isfinite(terminal).all()
    if not hasattr(rt, 'pasted_semantic'):
        rt.pasted_semantic = semantic.SemanticReadout(rt)
    full, _ = rt.pasted_semantic.probabilities(terminal)
    assert torch.isfinite(full).all()
    original = rt.pasted_semantic.original_labels[labels]
    q = full.gather(1, original[:, None]).squeeze(1)
    logs = full.clamp_min(1e-12).log()
    target = logs.gather(1, original[:, None]).squeeze(1)
    competitor = logs.clone()
    competitor.scatter_(1, original[:, None], -torch.inf)
    means = terminal.mean((2, 3))
    stds = (terminal-means[:, :, None, None]).square().mean((2, 3)).clamp_min(1e-10).sqrt()
    moments = torch.cat((means, stds), 1)
    return dict(q=q, margin=target-competitor.max(1).values,
                target_top1=(full.argmax(1) == original), moments=moments,
                energy=terminal.square().flatten(1).mean(1))


def numpy(tensor):
    return tensor.detach().cpu().numpy()


def unit(x):
    return x/norm(x).clamp_min(1e-12)


def random_unit(x, generator):
    return torch.randint(0, 2, x.shape, device=x.device, generator=generator,
                         dtype=torch.int32).to(x.dtype).mul(2).sub(1)/math.sqrt(x[0].numel())
