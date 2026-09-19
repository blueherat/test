from __future__ import annotations
from dataclasses import replace
import numpy as np
import torch
from . import catalog, train
from experiments import sit_guidance_fusion_20260910 as previous
from experiments.lifting_scale_sweep_20260909 import read, sha

FAMILIES = catalog.FAMILIES
configurations = catalog.configurations
install = previous.install
limiting_checks = previous.limiting_checks


def get_head(rt, method):
    source, target = method.split('_')
    if target == 'original':
        return rt.head.module if source == 'ig' else rt.model.final_layer
    if not hasattr(rt, 'strong_reference_heads'):
        rt.strong_reference_heads = {}
    if method not in rt.strong_reference_heads:
        folder = catalog.head_folder(method)
        receipt = read(folder / 'complete.json')
        assert receipt['checkpoint_sha256'] == sha(folder / 'model.pt')
        saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        assert saved['method'] == method and saved['step'] == catalog.STEPS
        assert saved['request_sha256'] == receipt['request_sha256']
        head = train.initial_head(rt, source)
        head.load_state_dict(saved['ema'], strict=True)
        rt.strong_reference_heads[method] = head.eval().requires_grad_(False)
    return rt.strong_reference_heads[method]


@torch.inference_mode()
def sample(rt, noise, labels, config, *, zero=False):
    if config['parameters'].get('inherited_exact'):
        return previous.sample(rt, noise, labels, config, zero=zero)
    rt.labels = labels
    original_head, original_final = rt.head, rt.model.final_layer
    method, source = config['parameters']['head_method'], config['source']
    module = get_head(rt, method) if not zero else None
    if source == 'ig' and not zero:
        rt.head = replace(rt.head, module=module)
    before = rt.counts.copy(); z = noise.clone()
    try:
        for k in range(64):
            t, h = k / 64, 1 / 64
            amount = 0. if zero else previous.old.amount_at(config, t)
            def velocity(x, tv):
                ts = x.new_tensor(tv)
                if not amount:
                    return rt.field(x, ts, 'full')
                if source == 'ig':
                    strong, weak = rt.pair(x, ts)
                else:
                    strong = rt.field(x, ts, 'full')
                    try:
                        rt.labels = torch.full_like(labels, 100)
                        rt.model.final_layer = module
                        weak = rt.field(x, ts, 'full')
                    finally:
                        rt.labels = labels
                        rt.model.final_layer = original_final
                return strong + amount * (strong - weak)
            first = velocity(z, t)
            second = velocity(z + h * first, t + h)
            z = z + (h / 2) * (first + second)
            if not torch.isfinite(z).all() or z.abs().max() > 1e6:
                raise FloatingPointError(f'{config["arm"]}: invalid state at step {k}')
    finally:
        rt.head, rt.model.final_layer, rt.labels = original_head, original_final, labels
    full = rt.counts['full'] - before['full']
    prefix = rt.counts['prefix'] - before['prefix']
    assert full == (128 if zero or source == 'ig' else 224) and prefix == 0, (full, prefix)
    return z, dict(full_calls=full, prefix_calls=prefix, auxiliary_full_calls=0,
        diagnostic_queries=0, strang_active_steps=0, diagnostics=np.zeros((len(z),5)),
        head_method=method, independent_weak_prefix=False, guidance_uses_external_semantics=False)
