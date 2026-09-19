"""Released SG velocity extrapolation on opposite native time conventions.

JiT: clean at t=1, v=(clean-z)/max(1-t,.05).
RAEv2: clean at t=0, v=(z-clean)/max(t,t_eps).
Each shifted prediction is converted at ITS OWN query time. SG-prev caches
the unguided conditional velocity from the previous accepted Euler state.
This is the released flow/velocity rule, not a cross-time exact score ratio.
"""
import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
from pathlib import Path
import math
import torch
import numpy as np
from experiments import lifting_scale_sweep_20260909 as common

ROOT = common.EXPS/'self_guidance_cross_model_20260913'
SHAPES = {'jit': (3,256,256), 'raev2': (1024,16,16)}


class Runtime:
    def __init__(self, name):
        if name not in SHAPES:
            raise ValueError(name)
        self.name, self.calls = name, 0
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if name == 'jit':
            from experiments import jit_internal_guidance as jig
            self.jig = jig
            self.model = jig.load_source('cuda')
            self.t_floor = .05
        else:
            # Existing strictly loaded model/decoder assets and compatibility shim.
            self.rt = common.Runtime('raev2')
            self.model = self.rt.model
            self.t_floor = float(self.rt.cfg.transport.t_eps)
        self.model.register_forward_pre_hook(self.count)

    def count(self, module, args):
        self.calls += 1

    def grid(self, steps, device):
        base = torch.linspace(0,1,steps+1,device=device)
        if self.name == 'jit':
            return base
        return self.rt.native.shifted_time_grid(steps, 8., device)

    def query(self, z, t, labels):
        times = t.expand(len(z))
        if self.name == 'jit':
            clean = self.model(z, times, labels)
            return self.jig.velocity(clean, z, times), None
        full, weak = self.model(z, times, context=labels, attn_mask=None)
        convert = lambda clean: self.rt.native.clean_to_velocity(clean, z, times,
                                                               denominator_floor=self.t_floor)
        return convert(full), convert(weak)

    def decode(self, z):
        if self.name == 'raev2':
            return self.rt.decode(z)
        return np.round(np.clip((z.float().cpu().numpy().transpose(0,2,3,1)+1)*127.5,0,255)).astype(np.uint8)


def configs(name):
    rows = []
    guided = 'cfg' if name == 'jit' else 'ig'
    for base in ('strong', guided):
        rows.append(dict(arm=base, base=base, kind='baseline', omega=0., steps=100))
        for kind in ('sg', 'sg_prev'):
            for omega in ([1.] if base == 'strong' else [1.,3.]):
                rows.append(dict(arm=f'{base}_{kind}_w{int(omega)}', base=base, kind=kind,
                                 omega=omega, shift=.01, prev_start=.5, steps=100))
    if name == 'jit':
        rows.append(dict(arm='cfg_heun50', base='cfg', kind='baseline', omega=0., steps=50, solver='heun'))
    return rows


def field(rt, z, t, labels, config, previous=None):
    full, weak = rt.query(z, t, labels)
    value = full
    base = config['base']
    if base == 'cfg':
        # Preserve official JiT arithmetic and even its inactive null query.
        null, _ = rt.query(z, t, torch.full_like(labels, 1000))
        scale = torch.where((t < 1.) & (t > .1), 3., 1.)
        value = null + scale * (full-null)
    elif base == 'ig' and .1 <= float(t) <= 1.:
        # Existing RAEv2 velocity-space native baseline, IG scale=1.78.
        value = weak + 1.78*(full-weak)
    omega = float(config['omega'])
    active = False
    delta = torch.zeros_like(full)
    if omega and config['kind'] == 'sg':
        shift = config.get('shift', .01)
        ref_time = (t-shift).clamp_min(0) if rt.name == 'jit' else (t+shift).clamp_max(1)
        if not torch.equal(ref_time, t):
            reference, _ = rt.query(z, ref_time, labels)
            delta, active = omega*(full-reference), True
    elif omega and config['kind'] == 'sg_prev' and previous is not None:
        noise_time = 1-float(t) if rt.name == 'jit' else float(t)
        if noise_time < config.get('prev_start', .5):
            delta, active = omega*(full-previous), True
    if active:
        value = value+delta
    return value, full.detach().clone(), delta


@torch.inference_mode()
def sample(rt, noise, labels, config, trace=False):
    if config['kind'] not in ('baseline','sg','sg_prev'):
        raise ValueError(config)
    allowed = ('strong','cfg') if rt.name == 'jit' else ('strong','ig')
    if config['base'] not in allowed:
        raise ValueError(config)
    for k, default in [('omega',0),('shift',.01)]:
        if not math.isfinite(config.get(k,default)) or config.get(k,default) < 0:
            raise ValueError(config)
    if config.get('solver','euler') not in ('euler','heun'):
        raise ValueError(config)
    if config.get('solver') == 'heun' and (rt.name != 'jit' or config['kind'] != 'baseline'):
        raise ValueError('SG-prev uses accepted Euler steps; Heun is an official JiT control.')
    if noise.dtype != torch.float32 or labels.dtype != torch.long or len(noise) != len(labels):
        raise ValueError('Expected FP32 noise paired with int64 labels.')
    before = rt.calls
    z, previous, rows = noise.clone(), None, []
    grid = rt.grid(int(config['steps']), noise.device)
    expected = 0
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=noise.is_cuda):
        for k, (t,u) in enumerate(zip(grid[:-1],grid[1:])):
            v, proposal, correction = field(rt,z,t,labels,config,previous)
            expected += 2 if config['base']=='cfg' else 1
            if config['kind']=='sg' and config['omega'] and config.get('shift',.01)>0 and k>0:
                expected += 1
            predicted = z+(u-t)*v
            if config.get('solver') == 'heun' and k < len(grid)-2:
                second,_,_ = field(rt,predicted,u,labels,config)
                z = z+(u-t)*(.5*(v+second))
                expected += 2 if config['base']=='cfg' else 1
            else:
                z = predicted
            previous = proposal
            if not torch.isfinite(z).all() or z.abs().max()>1e6:
                raise FloatingPointError(f"{rt.name}/{config['arm']} at step {k}")
            if trace:
                rms = lambda x: x.float().square().flatten(1).mean(1).sqrt()
                rows.append(torch.stack((torch.ones_like(rms(z))*t,rms(z),rms(correction)),1).cpu().numpy())
    calls = rt.calls-before
    assert calls == expected, (calls,expected)
    return z, dict(full=calls, prefix=0, trace=np.stack(rows) if rows else np.empty((0,)))
