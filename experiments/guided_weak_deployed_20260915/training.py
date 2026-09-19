import torch
from experiments.guidance_dynamic_50k_20260915 import training as original
from experiments.guidance_dynamic_50k_20260915 import config as k
from . import NAME, REQUEST


def anchor(model, t):
    profile = torch.where(t < .25, 6/7, 1.) if model == 'sit_small' else torch.ones_like(t)
    return k.settings(model)['alpha'] * profile


def active_time(model, t, generator):
    if model == 'sit_small':
        return t * .5
    t = t.clone()
    while bool((t >= .5).any()):
        mask = t >= .5
        t[mask] = torch.sigmoid(torch.randn(int(mask.sum()),device=t.device,generator=generator)*.8-.8)
    return t


def install_training():
    original_compute = original.compute_loss
    original_train = original.train
    original_stream = original.Stream
    original_save = original.save_torch

    class ActiveStream(original_stream):
        def draw(self,generator):
            batch = super().draw(generator)
            batch['t'] = active_time(self.model,batch['t'],generator)
            return batch

    def compute(adapter,net,batch,spec,*args,**kwargs):
        if not spec.get('deployed_objective'):
            return original_compute(adapter,net,batch,spec,*args,**kwargs)
        saved = adapter.cfg['alpha']
        adapter.cfg['alpha'] = anchor(adapter.name,batch['t'])[:,None,None]
        try:
            return original_compute(adapter,net,batch,spec,*args,**kwargs)
        finally:
            adapter.cfg['alpha'] = saved

    def train(model,arm,context,source=None,fold=None):
        if arm != NAME:
            return original_train(model,arm,context,source,fold)
        original.Stream = ActiveStream
        def save(path,value):
            return original_save(path,dict(value,candidate_request_sha256=k.sha(REQUEST)))
        original.save_torch = save
        try:
            return original_train(model,arm,context,source,fold)
        finally:
            original.Stream = original_stream
            original.save_torch = original_save

    original.compute_loss = compute
    original.train = train

