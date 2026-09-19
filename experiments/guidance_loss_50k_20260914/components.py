"""Source sampling, losses and shared-feature readouts."""
import copy
import numpy as np
import torch
from torch import nn

from . import config as k
from experiments.weak_reference_loss_20260914 import runner as legacy
from experiments.weak_reference_loss_20260914.objectives import smoothing_pair, weighted_mse
from experiments.endpoint_contrast_20260914.objectives import contrast_loss, covariance_target

# Configuration is scoped to this worker process; legacy files stay immutable.
legacy.k = k
c, local = legacy.c, legacy.local


class SourceHead(nn.Module):
    def __init__(self, template):
        super().__init__()
        width = template.token_mean.numel()
        for name in ('token_mean', 'token_std', 'condition_mean', 'condition_std'):
            self.register_buffer(name, getattr(template, name).detach().clone())
        self.layers = nn.Sequential(nn.Linear(3*width+2, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, tokens, condition, t):
        tokens = (tokens.float()-self.token_mean)/self.token_std
        condition = (condition.float()-self.condition_mean)/self.condition_std
        value = torch.cat((tokens.mean(1), tokens.std(1, correction=0), condition,
                           t[:,None], t[:,None].square()), 1)
        return self.layers(value).squeeze(1)


def ema_update(averaged, online):
    with torch.no_grad():
        for left, right in zip(averaged.parameters(), online.parameters()):
            left.lerp_(right, .005)


def optimizer(module):
    return torch.optim.AdamW(module.parameters(), lr=3e-4, weight_decay=1e-4, betas=(.9,.999))


def read_negative(source):
    if source in ('real','strong'):
        path = k.DATA / f'{source}_clean.npy'
        return np.load(path), {str(path): k.sha(path)}
    if source == 'weakmix':
        first, provenance = read_negative('strong')
        second, extra = read_negative('weak')
        return np.concatenate((first,second), axis=1), dict(provenance, **extra)
    root = k.ROOT / 'endpoints' / source
    receipt = k.read(root / 'complete.json')
    path = root / 'clean.npy'
    assert receipt['complete'] and receipt['request_sha256']==k.sha(k.ROOT/'request.json')
    assert receipt['files'][str(path)]==k.sha(path)
    for asset,digest in receipt.get('dependencies',{}).items():
        assert k.sha(asset)==digest
    return np.load(path), {str(path):k.sha(path), str(root/'complete.json'):k.sha(root/'complete.json')}


class SourceBank:
    def __init__(self, source, device='cuda'):
        real, p = read_negative('real')
        negative, q = read_negative(source)
        assert real.shape[0] == negative.shape[0] == 100
        assert real.shape[2:] == negative.shape[2:] == (4,32,32)
        self.positive = torch.from_numpy(real).to(device)
        self.negative = torch.from_numpy(negative).to(device)
        self.provenance = dict(p, **q)
        self.device = device

    def draw(self, generator, n=k.BATCH, train_excluding_fold=None, validation=False):
        device = self.device
        labels = torch.randint(100,(n,),device=device,generator=generator)
        def indices(size):
            assert size % 20 == 0
            component = torch.randint(size//20,(n,),device=device,generator=generator)
            if validation:
                within = 18+torch.randint(2,(n,),device=device,generator=generator)
            elif train_excluding_fold is None:
                within = torch.randint(18,(n,),device=device,generator=generator)
            else:
                within = 2*torch.randint(9,(n,),device=device,generator=generator)+(1-train_excluding_fold)
            return 20*component+within
        pos_id = indices(self.positive.shape[1])
        neg_id = indices(self.negative.shape[1])
        source = (torch.randperm(n,device=device,generator=generator)%2).float()
        time = .01+.98*torch.rand(n,device=device,generator=generator)
        noise = torch.randn((n,4,32,32),device=device,generator=generator)
        coin = torch.rand(n,device=device,generator=generator)
        null_source = (torch.randperm(n,device=device,generator=generator)%2).float()
        positive, negative = self.positive[labels,pos_id], self.negative[labels,neg_id]
        clean = torch.where(source[:,None,None,None].bool(),positive,negative)
        fold = torch.where(source.bool(),pos_id,neg_id)%2
        return dict(labels=labels,source=source,t=time,noise=noise,coin=coin,null_source=null_source,
                    positive=positive,negative=negative,clean=clean,fold=fold,pos_id=pos_id,neg_id=neg_id)


def native_amount(t):
    return torch.where(t<.25,k.ALPHA*6/7,k.ALPHA)*(t<.5)


def unconditional(rt, z, t):
    labels = rt.labels
    rt.labels = torch.full_like(labels, 100)
    try:
        return rt.field(z,t,'full')
    finally:
        rt.labels = labels


@torch.no_grad()
def base_velocity(rt, z, t, labels, kind):
    """Per-state training baseline; no source label enters this function."""
    rt.labels = labels
    if kind == 'ig':
        full, weak = rt.pair(z,t)
        return full+native_amount(t)[:,None,None,None]*(full-weak)
    full = rt.field(z,t,'full')
    if kind == 'cfg':
        uncond = unconditional(rt,z,t)
        full = full+(1.25*(t<.75))[:,None,None,None]*(full-uncond)
    return full


def load_nuisances(rt, source):
    models, provenance = [], {}
    for fold in (0,1):
        root = k.ROOT/'nuisance'/source/f'fold{fold}'
        done = k.read(root/'complete.json')
        assert done['complete'] and done['steps']==k.STEPS
        assert done['head_sha256']==k.sha(root/'head.pt')
        state = legacy.load_torch(root/'head.pt')
        mean = legacy.make_head(rt)
        eta = SourceHead(mean).cuda()
        mean.load_state_dict(state['mean_ema'])
        eta.load_state_dict(state['eta_ema'])
        models.append((mean.eval().requires_grad_(False),eta.eval().requires_grad_(False)))
        provenance[str(root/'head.pt')] = k.sha(root/'head.pt')
    return models, provenance


@torch.no_grad()
def nuisance_prediction(models, features, t, folds):
    eta = torch.empty(len(t),device=t.device)
    mean = None
    for fold,(mean_model,eta_model) in enumerate(models):
        mask = folds == fold
        if not mask.any():
            continue
        tokens, condition = features['context'][mask], features['condition'][mask]
        output = mean_model(tokens,condition)
        if mean is None:
            mean = torch.empty((len(t),*output.shape[1:]),device=t.device,dtype=output.dtype)
        mean[mask] = output
        eta[mask] = torch.sigmoid(eta_model(tokens,condition,t[mask]))
    assert mean is not None and torch.isfinite(eta).all() and torch.isfinite(mean).all()
    return eta, mean


def objective(prediction, target, source, eta, baseline, kind, weights=None):
    target, baseline = target.detach(), baseline.detach()
    if kind == 'fm':
        return weighted_mse(prediction,target,weights if weights is not None else torch.ones(len(target),device=target.device))
    if kind == 'residual':
        return (prediction-(target-baseline).detach()).square().mean()
    if kind == 'guided_weak':
        return (baseline+k.ALPHA*(baseline-prediction)-target).square().mean()
    if kind in ('contrast','null','contrast_weak'):
        correction = k.ALPHA*(baseline-prediction) if kind=='contrast_weak' else prediction
        return contrast_loss(correction,target,source,eta,baseline)
    if kind == 'covariance':
        desired = covariance_target(target,source,eta,baseline)
        return (prediction-desired).square().mean()
    raise ValueError(kind)


def guided_field(rt, head, capture, arm, z, t, left):
    value = k.spec(arm)
    a = c.amount(rt,left,'ig')
    need_head = head is not None and a != 0
    if value['base']=='ig' and a:
        full, weak = rt.pair(z,t)
        base = full+a*(full-weak)
    else:
        full = rt.field(z,t,'full')
        base = full
    # Preserve the conditional features before a possible unconditional pass.
    features = dict(capture.values) if need_head else None
    if value['base']=='cfg':
        cfg = c.amount(rt,left,'cfg')
        if cfg:
            uncond = unconditional(rt,z,t)
            base = full+cfg*(full-uncond)
    if not need_head:
        return base
    pred = local.unpatchify(rt,head(features['context'],features['condition'])).float()
    if value['loss'] in k.WEAK_LOSSES:
        return base+(a*(.5 if arm.endswith('_half') else 1.))*(full-pred)
    factor = 4. if value['loss']=='covariance' else 1.
    return base+(a/k.ALPHA)*factor*pred
