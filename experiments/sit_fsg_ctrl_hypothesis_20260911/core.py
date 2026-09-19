"""Frozen-field interventions; all model and oracle calls use the counted runtime."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch
from experiments.sit_guidance_portfolio_20260910 import operators as old
from experiments.sit_fsg_followup_20260910 import core as legacy_fsg
from experiments.sit_control_50_20260910 import core as legacy
from experiments.lifting_scale_sweep_20260909 import array_sha

norm, cap, inner, projection = old.norm, old.cap, old.inner, old.projection
EPS = 1e-10
METHODS = {
    'cfg_tuned': dict(kind='cfg', amount=1.25),
    'cfg_high': dict(kind='cfg', amount=2.75),
    'fsg_high': dict(kind='fsg', amount=2.75, horizon=.125),
    'fsg_length_high': dict(kind='fsg_length', amount=2.75, horizon=.125),
    'fsg_debiased_high': dict(kind='fsg_debiased', amount=2.75, horizon=.125),
    'cycle_high': dict(kind='cycle', amount=2.75, horizon=.125),
    'smc_high': dict(kind='smc', amount=2.75, gain=.2, decay=5.),
    'instant_high': dict(kind='instant', amount=2.75, gain=.2, decay=5.),
    'soft_high': dict(kind='soft', amount=2.75, gain=.2, decay=5.),
    'norm_high': dict(kind='norm', amount=2.75, gain=.2, decay=5.),
    'instant01_high': dict(kind='instant', amount=2.75, gain=.1, decay=5.),
    'refined_high': dict(kind='refined', amount=2.75),
    'local_fit': dict(kind='fit', amount=1.25, objective='local'),
    'agreement_fit': dict(kind='fit', amount=1.25, objective='agreement'),
    'write_fit': dict(kind='fit', amount=1.25, objective='write'),
    'guided_write_fit': dict(kind='fit', amount=1.25, objective='guided_write'),
}
REFINE_STEPS = (0, 6, 12, 18, 24, 30, 36, 42, 47)
OBJECTIVES = ('local', 'agreement', 'write', 'guided_write')


def flat(x):
    return x.flatten(1)


def rms(x):
    return flat(x).square().mean(1).sqrt()


def unit(x):
    return x/norm(x).clamp_min(EPS)


def field(rt, x, t, labels, null=False):
    before = rt.labels
    rt.labels = torch.full_like(labels, 100) if null else labels
    try:
        return rt.field(x, x.new_tensor(float(t)), 'full')
    finally:
        rt.labels = before


def pair(rt, x, t, labels):
    return field(rt, x, t, labels), field(rt, x, t, labels, True)


def correction(gap, previous, kind, gain=.1, decay=5.):
    previous = gap if previous is None else previous
    sliding = (gap-previous)+decay*previous
    instantaneous = gap-gain*gap.sign()
    if kind == 'smc':
        modified = gap-gain*sliding.sign()
    elif kind == 'instant':
        modified = instantaneous
    elif kind == 'soft':
        modified = gap.sign()*(gap.abs()-gain).clamp_min(0)
    elif kind == 'norm':
        modified = gap*(norm(instantaneous)/norm(gap).clamp_min(EPS))
    else:
        modified = gap
    return modified, sliding


def model_field(rt, x, t, labels, method, memory=None, active=True):
    amount = method['amount'] if active else 0.
    c = field(rt, x, t, labels)
    if not amount:
        return c, dict(memory=memory)
    u = field(rt, x, t, labels, True)
    g = c-u
    kind = method['kind']
    if kind in ('smc', 'instant', 'soft', 'norm'):
        m, sliding = correction(g, memory, kind, method['gain'], method['decay'])
        value = u+(1+amount)*m
    else:
        m, sliding = g, g
        value = c+amount*g
    return value, dict(memory=m.detach(), gap=g, modified=m, sliding=sliding, conditional=c, null=u)


def fsg_delta(rt, x, t, labels, amount, horizon=.125, same_field=False):
    before = rt.labels
    rt.labels = labels
    try:
        # Retain the existing adapter's counted prefix query, so its cost and
        # arithmetic can be checked against the previously committed FSG control.
        p = legacy_fsg.Probe(rt, x, t, SimpleNamespace(xi=x))
        h = min(horizon, 1-t-1e-5)
        moved = x+h*(p.u if same_field else p.c+amount*p.d)
        raw = moved-h*p.field(moved, t+h, 'u')-x
        radius = (4/64)*amount*norm(p.d)
        return raw, radius, p.d
    finally:
        rt.labels = before


def heun_step(rt, x, k, labels, method, memory=None, *, calibrate=True):
    t, h = k/64, 1/64
    active = k < 48
    if calibrate and active and method['kind'] in ('fsg', 'cycle', 'fsg_length', 'fsg_debiased') and k % 4 == 0:
        delta, radius, g = fsg_delta(rt, x, t, labels, method['amount'],
                                   method['horizon'], method['kind'] == 'cycle')
        if method['kind'] == 'fsg_length':
            delta = norm(delta)*unit(g)
        elif method['kind'] == 'fsg_debiased':
            cycle, _, _ = fsg_delta(rt,x,t,labels,method['amount'],method['horizon'],True)
            delta = delta-cycle
        x = x+cap(delta, radius)
    if calibrate and active and method['kind'] == 'fit' and k == 24:
        x, _ = fit_update(rt, x, k, labels, method['amount'], method['objective'])
    subdivisions = 2 if method['kind'] == 'refined' and k in REFINE_STEPS else 1
    left_info = None
    for j in range(subdivisions):
        width = h/subdivisions
        left = t+j*width
        first, info = model_field(rt, x, left, labels, method, memory, active)
        second, _ = model_field(rt, x+width*first, left+width, labels, method, memory, active)
        x = x+(width/2)*(first+second)
        if j == 0:
            left_info = info
    if method['kind'] == 'smc' and active:
        memory = left_info['memory']
    if not torch.isfinite(x).all() or x.abs().max() > 1e6:
        raise FloatingPointError(f'Invalid trajectory for {method} at {k}')
    return x, memory, left_info


def future(rt, x, k, labels, kind='null', steps=None, amount=1.25, return_path=False):
    steps = (64-k) if steps is None else int(steps)
    t0 = k/64
    # Reuse the exact main grid for discrete semigroup/replay checks.
    grid = [j/64 for j in range(k,65)] if steps == 64-k else [t0+(1-t0)*j/steps for j in range(steps+1)]
    state, path = x, [x]
    for index, (left, right) in enumerate(zip(grid[:-1], grid[1:])):
        h = right-left
        if kind == 'null':
            first = field(rt, state, left, labels, True)
            second = field(rt, state+h*first, right, labels, True)
        elif kind == 'conditional':
            first = field(rt, state, left, labels)
            second = field(rt, state+h*first, right, labels)
        elif kind == 'guided':
            method = dict(kind='cfg', amount=amount)
            first, _ = model_field(rt, state, left, labels, method, active=left < .75)
            second, _ = model_field(rt, state+h*first, right, labels, method, active=left < .75)
        else:
            raise ValueError(kind)
        state = state+(h/2)*(first+second)
        if return_path:
            path.append(state)
    return path if return_path else state


def inverse_null(rt, endpoint, k, labels, subdivisions=1):
    state = endpoint
    grid = [j/(64*subdivisions) for j in range(64*subdivisions, k*subdivisions-1, -1)]
    for left, right in zip(grid[:-1], grid[1:]):
        h = right-left
        first = field(rt, state, left, labels, True)
        second = field(rt, state+h*first, right, labels, True)
        state = state+(h/2)*(first+second)
    if not torch.isfinite(state).all():
        raise FloatingPointError('Nonfinite null inversion')
    return state


def interval(rt, state, left, right, labels, *, null=False, amount=0., steps=8):
    """Same-time forward/inverse comparisons, with a declared numerical grid."""
    grid = [left+(right-left)*j/steps for j in range(steps+1)]
    for t,end in zip(grid[:-1],grid[1:]):
        h = end-t
        if null:
            first=field(rt,state,t,labels,True)
            second=field(rt,state+h*first,end,labels,True)
        else:
            method=dict(kind='cfg',amount=amount)
            first,_=model_field(rt,state,t,labels,method,active=amount!=0.)
            second,_=model_field(rt,state+h*first,end,labels,method,active=amount!=0.)
        state=state+(h/2)*(first+second)
    return state


def trajectory(rt, noise, labels, method, *, capture=(), start=0, state=None, memory=None,
               handoff=None, tail='null', trace=False):
    method = METHODS[method] if isinstance(method, str) else method
    rt.labels = labels
    state = noise.clone() if state is None else state.clone()
    memory = None if memory is None else memory.clone()
    initial = rt.counts.copy()
    snapshots, traces = {}, []
    for k in range(start,64):
        if k in capture:
            snapshots[k] = dict(state=state.clone(), memory=None if memory is None else memory.clone(),
                full_calls=rt.counts['full']-initial['full'], prefix_calls=rt.counts['prefix']-initial['prefix'])
        if handoff is not None and k == handoff:
            state = future(rt, state, k, labels, tail)
            break
        state, memory, info = heun_step(rt, state, k, labels, method, memory)
        if trace and 'gap' in info:
            g, m, s = info['gap'], info['modified'], info['sliding']
            traces.append(dict(k=k, gap_rms=rms(g), clean_gap_rms=(1-k/64)*rms(g),
                noise_gap_rms=(k/64)*rms(g), modified_rms=rms(m), correction_rms=rms(m-g),
                sign_disagreement=flat(s.sign()!=g.sign()).float().mean(1),
                guidance_flip_fraction=flat(m*g<0).float().mean(1)))
    stats = dict(full_calls=rt.counts['full']-initial['full'], prefix_calls=rt.counts['prefix']-initial['prefix'],
        auxiliary_full_calls=0, diagnostic_queries=0, strang_active_steps=0,
        diagnostics=np.zeros((len(noise),5),dtype=np.float64), auxiliary_decoder_images=0,
        auxiliary_classifier_images=0)
    stop = 64 if handoff is None else handoff
    base_calls = sum(4 if k<48 and method['amount'] else 2 for k in range(start,stop))
    if handoff is not None:
        base_calls += 2*(64-handoff)
    stats['auxiliary_full_calls'] = max(0, stats['full_calls']-base_calls)
    return state, stats, snapshots, traces


def oracle(rt, state, k, labels, steps=16):
    c, u = pair(rt, state, k/64, labels)
    return dict(g=c-u, c=future(rt,state,k,labels,'conditional',steps),
                u=future(rt,state,k,labels,'null',steps))


def basis_vectors(gap, fsg_direction, noise):
    seed = int(array_sha(noise.detach().cpu().numpy())[:15],16)
    generator = torch.Generator(device=noise.device).manual_seed((seed+202611211) % (2**63-1))
    random = torch.randn((2,*noise.shape), device=noise.device, dtype=noise.dtype, generator=generator)
    vectors = [gap, fsg_direction, *random.unbind(0)]
    q = []
    for vector in vectors:
        for _ in range(2):
            for previous in q:
                vector = vector-projection(vector,previous)
        q.append(unit(vector))
    return torch.stack([flat(v) for v in q],-1)


def combine(q, coefficient, like):
    with old.exact_matmul():
        return torch.bmm(q, coefficient.unsqueeze(-1)).squeeze(-1).reshape_as(like)


def least_squares(residual, jacobian):
    with old.exact_matmul():
        j, r = jacobian.double(), flat(residual).double()
        gram = j.transpose(1,2)@j
        ridge = .001*gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
        system = gram+ridge[:,None,None]*torch.eye(j.shape[-1],device=j.device,dtype=j.dtype)
        coefficient = torch.linalg.solve(system,-(j.transpose(1,2)@r[:,:,None])).squeeze(-1)
    return coefficient.to(residual.dtype)


def loss(bundle, name, reference, guided_reference):
    error = {'local':bundle['g'], 'agreement':bundle['c']-bundle['u'],
             'write':bundle['u']-reference, 'guided_write':bundle['u']-guided_reference}[name]
    return flat(error).square().mean(1)


@old.exact_matmul()
def fit_setup(rt, state, k, labels, amount):
    raw, radius, gap = fsg_delta(rt,state,k/64,labels,amount)
    q = basis_vectors(gap,raw,state)
    epsilon = .001*norm(state).clamp_min(64.)
    baseline = oracle(rt,state,k,labels)
    guided_reference = future(rt,state,k,labels,'guided',16,amount)
    differences = {name:[] for name in ('g','c','u')}
    for index in range(q.shape[-1]):
        direction = q[:,:,index].reshape_as(state)
        plus = oracle(rt,state+epsilon*direction,k,labels)
        minus = oracle(rt,state-epsilon*direction,k,labels)
        for name in differences:
            differences[name].append(flat((plus[name]-minus[name])/(2*epsilon)))
    jacobians = {name:torch.stack(values,-1) for name,values in differences.items()}
    residuals = dict(local=baseline['g'], agreement=baseline['c']-baseline['u'],
                    write=baseline['u']-baseline['c'], guided_write=baseline['u']-guided_reference)
    responses = dict(local=jacobians['g'],agreement=jacobians['c']-jacobians['u'],
                     write=jacobians['u'],guided_write=jacobians['u'])
    return dict(raw=raw,radius=radius,gap=gap,q=q,epsilon=epsilon,baseline=baseline,
        guided_reference=guided_reference,jacobians=jacobians,residuals=residuals,responses=responses)


@old.exact_matmul()
def fit_from_setup(rt, state, k, labels, name, setup):
    coefficient = least_squares(setup['residuals'][name],setup['responses'][name])
    delta = cap(combine(setup['q'],coefficient,state),setup['radius'])
    candidates = [state,state+.5*delta,state+delta]
    values = [setup['baseline']]+[oracle(rt,z,k,labels) for z in candidates[1:]]
    losses = torch.stack([loss(v,name,setup['baseline']['c'],setup['guided_reference']) for v in values],1)
    index = losses.argmin(1)
    stack = torch.stack(candidates,1)
    result = stack[torch.arange(len(state),device=state.device),index]
    selected = losses.gather(1,index[:,None]).squeeze(1)
    assert (selected <= losses[:,0]+1e-10).all()
    return result,dict(losses=losses,index=index,delta=delta,
        before=losses[:,0],after=selected,shift_rms=rms(result-state))


def fit_update(rt, state, k, labels, amount, name):
    setup = fit_setup(rt,state,k,labels,amount)
    return fit_from_setup(rt,state,k,labels,name,setup)


@old.exact_matmul()
def proposals(rt, state, k, labels, amount):
    setup = fit_setup(rt,state,k,labels,amount)
    radius = setup['radius']
    same, _, _ = fsg_delta(rt,state,k/64,labels,amount,same_field=True)
    candidates = dict(unchanged=state, cfg_radius=state+radius*unit(setup['gap']),
        fsg_capped=state+cap(setup['raw'],radius), fsg_radius=state+radius*unit(setup['raw']),
        cycle_radius=state+radius*unit(same))
    records = {}
    for name in OBJECTIVES:
        with old.exact_matmul():
            gradient = -(setup['responses'][name].transpose(1,2)@flat(setup['residuals'][name])[:,:,None]).squeeze(-1)
        direction = combine(setup['q'],gradient,state)
        candidates[name+'_radius'] = state+radius*unit(direction)
        candidates[name+'_fit'], records[name] = fit_from_setup(rt,state,k,labels,name,setup)
    return candidates, setup, records


def install():
    pass


def configurations():
    # The original engine uses this exact anchor only for its historical replay check.
    anchor = next(c for c in legacy.configurations() if c['family']=='ig_local' and c['strength']==.8)
    return [anchor]


def sample(rt, noise, labels, config, *, zero=False):
    if config['key']=='ig_local_attention':
        return legacy.sample(rt,noise,labels,config,zero=zero)
    method = dict(METHODS[config['parameters']['method']])
    if zero:
        method = dict(kind='cfg',amount=0.)
    endpoint, stats, _, _ = trajectory(rt,noise,labels,method,
        handoff=config['parameters'].get('handoff'),tail=config['parameters'].get('tail','null'))
    return endpoint, stats


def limiting_checks(rt, noise, labels):
    rt.labels=labels
    x = noise[:2]
    y = labels[:2]
    c,u=pair(rt,x,.25,y)
    assert rt.labels is labels
    for kind in ('smc','instant','soft','norm'):
        m,_=correction(c-u,None,kind,0.,5.)
        torch.testing.assert_close(m,c-u,rtol=1e-6,atol=1e-7)
    v,_=model_field(rt,x,.25,y,dict(kind='cfg',amount=0.))
    assert torch.equal(v,c)
    return dict(zero_control_exact=True,labels_restored=True,zero_correction_checked=True)
