"""Same-query CFG-CTRL/APG interpolation with separate accepted histories.

This is an empirical complementarity screen, not a novel method claim. The
parallel control keeps the signed component of the interpolation change along
the existing APG guidance. No controller output enters the other controller.
"""
from pathlib import Path
import numpy as np
import torch
from . import baselines as b

SOURCE_FILES = [Path(b.__file__).resolve()]


def directions(z, t, conditional, unconditional, histories, config):
    gap = conditional - unconditional
    apg_previous, ctrl_previous = histories
    apg_previous = torch.zeros_like(gap) if apg_previous is None else apg_previous
    ctrl_previous = gap if ctrl_previous is None else ctrl_previous
    momentum = gap + float(config.get('beta', -.5)) * apg_previous
    bounded = b.cap(momentum, 2 * b.norm(gap))
    clean = z + (1-t) * conditional
    apg = float(config.get('apg_alpha', 2.)) * (bounded - b.projection(bounded, clean))
    # Preserve the baseline operation order around the discontinuous relay.
    sliding = gap - ctrl_previous + float(config.get('lambda_ctrl', 5.)) * ctrl_previous
    modified = gap - float(config.get('K', .2)) * sliding.sign()
    # Subtract the conditional anchor: v_CTRL - v_c = (1+a)*m - gap.
    ctrl = (1+float(config.get('ctrl_alpha', 2.75))) * modified - gap
    eta = float(config['eta'])
    if config['kind'] == 'blend_parallel':
        update = apg + eta * b.projection(ctrl-apg, apg)
    elif eta == 0.:
        update = apg
    elif eta == 1.:
        update = ctrl
    else:
        update = (1-eta) * apg + eta * ctrl
    return conditional + update, (momentum.detach(), modified.detach())


def field(rt, z, time_value, labels, histories, config, active):
    old_labels = rt.labels
    try:
        rt.labels = labels
        t = z.new_tensor(time_value)
        conditional = rt.field(z,t,'full')
        if not active:
            return conditional, histories
        rt.labels = torch.full_like(labels, 100)
        unconditional = rt.field(z,t,'full')
    finally:
        rt.labels = old_labels
    # Endpoints execute the historical floating-point expression exactly.
    value, proposal = directions(z,t,conditional,unconditional,histories,config)
    if config['kind'] == 'blend' and float(config['eta']) == 1.:
        value = unconditional + (1+float(config.get('ctrl_alpha',2.75))) * proposal[1]
    return value, proposal


@torch.inference_mode()
def sample(rt, noise, labels, config, snapshots=False):
    if config['kind'] in ('cfg','apg','ctrl'):
        return b.sample(rt,noise,labels,config,snapshots)
    assert config['kind'] in ('blend','blend_parallel')
    assert 0 <= float(config['eta']) <= 1
    steps = int(config.get('steps',64)); cutoff = float(config.get('cutoff',.75))
    assert steps in (64,96) and 0 <= cutoff <= 1
    before = rt.counts.copy(); rt.labels=labels
    z=noise.clone(); histories=(None,None); trace=[]; active_steps=0
    saved={'step_000':z.detach().clone()} if snapshots else None
    save_steps={round(steps*f) for f in (.25,.5,.75,1.)}
    with rt.context():
        for k in range(steps):
            t,h=k/steps,1/steps; active=t<cutoff
            first,proposal=field(rt,z,t,labels,histories,config,active)
            second,_=field(rt,z+h*first,t+h,labels,histories,config,active)
            z=z+(h/2)*(first+second)
            if active:histories=proposal;active_steps+=1
            if not torch.isfinite(z).all() or z.abs().max()>1e6:
                raise FloatingPointError((config['arm'],k))
            if snapshots:
                trace.append(torch.stack((z.new_full((len(z),),k),z.new_full((len(z),),t),
                    z.new_full((len(z),),h),z.new_full((len(z),),float(config['eta'])),
                    z.square().flatten(1).mean(1).sqrt(),first.square().flatten(1).mean(1).sqrt()),-1))
                if k+1 in save_steps:saved[f'step_{k+1:03d}']=z.detach().clone()
    counts={key:rt.counts[key]-before[key] for key in ('full','prefix')}
    assert counts==dict(full=2*steps+2*active_steps,prefix=0),counts
    result=dict(latents=z,counts=counts)
    if snapshots:result.update(snapshots=saved,trace=torch.stack(trace).cpu().numpy())
    return result


def check(device):
    """Meaningful endpoint parity and signed-direction checks, no quality claim."""
    from experiments.guidance_pasted_20260912 import common as c
    torch.manual_seed(2026091399)
    if device=='cpu':
        z=torch.randn(3,4,8,8);vc=torch.randn_like(z);vu=torch.randn_like(z)
        h=(torch.randn_like(z),torch.randn_like(z));cfg=dict(kind='blend',eta=.5)
        mix,_=directions(z,.5,vc,vu,h,cfg)
        left,_=directions(z,.5,vc,vu,h,dict(cfg,eta=0.))
        right,_=directions(z,.5,vc,vu,h,dict(cfg,eta=1.))
        torch.testing.assert_close(mix,(left+right)/2,rtol=2e-6,atol=2e-6)
        parallel,_=directions(z,.5,vc,vu,h,dict(cfg,kind='blend_parallel'))
        residual=(mix-parallel);p=left-vc
        assert ((residual*p).flatten(1).sum(1).abs()<1e-3).all()
        return dict(cpu=True,blend_identity=True,signed_projection=True)
    rt=c.runtime('sit_small')
    root=c.EXPS/'cfg_transport_search_20260913/baseline_1k'
    with np.load(root/'inputs.npz') as d:
        noise=c.cuda(d['noise'][:8]);labels=c.cuda(d['labels'][:8])
    comparisons=[]
    for eta,kind,alpha in [(0.,'apg',2.),(1.,'ctrl',2.75)]:
        base=b.sample(rt,noise,labels,dict(kind=kind,alpha=alpha))
        blend=sample(rt,noise,labels,dict(arm='parity',kind='blend',eta=eta))
        assert torch.equal(base['latents'],blend['latents'])
        assert base['counts']==blend['counts']==dict(full=224,prefix=0)
        comparisons.append(dict(eta=eta,baseline=kind,latent_bitwise=True,counts=blend['counts']))
    for kind in ['blend','blend_parallel']:
        result=sample(rt,noise,labels,dict(arm='preflight',kind=kind,eta=.5),True)
        assert len(result['snapshots'])==5 and result['trace'].shape==(64,8,6)
    return dict(actual_model=True,comparisons=comparisons,finite_candidate_trajectories=2)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output')
    args=p.parse_args();result=check(args.device)
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
