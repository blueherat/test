"""Verify IG arithmetic, native paths, raw-strong references and query counts."""
import argparse
from contextlib import nullcontext
import numpy as np
import torch
from experiments.ig_sg_20260914 import core as c


class Analytic:
    def __init__(self,name):
        self.name=name;self.observed_queries=0;self.rt=self;self.counts=dict(full=0,prefix=0);self.queries=[]
    @property
    def calls(self):return self.counts['full']
    def context(self):return nullcontext()
    def grid(self,n,device):
        t=torch.linspace(0,1,n+1,device=device)
        return 8*(1-t)/(1+7*(1-t)) if self.name=='raev2' else t
    def query(self,z,t,labels,need_weak=True):
        self.counts['full']+=1;self.observed_queries+=1;self.queries.append((z.clone(),float(t),labels.clone(),need_weak))
        full=-.1*z+.04*z.square()+t
        return full,full*.7+.02 if need_weak else None


def cpu():
    torch.manual_seed(1422);x=torch.randn(2,2,4,4);y=torch.tensor([1,3])
    for name in c.SHAPES:
        cfg=c.configs(name)[0];cfg=dict(cfg,steps=4)
        ref,n=c.sample(Analytic(name),x,y,cfg)
        for kind in ['sg','log']:
            rt=Analytic(name)
            zero,count=c.sample(rt,x,y,dict(cfg,kind=kind,omega=0))
            assert torch.equal(zero,ref) and count['full']==n['full']
        rt=Analytic(name);sg,_=c.sample(rt,x,y,dict(cfg,kind='sg',omega=1,shift=.01))
        assert not torch.equal(sg,ref)
        # All reference calls are raw full predictions with the original labels.
        assert all(torch.equal(q[2],y) for q in rt.queries)
        rt=Analytic(name);log,_=c.sample(rt,x,y,dict(cfg,kind='log',omega=1,kappa=.2))
        assert not torch.equal(log,ref)
        q=rt.queries[:3];torch.testing.assert_close((q[1][0]+q[2][0])*.5,x)
        assert q[0][1]==q[1][1]==q[2][1]
        assert not q[1][3] and not q[2][3]
        repeated,_=c.sample(Analytic(name),x,y,dict(cfg,kind='log',omega=1,kappa=.2))
        assert torch.equal(repeated,log)
    return dict(cpu_passed=True,raw_strong_reference=True,symmetric_same_time_probes=True,repeatable=True)


@torch.inference_mode()
def real(name):
    result=cpu();rt=c.Runtime(name);torch.manual_seed(1423)
    x=torch.randn((2,*c.SHAPES[name]),device='cuda');y=torch.tensor([2,61],device='cuda')
    base=c.configs(name)[0]
    actual,count=c.sample(rt,x,y,base)
    if name=='sit_small':
        from experiments.guidance_pasted_20260912 import common as old
        rt.rt.grid=rt.grid(64,x.device)
        def field(z,t,left,*_):return rt.rt.guided(z,t,old.amount(rt.rt,left,'ig'))
        expected,_=old.integrate(rt.rt,x,y,field)
    elif name=='jit':
        expected,_=rt.rt.sample(x,y,'ig',.3)
    else:
        from experiments.self_guidance_cross_model_20260913 import core as old
        expected,_=old.sample(rt,x,y,dict(base,base='ig'))
    assert torch.equal(expected,actual),float((expected-actual).abs().max())
    checks={}
    for cfg in c.configs(name)[1:3]:
        zero,n=c.sample(rt,x,y,dict(cfg,omega=0));assert torch.equal(zero,actual)
        generated,n=c.sample(rt,x,y,cfg,trace=True)
        assert not torch.equal(generated,actual) and np.isfinite(n['trace']).all()
        if name=='raev2':
            if cfg['kind']=='sg':ref,_=old.sample(rt,x,y,dict(cfg,base='ig'))
            else:
                from experiments.weak_reference_20260914 import cross_core as previous
                ref,_=previous.sample(rt,x,y,dict(cfg,base='ig'))
            assert torch.equal(ref,generated),cfg
        assert rt.decode(generated).shape==(2,256,256,3)
        checks[cfg['kind']]=dict(full=n['full'],delta_rms=float((generated-actual).square().mean().sqrt()))
    result.update(model=name,model_passed=True,native_baseline_bitwise=True,
        zero_strength_bitwise=True,actual_block_hook_matches_counts=True,
        raw_source_counts=count['full'],checks=checks,runtime_sources=rt.rt.sources,
        maximum_allocated_bytes=torch.cuda.max_memory_allocated(),raev2_prior_implementation_bitwise=name=='raev2')
    c.common.atomic(c.ROOT/'checks'/f'{name}.json',result)
    print(result,flush=True);rt.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(c.SHAPES),required=True);real(p.parse_args().model)
