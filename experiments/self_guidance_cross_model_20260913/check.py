"""Time, history, baseline-parity, and actual full-forward checks."""
import argparse
import json
import time
import torch
from experiments.self_guidance_cross_model_20260913 import core as c


class Fake:
    def __init__(self,name): self.name,self.calls,self.queries=name,0,[]
    def grid(self,steps,device):
        x=torch.linspace(0,1,steps+1,device=device)
        return x if self.name=='jit' else x.flip(0)
    def query(self,z,t,labels):
        self.calls+=1;self.queries.append((z.clone(),float(t)))
        sign=1 if self.name=='jit' else -1
        full=sign*(.1*z+t.square()+labels[:,None,None,None]*.0001)
        return full,.8*full


def cpu():
    x=torch.ones(2,1,2,2);y=torch.tensor([3,7])
    for name in ('jit','raev2'):
        base='cfg' if name=='jit' else 'ig'
        rt=Fake(name)
        conf=dict(arm='check',base=base,kind='sg',omega=1.,shift=.1,steps=8)
        c.field(rt,x,torch.tensor(.4),y,conf)
        assert abs(rt.queries[-1][1]-(.3 if name=='jit' else .5))<1e-6
        assert torch.equal(rt.queries[-1][0],x)
        baseline,_=c.sample(Fake(name),x,y,dict(conf,kind='baseline',omega=0.))
        for kind in ('sg','sg_prev'):
            zero,counts=c.sample(Fake(name),x,y,dict(conf,kind=kind,omega=0.))
            assert torch.equal(zero,baseline)
            assert counts['full']==(16 if name=='jit' else 8)
        conf=dict(conf,kind='sg_prev')
        a,_=c.sample(rt,x,y,conf);b,_=c.sample(rt,x,y,conf)
        assert torch.equal(a,b)
        z,old=x.clone(),None
        rt2=Fake(name);grid=rt2.grid(8,x.device)
        for t,u in zip(grid[:-1],grid[1:]):
            full,weak=rt2.query(z,t,y)
            if name=='jit':
                null,_=rt2.query(z,t,torch.full_like(y,1000))
                guided=null+torch.where((t>.1)&(t<1),3.,1.)*(full-null)
            else: guided=weak+1.78*(full-weak) if .1<=float(t)<=1 else full
            active=(float(t)>.5 if name=='jit' else float(t)<.5)
            if old is not None and active: guided=guided+(full-old)
            z,old=z+(u-t)*guided,full.clone()
        assert torch.equal(a,z)
    return dict(cpu_passed=True, noisier_query_direction=True,same_state=True,
                raw_conditional_history=True,history_reset=True,zero_scale_exact=True)


@torch.inference_mode()
def model(name):
    rt=c.Runtime(name)
    gen=torch.Generator(device='cuda').manual_seed(2026091321)
    x=torch.randn((1,*c.SHAPES[name]),device='cuda',generator=gen)
    y=torch.tensor([267],device='cuda')
    base='cfg' if name=='jit' else 'ig'
    conf=dict(arm='check',kind='baseline',base=base,omega=0.,steps=100)
    expected=x.clone();grid=rt.grid(100,x.device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if name=='jit':
            from denoiser import Denoiser
            official=Denoiser.__new__(Denoiser);torch.nn.Module.__init__(official)
            official.net=rt.model;official.num_classes=1000;official.cfg_scale=3.
            official.cfg_interval=(.1,1.);official.t_eps=.05
            for t,u in zip(grid[:-1],grid[1:]):
                expected=official._euler_step(expected,t.expand(1,1,1,1),u.expand(1,1,1,1),y)
        else:
            for t,u in zip(grid[:-1],grid[1:]):
                full,weak=rt.model(expected,t.expand(1),context=y,attn_mask=None)
                fv=rt.rt.native.clean_to_velocity(full,expected,t.expand(1),denominator_floor=rt.t_floor)
                bv=rt.rt.native.clean_to_velocity(weak,expected,t.expand(1),denominator_floor=rt.t_floor)
                v=bv+1.78*(fv-bv) if .1<=float(t)<=1 else fv
                expected=expected+(u-t)*v
    actual,_=c.sample(rt,x,y,conf)
    assert torch.equal(actual,expected),float((actual-expected).abs().max())
    for kind in ('sg','sg_prev'):
        zero,_=c.sample(rt,x,y,dict(conf,kind=kind))
        assert torch.equal(zero,actual)
    records={}
    for kind in ('sg','sg_prev'):
        nonzero,counts=c.sample(rt,x,y,dict(conf,kind=kind,omega=1.))
        assert not torch.equal(nonzero,actual)
        records[kind]=dict(full=counts['full'],rms=float(nonzero.square().mean().sqrt()))
    # A full B8 path checks production memory and gives a real throughput estimate.
    batch=8
    x=x.expand(batch,-1,-1,-1).clone();y=y.expand(batch).clone()
    torch.cuda.synchronize();start=time.perf_counter()
    z,counts=c.sample(rt,x,y,dict(conf,kind='sg',omega=1.))
    pixels=rt.decode(z)
    torch.cuda.synchronize()
    assert pixels.shape==(batch,256,256,3)
    return dict(model=name,model_passed=True,baseline_bitwise=True,zero_scale_bitwise=True,
                records=records,batch=batch,sg_sample_decode_seconds=time.perf_counter()-start,
                peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,t_floor=rt.t_floor)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['jit','raev2']);a=p.parse_args()
    result=cpu()
    if a.model: result.update(model(a.model))
    path=c.ROOT/'checks'/f'{a.model or "cpu"}.json'
    c.common.atomic(path,result)
    print(json.dumps(result,indent=2))
