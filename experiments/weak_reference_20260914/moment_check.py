"""Calibrate diagonal moments; check analytical covariance and query coordinates."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from experiments.weak_reference_20260914 import moment as m
from experiments.guidance_pasted_20260912 import common as c
from experiments.self_guidance_20260913.check import Analytic


def prepare():
    parent=c.EXPS/'sit_measure_guidance_20260912';path=parent/'train_moments.npy'
    data=np.load(path,mmap_mode='r');means=[];variances=[]
    for block in data:
        clean_mean=block[:,:4].astype(np.float64)*.18215
        posterior_std=block[:,4:].astype(np.float64)*.18215
        mean=clean_mean.mean(0);var=((clean_mean-mean)**2+posterior_std**2).mean(0)
        means.append(mean.astype(np.float32));variances.append(var.astype(np.float32))
    means=np.stack(means);variances=np.stack(variances)
    assert variances.min()>0 and np.allclose(means,np.load(parent/'class_means.npy'),atol=1e-6)
    m.STATS.parent.mkdir(parents=True,exist_ok=True)
    np.savez(m.STATS,mean=means,variance=variances)
    c.atomic(m.STATS.with_suffix('.json'),dict(source=str(path),source_sha256=c.sha(path),
        samples_per_class=int(data.shape[1]),classes=len(means),scaling=.18215,
        diagonal_covariance=True,posterior_variance_included=True,stats_sha256=c.sha(m.STATS)))


def checks(real=False):
    torch.manual_seed(1409);z=torch.randn(3,2,4,4);mean=torch.randn_like(z);variance=torch.rand_like(z)*2+.1
    class Gaussian(Analytic):
        def field(self,z,t,kind):
            self.counts['full']+=1
            return mean+(t*variance-(1-t))/(t*t*variance+(1-t)**2)*(z-t*mean)
    errors=[];rt=Gaussian();labels=torch.arange(3)
    for t in (.015625,.2,.7,.95):
        full=rt.field(z,z.new_tensor(t),'full')
        delta=m.correction(rt,z,t,labels,full,.2,mean,variance)
        errors.append(float(delta.abs().max()));assert errors[-1]<5e-5
    result=dict(cpu_passed=True,diagonal_gaussian_max_error=max(errors),
                includes_input_time_score_coordinate_transforms=True)
    if real:
        rt=c.runtime('sit_small');z=torch.randn(2,4,32,32,device='cuda');labels=torch.tensor([2,61],device='cuda')
        cfg=dict(arm='check',kind='moment',alpha=1.25,omega=0,kappa=.2,steps=64)
        base=m.sample(rt,z,labels,cfg);ref=m.plain.sample(rt,z,labels,dict(cfg,kind='baseline'))
        assert torch.equal(base['latents'],ref['latents'])
        out=m.sample(rt,z,labels,dict(cfg,omega=1))
        assert out['counts']['full']==175 and not torch.equal(out['latents'],ref['latents'])
        result.update(model_passed=True,zero_native_bitwise=True,counts=out['counts'])
    c.atomic(c.WORK/'docs/data/weak_reference_20260914/moment_check.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--model',action='store_true')
    args=p.parse_args()
    if args.prepare:prepare()
    checks(args.model)
