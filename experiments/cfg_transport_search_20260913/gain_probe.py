"""Observe signed-parallel gains without changing its fields or model queries.

This freezes an equal-image, per-Heun-stage time schedule on 100 independent,
class-balanced calibration noises. No FID is read or computed. directions is
wrapped only inside this process and its original return value is preserved.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import independent_blend as blend
from . import baselines as b

c = b.common
ROOT = c.EXPS / 'cfg_transport_search_20260913/gain_probe'
SEED = 2026091321
N = 100
BATCH = 8
CONFIG = dict(arm='parallel_gain_calibration', kind='blend_parallel', eta=.5,
              apg_alpha=2., beta=-.5, ctrl_alpha=2.75, lambda_ctrl=5., K=.2,
              steps=64, cutoff=.75)
SOURCE_FILES = [Path(blend.__file__).resolve(), Path(b.__file__).resolve(),
                Path(c.__file__).resolve(), c.WORK/'experiments/lifting_scale_sweep_20260909.py']
FIELDS = ('gain', 'apg_norm', 'ctrl_norm', 'gap_norm', 'clean_norm',
          'apg_clean_cosine', 'ctrl_clean_cosine', 'apg_gap_cosine',
          'ctrl_apg_cosine', 'actual_update_clean_cosine', 'small_apg',
          'gain_numerator', 'gain_denominator')


def dot(x, y):
    return (x*y).flatten(1).sum(1)


def length(x):
    return x.square().flatten(1).sum(1).sqrt()


def cosine(x, y):
    return dot(x,y)/(length(x)*length(y)).clamp_min(b.EPS)


class Recorder:
    def __init__(self):
        self.original = blend.directions
        self.records = []
        self.errors = []

    def directions(self, z, t, conditional, unconditional, histories, config):
        index = len(self.records)
        assert index < 96 and abs(float(t)-((index//2+index%2)/64)) < 1e-12
        value, proposal = self.original(z,t,conditional,unconditional,histories,config)
        gap = conditional-unconditional
        momentum, modified = proposal
        clean = z+(1-t)*conditional
        bounded = b.cap(momentum,2*b.norm(gap))
        apg = float(config['apg_alpha'])*(bounded-b.projection(bounded,clean))
        ctrl = (1+float(config['ctrl_alpha']))*modified-gap
        numerator = dot(ctrl-apg,apg)
        denominator = dot(apg,apg)
        # The floor matches the ACTUAL signed projection implementation.
        gain = 1+float(config['eta'])*numerator/denominator.clamp_min(b.EPS)
        update = apg+float(config['eta'])*b.projection(ctrl-apg,apg)
        self.errors.append((value-(conditional+update)).abs().max())
        self.records.append(torch.stack((gain,length(apg),length(ctrl),length(gap),length(clean),
            cosine(apg,clean),cosine(ctrl,clean),cosine(apg,gap),cosine(ctrl,apg),
            cosine(update,clean),(denominator<=b.EPS).to(gain.dtype),numerator,denominator),-1))
        return value, proposal

    @contextlib.contextmanager
    def capture(self):
        assert blend.directions is self.original
        blend.directions = self.directions
        try:
            yield
        finally:
            blend.directions = self.original

    def arrays(self):
        assert len(self.records)==96
        values=torch.stack(self.records,1).reshape(len(self.records[0]),48,2,len(FIELDS))
        error=float(torch.stack(self.errors).max())
        assert error==0., error
        return values.float().cpu().numpy(), error


def source_manifest():
    paths={Path(__file__).resolve(),*SOURCE_FILES,*map(Path,c.source_paths('sit_small'))}
    return {str(p):c.sha(p) for p in sorted(paths)}


def save_npz(path, **values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f,**values)
    tmp.replace(path)


def summarize(records):
    gain=records[...,0].astype(np.float64)
    p2=records[...,12].astype(np.float64)
    mean=gain.mean(0)
    within=float(np.square(gain-mean[None]).mean())
    between=float(np.square(mean-mean.mean()).mean())
    std=gain.std(0,ddof=1)
    weighted=(gain*p2).sum(0)/p2.sum(0).clip(1e-12)
    quantiles=np.quantile(gain,[0,.01,.05,.25,.5,.75,.95,.99,1.])
    result=dict(samples=N,stages_per_image=96,total_gain_records=int(gain.size),
        gain_quantiles=dict(zip(('min','p01','p05','p25','p50','p75','p95','p99','max'),map(float,quantiles))),
        mean_gain=float(gain.mean()),negative_fraction=float((gain<0).mean()),
        below_one_fraction=float((gain<1).mean()),small_apg_count=int(records[...,10].sum()),
        per_time_mean_min=float(mean.min()),per_time_mean_max=float(mean.max()),
        within_time_std_median=float(np.median(std)),within_time_std_max=float(std.max()),
        total_variance=float(gain.var()),within_time_variance=within,between_time_variance=between,
        fraction_variance_between_time=between/(between+within) if between+within else 0.,
        left_right_mean_abs_difference=float(np.abs(mean[:,0]-mean[:,1]).mean()),
        p2_weighted_vs_equal_image_gain_mean_abs_difference=float(np.abs(weighted-mean).mean()),
        apg_clean_cosine_abs_max=float(np.abs(records[...,5]).max()),
        actual_update_clean_cosine_abs_max=float(np.abs(records[...,9]).max()),
        absolute_norms='Whole 4x32x32 latent L2 norms; divide by 64 for RMS.',
        calibration_rule='Arithmetic mean over 100 images separately for each accepted step and Heun stage; no clipping, weighting or FID selection.',
        limitations=['One noise per class; between-image variation mixes class and noise/state effects.',
                    'Time means are measured on signed-parallel paths; time-only control follows its own states.',
                    'Equal-image mean gain does not exactly match p-norm-weighted guidance action.',
                    'No image-quality evaluation or statistical significance claim.'])
    return mean,std,weighted,result


def check():
    class FakeRuntime:
        name='sit_small'
        def __init__(self): self.labels=None;self.counts=dict(full=0,prefix=0)
        def context(self):return contextlib.nullcontext()
        def field(self,z,t,kind):
            assert kind=='full';self.counts['full']+=1
            label=self.labels.to(z.dtype).reshape(-1,1,1,1)
            return .14*torch.sin(z+.3*t)+.03*z+.003*label*torch.cos(z+.1*t)
    gen=torch.Generator().manual_seed(SEED)
    noise=torch.randn((4,4,8,8),generator=gen);labels=torch.tensor([1,25,52,91])
    rt=FakeRuntime();expected=blend.sample(rt,noise,labels,CONFIG)
    recorder=Recorder();rt=FakeRuntime()
    with recorder.capture(): actual=blend.sample(rt,noise,labels,CONFIG)
    values,error=recorder.arrays()
    assert torch.equal(actual['latents'],expected['latents'])
    assert actual['counts']==expected['counts']==dict(full=224,prefix=0)
    assert values.shape==(4,48,2,len(FIELDS)) and np.isfinite(values).all()
    result=dict(passed=True,cuda_used=False,exact_unchanged_trajectory=True,counts=actual['counts'],
                reconstruction_error=error,shape=list(values.shape),sources=source_manifest())
    ROOT.mkdir(parents=True,exist_ok=True);c.atomic(ROOT/'cpu_check.json',result)
    print(json.dumps(dict(passed=True,counts=actual['counts'],source_sha256=c.sha(__file__))),flush=True)


@torch.inference_mode()
def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    checked=c.read(ROOT/'cpu_check.json');assert checked['passed']
    for path,digest in checked['sources'].items():assert c.sha(path)==digest,path
    rng=np.random.default_rng(SEED)
    labels=np.arange(N,dtype=np.int64);rng.shuffle(labels)
    noise=rng.standard_normal((N,4,32,32),dtype=np.float32)
    parent=c.read(c.EXPS/'cfg_transport_search_20260913/blend_1k/request.json')
    assert c.array_sha(noise)!=parent['noise_sha256']
    request=dict(samples=N,seed=SEED,batch=BATCH,config=CONFIG,sources=source_manifest(),
                 assets=parent['assets'],noise_sha256=c.array_sha(noise),labels_sha256=c.array_sha(labels),
                 parent_request_sha256=c.sha(c.EXPS/'cfg_transport_search_20260913/blend_1k/request.json'),
                 uses_fid=False,calibration='100 classes, exactly one independent noise each; equal-image mean by accepted step and Heun stage.')
    target=ROOT/'request.json'
    if target.exists():assert c.read(target)==request,'Frozen probe request changed'
    else:c.atomic(target,request)
    save_npz(ROOT/'inputs.npz',noise=noise,labels=labels)
    rt=b.make_runtime();all_values=[];endpoints=[];times=[]
    # An unchanged first 8-input trajectory gives a real-model wrapper check.
    plain=blend.sample(rt,c.cuda(noise[:BATCH]),c.cuda(labels[:BATCH]),CONFIG)
    original_endpoint=plain['latents'].detach().clone()
    for start in range(0,N,BATCH):
        stop=min(start+BATCH,N);recorder=Recorder()
        torch.cuda.synchronize();begin=time.perf_counter()
        with recorder.capture():
            result=blend.sample(rt,c.cuda(noise[start:stop]),c.cuda(labels[start:stop]),CONFIG)
        values,error=recorder.arrays()
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        assert result['counts']==dict(full=224,prefix=0)
        if start==0:assert torch.equal(result['latents'],original_endpoint)
        all_values.append(values);endpoints.append(result['latents'].float().cpu().numpy());times.append(seconds)
        save_npz(ROOT/f'batch{start:04d}.npz',records=values,labels=labels[start:stop],
                 latents=endpoints[-1],seconds=seconds,full=224,prefix=0,reconstruction_error=error)
        print(json.dumps(dict(completed=stop,total=N,seconds=seconds)),flush=True)
    records=np.concatenate(all_values);assert records.shape==(N,48,2,len(FIELDS))
    mean,std,weighted,summary=summarize(records)
    time_values=np.array([[(k+stage)/64 for stage in (0,1)] for k in range(48)],dtype=np.float64)
    save_npz(ROOT/'calibration.npz',mean_gain=mean,gain=records[...,0],times=time_values,labels=labels,
             noise_sha256=np.asarray(request['noise_sha256']),request_sha256=np.asarray(c.sha(target)))
    save_npz(ROOT/'records.npz',records=records,fields=np.asarray(FIELDS),labels=labels,
             latents=np.concatenate(endpoints),std_gain=std,p2_weighted_gain=weighted)
    with (ROOT/'time_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','stage','time','mean_gain','std_gain','negative_fraction','mean_apg_norm','mean_ctrl_norm','p2_weighted_gain'])
        for k in range(48):
            for stage in (0,1):
                r=records[:,k,stage]
                writer.writerow([k,stage,time_values[k,stage],mean[k,stage],std[k,stage],
                                 float((r[:,0]<0).mean()),float(r[:,1].mean()),float(r[:,2].mean()),weighted[k,stage]])
    summary.update(complete=True,actual_model_wrapper_bitwise_exact=True,full_calls_per_output=224,
                   extra_model_calls_for_recording=0,parity_check_images=BATCH,
                   parity_check_additional_full_branch_image_evaluations=BATCH*224,
                   sampling_recording_seconds=sum(times),request_sha256=c.sha(target),
                   calibration_npz=str(ROOT/'calibration.npz'),calibration_sha256=c.sha(ROOT/'calibration.npz'),
                   records_sha256=c.sha(ROOT/'records.npz'))
    c.atomic(ROOT/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['check','run'])
    args=parser.parse_args()
    check() if args.action=='check' else run()
