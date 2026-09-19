"""Matched strength curves: native IG, fixed residual direction, and ADG."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

from experiments import small_sit_carrier_flow_20260909 as infrastructure
from experiments import small_sit_predictable_gap_20260909 as residual
from experiments.lifting_scale_sweep_20260909 import EXPS,WORK,array_sha,atomic,read,sha

ROOT=EXPS/'small_sit_guidance_tuning_20260910'
BANK_ROOT=infrastructure.BANK_ROOT
PROTOCOL=WORK/'docs/SMALL_SIT_GUIDANCE_TUNING_PROTOCOL_20260910_ZH.md'
FAMILIES=('ig','residual_norm','adg')
PEAKS=(.4,.55,.7,.85,1.,1.2)
ARMS=tuple(f'{family}_a{round(100*peak):03d}' for peak in PEAKS for family in FAMILIES)
SPECS={f'{family}_a{round(100*peak):03d}':dict(family=family,peak=peak)
       for peak in PEAKS for family in FAMILIES}
GRID,ALPHAS=residual.GRID,residual.ALPHAS
SAMPLES,BATCH,RANKS=1000,8,4
DIAGNOSTICS=('correction_to_native_norm','off_affine_energy','clean_norm_ratio',
             'angle_radians','cap_fraction','fallback_fraction')
COMMON_VERIFY=infrastructure.verify_request


def adg_field(z,strong,weak,t,alpha):
    """Paper ADG clean-space formula, with exact a=0 and explicit degeneracies.

    q=(m-proj_n(m))/sin(theta), angle=min(alpha*theta,pi/3),
    G=cos(angle)*m+sin(angle)*q. q is perpendicular to Weak, not Strong.
    """
    if alpha==0:return strong,None
    b=1-t
    m,n=z+b*strong,z+b*weak
    axes=tuple(range(1,z.ndim))
    sq=lambda x:x.square().sum(axes,keepdim=True)
    floor=float(np.prod(z.shape[1:]))*1e-24
    m2,n2=sq(m),sq(n)
    mn=(m*n).sum(axes,keepdim=True)
    p=m-mn/n2.clamp_min(floor)*n
    sine=(sq(p)/m2.clamp_min(floor)).sqrt()
    cosine=mn/(m2*n2).clamp_min(floor**2).sqrt()
    theta=torch.atan2(sine,cosine)
    angle=(alpha*theta).clamp_max(torch.pi/3)
    near=(b.abs()<1e-5)
    denominator=torch.where(near,torch.ones_like(b),b)
    correction=(-2*torch.sin(angle/2).square()*m+
                torch.sin(angle)/sine.clamp_min(1e-6)*p)/denominator
    d=strong-weak
    tangent=d-(m*d).sum(axes,keepdim=True)/m2.clamp_min(floor)*m
    correction=torch.where(near,alpha*tangent,correction)
    invalid=(m2<=floor)|(n2<=floor)|((sine<1e-6)&~near)
    value=torch.where(invalid,strong,strong+correction)
    details=torch.cat((angle,(alpha*theta>=torch.pi/3).to(z.dtype),invalid.to(z.dtype)),dim=-1).reshape(len(z),3)
    return value,details


def field(rt,z,t,alpha,family):
    s,w=rt.pair(z,t)
    if family=='residual_norm':
        d=s-w
        c=residual.cache_source.unpatchify(residual.project(rt.capture_weak.value,rt.projection))
        _,direction,_,fallback=residual.directions(d,c)
        value=s+alpha*direction
        special=torch.zeros((len(z),3),device=z.device,dtype=z.dtype)
        special[:,-1]=fallback.flatten(1).any(1).to(z.dtype)
    else:
        assert family=='adg'
        value,special=adg_field(z,s,w,t,alpha)
    axes=tuple(range(1,z.ndim));sq=lambda x:x.square().sum(axes)
    d,correction=s-w,value-s
    d2=sq(d).clamp_min(1e-24)
    coefficient=(d*correction).sum(axes)/d2
    off=correction-coefficient.reshape((-1,)+(1,)*(z.ndim-1))*d
    m=z+(1-t)*s
    stats=torch.stack(((sq(correction)/(alpha**2*d2)).sqrt(),
        sq(off)/sq(correction).clamp_min(1e-24),
        (sq(z+(1-t)*value)/sq(m).clamp_min(1e-24)).sqrt()),-1)
    return value,torch.cat((stats,special),-1)


@torch.inference_mode()
def sample(rt,noise,labels,arm,*,zero=False):
    spec=SPECS[arm]
    rt.labels=labels;z=noise.clone();grid=z.new_tensor(GRID)
    begin=rt.counts.copy()
    calls,geometry_calls,diagnostics=[],[],[]
    # Preserve exactly the already measured .6/.7 anchor arithmetic.
    profile=ALPHAS if spec['peak']==.7 else tuple(a*(spec['peak']/.7) for a in ALPHAS)
    for k,amount in enumerate(profile):
        alpha=0. if zero else amount
        before=rt.counts['full'];count=0
        acc=torch.zeros((len(z),len(DIAGNOSTICS)),device=z.device,dtype=torch.float64)
        def rhs(t,x):
            nonlocal count
            if alpha==0 or spec['family']=='ig':return rt.guided(x,t,alpha)
            v,diag=field(rt,x,t,alpha,spec['family'])
            acc.add_(diag.double());count+=1
            return v
        z=odeint(rhs,z,grid[k:k+2],method='dopri5',rtol=.001,atol=1e-6)[-1]
        if not torch.isfinite(z).all():raise FloatingPointError(f'{arm}: nonfinite block {k}')
        calls.append(rt.counts['full']-before);geometry_calls.append(count)
        diagnostics.append((acc/max(count,1)).cpu().numpy())
    assert rt.counts['prefix']==begin['prefix']
    return z,dict(full_calls=rt.counts['full']-begin['full'],prefix_calls=0,auxiliary_full_calls=0,
        geometry_calls=sum(geometry_calls),block_full_calls=np.asarray(calls),
        block_geometry_calls=np.asarray(geometry_calls),block_geometry_diagnostics=np.asarray(diagnostics))


def cpu_checks():
    rng=np.random.default_rng(202610002)
    errors=[]
    for _ in range(32):
        m,n,z=rng.normal(size=(3,7))
        r,s=np.linalg.norm(m),np.linalg.norm(n)
        cosine=m@n/(r*s);theta=np.arccos(np.clip(cosine,-1,1))
        q=(m-(m@n)/(n@n)*n)/np.sin(theta)
        for alpha in PEAKS:
            angle=min(alpha*theta,np.pi/3)
            expected=np.cos(angle)*m+np.sin(angle)*q
            for t in (0.,.4):
                b=1-t;strong=(m-z)/b;weak=(n-z)/b
                tensors=[torch.tensor(x[None],dtype=torch.float64) for x in (z,strong,weak)]
                v,_=adg_field(*tensors,torch.tensor(t,dtype=torch.float64),alpha)
                actual=z+b*v.numpy()[0]
                errors.append(float(np.abs(actual-expected).max()))
                assert abs(np.linalg.norm(actual)**2/(r*r)-(1+np.sin(2*angle)*np.sin(theta)))<1e-12
    assert max(errors)<1e-12
    z=torch.tensor([[.2,-.1]],dtype=torch.float64)
    s=torch.tensor([[1.,.3]],dtype=torch.float64)
    for t in (0.,1.):
        value,_=adg_field(z,s,s,torch.tensor(t),.7)
        torch.testing.assert_close(value,s,rtol=0,atol=0)
    v,_=adg_field(z,s,-s,torch.tensor(.2),0.)
    assert torch.equal(v,s)
    return dict(passed=True,direct_paper_formula_cases=len(errors),max_error=max(errors),
        norm_formula_passed=True,zero_scale_exact=True,equal_heads_exact=True)


@torch.inference_mode()
def preflight(rt,noise,labels,rank,request):
    rt.labels=labels;probes=[]
    for tv in (0.,.125,.375,.875):
        t=noise.new_tensor(tv);s,w=rt.pair(noise,t)
        assert torch.equal(s,rt.field(noise,t,'full'))
        assert torch.equal(w,rt.field(noise,t,'base'))
        v,diag=adg_field(noise,s,w,t,.7)
        vd,_=adg_field(noise.double(),s.double(),w.double(),t.double(),.7)
        error=float((v.double()-vd).abs().max());assert error<2e-5
        ratio=(noise+(1-t)*v).flatten(1).norm(dim=1)/(noise+(1-t)*s).flatten(1).norm(dim=1)
        assert float(ratio.min())>1-2e-5 and float(ratio.max())<2**.5+2e-5
        probes.append(dict(t=tv,float64_max_error=error,mean_angle=float(diag[:,0].mean()),
            fallback_count=float(diag[:,-1].sum()),norm_ratio_mean=float(ratio.mean())))
    zs=[sample(rt,noise,labels,f'{family}_a070',zero=True)[0] for family in FAMILIES]
    assert all(torch.equal(zs[0],v) for v in zs[1:])
    golden=[]
    for arm,previous in (('ig_a070',residual.OLD_ROOT/'ig_restarted'),
                         ('residual_norm_a070',residual.ROOT/'residual_norm')):
        value,_=sample(rt,noise,labels,arm)
        path=previous/f'rank{rank}/batch{rank*BATCH:04d}.npz'
        with np.load(path) as data:np.testing.assert_array_equal(value.cpu().numpy(),data['latents'])
        golden.append(dict(arm=arm,old_batch=str(path),old_batch_sha256=sha(path),exact=True))
    for path,h in rt.sources.items():assert request['sources'].get(path)==h,path
    return dict(passed=True,rank=rank,pair_prefix_exact=True,zero_three_exact=True,
        golden=golden,probes=probes,runtime_sources=rt.sources,metadata=rt.metadata,
        cuda_device=torch.cuda.get_device_name(),tf32=torch.backends.cuda.matmul.allow_tf32)


def prepare():
    residual.install_infrastructure()
    parent,parent_hash=infrastructure.verify_request();residual.verify_fit()
    assert read(residual.ROOT/'status.json')['phase']=='complete'
    checks=cpu_checks();assert not ROOT.exists();ROOT.mkdir(parents=True)
    sources=dict(parent['sources'])
    for path in (Path(__file__),PROTOCOL):sources[str(path)]=sha(path)
    refs={}
    for rank in range(RANKS):
        for previous in (residual.OLD_ROOT/'ig_restarted',residual.ROOT/'residual_norm'):
            path=previous/f'rank{rank}/batch{rank*BATCH:04d}.npz';refs[str(path)]=sha(path)
    snap=ROOT/'sources';snap.mkdir()
    for i,p in enumerate(sorted(sources)):(snap/f'{i:02d}_{Path(p).name}').write_bytes(Path(p).read_bytes())
    request=dict(parent,arms=ARMS,specs=SPECS,peaks=PEAKS,sources=sources,parent_request_sha256=parent_hash,
        preflight_reference_files=refs,cpu_checks=checks,diagnostic_names=DIAGNOSTICS,
        independent_confirmation=False,known_residual_confirmation_used_for_family_choice=True,
        adg=dict(reference='https://arxiv.org/html/2506.11039v1',angle_cap=float(np.pi/3),
            whole_image_geometry=True,author_extra_scale_epsilon_used=False,near_endpoint_limit=1e-5))
    request.pop('baseline',None)
    atomic(ROOT/'request.json',request);atomic(ROOT/'status.json',dict(phase='prepared',research_goal_achieved=False))
    print(json.dumps(dict(prepared=True,request_sha256=sha(ROOT/'request.json'),arms=ARMS,cpu_checks=checks)),flush=True)


def verify_request():
    request,h=COMMON_VERIFY()
    for p,digest in request['preflight_reference_files'].items():assert sha(p)==digest,p
    residual.verify_fit()
    return request,h


def install_infrastructure():
    infrastructure.ROOT,infrastructure.BANK_ROOT=ROOT,BANK_ROOT
    infrastructure.ARMS,infrastructure.SAMPLES=ARMS,SAMPLES
    infrastructure.Runtime=residual.make_runtime
    infrastructure.sample,infrastructure.preflight=sample,preflight
    infrastructure.verify_request=verify_request


def run():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request,h=verify_request();assert read(ROOT/'status.json')['phase']=='prepared'
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb')==request['inception_graph_sha256']
    workers,streams,results=[],[],[];begin=time.perf_counter()
    def interrupted(signum,frame):raise RuntimeError(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for rank in range(RANKS):
            stream=(ROOT/f'worker{rank}.log').open('w');streams.append(stream)
            workers.append(subprocess.Popen([sys.executable,'-u','-m',
                'experiments.small_sit_guidance_tuning_20260910','--rank',str(rank)],cwd=WORK,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT))
        status=dict(controller_pid=os.getpid(),worker_pids=[p.pid for p in workers],research_goal_achieved=False)
        atomic(ROOT/'status.json',dict(phase='preflight',**status))
        infrastructure.wait_files([ROOT/f'preflight_rank{r}.json' for r in range(RANKS)],workers)
        checks=[read(ROOT/f'preflight_rank{r}.json') for r in range(RANKS)]
        assert all(c['passed'] for c in checks)
        assert all(c['runtime_sources']==checks[0]['runtime_sources'] for c in checks)
        atomic(ROOT/'preflight_passed.json',dict(passed=True,checks=checks,request_sha256=h))
        for arm in ARMS:
            atomic(ROOT/'status.json',dict(phase='sampling',arm=arm,**status))
            infrastructure.wait_files([ROOT/arm/f'rank{r}/summary.json' for r in range(RANKS)],workers)
            atomic(ROOT/'status.json',dict(phase='evaluating',arm=arm,**status))
            result=infrastructure.evaluate(arm,request,h);result.update(SPECS[arm])
            atomic(ROOT/arm/'result.json',result);results.append(result);atomic(ROOT/'results.json',results)
            print(json.dumps(result),flush=True)
            atomic(ROOT/arm/'advance.json',dict(complete=True,request_sha256=h))
        codes=[p.wait() for p in workers];assert codes==[0]*RANKS,codes
        verify_request();atomic(ROOT/'status.json',dict(phase='complete',results=len(results),
            wall_seconds=time.perf_counter()-begin,numerical_failures=sum(not r['complete'] for r in results),
            worker_exit_codes=codes,**status))
    except BaseException as error:
        atomic(ROOT/'status.json',dict(phase='failed',error=repr(error),controller_pid=os.getpid(),
            worker_pids=[p.pid for p in workers],research_goal_achieved=False));raise
    finally:
        for p in workers:
            if p.poll() is None:p.terminate()
        for p in workers:p.wait()
        for stream in streams:stream.close()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-checks',action='store_true');parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run-prepared',action='store_true');parser.add_argument('--rank',type=int,choices=range(RANKS))
    args=parser.parse_args();assert sum((args.cpu_checks,args.prepare,args.run_prepared,args.rank is not None))==1
    if args.cpu_checks:print(json.dumps(cpu_checks()))
    elif args.prepare:prepare()
    else:
        install_infrastructure();run() if args.run_prepared else infrastructure.worker(args.rank)
