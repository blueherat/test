"""Paired clean-prediction rotations and multiplicative radial guidance."""
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
from experiments.lifting_scale_sweep_20260909 import EXPS, WORK, Runtime, array_sha, atomic, read, sha

ROOT = EXPS / 'small_sit_nonlinear_prediction_20260909'
OLD_ROOT = infrastructure.ROOT
BANK_ROOT = infrastructure.BANK_ROOT
PROTOCOL = WORK / 'docs/SMALL_SIT_NONLINEAR_PREDICTION_PROTOCOL_20260909_ZH.md'
ARMS = ('ig_restarted', 'polar_exp', 'sphere_exp', 'sphere_retraction')
GRID, ALPHAS = infrastructure.GRID, infrastructure.ALPHAS
SAMPLES, BATCH, RANKS = 1000, 8, 4
DIAGNOSTICS = ('radial_log_gain', 'absolute_rotation_radians', 'clean_norm_ratio',
    'correction_to_linear_gap_norm', 'off_affine_line_energy_fraction',
    'difference_from_ig_to_linear_gap_norm', 'fallback_fraction', 'off_line_fraction')
COMMON_VERIFY = infrastructure.verify_request


def exprel(x):
    """expm1(x)/x with its continuous value and stable small-x polynomial."""
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1+x*(.5+x*(1/6+x/24)), torch.expm1(x)/safe)


def nonlinear_field(z, strong, weak, t, alpha, arm, *, diagnostics=False):
    """Closed-form finite map in clean coordinates, returned as native velocity.

    This model uses z_t=(1-t)*noise+t*data, m=z+(1-t)*S.
    p=<m,S-W>/||m||^2, Q=(S-W)-p*m, k=||Q||/||m||.
    The polar endpoint is exp(a*b*p)*(cos(a*b*k)*m+a*b*sinc(a*b*k)*Q).
    Entire-function quotients avoid cancellation as b=1-t approaches zero.
    """
    assert arm in ARMS and arm != 'ig_restarted'
    if alpha == 0:
        return strong, None
    b = 1-t
    m, d = z+b*strong, strong-weak
    axes = tuple(range(1, z.ndim))
    sq = lambda x: x.square().sum(axes, keepdim=True)
    r2 = sq(m)
    floor = float(np.prod(z.shape[1:]))*1e-24
    invalid = r2 <= floor
    safe_r2 = r2.clamp_min(floor)
    p = (m*d).sum(axes, keepdim=True)/safe_r2
    q = d-p*m
    k2 = sq(q)/safe_r2
    k = k2.sqrt()
    phi = alpha*b*k
    eta = alpha*b*p if arm == 'polar_exp' else torch.zeros_like(p)
    if arm == 'sphere_retraction':
        h = m+(alpha*b)*d
        h2 = sq(h)
        invalid = invalid | (h2 <= floor)
        ratio = (h2.clamp_min(floor)/safe_r2).sqrt()
        coefficient = -(2*alpha*p+alpha**2*b*sq(d)/safe_r2)/(ratio*(1+ratio))
        value = strong+coefficient*m+(alpha/ratio)*d
        phi = torch.atan2(alpha*b*k,1+alpha*b*p)
    else:
        sinc = torch.sinc(phi/torch.pi)
        coefficient = -.5*alpha**2*b*k2*torch.sinc(phi/(2*torch.pi)).square()
        if arm == 'polar_exp':
            coefficient = coefficient+alpha*p*exprel(eta)*torch.cos(phi)
        value = strong+coefficient*m+torch.exp(eta)*alpha*sinc*q
    # An undefined polar origin (or zero retraction input) returns Strong.
    value = torch.where(invalid, strong, value)
    if not diagnostics:
        return value, None
    correction = value-strong
    linear = alpha*d
    corr2, lin2 = sq(correction), sq(linear)
    projection = (correction*d).sum(axes, keepdim=True)/sq(d).clamp_min(floor)*d
    off = sq(correction-projection)/corr2.clamp_min(floor)
    actual_ratio = (sq(z+b*value)/safe_r2).sqrt()
    stats = torch.cat((eta, phi.abs(), actual_ratio,
        (corr2/lin2.clamp_min(floor)).sqrt(), off,
        (sq(correction-linear)/lin2.clamp_min(floor)).sqrt(),
        invalid.to(z.dtype), (off > 1e-6).to(z.dtype)), dim=-1).reshape(len(z),len(DIAGNOSTICS))
    return value, stats


@torch.inference_mode()
def sample(rt, noise, labels, arm, *, zero=False):
    assert arm in ARMS
    rt.labels = labels
    z = noise.clone()
    grid = z.new_tensor(GRID)
    before = rt.counts.copy()
    full_calls, geometry_calls, geometry_mean = [], [], []
    for k, scheduled in enumerate(ALPHAS):
        alpha = 0. if zero else scheduled
        nfe = rt.counts['full']
        count = 0
        accum = torch.zeros((len(z),len(DIAGNOSTICS)),device=z.device,dtype=torch.float64)
        def rhs(t,x):
            nonlocal count
            if arm == 'ig_restarted' or alpha == 0:
                return rt.guided(x,t,alpha)
            s,w = rt.pair(x,t)
            v,stats = nonlinear_field(x,s,w,t,alpha,arm,diagnostics=True)
            accum.add_(stats.double())
            count += 1
            return v
        z = odeint(rhs,z,grid[k:k+2],method='dopri5',rtol=.001,atol=1e-6)[-1]
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'{arm}: nonfinite state at block {k}')
        full_calls.append(rt.counts['full']-nfe)
        geometry_calls.append(count)
        geometry_mean.append((accum/max(count,1)).cpu().numpy())
    assert rt.counts['prefix'] == before['prefix']
    return z,dict(full_calls=rt.counts['full']-before['full'],prefix_calls=0,auxiliary_full_calls=0,
        geometry_calls=sum(geometry_calls),block_full_calls=np.asarray(full_calls),
        block_geometry_calls=np.asarray(geometry_calls),block_geometry_diagnostics=np.asarray(geometry_mean))


def cpu_checks():
    """Independent dense matrix exponentials check the finite maps in FP64."""
    from scipy.linalg import expm
    rng = np.random.default_rng(202609991)
    errors, norm_errors = [], []
    for dim in (2,7):
        for _ in range(12):
            m,d,z = rng.normal(size=(3,dim))
            r2 = m@m
            rho = m@d/r2
            q = d-rho*m
            skew = (np.outer(q,m)-np.outer(m,q))/r2
            for t in (0.,.25,.5,.999):
                b = 1-t
                s = (m-z)/b
                w = s-d/b
                tensors = [torch.tensor(x[None],dtype=torch.float64) for x in (z,s,w)]
                for alpha in (0.,.35,.7,1.4):
                    for arm in ARMS[1:]:
                        v,_ = nonlinear_field(*tensors,t,alpha,arm)
                        if arm == 'polar_exp':
                            expected = expm(alpha*(rho*np.eye(dim)+skew))@m
                            radius = np.linalg.norm(m)*np.exp(alpha*rho)
                        elif arm == 'sphere_exp':
                            expected = expm(alpha*skew)@m
                            radius = np.linalg.norm(m)
                        else:
                            h = m+alpha*d
                            expected = np.linalg.norm(m)*h/np.linalg.norm(h)
                            radius = np.linalg.norm(m)
                        actual = z+b*v.numpy()[0]
                        errors.append(float(np.max(np.abs(actual-expected))))
                        norm_errors.append(float(abs(np.linalg.norm(actual)-radius)))
    assert max(errors) < 1e-10 and max(norm_errors) < 1e-10
    z = torch.tensor([[.4,-.2]],dtype=torch.float64)
    m = torch.tensor([[1.,0.]],dtype=torch.float64)
    d = torch.tensor([[.3,.7]],dtype=torch.float64)
    s,w = m-z,m-z-d
    value,diag = nonlinear_field(z,s,w,0.,.7,'polar_exp',diagnostics=True)
    assert float(diag[0,4]) > .001
    a = 1e-5
    vp,_ = nonlinear_field(z,s,w,0.,a,'polar_exp')
    vm,_ = nonlinear_field(z,s,w,0.,-a,'polar_exp')
    derivative_error = float(((vp-vm)/(2*a)-d).abs().max())
    assert derivative_error < 1e-9
    special = []
    for arm in ARMS[1:]:
        for name,ss,ww in (('equal',s,s),('zero_clean',-z,-z-d),
                           ('radial',s,s-torch.tensor([[.2,0.]],dtype=torch.float64))):
            value,_ = nonlinear_field(z,ss,ww,0.,.7,arm)
            assert torch.isfinite(value).all()
            if name in ('equal','zero_clean'):
                torch.testing.assert_close(value,ss,rtol=0,atol=0)
            special.append(dict(arm=arm,case=name,finite=True))
        v,_ = nonlinear_field(z,s,w,1.,.7,arm)
        assert torch.isfinite(v).all()
        zero,_ = nonlinear_field(z,s,w,.2,0.,arm)
        assert torch.equal(zero,s)
    retract_zero,_ = nonlinear_field(z,s,s+m/.5,0.,.5,'sphere_retraction')
    torch.testing.assert_close(retract_zero,s,rtol=0,atol=0)
    return dict(passed=True,matrix_exponential_cases=len(errors),max_clean_error=max(errors),
        max_norm_error=max(norm_errors),first_derivative_error=derivative_error,
        off_affine_line_energy_fraction=float(diag[0,4]),special_cases=special,
        zero_alpha_exact=True,time_endpoint_finite=True,zero_retraction_returns_strong=True)


@torch.inference_mode()
def preflight(rt,noise,labels,rank,request):
    rt.labels = labels
    probes=[]
    for time_value in (0.,.125,.25,.5,.875):
        t=noise.new_tensor(time_value)
        s,w=rt.pair(noise,t)
        torch.testing.assert_close(s,rt.field(noise,t,'full'),rtol=0,atol=0)
        torch.testing.assert_close(w,rt.field(noise,t,'base'),rtol=0,atol=0)
        for arm in ARMS[1:]:
            v,diag=nonlinear_field(noise,s,w,t,.7,arm,diagnostics=True)
            vd,_=nonlinear_field(noise.double(),s.double(),w.double(),t.double(),.7,arm)
            error=float((v.double()-vd).abs().max())
            assert error < 2e-5
            desired=torch.exp(diag[:,0]) if arm=='polar_exp' else torch.ones_like(diag[:,0])
            norm_error=float((diag[:,2]-desired).abs().max())
            assert norm_error < 2e-5 and torch.isfinite(v).all()
            probes.append(dict(t=time_value,arm=arm,float64_max_error=error,norm_error=norm_error,
                mean_diagnostics=diag.double().mean(0).tolist()))
    zeros=[sample(rt,noise,labels,arm,zero=True)[0] for arm in ARMS]
    assert all(torch.equal(zeros[0],v) for v in zeros[1:])
    ordinary,_=sample(rt,noise,labels,'ig_restarted')
    path=OLD_ROOT/'ig_restarted'/f'rank{rank}/batch{rank*BATCH:04d}.npz'
    with np.load(path) as old:
        np.testing.assert_array_equal(ordinary.cpu().numpy(),old['latents'])
    for p,h in rt.sources.items():
        assert request['sources'].get(p)==h,p
    return dict(passed=True,rank=rank,pair_prefix_exact=True,zero_all_four_exact=True,
        ordinary_ig_latents_exact=True,old_batch=str(path),old_batch_sha256=sha(path),
        probes=probes,runtime_sources=rt.sources,metadata=rt.metadata,
        cuda_device=torch.cuda.get_device_name(),tf32=torch.backends.cuda.matmul.allow_tf32)


def prepare():
    parent,parent_hash=COMMON_VERIFY()
    assert read(OLD_ROOT/'status.json')['phase']=='complete'
    checks=cpu_checks()
    assert not ROOT.exists()
    ROOT.mkdir(parents=True)
    sources=dict(parent['sources'])
    for path in (Path(__file__),PROTOCOL):
        sources[str(path)]=sha(path)
    references={}
    for rank in range(RANKS):
        path=OLD_ROOT/'ig_restarted'/f'rank{rank}/batch{rank*BATCH:04d}.npz'
        references[str(path)]=sha(path)
    snap=ROOT/'sources';snap.mkdir()
    for i,p in enumerate(sorted(sources)):
        (snap/f'{i:02d}_{Path(p).name}').write_bytes(Path(p).read_bytes())
    request=dict(parent,arms=ARMS,sources=sources,parent_request_sha256=parent_hash,
        backbone_evaluations_per_rhs=1,learned_parameters=0,geometry_diagnostics=DIAGNOSTICS,
        preflight_reference_files=references,cpu_checks=checks,
        old_baseline_for_parity=read(OLD_ROOT/'ig_restarted/result.json'),
        inception_graph_sha256=sha('/data/shared/adm_refs/classify_image_graph_def.pb'))
    request.pop('heun_steps')
    atomic(ROOT/'request.json',request)
    atomic(ROOT/'status.json',dict(phase='prepared',research_goal_achieved=False))
    print(json.dumps(dict(prepared=True,request_sha256=sha(ROOT/'request.json'),cpu_checks=checks)),flush=True)


def verify_request():
    request,request_hash=COMMON_VERIFY()
    for p,h in request['preflight_reference_files'].items():
        assert sha(p)==h,p
    return request,request_hash


def install_infrastructure():
    infrastructure.ROOT,infrastructure.BANK_ROOT=ROOT,BANK_ROOT
    infrastructure.ARMS,infrastructure.SAMPLES=ARMS,SAMPLES
    infrastructure.Runtime,infrastructure.sample,infrastructure.preflight=Runtime,sample,preflight
    infrastructure.verify_request=verify_request


def run():
    lock=(ROOT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request,request_hash=verify_request()
    assert read(ROOT/'status.json')['phase']=='prepared'
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb')==request['inception_graph_sha256']
    workers,streams,results=[],[],[]
    begin=time.perf_counter()
    def interrupted(signum,frame):
        raise RuntimeError(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for rank in range(RANKS):
            stream=(ROOT/f'worker{rank}.log').open('w');streams.append(stream)
            workers.append(subprocess.Popen([sys.executable,'-u','-m',
                'experiments.small_sit_nonlinear_prediction_20260909','--rank',str(rank)],cwd=WORK,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT))
        status=dict(controller_pid=os.getpid(),worker_pids=[p.pid for p in workers],research_goal_achieved=False)
        atomic(ROOT/'status.json',dict(phase='preflight',**status))
        infrastructure.wait_files([ROOT/f'preflight_rank{r}.json' for r in range(RANKS)],workers)
        checks=[read(ROOT/f'preflight_rank{r}.json') for r in range(RANKS)]
        assert all(c['passed'] for c in checks)
        assert all(c['runtime_sources']==checks[0]['runtime_sources'] for c in checks)
        atomic(ROOT/'preflight_passed.json',dict(passed=True,checks=checks,request_sha256=request_hash))
        for arm in ARMS:
            atomic(ROOT/'status.json',dict(phase='sampling',arm=arm,**status))
            infrastructure.wait_files([ROOT/arm/f'rank{r}/summary.json' for r in range(RANKS)],workers)
            atomic(ROOT/'status.json',dict(phase='evaluating',arm=arm,**status))
            result=infrastructure.evaluate(arm,request,request_hash)
            results.append(result);atomic(ROOT/'results.json',results)
            print(json.dumps(result),flush=True)
            atomic(ROOT/arm/'advance.json',dict(complete=True,request_sha256=request_hash))
        codes=[p.wait() for p in workers]
        assert codes==[0]*RANKS,codes
        verify_request()
        atomic(ROOT/'status.json',dict(phase='complete',results=len(results),
            wall_seconds=time.perf_counter()-begin,numerical_failures=sum(not r['complete'] for r in results),
            worker_exit_codes=codes,**status))
    except BaseException as error:
        atomic(ROOT/'status.json',dict(phase='failed',error=repr(error),controller_pid=os.getpid(),
            worker_pids=[p.pid for p in workers],research_goal_achieved=False))
        raise
    finally:
        for p in workers:
            if p.poll() is None:p.terminate()
        for p in workers:p.wait()
        for stream in streams:stream.close()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-checks',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run-prepared',action='store_true')
    parser.add_argument('--rank',type=int,choices=range(RANKS))
    args=parser.parse_args()
    assert sum((args.cpu_checks,args.prepare,args.run_prepared,args.rank is not None))==1
    if args.cpu_checks:print(json.dumps(cpu_checks()))
    elif args.prepare:prepare()
    else:
        install_infrastructure()
        run() if args.run_prepared else infrastructure.worker(args.rank)
