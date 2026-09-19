"""Direct same-state C=U hypothesis checks, independent of the FSG adapter."""
from __future__ import annotations
import argparse
import contextlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from experiments.sit_fsg_ctrl_hypothesis_20260911 import core,pipeline as p,study
from experiments import audit_sit_fsg_own_target_20260911 as barrier
from experiments.lifting_scale_sweep_20260909 import WORK,atomic,read,sha

ROOT=p.ROOT/'golden_path_direct'
PROTOCOL=WORK/'docs/SIT_GOLDEN_PATH_DIRECT_PROTOCOL_20260911_ZH.md'
MODULE='experiments.sit_golden_path_hypothesis_20260911'
ITERATIONS=(0,1,4,16,32,64)
TIMES=(8,24,40)
OBJECTIVES=('agreement','write','local')


def math_attention():
    return torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,
        enable_mem_efficient=False,enable_cudnn=False)


def differentiable_future(rt,x,k,y,kind,steps=12):
    grid=[k/64+(1-k/64)*j/steps for j in range(steps+1)]
    for t,end in zip(grid[:-1],grid[1:]):
        def block(z,t=t,end=end):
            h=end-t
            v=core.field(rt,z,t,y,kind=='null')
            w=core.field(rt,z+h*v,end,y,kind=='null')
            return z+(h/2)*(v+w)
        x=checkpoint(block,x,use_reentrant=False) if torch.is_grad_enabled() else block(x)
    return x


def residual(rt,x,k,y,objective,reference):
    if objective=='local':
        c,u=core.pair(rt,x,k/64,y);return c-u
    u=differentiable_future(rt,x,k,y,'null')
    c=differentiable_future(rt,x,k,y,'conditional') if objective=='agreement' else reference
    return u-c


def optimize(rt,x,k,y,objective,iterations=64):
    with torch.no_grad():
        reference=differentiable_future(rt,x,k,y,'conditional')
        initial=residual(rt,x,k,y,objective,reference).square().flatten(1).mean(1).clamp_min(1e-12)
    z=x.clone().requires_grad_(True)
    optimizer=torch.optim.LBFGS([z],lr=1.,max_iter=1,max_eval=8,tolerance_grad=1e-10,
        tolerance_change=1e-12,history_size=12,line_search_fn='strong_wolfe')
    records=[];states={0:x.clone()};closures=0
    def values():return residual(rt,z,k,y,objective,reference).square().flatten(1).mean(1)/initial
    def closure():
        nonlocal closures
        optimizer.zero_grad(set_to_none=True);value=values().sum()
        value.backward();closures+=1
        assert torch.isfinite(value) and torch.isfinite(z.grad).all()
        return value
    for iteration in range(1,iterations+1):
        previous=z.detach().clone()
        with torch.no_grad():before=values()
        optimizer.step(closure)
        with torch.no_grad():
            after=values();bad=after>before+1e-6
            if bad.any():
                z[bad]=previous[bad]
                optimizer.state.clear()
                after=values()
            assert (after<=before+1e-6).all(),(objective,iteration,before,after)
            records.append(dict(iteration=iteration,relative_mse=after.cpu().tolist(),
                shift_rms=core.rms(z-x).cpu().tolist(),closure_calls=closures,
                per_sample_rejections=int(bad.sum())))
            if iteration in ITERATIONS:states[iteration]=z.detach().clone()
    return states,records


@torch.no_grad()
def assess(rt,reader,x0,moved,k,y,reference_c,reference_u,reference_g,context,folder):
    c=core.future(rt,moved,k,y,'conditional')
    u=core.future(rt,moved,k,y,'null')
    g=core.future(rt,moved,k,y,'guided',amount=1.25)
    vc,vu=core.pair(rt,moved,k/64,y)
    c0,u0=core.pair(rt,x0,k/64,y)
    values=dict(shift_rms=core.rms(moved-x0),state_rms=core.rms(moved),
        agreement_rms=core.rms(u-c),write_rms=core.rms(u-reference_c),
        local_gap_rms=core.rms(vc-vu),
        agreement_ratio=study.relative(u-c,reference_u-reference_c),
        write_ratio=study.relative(u-reference_c,reference_u-reference_c),
        local_gap_ratio=study.relative(vc-vu,c0-u0),
        conditional_drift_rms=core.rms(c-reference_c),guided_drift_rms=core.rms(g-reference_g))
    values.update(study.bundle_readouts(reader,dict(u=u,c=c,g=g),y,folder,context['start']))
    return study.encode_records(len(moved),context,values)


def sources():
    return [Path(__file__).resolve(),PROTOCOL,Path(core.__file__).resolve(),
        Path(study.__file__).resolve(),Path(barrier.__file__).resolve()]


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    value=dict(sources={str(path):sha(path) for path in sources()},
        original_request_sha256=sha(p.ROOT/'study_request.json'),samples=16,batch=2,ranks=4,
        indices=list(range(16)),base='cfg_tuned',times=list(TIMES),objectives=list(OBJECTIVES),
        iterations=list(ITERATIONS),optimizer='full latent L-BFGS, no class-readout optimization',
        optimization_grid='12 uniform remaining Heun intervals',
        assessment_grid='original remaining Heun64 grid',
        precision='FP32, TF32 matmul off, differentiable math attention',created_unix=time.time())
    path=ROOT/'request.json'
    if path.exists():
        existing=read(path);value['created_unix']=existing['created_unix'];assert value==existing
    else:atomic(path,value)
    return value


def verify():
    value=read(ROOT/'request.json')
    for path,digest in value['sources'].items():assert sha(path)==digest,path
    assert value['original_request_sha256']==sha(p.ROOT/'study_request.json')
    return value


def checks(rt,x,y):
    records=[]
    for objective in OBJECTIVES:
        k=24
        with torch.no_grad():reference=differentiable_future(rt,x,k,y,'conditional')
        z=x.clone().requires_grad_(True)
        loss=residual(rt,z,k,y,objective,reference).square().sum()
        gradient=torch.autograd.grad(loss,z)[0]
        direction=core.unit(gradient);epsilon=.001
        with torch.no_grad():
            plus=residual(rt,x+epsilon*direction,k,y,objective,reference).square().sum()
            minus=residual(rt,x-epsilon*direction,k,y,objective,reference).square().sum()
            fd=(plus-minus)/(2*epsilon);ad=core.inner(gradient,direction).sum()
            relative=float(((fd-ad).abs()/ad.abs().clamp_min(1e-8)).item())
            assert relative<.02,(objective,relative)
            records.append(dict(objective=objective,scalar_loss_autograd_vs_finite_relative_error=relative))
    # The differentiable checkpoint path and the ordinary 12-step path must coincide.
    for kind in ('null','conditional'):
        z=x.clone().requires_grad_(True)
        observed=differentiable_future(rt,z,24,y,kind).detach()
        with torch.no_grad():expected=core.future(rt,x,24,y,kind,steps=12)
        torch.testing.assert_close(observed,expected,rtol=0,atol=0)
    return records


def worker(rank,pilot=False):
    request=verify();request_hash=sha(ROOT/'request.json')
    rt=core.old.make_runtime();rt.model.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False
    reader=study.Readouts(rt);labels=np.load(p.ROOT/'inputs/labels.npy')
    directory=ROOT/'pilot' if pilot else ROOT/'runs';directory.mkdir(exist_ok=True)
    with core.old.exact_matmul(),math_attention():
        for start in range(rank*2,2 if pilot else 16,8):
            source_start=(start//8)*8
            y=torch.from_numpy(labels[start:start+2].copy()).cuda()
            for k in ((24,) if pilot else TIMES):
                path=directory/f'{k:02d}_{start:03d}.json'
                if path.exists():
                    assert read(path)['request_sha256']==request_hash;continue
                begin=time.perf_counter();before=rt.counts.copy()
                x=study.load_state('cfg_tuned',source_start,k)[start-source_start:start-source_start+2]
                audit=checks(rt,x,y) if k==24 and start==rank*2 else []
                with torch.no_grad():
                    c0=core.future(rt,x,k,y,'conditional');u0=core.future(rt,x,k,y,'null')
                    g0=core.future(rt,x,k,y,'guided',amount=1.25)
                    vc,vu=core.pair(rt,x,k/64,y);gap=vc-vu
                    radius=(4/64)*1.25*core.norm(gap)
                rows=[];optimizations={};states_out={}
                for objective in (('agreement',) if pilot else OBJECTIVES):
                    states,trace=optimize(rt,x,k,y,objective,iterations=16 if pilot else 64)
                    optimizations[objective]=trace
                    for iteration,z in states.items():
                        key=f'{objective}_{iteration:02d}'
                        folder=ROOT/'images'/f'{k:02d}'/key
                        context=dict(start=start,k=k,objective=objective,iteration=iteration,variant='optimized')
                        rows+=assess(rt,reader,x,z,k,y,c0,u0,g0,context,folder)
                        states_out[key]=z.cpu().numpy()
                    if not pilot:
                        z=states[64]
                        comparisons=dict(cfg_equal=x+core.norm(z-x)*core.unit(gap),
                            radius_projected=x+core.cap(z-x,radius),
                            radius4_projected=x+core.cap(z-x,4*radius))
                        for variant,moved in comparisons.items():
                            context=dict(start=start,k=k,objective=objective,iteration=64,variant=variant)
                            rows+=assess(rt,reader,x,moved,k,y,c0,u0,g0,context,
                                ROOT/'images'/f'{k:02d}'/f'{objective}_{variant}')
                    print(dict(rank=rank,start=start,k=k,objective=objective,
                        relative_mse=trace[-1]['relative_mse'],seconds=time.perf_counter()-begin),flush=True)
                if not pilot:
                    with torch.no_grad():inverse=core.inverse_null(rt,c0,k,y)
                    rows+=assess(rt,reader,x,inverse,k,y,c0,u0,g0,
                        dict(start=start,k=k,objective='frozen_inverse',iteration=1,variant='oracle'),
                        ROOT/'images'/f'{k:02d}'/'frozen_inverse')
                np.savez(path.with_suffix('.npz'),x=x.cpu().numpy(),labels=y.cpu().numpy(),**states_out)
                verify();atomic(path,dict(complete=True,request_sha256=request_hash,rows=rows,
                    optimization_traces=optimizations,gradient_checks=audit,
                    total_full_calls=rt.counts['full']-before['full'],elapsed_seconds=time.perf_counter()-begin,
                    state_arrays_sha256=sha(path.with_suffix('.npz'))))
    atomic(directory/f'rank{rank}.json',dict(complete=True,pilot=pilot,request_sha256=request_hash))


def run(pilot=False,pause=False):
    prepare();barrier.ROOT=ROOT;pid=barrier.pause_at_sampling_barrier() if pause else None
    processes=[];streams=[]
    try:
        for rank in range(1 if pilot else 4):
            stream=(ROOT/f'{"pilot" if pilot else "run"}_rank{rank}.log').open('a');streams.append(stream)
            command=[sys.executable,'-u','-m',MODULE,'--worker',str(rank)]
            if pilot:command.append('--pilot')
            processes.append(subprocess.Popen(command,cwd=WORK,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                stdout=stream,stderr=subprocess.STDOUT))
        while any(x.poll() is None for x in processes):
            if any(x.poll() not in (None,0) for x in processes):raise RuntimeError('Golden-path worker failed; inspect logs')
            time.sleep(1)
        assert all(x.returncode==0 for x in processes)
        atomic(ROOT/('pilot_complete.json' if pilot else 'complete.json'),dict(complete=True,pilot=pilot,
            request_sha256=sha(ROOT/'request.json'),finished_unix=time.time()))
    finally:
        for child in processes:
            if child.poll() is None:child.terminate()
        for child in processes:child.wait(timeout=30)
        for stream in streams:stream.close()
        if pid is not None:
            os.kill(pid,signal.SIGCONT);atomic(ROOT/'resumed.json',dict(controller_pid=pid,resumed_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--pause-small',action='store_true');parser.add_argument('--worker',type=int,choices=range(4))
    args=parser.parse_args()
    if args.worker is None:run(args.pilot,args.pause_small)
    else:worker(args.worker,args.pilot)
