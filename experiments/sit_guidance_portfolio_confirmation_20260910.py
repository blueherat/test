"""Frozen portfolio winners and native controls on a fresh balanced 5K bank."""
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
from experiments.sit_guidance_portfolio_20260910 import runner as pilot
from experiments import small_sit_carrier_flow_20260909 as infrastructure
from experiments.lifting_scale_sweep_20260909 import EXPS,WORK,array_sha,atomic,read,sha
ROOT=EXPS/'sit_guidance_portfolio_confirmation_20260910'
BANK_ROOT=ROOT/'inputs'
PROTOCOL=WORK/'docs/SIT_GUIDANCE_PORTFOLIO_CONFIRMATION_PROTOCOL_20260910_ZH.md'
SAMPLES,BATCH,RANKS,SEED=5000,8,4,202610060
DIAGNOSTICS=pilot.DIAGNOSTICS
SPECS=pilot.SPECS
SELECTED=(
    "control_ig_heun64_05", "i40_ig_local_attention_g2_p1",
    "control_ig_dopri5_05", "control_cfg_c75_04",
    "i01_cfg_apg_momentum_g3_p1", "i38_cfg_condition_secant_g2_p0",
    "i03_cfg_channel_rescale_g2_p2",
)
ARMS=()
COMMON_VERIFY=infrastructure.verify_request
sample=pilot.sample


def prepare():
    pilot.install_infrastructure()
    parent,h=pilot.verify_request()
    assert read(pilot.ROOT/'status.json')['phase']=='complete'
    records=read(pilot.ROOT/'results.json')
    assert len(records)==len(pilot.ARMS) and all(r['complete'] for r in records)
    selected=[next(r for r in records if r['arm']==arm) for arm in SELECTED]
    for row in selected:
        candidates=[r for r in records if r['key']==row['key'] and r['solver']==row['solver']]
        assert row['fid']==min(r['fid'] for r in candidates),row['arm']
    assert not ROOT.exists();ROOT.mkdir(parents=True);BANK_ROOT.mkdir()
    generator=torch.Generator(device='cuda').manual_seed(SEED)
    noise=np.lib.format.open_memmap(BANK_ROOT/'noise.npy',mode='w+',dtype=np.float32,shape=(SAMPLES,4,32,32))
    for start in range(0,SAMPLES,BATCH):
        noise[start:start+BATCH]=torch.randn((BATCH,4,32,32),device='cuda',generator=generator).cpu().numpy()
    noise.flush()
    labels=np.random.default_rng(SEED+1).permutation(np.repeat(np.arange(100,dtype=np.int64),50))
    assert np.all(np.bincount(labels,minlength=100)==50)
    np.save(BANK_ROOT/'labels.npy',labels)
    files={name:sha(BANK_ROOT/name) for name in ('noise.npy','labels.npy')}
    sources=dict(parent['sources'])
    for path in (Path(__file__),PROTOCOL):sources[str(path)]=sha(path)
    snap=ROOT/'sources';snap.mkdir()
    for i,path in enumerate(sorted(sources)):(snap/f'{i:02d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    references={}
    for path in (pilot.ROOT/'request.json',pilot.ROOT/'results.json',pilot.BANK_ROOT/'noise.npy',pilot.BANK_ROOT/'labels.npy'):
        references[str(path)]=sha(path)
    for record in selected:
        for rank in range(RANKS):
            path=pilot.ROOT/record['arm']/f'rank{rank}/batch{rank*BATCH:04d}.npz'
            references[str(path)]=sha(path)
    request=dict(parent,arms=[r['arm'] for r in selected],configs=[SPECS[r['arm']] for r in selected],
        sources=sources,samples=SAMPLES,
        bank_root=str(BANK_ROOT),bank_files=files,preflight_reference_files=references,
        bank=dict(samples=SAMPLES,batch=BATCH,shape=[SAMPLES,4,32,32],noise_seed=SEED,label_seed=SEED+1,
            noise_sha256=array_sha(noise),label_sha256=array_sha(labels),classes_balanced=True,
            noise_file_sha256=files['noise.npy'],labels_file_sha256=files['labels.npy'],
            noise_generation='one continuous CUDA generator, sequential B8 draws'),
        selection=dict(rule='lowest completed 1K FID for native Heun IG, Dopri5 IG, native CFG, and candidate families 40/1/38/3',
            parent_request_sha256=h,records=selected),independent_confirmation=True,
        independent_training=False,independent_reference=False)
    atomic(ROOT/'request.json',request);atomic(ROOT/'status.json',dict(phase='prepared',research_goal_achieved=False))
    print(json.dumps(dict(prepared=True,arms=request['arms'],request_sha256=sha(ROOT/'request.json'),bank=request['bank'])),flush=True)


def verify_request():
    request,h=COMMON_VERIFY()
    for path,digest in request['preflight_reference_files'].items():assert sha(path)==digest,path
    pilot.assets.verify()
    return request,h


@torch.inference_mode()
def preflight(rt,noise,labels,rank,request):
    begin=rank*BATCH
    n=torch.from_numpy(np.load(pilot.BANK_ROOT/'noise.npy')[begin:begin+BATCH].copy()).cuda()
    y=torch.from_numpy(np.load(pilot.BANK_ROOT/'labels.npy')[begin:begin+BATCH].copy()).cuda()
    rt.labels=y
    for tv in (0.,.375):
        t=n.new_tensor(tv);strong,weak=rt.pair(n,t)
        assert torch.equal(strong,rt.field(n,t,'full'))
        assert torch.equal(weak,rt.field(n,t,'base'))
    for path,digest in rt.sources.items():assert request['sources'].get(path)==digest,path
    hooks=pilot.hook_counts(rt)
    original,zero_stats=sample(rt,n,y,'control_ig_heun64_05',zero=True)
    assert zero_stats['full_calls']==128 and zero_stats['prefix_calls']==0
    result=dict(passed=True,rank=rank,native_pair_prefix_exact=True,runtime_sources=rt.sources)
    selected=[]
    for arm in ARMS:
        value,_=sample(rt,n,y,arm)
        path=pilot.ROOT/arm/f'rank{rank}/batch{begin:04d}.npz'
        with np.load(path) as old:np.testing.assert_array_equal(value.cpu().numpy(),old['latents'])
        assert pilot.hook_counts(rt)==hooks
        selected.append(dict(arm=arm,path=str(path),sha256=sha(path),exact=True))
    repeated,_=sample(rt,n,y,'i40_ig_local_attention_g2_p1',zero=True)
    assert torch.equal(original,repeated)
    result['zero_scale_and_native_after_candidates_exact']=True
    rt.labels=labels
    s,w=rt.pair(noise,noise.new_tensor(.25))
    assert torch.isfinite(s).all() and torch.isfinite(w).all()
    result.update(selected_old_endpoints_exact=selected,fresh_noise_sha256=array_sha(noise.cpu().numpy()),
        fresh_labels_sha256=array_sha(labels.cpu().numpy()))
    return result


def install_infrastructure():
    global ARMS
    ARMS=tuple(read(ROOT/'request.json')['arms'])
    infrastructure.ROOT,infrastructure.BANK_ROOT=ROOT,BANK_ROOT
    infrastructure.ARMS,infrastructure.SAMPLES=ARMS,SAMPLES
    infrastructure.Runtime=lambda name:pilot.operators.make_runtime()
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
                'experiments.sit_guidance_portfolio_confirmation_20260910','--rank',str(rank)],cwd=WORK,
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
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run-prepared',action='store_true');parser.add_argument('--rank',type=int,choices=range(RANKS))
    args=parser.parse_args();assert sum((args.prepare,args.run_prepared,args.rank is not None))==1
    if args.prepare:prepare()
    else:
        install_infrastructure();run() if args.run_prepared else infrastructure.worker(args.rank)
