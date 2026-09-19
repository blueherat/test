import argparse
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import core as m
from experiments.guidance_pasted_20260912 import evaluate
from experiments.guidance_pasted_20260912.audit import REFS, fid_from_features

c = m.c


def evaluate_track(track, n):
    m.configure()
    rows = []
    for arm in m.ARMS:
        rows.append(evaluate.evaluate('sit_small', m.stage(track, n), arm))
        c.atomic(m.ROOT/'sit_small'/m.stage(track, n)/'results.json', rows)


def audit(n):
    with np.load(REFS['sit_small']) as d:
        mu, cov = d['mu'], d['sigma']
    records, decisions = [], {}
    for track in m.TRACKS:
        rp, request = m.verify(track, n)
        root = m.ROOT/'sit_small'/m.stage(track, n)
        rows = c.read(root/'results.json')
        assert {r['arm'] for r in rows} == set(m.ARMS)
        for row in rows:
            out = root/row['arm']
            with np.load(out/'activations.npz') as d:
                features = d['pool_3']
            assert len(features) == n
            reconstructed = fid_from_features(features, mu, cov)
            error = abs(reconstructed-row['fid'])
            assert error < .002, (track, row['arm'], error)
            assert c.sha(out/'samples.npz') == row['samples_sha256']
            records.append(dict(track=track, arm=row['arm'], fid=row['fid'], fid_recomputed=reconstructed,
                fid_absolute_error=error, samples=n, features_sha256=c.sha(out/'activations.npz')))
        by = {r['arm']:r for r in rows}
        candidate = by['positive']
        best = min((r for r in rows if r['arm'] != 'positive'), key=lambda r:r['fid'])
        decision = dict(passes_gate=candidate['fid'] <= best['fid']-2 and candidate['inception_score'] >= .9*by['ode']['inception_score'],
            candidate_fid=candidate['fid'], best_control=best['arm'], best_control_fid=best['fid'],
            difference=candidate['fid']-best['fid'], differences={k:candidate['fid']-r['fid'] for k,r in by.items() if k!='positive'},
            statistical_significance_established=False, mechanism_established=False)
        decisions[track] = decision
    c.atomic(m.ROOT/'audit.json',dict(passed=True, samples=n*len(m.ARMS)*len(m.TRACKS), records=records,
        independent_feature_extractor=False, max_fid_error=max(r['fid_absolute_error'] for r in records)))
    c.atomic(m.ROOT/'decision.json', decisions)
    print('Fixed decisions', decisions, flush=True)


@torch.inference_mode()
def benchmark(n):
    m.configure()
    rt, lut = m.runtime(), m.kernel.Kernel(m.ROOT/'kernel_lut.npz')
    timings = {}
    for track in m.TRACKS:
        noise, _, labels = c.bank('sit_small', m.stage(track, n))
        x,y=c.cuda(noise[:m.BATCH]),c.cuda(labels[:m.BATCH])
        times={arm:[] for arm in m.ARMS}
        for repeat in range(4):
            order=m.ARMS[repeat:]+m.ARMS[:repeat]
            for arm in order:
                torch.cuda.synchronize(); begin=time.perf_counter()
                z, counts=m.sample(rt,lut,x,y,track,arm,m.seed(track,n)+100000)
                rt.decode(z);torch.cuda.synchronize()
                elapsed=time.perf_counter()-begin
                if repeat:times[arm].append(elapsed)
        timings[track]=dict(seconds=times,medians={a:float(np.median(v)) for a,v in times.items()})
        c.atomic(m.ROOT/'benchmark.json',dict(complete=False,batch=m.BATCH,repeats=3,tracks=timings))
    c.atomic(m.ROOT/'benchmark.json',dict(complete=True,batch=m.BATCH,repeats=3,tracks=timings,
        includes_vae=True,provenance=rt.provenance,lookup_bytes=lut.tables.numel()*lut.tables.element_size()))


def launch(n):
    m.configure()
    assert c.read(m.ROOT/'implementation_checks.json')['passed']
    for track in m.TRACKS:
        m.prepare(track,n)
    jobs=[]
    for rank,gpu in enumerate((1,2,3)):
        log=(m.ROOT/f'worker{rank}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        cmd=[c.PYTHON,'-u','-m','experiments.transition_decontamination_20260913.core','--worker','--rank',str(rank),
             '--world','3','--parent',str(os.getpid()),'--n',str(n)]
        p=subprocess.Popen(cmd,cwd=c.WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log))
    c.atomic(m.ROOT/'workers.json',dict(parent=os.getpid(),children=[dict(pid=p.pid,phase='sampling') for p,_ in jobs]))
    c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='sampling',samples_per_arm=n,arms=12))
    try:
        for p,_ in jobs:
            if p.wait()!=0:raise RuntimeError(f'Sampling worker {p.pid} failed')
    finally:
        for p,log in jobs:
            if p.poll() is None:p.terminate()
            log.close()
    for track in m.TRACKS:
        for arm in m.ARMS:m.collect(track,arm,n)
    c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='evaluation',images=12*n))
    jobs=[]
    for track in m.TRACKS:
        log=(m.ROOT/f'evaluation_{track}.log').open('a')
        p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.transition_decontamination_20260913.run','--evaluate',track,'--n',str(n)],
            cwd=c.WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),
            stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log))
    for p,log in jobs:
        result=p.wait();log.close()
        if result:raise RuntimeError(f'Evaluation {p.pid} failed')
    audit(n)
    c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='benchmark',images=12*n))
    subprocess.run([c.PYTHON,'-u','-m','experiments.transition_decontamination_20260913.run','--benchmark','--n',str(n)],
        cwd=c.WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2'),check=True)
    c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='complete',images=12*n,arms=12,goal_complete=False))
    print('Completed all fixed image arms, evaluation, and timing',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',choices=m.TRACKS)
    p.add_argument('--benchmark',action='store_true');p.add_argument('--n',type=int,choices=(400,1000),default=400)
    a=p.parse_args()
    if a.evaluate:evaluate_track(a.evaluate,a.n)
    elif a.benchmark:benchmark(a.n)
    else:launch(a.n)
