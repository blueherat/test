"""Complete both requested 5K arms before reporting a comparison."""
import os
import subprocess
import time

from experiments.guidance_pasted_20260912 import common as c
from . import core as m


def env(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')


def evaluate():
    rows=[]
    for arm in m.ARMS:
        root=m.BASE/arm;summary=c.read(root/'summary.json')
        output=root/'metrics.json'
        if output.exists():
            rows.append(c.read(output));continue
        assert summary['complete'] and c.sha(root/'samples.npz')==summary['samples_sha256']
        command=[c.PYTHON,str(c.WORK/'experiments/evaluate_raev2_official_samples.py'),
            '--branch',arm+'='+str(root/'samples.npz'),'--output',str(root/'official.csv'),
            '--batch-size','64','--device','cuda','--feature-cache-dir',str(root/'features')]
        c.atomic(root/'evaluation_request.json',dict(command=command,gpu=2,samples_sha256=summary['samples_sha256'],
            same_evaluator_device_and_batch_for_both=True))
        with (root/'evaluation.log').open('a') as log:
            subprocess.run(command,cwd=c.WORK,env=env(2),stdout=log,stderr=subprocess.STDOUT,check=True)
        metric=c.read(root/'official.json')[0]
        row=dict(summary,fid=metric['fid'],inception_score=metric['inception_score'],metrics=metric)
        c.atomic(output,row);rows.append(row)
        c.atomic(m.BASE/'results.json',rows)
        print(arm,'5K FID',row['fid'],'IS',row['inception_score'],flush=True)
    c.atomic(m.BASE/'results.json',rows)
    return rows


def controller():
    m.prepare()
    if not (m.ROOT/'preflight.json').exists():
        with (m.ROOT/'preflight.log').open('a') as log:
            subprocess.run([c.PYTHON,'-u','-m','experiments.raev2_context_5k_20260914.core','--preflight'],
                cwd=c.WORK,env=env(1),stdout=log,stderr=subprocess.STDOUT,check=True)
    assert c.read(m.ROOT/'preflight.json')['passed']
    jobs=[]
    for rank in range(3):
        log=(m.ROOT/f'rank{rank}.log').open('a')
        p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.raev2_context_5k_20260914.core',
            '--rank',str(rank),'--world','3','--parent',str(os.getpid())],
            cwd=c.WORK,env=env(rank+1),stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log,rank))
    c.atomic(m.ROOT/'workers.json',dict(parent=os.getpid(),children=[dict(pid=p.pid,rank=r,gpu=r+1) for p,_,r in jobs]))
    try:
        while any(p.poll() is None for p,_,_ in jobs):
            rows=[dict(pid=p.pid,rank=r,returncode=p.poll()) for p,_,r in jobs]
            c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='sampling',jobs=rows,samples_per_arm=m.N))
            failures=[r for r in rows if r['returncode'] not in (None,0)]
            if failures:raise RuntimeError(failures)
            time.sleep(5)
        assert all(p.returncode==0 for p,_,_ in jobs)
    finally:
        for p,log,_ in jobs:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
            log.close()
    c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='collecting',quality_images=10000))
    for arm in m.ARMS:m.collect(arm)
    c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='evaluation',quality_images=10000))
    rows=evaluate()
    c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='benchmark',quality_images=10000))
    if not (m.ROOT/'benchmark.json').exists():
        with (m.ROOT/'benchmark.log').open('a') as log:
            subprocess.run([c.PYTHON,'-u','-m','experiments.raev2_context_5k_20260914.core','--benchmark'],
                cwd=c.WORK,env=env(1),stdout=log,stderr=subprocess.STDOUT,check=True)
    c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='complete',quality_images=10000,
        old_replay_images=8,additional_training=False,goal_complete=False))
    print('Both 5K configurations complete; FID delta MLP-native:',rows[1]['fid']-rows[0]['fid'],flush=True)


if __name__=='__main__':controller()
