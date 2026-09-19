"""Run the requested paired resampling and a gated independent 1K confirmation."""
import argparse
import os
import subprocess
import time

from experiments.guidance_pasted_20260912 import common as c,evaluate
from . import sample as m


def env(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')


def watch(stage):
    m.configure()
    request = c.read(m.ROOT/'raev2'/stage/'request.json')
    done = {}
    while len(done) < len(request['arms']):
        for arm in request['arms']:
            if arm in done or m.collect(stage,arm) is None:
                continue
            done[arm] = evaluate.evaluate('raev2',stage,arm)
            c.atomic(m.ROOT/'raev2'/stage/'results.json',[done[a] for a in request['arms'] if a in done])
        if len(done) < len(request['arms']):
            time.sleep(5)


def run_stage(stage):
    jobs = []
    for rank in range(3):
        log = (m.ROOT/f'{stage}_rank{rank}.log').open('a')
        p = subprocess.Popen([c.PYTHON,'-u','-m','experiments.raev2_context_20k_20260914.sample',
            '--stage',stage,'--rank',str(rank),'--world','3','--parent',str(os.getpid())],
            cwd=c.WORK,env=env(rank+1),stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log,'sample',rank))
    log = (m.ROOT/f'{stage}_evaluation.log').open('a')
    p = subprocess.Popen([c.PYTHON,'-u','-m','experiments.raev2_context_20k_20260914.run','--evaluate',stage],
        cwd=c.WORK,env=env(''),stdout=log,stderr=subprocess.STDOUT)
    jobs.append((p,log,'evaluate',None))
    c.atomic(m.ROOT/f'{stage}_workers.json',dict(parent=os.getpid(),children=[dict(pid=p.pid,kind=k,rank=r) for p,_,k,r in jobs]))
    try:
        while any(p.poll() is None for p,_,_,_ in jobs):
            rows = [dict(pid=p.pid,kind=k,rank=r,returncode=p.poll()) for p,_,k,r in jobs]
            c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='sampling_and_evaluation',stage=stage,jobs=rows))
            failed = [r for r in rows if r['returncode'] not in (None,0)]
            if failed:
                raise RuntimeError(failed)
            time.sleep(5)
        assert all(p.returncode == 0 for p,_,_,_ in jobs)
    finally:
        for p,log,_,_ in jobs:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=30)
            log.close()


def decide_screen():
    new = c.read(m.ROOT/'raev2'/m.SCREEN/'results.json')
    old = {a:c.read(m.OLD_SCREEN/a/'metrics.json') for a in m.CONTROL_ARMS}
    best = min(v['fid'] for v in old.values())
    decisions = []
    for row in new:
        differences = {a:row['fid']-r['fid'] for a,r in old.items()}
        decisions.append(dict(arm=row['arm'],fid=row['fid'],differences=differences,
            inception_ratio=row['inception_score']/old['native_base']['inception_score'],
            passes_gate=row['fid'] <= best-2 and row['inception_score'] >= .9*old['native_base']['inception_score']))
    chosen = next((a for a in m.SCREEN_ARMS if any(r['arm']==a and r['passes_gate'] for r in decisions)),None)
    record = dict(decisions=decisions,chosen=chosen,rule='At least 2 FID better than all four retained controls; IS at least 90% of native base.')
    c.atomic(m.ROOT/'screen_decision.json',record)
    print('400-image decision',record,flush=True)
    return chosen


def controller():
    m.prepare(m.SCREEN)
    preflight = m.ROOT/'preflight.json'
    if not preflight.exists():
        with (m.ROOT/'preflight.log').open('a') as log:
            subprocess.run([c.PYTHON,'-u','-m','experiments.raev2_context_20k_20260914.sample','--preflight'],
                cwd=c.WORK,env=env(1),stdout=log,stderr=subprocess.STDOUT,check=True)
    assert c.read(preflight)['passed']
    run_stage(m.SCREEN)
    chosen = decide_screen()
    if chosen:
        m.prepare(m.CONFIRM,chosen)
        run_stage(m.CONFIRM)
    else:
        (m.ROOT/'STOP_AFTER_SCREEN').write_text('Neither fixed 20K strength passed the paired 400-image gate; no 1K confirmation or strength search.\n')
    c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='complete',training_steps=20000,
        quality_images=800+(4000 if chosen else 0),old_replay_images=8,confirmed=bool(chosen),goal_complete=False))
    print('Requested 20K training and sampling complete; independent 1K run:',bool(chosen),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate',choices=(m.SCREEN,m.CONFIRM))
    args = parser.parse_args()
    if args.evaluate:
        watch(args.evaluate)
    else:
        controller()
