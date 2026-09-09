"""Finish the user-requested 0.78 arm after the four constant settings, reusing saved batches."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/home/zhoushunyu/data/eqvae/experiments')
ARMROOT=BASE/'ig_capacity_lifting_fourcard_20260908/a078_constant'
FOUR=BASE/'ig_capacity_lifting_constant_fourcard_20260908'
LOG=Path('experiments/results/terminal_defect_20260908/capacity_lifting_constant')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**kw):
    p=LOG/'a078_append_status.json';temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(time=time.time(),**kw),indent=2)+'\n');temp.replace(p)
    print(json.dumps(kw),flush=True)

def main():
    saved={str(p):sha(p) for p in ARMROOT.glob('rank*/batch*.npz')}
    for d in ARMROOT.glob('rank*'):
        req=json.loads((d/'request.json').read_text())
        for name,value in req['sources'].items():assert sha(Path(name))==value,name
        assert sha(Path('docs/IG_CAPACITY_LIFTING_20260908_ZH.md'))==req['protocol_sha256']
    (LOG/'a078_reuse_manifest.json').write_text(json.dumps(dict(saved_samples=len(saved)*4,batches=saved),indent=2)+'\n')
    status(phase='waiting_for_four_settings',reused_samples=len(saved)*4)
    while True:
        q=json.loads((LOG/'queue_status.json').read_text())
        if q['phase']=='failed':raise RuntimeError(f'Preceding queue failed: {q}')
        if q['phase']=='complete':break
        time.sleep(15)
    for arm in ['a050_constant','a065_constant','a085_constant','a090_constant']:
        r=json.loads((FOUR/arm/'result.json').read_text());assert r['complete']
    workers=[];streams=[]
    try:
        for rank in range(4):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(rank)
            f=(LOG/f'a078_resume_rank{rank}.log').open('a');streams.append(f)
            workers.append(subprocess.Popen([sys.executable,'-m','experiments.sample_raev2_capacity_lifting_shard','--rank',str(rank),'--arm','a078_constant'],env=env,stdout=f,stderr=subprocess.STDOUT))
        status(phase='sampling',pids=[p.pid for p in workers],reused_samples=len(saved)*4)
        while True:
            codes=[p.poll() for p in workers]
            if any(c is not None and c!=0 for c in codes):raise RuntimeError(f'0.78 workers failed: {codes}')
            if all(c==0 for c in codes):break
            time.sleep(5)
    finally:
        for p in workers:
            if p.poll() is None:p.terminate()
        for p in workers:p.wait()
        for f in streams:f.close()
    for name,value in saved.items():assert sha(Path(name))==value
    status(phase='evaluation',reused_samples=len(saved)*4,reuse_hashes_verified=True)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    subprocess.run([sys.executable,'-m','experiments.evaluate_raev2_capacity_lifting_shards','--arm','a078_constant'],env=env,check=True)
    rows=[]
    for a,root in [(.5,FOUR/'a050_constant'),(.65,FOUR/'a065_constant'),(.78,ARMROOT),(.85,FOUR/'a085_constant'),(.9,FOUR/'a090_constant')]:
        r=json.loads((root/'result.json').read_text())
        rows.append(dict(alpha=a,fid=r['method']['fid'],baseline_fid=r['reused_baseline']['fid'],
            relative_improvement=r['relative_fid_improvement'],sum_batch_gpu_seconds=r['sum_batch_gpu_seconds'],
            result_path=str(root/'result.json')))
    report=dict(complete=True,rows=rows,reused_a078_samples=len(saved)*4,reuse_hashes_verified=True,
                note='Five exploratory constant-lifting coefficients on the same paired1K; no independent success claim.')
    (LOG/'five_settings.json').write_text(json.dumps(report,indent=2)+'\n');status(phase='complete',report=report)

if __name__=='__main__':
    try:main()
    except BaseException as e:status(phase='failed',error=repr(e));raise
