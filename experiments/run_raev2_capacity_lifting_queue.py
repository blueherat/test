"""Run four candidates sequentially, four GPUs cooperate on each candidate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/ig_capacity_lifting_fourcard_20260908')
LOG=Path('experiments/results/terminal_defect_20260908/capacity_lifting')
ARMS=['a078_constant','a078_fade','a125_constant','a125_fade']

def status(**values):
    p=LOG/'queue_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(time=time.time(),**values),indent=2)+'\n');tmp.replace(p)
    print(json.dumps(values),flush=True)


def main():
    results=[]
    for arm in ARMS:
        if (ROOT/arm/'result.json').exists():
            results.append(json.loads((ROOT/arm/'result.json').read_text()));continue
        workers=[];logs=[]
        try:
            for rank in range(4):
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(rank)
                f=(LOG/f'{arm}_rank{rank}.log').open('a');logs.append(f)
                proc=subprocess.Popen([sys.executable,'-m','experiments.sample_raev2_capacity_lifting_shard',
                    '--rank',str(rank),'--arm',arm],env=env,stdout=f,stderr=subprocess.STDOUT)
                workers.append(proc)
            status(phase='sampling',arm=arm,pids=[p.pid for p in workers])
            while True:
                codes=[p.poll() for p in workers]
                if any(c is not None and c!=0 for c in codes):raise RuntimeError(f'{arm} workers failed: {codes}')
                if all(c==0 for c in codes):break
                time.sleep(5)
        finally:
            for p in workers:
                if p.poll() is None:p.terminate()
            for p in workers:p.wait()
            for f in logs:f.close()
        status(phase='evaluation',arm=arm)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
        subprocess.run([sys.executable,'-m','experiments.evaluate_raev2_capacity_lifting_shards','--arm',arm],env=env,check=True)
        result=json.loads((ROOT/arm/'result.json').read_text());results.append(result)
        status(phase='candidate_complete',arm=arm,fid=result['method']['fid'],reused_baseline_fid=result['reused_baseline']['fid'])
    report=dict(complete=True,rows=[dict(arm=r['method']['branch'],fid=r['method']['fid'],
                baseline_fid=r['reused_baseline']['fid'],relative_improvement=r['relative_fid_improvement'],
                sum_batch_gpu_seconds=r['sum_batch_gpu_seconds']) for r in results],
                note='Four exploratory settings on one paired 1K bank; best selection is not independent confirmation.')
    (LOG/'four_settings.json').write_text(json.dumps(report,indent=2)+'\n');status(phase='complete',report=report)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        status(phase='failed',error=repr(e));raise
