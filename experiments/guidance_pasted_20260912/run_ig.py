import os
import subprocess
import time
import numpy as np
from . import common as c,ig


def launch(gpu,model,source,rank=0,world=1):
    out=c.ROOT/model/(ig.DATA if source else ig.STAGE)
    log=(out/('source.log' if source else f'worker{rank}.log')).open('a')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    cmd=[c.PYTHON,'-u','-m','experiments.guidance_pasted_20260912.ig','--model',model,
        '--rank',str(rank),'--world',str(world),'--parent-pid',str(os.getpid())]
    if source:cmd.append('--source')
    p=subprocess.Popen(cmd,env=env,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT)
    return p,log


def aggregate(model,world):
    root=c.ROOT/model/ig.STAGE
    for spec in ig.CONFIGS:
        arm=spec['arm']
        if not (root/arm/'summary.json').exists() and all((root/arm/f'rank{r}/complete.json').exists() for r in range(world)):
            assert c.collect(model,ig.STAGE,arm)
            if arm=='native_base':
                values=[]
                for p in (root/arm).glob('rank*/batch*.npz'):
                    with np.load(p) as d:values.append(d['source_probabilities'])
                profile=np.concatenate(values).mean(0,dtype=np.float64).astype(np.float32)
                np.save(root/'time_profile.npy',profile)
                c.atomic(root/'time_profile.json',dict(source='native IG actual queries, both pair branches',
                    sha256=c.sha(root/'time_profile.npy'),samples=2*c.SAMPLES,selected_using_quality=False))
            print(model,arm,'aggregated',flush=True)


def main():
    ig.prepare();jobs=[]
    # Reuse the two GPUs freed by the SiT CFG screen. RAE CFG remains on GPUs 2/3.
    while True:
        status=c.read(c.ROOT/'cfg_status.json')
        sit=[w for w in status['workers'] if w['model']=='sit_small']
        if all(w['returncode']==0 for w in sit):break
        if any(w['returncode'] not in (None,0) for w in sit):raise RuntimeError('CFG SiT worker failed; investigate before reusing its GPU')
        c.atomic(c.ROOT/'ig_status.json',dict(phase='waiting_for_sit_cfg_gpus',pid=os.getpid()))
        time.sleep(3)
    try:
        sources={m:launch(i,m,True) for i,m in enumerate(c.MODELS)};jobs.extend(sources.values())
        c.atomic(c.ROOT/'ig_status.json',dict(phase='source_data',pid=os.getpid(),jobs=[dict(model=m,pid=p.pid) for m,(p,l) in sources.items()]))
        while sources['sit_small'][0].poll() is None:
            if sources['raev2'][0].poll() not in (None,0):raise RuntimeError('RAE source failed')
            time.sleep(3)
        if sources['sit_small'][0].returncode!=0:raise RuntimeError('SiT source failed')
        sit_quality=launch(0,'sit_small',False);jobs.append(sit_quality)
        c.atomic(c.ROOT/'ig_status.json',dict(phase='sit_quality_and_rae_sources',pid=os.getpid(),sit_pid=sit_quality[0].pid,rae_source_pid=sources['raev2'][0].pid))
        while sit_quality[0].poll() is None or sources['raev2'][0].poll() is None:
            aggregate('sit_small',1)
            if sit_quality[0].poll() not in (None,0) or sources['raev2'][0].poll() not in (None,0):raise RuntimeError('IG source/quality worker failed')
            time.sleep(3)
        aggregate('sit_small',1)
        rae=[launch(r,'raev2',False,r,2) for r in range(2)];jobs.extend(rae)
        c.atomic(c.ROOT/'ig_status.json',dict(phase='rae_quality',pid=os.getpid(),workers=[p.pid for p,l in rae]))
        while any(p.poll() is None for p,l in rae):
            aggregate('raev2',2)
            if any(p.poll() not in (None,0) for p,l in rae):raise RuntimeError('RAE quality worker failed')
            time.sleep(3)
        aggregate('raev2',2)
        c.atomic(c.ROOT/'ig_status.json',dict(phase='samples_complete',pid=os.getpid()))
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate()
        for p,l in jobs:p.wait();l.close()


if __name__=='__main__':main()
