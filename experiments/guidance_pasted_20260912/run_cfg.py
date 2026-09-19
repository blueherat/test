import os
import subprocess
import time
from . import common as c,cfg


def main():
    cfg.prepare();jobs=[]
    for gpu,(model,rank) in enumerate([(m,r) for m in c.MODELS for r in range(2)]):
        log=(c.ROOT/model/cfg.STAGE/f'worker{rank}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        cmd=[c.PYTHON,'-u','-m','experiments.guidance_pasted_20260912.cfg','--model',model,'--rank',str(rank),'--parent-pid',str(os.getpid())]
        process=subprocess.Popen(cmd,env=env,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((model,rank,process,log))
    c.atomic(c.ROOT/'cfg_controller.json',dict(pid=os.getpid(),jobs=[dict(model=m,rank=r,pid=p.pid) for m,r,p,l in jobs]))
    seen=set()
    try:
        while True:
            for model in c.MODELS:
                for spec in cfg.configs():
                    arm=spec['arm'];key=(model,arm)
                    if key not in seen and all((c.ROOT/model/cfg.STAGE/arm/f'rank{r}/complete.json').exists() for r in range(2)):
                        row=c.collect(model,cfg.STAGE,arm);assert row
                        seen.add(key);print(model,arm,'aggregated',flush=True)
            codes=[p.poll() for _,_,p,_ in jobs]
            if any(code not in (None,0) for code in codes):raise RuntimeError(f'worker failed: {codes}')
            c.atomic(c.ROOT/'cfg_status.json',dict(phase='sampling' if any(code is None for code in codes) else 'samples_complete',
                completed=len(seen),total=12,workers=[dict(model=m,rank=r,pid=p.pid,returncode=p.poll()) for m,r,p,l in jobs]))
            if all(code is not None for code in codes):break
            time.sleep(3)
        assert len(seen)==12
    finally:
        for m,r,p,l in jobs:
            if p.poll() is None:p.terminate()
        for m,r,p,l in jobs:
            p.wait();l.close()


if __name__=='__main__':main()
