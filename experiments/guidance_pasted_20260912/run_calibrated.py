import os
import subprocess
import time
from . import common as c,calibrate as cal
from .run_ig import aggregate


def main():
    for model in c.MODELS:
        path=c.ROOT/model/cal.DATA/'request.json'
        while not (c.ROOT/model/cal.RAW_DATA/'complete.json').exists():time.sleep(5)
        if not path.exists():cal.fit(model)
    c.atomic(c.ROOT/'calibrated_status.json',dict(phase='waiting_for_raw_experiments',pid=os.getpid()))
    while c.read(c.ROOT/'ig_status.json')['phase']!='samples_complete' or c.read(c.ROOT/'cfg_status.json')['phase']!='samples_complete':time.sleep(5)
    cal.configure();jobs=[];evaluators=[];worlds={}
    active=[m for m in c.MODELS if not (c.ROOT/m/cal.STAGE/'evaluation_complete.json').exists()
        and c.read(c.ROOT/m/cal.DATA/'request.json')['inverse_temperature']>0]
    try:
        for index,model in enumerate(c.MODELS):
            if (c.ROOT/model/cal.STAGE/'evaluation_complete.json').exists():
                print(model,'calibrated screen already complete; reuse',flush=True)
                continue
            request=c.read(c.ROOT/model/cal.DATA/'request.json')
            if request['inverse_temperature']==0:
                c.atomic(c.ROOT/model/cal.STAGE/'degenerate.json',dict(complete=True,all_methods_equal_native=True))
                continue
            world=4 if len(active)==1 else 2;worlds[model]=world
            for rank in range(world):
                log=(c.ROOT/model/cal.STAGE/f'worker{rank}.log').open('a')
                gpu=rank if world==4 else 2*index+rank
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                cmd=[c.PYTHON,'-u','-m','experiments.guidance_pasted_20260912.calibrate','--model',model,'--rank',str(rank),
                    '--world',str(world),'--parent-pid',str(os.getpid())]
                p=subprocess.Popen(cmd,env=env,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT);jobs.append((model,p,log))
            log=(c.ROOT/model/cal.STAGE/'evaluation_controller.log').open('a')
            p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.guidance_pasted_20260912.evaluate','--model',model,'--stage',cal.STAGE],
                cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT)
            evaluators.append((model,p,log))
        while any(p.poll() is None for m,p,l in jobs):
            for model in c.MODELS:aggregate(model,worlds.get(model,2))
            if any(p.poll() not in (None,0) for m,p,l in jobs):raise RuntimeError('Calibrated quality worker failed')
            c.atomic(c.ROOT/'calibrated_status.json',dict(phase='sampling',pid=os.getpid(),
                workers=[dict(model=m,pid=p.pid,returncode=p.poll()) for m,p,l in jobs]))
            time.sleep(3)
        for model in c.MODELS:aggregate(model,worlds.get(model,2))
        c.atomic(c.ROOT/'calibrated_status.json',dict(phase='evaluating',pid=os.getpid()))
        for model,p,log in evaluators:
            if p.wait()!=0:raise RuntimeError(f'{model} calibration evaluation failed')
        c.atomic(c.ROOT/'calibrated_status.json',dict(phase='complete',pid=os.getpid()))
    finally:
        for m,p,l in jobs+evaluators:
            if p.poll() is None:p.terminate()
        for m,p,l in jobs+evaluators:p.wait();l.close()


if __name__=='__main__':main()
