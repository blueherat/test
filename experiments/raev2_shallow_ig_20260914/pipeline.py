"""Finish head search with GPUs joining when the incumbent search releases them."""
import concurrent.futures,fcntl,os,subprocess,sys,time
from pathlib import Path
from experiments.raev2_shallow_ig_20260914 import sample as s
m,c=s.m,s.c


def run(action,stage=None,gpus='0,1,2,3'):
    command=[sys.executable,'-u','-m',s.MODULE,action,'--gpus',gpus]
    if stage:command+=['--stage',stage]
    env=dict(os.environ)
    if action=='preflight':env['CUDA_VISIBLE_DEVICES']='0'
    subprocess.run(command,check=True,cwd=c.WORK,env=env)


def flex_heads():
    stage='tune_heads';req=s.verify(stage);root=m.ROOT/stage
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[]
        for cfg in req['configs']:
            for start in range(0,400,128):
                stop=min(start+128,400)
                if any(not (root/cfg['arm']/f'batch{k:06d}.npz').exists() for k in range(start,stop,16)):
                    jobs.append(dict(config=cfg,start=start,stop=stop))
        c.atomic(root/'queue.json',jobs);processes={};streams=[]
        def spawn(gpu):
            f=(root/f'worker{gpu}.log').open('a');streams.append(f)
            processes[gpu]=subprocess.Popen([sys.executable,'-u','-m',s.MODULE,'worker','--stage',stage,'--rank',str(gpu)],
                cwd=c.WORK,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            print('Head search starts GPU',gpu,flush=True)
        for gpu in (0,1):spawn(gpu)
        while True:
            native=m.ROOT/'tune_native/controller_complete.json'
            if native.exists():
                assert c.read(native)['complete']
                for gpu in (2,3):
                    if gpu not in processes:spawn(gpu)
            codes={k:p.poll() for k,p in processes.items()}
            if any(code not in (None,0) for code in codes.values()):
                for p in processes.values():
                    if p.poll() is None:p.terminate()
                for p in processes.values():p.wait()
                c.atomic(root/'controller_complete.json',dict(complete=False,exit_codes=codes));raise RuntimeError(codes)
            if all(code==0 for code in codes.values()):break
            c.atomic(root/'status.json',dict(complete=False,phase='sampling',gpus=list(processes),remaining_shards=len(c.read(root/'queue.json'))))
            time.sleep(5)
        for f in streams:f.close()
        for cfg in req['configs']:s.collect(stage,cfg,req)
        while not (m.ROOT/'tune_native/controller_complete.json').exists():time.sleep(5)
        assert c.read(m.ROOT/'tune_native/controller_complete.json')['complete']
        def eval_gpu(gpu):
            for cfg in req['configs'][gpu::4]:s.evaluate(stage,cfg,gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(eval_gpu,range(4)))
        c.atomic(root/'controller_complete.json',dict(complete=True,exit_codes={k:p.returncode for k,p in processes.items()},arms=len(req['configs'])))
        c.atomic(root/'status.json',dict(complete=True,phase='complete',arms=len(req['configs'])))


def main():
    c.atomic(m.ROOT/'pipeline_manifest.json',dict(source_sha256=c.sha(__file__),stages=['tune_native','tune_heads','tune_refine','confirm_5k']))
    while not all((p/'summary.json').exists() for p in (m.TRAIN,s.d8.TRAIN)):time.sleep(5)
    for p in (m.TRAIN,s.d8.TRAIN):assert c.read(p/'summary.json')['complete'] and c.read(p/'summary.json')['steps']==50000
    if not (m.ROOT/'heads_preflight.json').exists():run('preflight')
    if not (m.ROOT/'tune_heads/controller_complete.json').exists():
        run('prepare','tune_heads');flex_heads()
    else:assert c.read(m.ROOT/'tune_heads/controller_complete.json')['complete']
    for stage in ('tune_refine','confirm_5k'):
        if not (m.ROOT/stage/'controller_complete.json').exists():
            run('prepare',stage);run('controller',stage)
        else:assert c.read(m.ROOT/stage/'controller_complete.json')['complete']
    c.atomic(m.ROOT/'pipeline_complete.json',dict(complete=True,training_steps=50000,heads=3,final_samples_per_family=5000))
    print('All head search and 5K sampling stages complete',flush=True)

if __name__=='__main__':main()
