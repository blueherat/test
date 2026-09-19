"""Resumable four-GPU queue with paired inputs and source-frozen evaluation."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from PIL import Image,ImageDraw
from experiments.self_guidance_cross_model_20260913 import core as c
from experiments.guidance_pasted_20260912.audit import REFS

EVAL_ROOT=c.common.DATA/'external_sources/nanogen-evals'
REFERENCE=REFS['raev2']
MODULE='experiments.self_guidance_cross_model_20260913.run'


def save_npz(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,**values)
    tmp.replace(path)


def sources(name):
    files=list(Path(__file__).parent.glob('*.py'))+[Path(c.common.__file__),
        c.common.WORK/'experiments/evaluate_raev2_official_samples.py']
    files+=list((EVAL_ROOT/'fd_evaluator/fd_evaluator').glob('*.py'))
    files+=[EVAL_ROOT/'fd_evaluator/fd_evaluator/catalogue.yaml']
    if name=='jit':
        from experiments import jit_internal_guidance as j
        files+=[Path(j.__file__),j.REPO/'model_jit.py',j.REPO/'denoiser.py',j.REPO/'util/model_util.py']
    else:
        files+=c.common.source_paths('raev2')
        files+=list((c.common.WORK/'external/RAEv2/src').rglob('*.py'))
    return sorted(set(p.resolve() for p in files))


def prepare(args):
    for name in args.models.split(','):
        root=c.ROOT/args.phase/name;root.mkdir(parents=True,exist_ok=True)
        assert c.common.read(c.ROOT/'checks'/f'{name}.json')['model_passed']
        configs=c.configs(name) if not args.config else c.common.read(args.config)[name]
        rng=np.random.default_rng(args.seed)
        noise_path=root/'noise.npy';label_path=root/'labels.npy'
        if not noise_path.exists():
            noise=np.lib.format.open_memmap(noise_path.with_suffix('.tmp'),mode='w+',
                    dtype=np.float32,shape=(args.samples,*c.SHAPES[name]))
            for start in range(0,args.samples,8):
                noise[start:start+8]=rng.standard_normal((min(8,args.samples-start),*c.SHAPES[name]),dtype=np.float32)
            noise.flush();del noise
            noise_path.with_suffix('.tmp').replace(noise_path)
            labels=np.arange(args.samples,dtype=np.int64)%1000;rng.shuffle(labels)
            np.save(label_path,labels)
        model_assets=[c.common.JIT_CKPT] if name=='jit' else c.common.asset_paths('raev2')
        assets=[*model_assets,REFERENCE]
        assets+=[Path('/home/zhoushunyu/.cache/torch/hub/checkpoints')/file for file in
                 ('weights-inception-2015-12-05-6726825d.pth','pt_inception-2015-12-05-6726825d.pth')]
        request=dict(model=name,phase=args.phase,samples=args.samples,seed=args.seed,batch=args.batch,
            configs=configs,sources={str(p):c.common.sha(p) for p in sources(name)},
            assets={str(p):c.common.sha(p) for p in assets},
            inputs={str(p):c.common.sha(p) for p in (noise_path,label_path)},
            reference=str(REFERENCE),precision='FP32 weights/state, BF16 autocast, TF32 on',
            upstream_commit='843bda799bb531ccf8d98164d8ccf1a3b8ce3f6d',
            scope='Transfer of released velocity SG rule; own-time clean-to-velocity conversion; paired 1K pilot')
        path=root/'request.json'
        if path.exists():assert c.common.read(path)==request,'Frozen phase changed'
        else:c.common.atomic(path,request)
        print('prepared',name,len(configs),args.samples,flush=True)


def verify(root):
    request=c.common.read(root/'request.json')
    for group in ('sources','assets','inputs'):
        for path,sha in request[group].items():assert c.common.sha(path)==sha,(group,path)
    return request


def pop_job(root):
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=c.common.read(root/'queue.json')
        if not queue:return None
        job=queue.pop(0);c.common.atomic(root/'queue.json',queue)
        return job


def collect(out,request,noise,labels):
    rows=[];pixels=[];records=[];total=0.
    for start in range(0,request['samples'],request['batch']):
        path=out/f'batch{start:06d}.npz'
        with np.load(path) as d:
            stop=min(start+request['batch'],request['samples'])
            assert int(d['start'])==start and str(d['noise_sha256'])==c.common.array_sha(noise[start:stop])
            assert str(d['request_sha256'])==c.common.sha(out.parent/'request.json')
            assert np.array_equal(d['labels'],labels[start:stop])
            assert d['pixels'].shape==(stop-start,256,256,3) and d['pixels'].dtype==np.uint8
            assert np.isfinite(d['latents']).all()
            pixels.append(d['pixels']);rows.append(int(d['full']));total+=float(d['seconds'])
        records.append(dict(path=str(path),sha256=c.common.sha(path)))
    assert len(set(rows))==1
    images=np.concatenate(pixels);save_npz(out/'samples.npz',arr_0=images)
    canvas=Image.new('RGB',(8*128,4*146),'white');draw=ImageDraw.Draw(canvas)
    for i,pix in enumerate(images[:32]):
        x,y=(i%8)*128,(i//8)*146
        canvas.paste(Image.fromarray(pix).resize((128,128)),(x,y))
        draw.text((x+3,y+128),f'ID {i}, class {labels[i]}',fill='black')
    canvas.save(out/'grid.png')
    c.common.atomic(out/'summary.json',dict(complete=True,samples=len(images),records=records,
        full_calls_per_output=rows[0],prefix_calls_per_output=0,seconds=total,
        request_sha256=c.common.sha(out.parent/'request.json'),samples_sha256=c.common.sha(out/'samples.npz')))


def evaluate(out):
    if (out/'metrics.json').exists():return
    command=[sys.executable,str(c.common.WORK/'experiments/evaluate_raev2_official_samples.py'),
        '--branch',out.name+'='+str(out/'samples.npz'),'--output',str(out/'metrics.csv'),
        '--device','cuda','--batch-size','32','--fid-reference',str(REFERENCE),
        '--feature-cache-dir',str(out/'features')]
    with (out/'evaluation.log').open('w') as stream:
        subprocess.run(command,check=True,cwd=c.common.WORK,stdout=stream,stderr=subprocess.STDOUT,
                       env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))


@torch.inference_mode()
def worker(args):
    root=c.ROOT/args.phase;rt=None
    while (job:=pop_job(root)) is not None:
        name,config=job['model'],job['config'];model_root=root/name
        request=verify(model_root)
        if rt is None or rt.name!=name:
            # Remove the counting hook before unloading a previous model.
            if rt is not None:
                rt.model._forward_pre_hooks.clear()
            del rt;gc.collect();torch.cuda.empty_cache();rt=c.Runtime(name)
        noise=np.load(model_root/'noise.npy',mmap_mode='r');labels=np.load(model_root/'labels.npy')
        out=model_root/config['arm'];out.mkdir(parents=True,exist_ok=True)
        if not (out/'summary.json').exists():
            c.sample(rt,torch.from_numpy(noise[:2].copy()).cuda(),torch.from_numpy(labels[:2]).cuda(),config)
            for start in range(0,request['samples'],request['batch']):
                path=out/f'batch{start:06d}.npz'
                if path.exists():continue
                stop=min(start+request['batch'],request['samples'])
                z=torch.from_numpy(noise[start:stop].copy()).cuda();y=torch.from_numpy(labels[start:stop]).cuda()
                torch.cuda.synchronize();begin=time.perf_counter()
                z,counts=c.sample(rt,z,y,config,trace=start==0)
                pixels=rt.decode(z)
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                save_npz(path,pixels=pixels,latents=z.float().cpu().numpy(),labels=labels[start:stop],
                    start=start,full=counts['full'],seconds=elapsed,trace=counts['trace'],
                    request_sha256=c.common.sha(model_root/'request.json'),
                    noise_sha256=c.common.array_sha(noise[start:stop]))
                c.common.atomic(root/f'progress{args.rank}.json',dict(model=name,arm=config['arm'],
                    completed=stop,total=request['samples'],phase='sampling'))
            collect(out,request,noise,labels)
        torch.cuda.empty_cache()
        c.common.atomic(root/f'progress{args.rank}.json',dict(model=name,arm=config['arm'],phase='evaluation'))
        evaluate(out)
        print(name,config['arm'],c.common.read(out/'metrics.json')[0]['fid'],flush=True)
    c.common.atomic(root/f'worker{args.rank}_complete.json',dict(complete=True))


def controller(args):
    root=c.ROOT/args.phase
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[]
        for name in args.models.split(','):
            request=verify(root/name)
            for config in request['configs']:
                if not (root/name/config['arm']/'metrics.json').exists():jobs.append(dict(model=name,config=config))
        c.common.atomic(root/'queue.json',jobs)
        workers=[];logs=[]
        for rank,gpu in enumerate(args.gpus.split(',')):
            stream=(root/f'worker{rank}.log').open('a');logs.append(stream)
            workers.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--phase',args.phase,
                '--rank',str(rank)],cwd=c.common.WORK,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')))
        while any(p.poll() is None for p in workers):
            codes=[p.poll() for p in workers]
            if any(code not in (None,0) for code in codes):
                for p in workers:
                    if p.poll() is None:p.terminate()
                for p in workers:p.wait()
                break
            completed=len(list(root.glob('*/*/metrics.json')))
            c.common.atomic(root/'status.json',dict(complete=False,completed=completed,remaining_queue=len(c.common.read(root/'queue.json')),
                                                   workers=[p.pid for p in workers]))
            time.sleep(5)
        codes=[p.wait() for p in workers]
        for stream in logs:stream.close()
        c.common.atomic(root/'controller_complete.json',dict(complete=all(code==0 for code in codes),exit_codes=codes))
        if any(codes):raise RuntimeError(codes)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','controller','worker'])
    p.add_argument('--phase',default='screen_1k');p.add_argument('--models',default='jit,raev2')
    p.add_argument('--samples',type=int,default=1000);p.add_argument('--seed',type=int,default=2026091327)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--config')
    args=p.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
