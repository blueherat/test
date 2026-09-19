"""Extend all three existing XL guidance curves using their frozen samplers."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.sample_official_sit_baseline_control_20260909 import sample_batch as ordinary_sample
from experiments.sample_official_sit_pfr_symmetric_20260909 import sample_batch as pfr_sample
from experiments.sample_official_sit_pfr_symmetric_20260909 import array_sha,write_json,CKPT
from experiments.run_official_sit_native_sde_20260909 import state_sha
from experiments.run_internal_guidance_sit_audit import load_model
from experiments.raev2_training_core import file_sha256

RAW=Path('/home/zhoushunyu/data/eqvae/experiments')
DATA=RAW/'official_sit_pfr_upper_20260909'
BASE=RAW/'official_sit_baseline_control_20260909'
PREVIOUS=RAW/'official_sit_pfr_symmetric_20260909'
ARMS={'ordinary200':('ordinary',2.),'time200':('time_only',2.),'projected200':('projected',2.),
      'ordinary250':('ordinary',2.5),'time250':('time_only',2.5),'projected250':('projected',2.5)}
REFERENCES={'ordinary':BASE/'ig175_all','time_only':PREVIOUS/'time175','projected':PREVIOUS/'projected175'}


class CountedModel:
    def __init__(self,model): self.model,self.batch_calls,self.sample_calls=model,0,0
    def __getattr__(self,name): return getattr(self.model,name)
    def __call__(self,x,*args,**kwargs):
        self.batch_calls+=1;self.sample_calls+=len(x)
        return self.model(x,*args,**kwargs)


def sample(model,vae,noise,labels,scale,method):
    if method=='ordinary': return ordinary_sample(model,vae,noise,labels,scale,1.),None,None,0
    pixels,latents,diagnostics,calls=pfr_sample(model,vae,noise,labels,scale,method)
    assert calls==(100,50,0)
    return pixels,latents,diagnostics,calls[1]


def prepare():
    DATA.mkdir(parents=True,exist_ok=False)
    old=json.loads((PREVIOUS/'request.json').read_text())
    assert json.loads((PREVIOUS/'status.json').read_text())['phase']=='complete'
    for path,digest in old['sources'].items(): assert file_sha256(Path(path))==digest,path
    assert file_sha256(CKPT)==old['checkpoint_sha256']
    assert file_sha256(PREVIOUS/'inputs.npz')==old['input_file_sha256']
    (DATA/'inputs.npz').write_bytes((PREVIOUS/'inputs.npz').read_bytes())
    sources=[Path(__file__),ROOT/'experiments/sample_official_sit_baseline_control_20260909.py',
        ROOT/'experiments/run_official_sit_native_sde_20260909.py',
        ROOT/'docs/OFFICIAL_SIT_PFR_UPPER_PROTOCOL_20260909_ZH.md']+[Path(p) for p in old['sources']]
    sources=list(dict.fromkeys(sources));(DATA/'sources').mkdir()
    for i,path in enumerate(sources): (DATA/'sources'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    references={}
    for method,directory in REFERENCES.items():
        metrics=json.loads((directory/'fid.json').read_text())[0]
        assert file_sha256(directory/'samples.npz')==metrics['sample_sha256']
        references[method]=dict(directory=str(directory),pixel_sha256=metrics['sample_sha256'])
    previous_csv=ROOT/'docs/data/guidance_goal_20260909/official_sit_pfr_symmetric.csv'
    previous_audit=ROOT/'docs/data/guidance_goal_20260909/official_sit_pfr_symmetric_audit.json'
    assert json.loads(previous_audit.read_text())['passed']
    write_json(DATA/'request.json',dict(arms=ARMS,samples=1000,batch_size=4,
        input_file_sha256=file_sha256(DATA/'inputs.npz'),noise_sha256=old['noise_sha256'],label_sha256=old['label_sha256'],
        checkpoint_sha256=old['checkpoint_sha256'],vae_state_sha256=old['vae_state_sha256'],
        sources={str(p):file_sha256(p) for p in sources},references=references,
        previous_request_sha256=file_sha256(PREVIOUS/'request.json'),
        previous_csv_sha256=file_sha256(previous_csv),previous_audit_sha256=file_sha256(previous_audit),
        sampling_bank_role='reused_1k_discovery'))
    print('Six upper-grid arms frozen; existing samplers and evidence hashes verified.',flush=True)


@torch.inference_mode()
def worker(rank):
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    req=json.loads((DATA/'request.json').read_text())
    for path,digest in req['sources'].items(): assert file_sha256(Path(path))==digest,path
    assert file_sha256(DATA/'inputs.npz')==req['input_file_sha256']
    with np.load(DATA/'inputs.npz') as bank: noise,labels=bank['noise'],bank['labels']
    assert array_sha(noise)==req['noise_sha256'] and array_sha(labels)==req['label_sha256']
    model,meta=load_model(repo=ROOT/'research_repos/internal_guidance_study/Internal-Guidance/SiT',
        checkpoint_path=CKPT,model_name='SiT-XL/2',encoder_depth=8,state_key='ema',device=torch.device('cuda'))
    from diffusers import AutoencoderKL
    vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse',local_files_only=True).cuda().eval().requires_grad_(False)
    assert state_sha(vae)==req['vae_state_sha256']
    records={};start=rank*4
    for method,ref in req['references'].items():
        counted=CountedModel(model)
        pixels,_,_,prefix=sample(counted,vae,torch.from_numpy(noise[start:start+4]).cuda(),
            torch.from_numpy(labels[start:start+4]).cuda(),1.75,method)
        with np.load(Path(ref['directory'])/'samples.npz') as expected:
            assert np.array_equal(pixels,expected['arr_0'][start:start+4]),(rank,method)
        assert counted.batch_calls==(115 if method=='ordinary' else 100)
        records[method]=dict(passed=True,pixel_sha256=array_sha(pixels),full_calls=counted.batch_calls,prefix_calls=prefix)
    write_json(DATA/f'parity_rank{rank}.json',dict(passed=True,indices=list(range(start,start+4)),records=records))
    while not (DATA/'parity_passed.json').exists(): time.sleep(.25)
    for arm,(method,scale) in ARMS.items():
        directory=DATA/arm;directory.mkdir(exist_ok=True)
        images,endpoints,stats,indices=[],[],[],[];prefix_calls=0;counted=CountedModel(model);batches=0
        torch.cuda.synchronize();began=time.perf_counter()
        for batch in range(rank,250,4):
            start=batch*4
            pixels,latent,diagnostics,prefix=sample(counted,vae,torch.from_numpy(noise[start:start+4]).cuda(),
                torch.from_numpy(labels[start:start+4]).cuda(),scale,method)
            images.append(pixels);indices.extend(range(start,start+4));prefix_calls+=prefix;batches+=1
            if latent is not None: endpoints.append(latent);stats.append(diagnostics)
            if batches%16==0: print(json.dumps(dict(arm=arm,rank=rank,images=batches*4,seconds=time.perf_counter()-began)),flush=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-began
        payload=dict(arr_0=np.concatenate(images),indices=np.array(indices,dtype=np.int64))
        if endpoints: payload.update(latents=np.concatenate(endpoints),diagnostics=np.concatenate(stats))
        path=directory/f'rank{rank}.npz';np.savez(path,**payload)
        write_json(directory/f'rank{rank}.json',dict(complete=True,rank=rank,arm=arm,method=method,scale=scale,
            samples=len(indices),batch_size=4,full_batch_calls=counted.batch_calls,full_sample_calls=counted.sample_calls,
            prefix_batch_calls=prefix_calls,prefix_sample_calls=prefix_calls*4,inference_seconds=elapsed,
            pixel_file_sha256=file_sha256(path),metadata=meta,noise_sha256=array_sha(noise[indices]),
            label_sha256=array_sha(labels[indices]),request_sha256=file_sha256(DATA/'request.json')))
        print(json.dumps(dict(arm=arm,rank=rank,complete=True,seconds=elapsed)),flush=True)
        while not (directory/'advance.json').exists(): time.sleep(.25)
    for path,digest in req['sources'].items(): assert file_sha256(Path(path))==digest,path


def wait_files(paths,workers):
    while not all(p.is_file() for p in paths):
        for w in workers:
            if w.poll() is not None: raise RuntimeError(f'Worker {w.pid} exited at barrier: {w.returncode}')
        time.sleep(1)


def main():
    torch.set_num_threads(4);prepare()
    req=json.loads((DATA/'request.json').read_text());workers,logs,results=[],[],[];began=time.perf_counter()
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    try:
        for rank in range(4):
            log=(DATA/f'worker{rank}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([sys.executable,str(Path(__file__)),'--rank',str(rank)],
                env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)),stdout=log,stderr=subprocess.STDOUT))
        write_json(DATA/'status.json',dict(phase='parity',pids=[w.pid for w in workers]))
        wait_files([DATA/f'parity_rank{r}.json' for r in range(4)],workers)
        parity=[json.loads((DATA/f'parity_rank{r}.json').read_text()) for r in range(4)]
        assert all(p['passed'] and set(p['records'])==set(REFERENCES) and all(x['passed'] for x in p['records'].values()) for p in parity)
        assert sorted(i for p in parity for i in p['indices'])==list(range(16))
        write_json(DATA/'parity_passed.json',dict(passed=True,unique_inputs=16,image_outputs=48,records=parity))
        print('All three methods reproduced 16 reference images each.',flush=True)
        with np.load(DATA/'inputs.npz') as bank: noise,labels=bank['noise'],bank['labels']
        for arm,(method,scale) in ARMS.items():
            write_json(DATA/'status.json',dict(phase='sampling',arm=arm,pids=[w.pid for w in workers]))
            directory=DATA/arm;wait_files([directory/f'rank{r}.json' for r in range(4)],workers)
            images=np.empty((1000,256,256,3),dtype=np.uint8);covered=np.zeros(1000,dtype=int);summaries=[]
            extras={} if method=='ordinary' else dict(latents=np.empty((1000,4,32,32),dtype=np.float32),diagnostics=np.empty((1000,5,4),dtype=np.float32))
            for rank in range(4):
                s=json.loads((directory/f'rank{rank}.json').read_text());summaries.append(s)
                assert s['method']==method and s['scale']==scale
                assert s['request_sha256']==file_sha256(DATA/'request.json')
                assert s['pixel_file_sha256']==file_sha256(directory/f'rank{rank}.npz')
                with np.load(directory/f'rank{rank}.npz') as shard:
                    idx=shard['indices']
                    assert np.array_equal(idx,np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,250,4)]))
                    assert array_sha(noise[idx])==s['noise_sha256'] and array_sha(labels[idx])==s['label_sha256']
                    assert shard['arr_0'].dtype==np.uint8
                    images[idx]=shard['arr_0'];covered[idx]+=1
                    for key,target in extras.items():
                        assert np.isfinite(shard[key]).all();target[idx]=shard[key]
            assert np.all(covered==1)
            full,prefix=(115,0) if method=='ordinary' else (100,50)
            assert sum(s['full_sample_calls'] for s in summaries)==full*1000
            assert sum(s['prefix_sample_calls'] for s in summaries)==prefix*1000
            np.savez(directory/'samples.npz',arr_0=images)
            if extras: np.savez(directory/'diagnostics.npz',**extras,labels=labels)
            write_json(DATA/'status.json',dict(phase='evaluation',arm=arm,pids=[w.pid for w in workers]))
            cmd=[sys.executable,str(ROOT/'experiments/evaluate_raev2_official_samples.py'),'--branch',arm+'='+str(directory/'samples.npz'),
                '--output',str(directory/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(directory/'features')]
            with (directory/'evaluation.log').open('x') as log:
                subprocess.run(cmd,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics=json.loads((directory/'fid.json').read_text())[0]
            assert metrics['sample_sha256']==file_sha256(directory/'samples.npz')
            row=dict(arm=arm,method=method,scale=scale,metrics=metrics,full_per_image=full,prefix_per_image=prefix,
                gpu_inference_seconds_sum=sum(s['inference_seconds'] for s in summaries),
                inference_seconds_max=max(s['inference_seconds'] for s in summaries))
            if extras: row['diagnostic_sha256']=file_sha256(directory/'diagnostics.npz')
            results.append(row);write_json(directory/'result.json',row);write_json(DATA/'results.json',results)
            print(json.dumps(row),flush=True);write_json(directory/'advance.json',dict(complete=True))
        for w in workers:
            if w.wait()!=0: raise RuntimeError('Worker exit failure')
        for path,digest in req['sources'].items(): assert file_sha256(Path(path))==digest,path
        write_json(DATA/'status.json',dict(phase='complete',seconds=time.perf_counter()-began,research_goal_achieved=False,results=results))
    except BaseException as error:
        write_json(DATA/'status.json',dict(phase='failed',error=repr(error)));raise
    finally:
        for w in workers:
            if w.poll() is None: w.terminate()
        for w in workers: w.wait()
        for log in logs: log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,choices=range(4));args=parser.parse_args()
    if args.rank is None: main()
    else: worker(args.rank)
