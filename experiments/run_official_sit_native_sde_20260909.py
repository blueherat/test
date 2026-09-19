"""Controlled author-parameter SDE baseline with paired EMA/MSE VAE decoding."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.raev2_training_core import file_sha256
from experiments.run_internal_guidance_sit_audit import load_model

DATA = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_native_sde_20260909')
BASE = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_baseline_control_20260909')
REPO = ROOT/'research_repos/internal_guidance_study/Internal-Guidance/SiT'
CKPT = Path('/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt')
KWARGS = dict(num_steps=250,cfg_scale=1.35,sg_scale=1.4,guidance_low=0.,guidance_high=.7,
              sg_guidance_low=0.,sg_guidance_high=1.,path_type='linear')
VAES = ('ema','mse')


def array_sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def brownian_seed(batch):
    digest=hashlib.sha256(f'official_sit_native_sde_20260909/brownian/batch/{batch}'.encode()).digest()
    return int.from_bytes(digest[:8],'big') % (2**63)


def expected_calls():
    grid=torch.linspace(1.,.04,250,dtype=torch.float64)
    return dict(batch_calls=250,full_per_image=sum(2 if 0 <= float(t) <= .7 else 1 for t in grid),
                gaussian_batch_calls=249)


class CountedModel:
    def __init__(self,model): self.model,self.batch_calls,self.sample_calls=model,0,0
    def __call__(self,x,*args,**kwargs):
        self.batch_calls+=1;self.sample_calls+=len(x)
        return self.model(x,*args,**kwargs)


def prepare():
    DATA.mkdir(parents=True,exist_ok=False)
    previous=json.loads((BASE/'request.json').read_text())
    assert file_sha256(BASE/'inputs.npz')==previous['input_file_sha256']
    assert file_sha256(CKPT)==previous['checkpoint_sha256']
    for path,digest in previous['sources'].items(): assert file_sha256(Path(path))==digest,path
    (DATA/'inputs.npz').write_bytes((BASE/'inputs.npz').read_bytes())
    sources=[Path(__file__),ROOT/'experiments/run_internal_guidance_sit_audit.py',
        ROOT/'experiments/evaluate_raev2_official_samples.py',REPO/'models/sit.py',REPO/'samplers.py',
        REPO/'generate.py',REPO/'gen.sh',REPO.parent/'README.md',
        ROOT/'docs/OFFICIAL_SIT_NATIVE_SDE_PROTOCOL_20260909_ZH.md']
    (DATA/'sources').mkdir()
    for i,path in enumerate(sources): (DATA/'sources'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    from diffusers import AutoencoderKL
    vae_hashes,vae_files={},{}
    for name in VAES:
        vae=AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{name}',local_files_only=True).eval()
        vae_hashes[name]=state_sha(vae)
        cache=Path('/home/zhoushunyu/.cache/huggingface/hub')/f'models--stabilityai--sd-vae-ft-{name}'
        files=sorted(cache.glob('snapshots/*/config.json'))+sorted(cache.glob('snapshots/*/diffusion_pytorch_model.safetensors'))
        assert files
        vae_files[name]={str(p):file_sha256(p) for p in files}
        del vae
    assert vae_hashes['mse']==previous['vae_state_sha256']
    seeds=[brownian_seed(b) for b in range(250)]
    assert len(set(seeds))==250
    write_json(DATA/'request.json',dict(samples=1000,batch_size=4,kwargs=KWARGS,expected_calls=expected_calls(),
        brownian_seeds=seeds,noise_sha256=previous['noise_sha256'],label_sha256=previous['label_sha256'],
        input_file_sha256=file_sha256(DATA/'inputs.npz'),checkpoint_sha256=previous['checkpoint_sha256'],
        vae_state_sha256=vae_hashes,vae_files=vae_files,sources={str(p):file_sha256(p) for p in sources},
        baseline_request_sha256=file_sha256(BASE/'request.json'),sampling_bank_role='reused_1k_discovery'))
    print('Native SDE request, two VAE states and input assets verified.',flush=True)


@torch.inference_mode()
def worker(rank):
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    req=json.loads((DATA/'request.json').read_text())
    for path,digest in req['sources'].items(): assert file_sha256(Path(path))==digest,path
    assert file_sha256(DATA/'inputs.npz')==req['input_file_sha256']
    with np.load(DATA/'inputs.npz') as bank: noise,labels=bank['noise'],bank['labels']
    assert array_sha(noise)==req['noise_sha256'] and array_sha(labels)==req['label_sha256']
    model,meta=load_model(repo=REPO,checkpoint_path=CKPT,model_name='SiT-XL/2',encoder_depth=8,state_key='ema',device=torch.device('cuda'))
    from samplers import euler_maruyama_ig_sampler
    from diffusers import AutoencoderKL
    vaes={name:AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{name}',local_files_only=True).cuda().eval().requires_grad_(False) for name in VAES}
    assert {name:state_sha(vae) for name,vae in vaes.items()}==req['vae_state_sha256']
    def sample(batch,net):
        torch.manual_seed(req['brownian_seeds'][batch])
        start=batch*4
        return euler_maruyama_ig_sampler(net,torch.from_numpy(noise[start:start+4]).cuda(),
            torch.from_numpy(labels[start:start+4]).cuda(),**req['kwargs'])
    counter=CountedModel(model)
    check=sample(rank,model)
    repeat=sample(rank,counter)
    assert torch.equal(check,repeat)
    assert counter.batch_calls==250 and counter.sample_calls==4*req['expected_calls']['full_per_image']
    write_json(DATA/f'parity_rank{rank}.json',dict(passed=True,indices=list(range(rank*4,rank*4+4)),
        native_fp64_endpoint_sha256=array_sha(check.cpu().numpy()),wrapper_calls=counter.batch_calls,
        wrapper_sample_calls=counter.sample_calls))
    del check,repeat,counter
    while not (DATA/'parity_passed.json').exists(): time.sleep(.25)
    images={name:[] for name in VAES};endpoints=[];indices=[];rng_hashes=[]
    trajectory_seconds=0.;decode_seconds={name:0. for name in VAES}
    batches=0;counter=CountedModel(model)
    for batch in range(rank,250,4):
        torch.cuda.synchronize();began=time.perf_counter()
        latent64=sample(batch,counter)
        torch.cuda.synchronize();trajectory_seconds+=time.perf_counter()-began
        assert torch.isfinite(latent64).all()
        latent=latent64.float();endpoints.append(latent.cpu().numpy())
        rng_hashes.append(array_sha(torch.cuda.get_rng_state().cpu().numpy()))
        for name,vae in vaes.items():
            torch.cuda.synchronize();began=time.perf_counter()
            decoded=vae.decode(latent/.18215).sample
            assert torch.isfinite(decoded).all()
            pixels=(255.*((decoded+1)/2.)).clamp(0,255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
            torch.cuda.synchronize();decode_seconds[name]+=time.perf_counter()-began
            images[name].append(pixels)
        indices.extend(range(batch*4,batch*4+4));batches+=1
        if batches%8==0: print(json.dumps(dict(rank=rank,images=batches*4,trajectory_seconds=trajectory_seconds)),flush=True)
    path=DATA/f'rank{rank}.npz'
    np.savez(path,**{name:np.concatenate(x) for name,x in images.items()},latents=np.concatenate(endpoints),
        indices=np.array(indices,dtype=np.int64),brownian_rng_end_sha256=np.array(rng_hashes))
    write_json(DATA/f'rank{rank}.json',dict(complete=True,rank=rank,samples=len(indices),batch_size=4,
        full_batch_calls=counter.batch_calls,full_sample_calls=counter.sample_calls,trajectory_seconds=trajectory_seconds,
        decode_seconds=decode_seconds,pixel_file_sha256=file_sha256(path),metadata=meta,
        noise_sha256=array_sha(noise[indices]),label_sha256=array_sha(labels[indices]),request_sha256=file_sha256(DATA/'request.json')))
    for p,d in req['sources'].items(): assert file_sha256(Path(p))==d,p
    del model,vaes,counter,latent,latent64,decoded
    torch.cuda.empty_cache()
    while not (DATA/'advance.json').exists(): time.sleep(.25)


def wait_files(paths,workers):
    while not all(p.is_file() for p in paths):
        for w in workers:
            if w.poll() is not None: raise RuntimeError(f'Worker {w.pid} exited at barrier, code {w.returncode}')
        time.sleep(1)


def main():
    torch.set_num_threads(4)
    prepare()
    req=json.loads((DATA/'request.json').read_text())
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    workers,logs=[],[];began=time.perf_counter()
    try:
        for rank in range(4):
            log=(DATA/f'worker{rank}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([sys.executable,str(Path(__file__)),'--rank',str(rank)],
                env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)),stdout=log,stderr=subprocess.STDOUT))
        write_json(DATA/'status.json',dict(phase='parity',pids=[w.pid for w in workers]))
        wait_files([DATA/f'parity_rank{r}.json' for r in range(4)],workers)
        parity=[json.loads((DATA/f'parity_rank{r}.json').read_text()) for r in range(4)]
        assert all(x['passed'] for x in parity)
        assert sorted(i for x in parity for i in x['indices'])==list(range(16))
        write_json(DATA/'parity_passed.json',dict(passed=True,images=16,records=parity))
        write_json(DATA/'status.json',dict(phase='sampling',pids=[w.pid for w in workers]))
        print('16 native FP64 endpoints reproduced; sampling original SDE250.',flush=True)
        wait_files([DATA/f'rank{r}.json' for r in range(4)],workers)
        with np.load(DATA/'inputs.npz') as bank: noise,labels=bank['noise'],bank['labels']
        images={n:np.empty((1000,256,256,3),dtype=np.uint8) for n in VAES}
        latents=np.empty((1000,4,32,32),dtype=np.float32);covered=np.zeros(1000,dtype=int);summaries=[]
        for rank in range(4):
            s=json.loads((DATA/f'rank{rank}.json').read_text());summaries.append(s)
            assert s['request_sha256']==file_sha256(DATA/'request.json')
            assert file_sha256(DATA/f'rank{rank}.npz')==s['pixel_file_sha256']
            with np.load(DATA/f'rank{rank}.npz') as shard:
                idx=shard['indices']
                assert np.array_equal(idx,np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,250,4)]))
                assert array_sha(noise[idx])==s['noise_sha256'] and array_sha(labels[idx])==s['label_sha256']
                for name in VAES: images[name][idx]=shard[name]
                latents[idx]=shard['latents'];covered[idx]+=1
        assert np.all(covered==1) and np.isfinite(latents).all()
        assert sum(s['full_batch_calls'] for s in summaries)==62500
        assert sum(s['full_sample_calls'] for s in summaries)==req['expected_calls']['full_per_image']*1000
        np.savez(DATA/'latents.npz',latents=latents,labels=labels)
        results=[]
        for name in VAES:
            directory=DATA/name;directory.mkdir()
            np.savez(directory/'samples.npz',arr_0=images[name])
            write_json(DATA/'status.json',dict(phase='evaluation',vae=name,pids=[w.pid for w in workers]))
            cmd=[sys.executable,str(ROOT/'experiments/evaluate_raev2_official_samples.py'),
                '--branch','native_sde250_'+name+'='+str(directory/'samples.npz'),'--output',str(directory/'fid.csv'),
                '--batch-size','32','--device','cuda','--feature-cache-dir',str(directory/'features')]
            with (directory/'evaluation.log').open('x') as log:
                subprocess.run(cmd,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics=json.loads((directory/'fid.json').read_text())[0]
            assert metrics['sample_sha256']==file_sha256(directory/'samples.npz')
            row=dict(vae=name,metrics=metrics,full_per_image=req['expected_calls']['full_per_image'],prefix_per_image=0,
                trajectory_gpu_seconds_sum=sum(s['trajectory_seconds'] for s in summaries),
                decode_gpu_seconds_sum=sum(s['decode_seconds'][name] for s in summaries))
            results.append(row);write_json(directory/'result.json',row);write_json(DATA/'results.json',results)
            print(json.dumps(row),flush=True)
        write_json(DATA/'advance.json',dict(complete=True))
        for w in workers:
            if w.wait()!=0: raise RuntimeError('Worker exit failure')
        for p,d in req['sources'].items(): assert file_sha256(Path(p))==d,p
        write_json(DATA/'status.json',dict(phase='complete',seconds=time.perf_counter()-began,
            research_goal_achieved=False,results=results))
    except BaseException as error:
        write_json(DATA/'status.json',dict(phase='failed',error=repr(error)));raise
    finally:
        for w in workers:
            if w.poll() is None: w.terminate()
        for w in workers: w.wait()
        for log in logs: log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,choices=range(4))
    args=parser.parse_args()
    if args.rank is None: main()
    else: worker(args.rank)
