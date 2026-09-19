"""Two predeclared SDE PFR/control arms, paired noise and equal block budget."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.raev2_training_core import file_sha256
from experiments.run_internal_guidance_sit_audit import load_model
from experiments.run_official_sit_native_sde_20260909 import array_sha,write_json,state_sha,CKPT,REPO,KWARGS
from experiments.official_sit_sde_pfr_control_core_20260909 import (
    CountedModel,counts,bridge_seed,make_noises,sample_sde)

RAW=Path('/home/zhoushunyu/data/eqvae/experiments')
DATA=RAW/'official_sit_sde_pfr_control_20260909'
NATIVE=RAW/'official_sit_native_sde_20260909'
ARMS={'ordinary281':dict(steps=281,pfr=False),'time250':dict(steps=250,pfr=True)}
VAES=('ema','mse')


class CheckedModel(CountedModel):
    def __init__(self, model):
        super().__init__(model)
        self.checked=[]
    def future_base(self,x,t,y):
        result=super().future_base(x,t,y)
        if self.prefix_batch_calls in (1,104):
            expected=self.model(x,t,y)[1]
            assert torch.equal(result,expected)
            self.checked.append(dict(prefix_call=self.prefix_batch_calls,images=len(x),time=float(t[0])))
        return result


def prepare():
    assert json.loads((RAW/'official_sit_pfr_upper_20260909/status.json').read_text())['phase']=='complete', 'Upper Euler cohort must finish first'
    old=json.loads((NATIVE/'request.json').read_text())
    assert json.loads((NATIVE/'status.json').read_text())['phase']=='complete'
    audit_path=ROOT/'docs/data/guidance_goal_20260909/official_sit_native_sde_audit.json'
    audit=json.loads(audit_path.read_text())
    assert audit['passed'] and audit['request_sha256']==file_sha256(NATIVE/'request.json')
    assert file_sha256(CKPT)==old['checkpoint_sha256']
    assert file_sha256(NATIVE/'inputs.npz')==old['input_file_sha256']
    for path,digest in old['sources'].items(): assert file_sha256(Path(path))==digest,path
    DATA.mkdir(parents=True,exist_ok=False)
    (DATA/'inputs.npz').write_bytes((NATIVE/'inputs.npz').read_bytes())
    sources=[Path(__file__),ROOT/'experiments/official_sit_sde_pfr_control_core_20260909.py',
        ROOT/'experiments/check_official_sit_sde_pfr_control_core_20260909.py',
        ROOT/'experiments/audit_official_sit_pfr_interface.py',
        ROOT/'docs/OFFICIAL_SIT_SDE_PFR_CONTROL_PROTOCOL_20260909_ZH.md']+[Path(p) for p in old['sources']]
    sources=list(dict.fromkeys(sources));(DATA/'sources').mkdir()
    for i,path in enumerate(sources): (DATA/'sources'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    rng_hashes={}
    for rank in range(4):
        metadata=json.loads((NATIVE/f'rank{rank}.json').read_text())
        assert metadata['pixel_file_sha256']==file_sha256(NATIVE/f'rank{rank}.npz')
        with np.load(NATIVE/f'rank{rank}.npz') as shard:
            for b,d in zip(range(rank,250,4),shard['brownian_rng_end_sha256']): rng_hashes[str(b)]=str(d)
    assert len(rng_hashes)==250
    costs={arm:counts(**cfg) for arm,cfg in ARMS.items()}
    assert {c['block_evaluations_per_image'] for c in costs.values()}=={13272}
    seeds=[bridge_seed(b) for b in range(250)];assert len(set(seeds))==250
    write_json(DATA/'request.json',dict(arms=ARMS,counts=costs,samples=1000,batch_size=4,
        input_file_sha256=file_sha256(DATA/'inputs.npz'),noise_sha256=old['noise_sha256'],label_sha256=old['label_sha256'],
        checkpoint_sha256=old['checkpoint_sha256'],vae_state_sha256=old['vae_state_sha256'],vae_files=old['vae_files'],
        brownian_seeds=old['brownian_seeds'],bridge_seeds=seeds,source_rng_end_hashes=rng_hashes,
        sources={str(p):file_sha256(p) for p in sources},native_request_sha256=file_sha256(NATIVE/'request.json'),
        native_audit_sha256=file_sha256(audit_path),native_csv_sha256=file_sha256(ROOT/'docs/data/guidance_goal_20260909/official_sit_native_sde.csv'),
        sampling_bank_role='reused_1k_discovery'))
    print('Two SDE arms frozen; native artifacts and equal block budgets verified.',flush=True)


@torch.inference_mode()
def worker(rank):
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    req=json.loads((DATA/'request.json').read_text())
    for p,d in req['sources'].items(): assert file_sha256(Path(p))==d,p
    assert file_sha256(DATA/'inputs.npz')==req['input_file_sha256']
    with np.load(DATA/'inputs.npz') as bank: noise,labels=bank['noise'],bank['labels']
    assert array_sha(noise)==req['noise_sha256'] and array_sha(labels)==req['label_sha256']
    model,meta=load_model(repo=REPO,checkpoint_path=CKPT,model_name='SiT-XL/2',encoder_depth=8,state_key='ema',device=torch.device('cuda'))
    from samplers import euler_maruyama_ig_sampler
    from experiments.check_official_sit_sde_pfr_control_core_20260909 import NoiseTorch
    from diffusers import AutoencoderKL
    vaes={n:AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{n}',local_files_only=True).cuda().eval().requires_grad_(False) for n in VAES}
    assert {n:state_sha(v) for n,v in vaes.items()}==req['vae_state_sha256']
    def decode(latent,vae):
        decoded=vae.decode(latent.float()/.18215).sample
        assert torch.isfinite(decoded).all()
        return (255.*((decoded+1)/2.)).clamp(0,255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
    def input_for(batch):
        i=batch*4
        return torch.from_numpy(noise[i:i+4]).cuda(),torch.from_numpy(labels[i:i+4]).cuda()
    def noises_for(batch,x):
        a,b,end=make_noises(x.double(),req['brownian_seeds'][batch],req['bridge_seeds'][batch])
        assert array_sha(end.cpu().numpy())==req['source_rng_end_hashes'][str(batch)], batch
        return a,b

    x,y=input_for(rank);a,b=noises_for(rank,x)
    torch.manual_seed(req['brownian_seeds'][rank])
    reference=euler_maruyama_ig_sampler(model,x,y,**KWARGS)
    old_parity=json.loads((NATIVE/f'parity_rank{rank}.json').read_text())
    assert array_sha(reference.cpu().numpy())==old_parity['native_fp64_endpoint_sha256']
    ordinary=sample_sde(CountedModel(model),x,y,a)
    assert torch.equal(reference,ordinary)
    checked=CheckedModel(model)
    zero=sample_sde(checked,x,y,a,pfr=True,response_scale=0.)
    assert torch.equal(reference,zero) and len(checked.checked)==2
    assert checked.sample_calls==4*422 and checked.prefix_sample_calls==4*182
    for name,vae in vaes.items():
        with np.load(NATIVE/name/'samples.npz') as f:
            assert np.array_equal(decode(reference,vae),f['arr_0'][rank*4:rank*4+4])
    proxy=NoiseTorch(b)
    original=euler_maruyama_ig_sampler
    injected=types.FunctionType(original.__code__,dict(original.__globals__,torch=proxy),
                                original.__name__,original.__defaults__,original.__closure__)
    reference281=injected(model,x,y,**dict(KWARGS,num_steps=281))
    assert next(proxy.noises,None) is None
    new281=sample_sde(CountedModel(model),x,y,b,steps=281)
    assert torch.equal(reference281,new281)
    preview={'ordinary281':new281,'time250':sample_sde(CountedModel(model),x,y,a,pfr=True)}
    preview_arrays={arm:z.float().cpu().numpy() for arm,z in preview.items()}
    assert all(np.isfinite(z).all() for z in preview_arrays.values())
    np.savez(DATA/f'preflight_rank{rank}.npz',**preview_arrays)
    write_json(DATA/f'parity_rank{rank}.json',dict(passed=True,rank=rank,indices=list(range(rank*4,rank*4+4)),
        native250_exact=True,copied250_exact=True,zero_response_exact=True,copied281_exact=True,
        vae_pixels_exact=True,prefix_checks=checked.checked,source_rng_exact=True,
        preview_sha256=file_sha256(DATA/f'preflight_rank{rank}.npz')))
    del reference,ordinary,zero,checked,reference281,new281,preview,a,b,x,y,proxy,injected
    while not (DATA/'parity_passed.json').exists(): time.sleep(.25)

    for arm,cfg in ARMS.items():
        directory=DATA/arm;directory.mkdir(exist_ok=True)
        images={n:[] for n in VAES};latents=[];indices=[]
        timings=dict(noise_coupling_seconds=0.,trajectory_seconds=0.,decode_seconds={n:0. for n in VAES})
        counter=CountedModel(model);n_batches=0
        for batch in range(rank,250,4):
            x,y=input_for(batch)
            torch.cuda.synchronize();began=time.perf_counter()
            a,b=noises_for(batch,x)
            torch.cuda.synchronize();timings['noise_coupling_seconds']+=time.perf_counter()-began
            torch.cuda.synchronize();began=time.perf_counter()
            latent=sample_sde(counter,x,y,a if cfg['pfr'] else b,**cfg)
            torch.cuda.synchronize();timings['trajectory_seconds']+=time.perf_counter()-began
            assert torch.isfinite(latent).all()
            z=latent.float();z_cpu=z.cpu().numpy()
            if batch==rank: assert np.array_equal(z_cpu,preview_arrays[arm]),(rank,arm)
            latents.append(z_cpu)
            for name,vae in vaes.items():
                torch.cuda.synchronize();began=time.perf_counter()
                images[name].append(decode(z,vae))
                torch.cuda.synchronize();timings['decode_seconds'][name]+=time.perf_counter()-began
            indices.extend(range(batch*4,batch*4+4));n_batches+=1
            if n_batches%8==0: print(json.dumps(dict(arm=arm,rank=rank,images=n_batches*4,trajectory_seconds=timings['trajectory_seconds'])),flush=True)
        cost=req['counts'][arm]
        assert counter.batch_calls==n_batches*cfg['steps']
        assert counter.sample_calls==len(indices)*cost['full_per_image']
        assert counter.prefix_sample_calls==len(indices)*cost['prefix_per_image']
        path=directory/f'rank{rank}.npz'
        np.savez(path,**{n:np.concatenate(v) for n,v in images.items()},latents=np.concatenate(latents),indices=np.array(indices,dtype=np.int64))
        write_json(directory/f'rank{rank}.json',dict(complete=True,arm=arm,rank=rank,samples=len(indices),batch_size=4,
            full_batch_calls=counter.batch_calls,full_sample_calls=counter.sample_calls,
            prefix_batch_calls=counter.prefix_batch_calls,prefix_sample_calls=counter.prefix_sample_calls,
            timings=timings,metadata=meta,pixel_file_sha256=file_sha256(path),
            noise_sha256=array_sha(noise[indices]),label_sha256=array_sha(labels[indices]),
            request_sha256=file_sha256(DATA/'request.json')))
        print(json.dumps(dict(arm=arm,rank=rank,complete=True,timings=timings)),flush=True)
        while not (directory/'advance.json').exists(): time.sleep(.25)
    for p,d in req['sources'].items(): assert file_sha256(Path(p))==d,p


def wait_files(paths,workers):
    while not all(p.is_file() for p in paths):
        for w in workers:
            if w.poll() is not None: raise RuntimeError(f'Worker {w.pid} exited at barrier, code {w.returncode}')
        time.sleep(1)


def main():
    torch.set_num_threads(4);prepare()
    req=json.loads((DATA/'request.json').read_text())
    workers,logs,results=[],[],[];began=time.perf_counter()
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    try:
        for rank in range(4):
            log=(DATA/f'worker{rank}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([sys.executable,str(Path(__file__)),'--rank',str(rank)],
                env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)),stdout=log,stderr=subprocess.STDOUT))
        write_json(DATA/'status.json',dict(phase='parity',pids=[w.pid for w in workers]))
        wait_files([DATA/f'parity_rank{r}.json' for r in range(4)],workers)
        parity=[json.loads((DATA/f'parity_rank{r}.json').read_text()) for r in range(4)]
        assert all(p['passed'] for p in parity)
        assert sorted(i for p in parity for i in p['indices'])==list(range(16))
        write_json(DATA/'parity_passed.json',dict(passed=True,unique_inputs=16,records=parity))
        print('16 native endpoints/pixels, copied281 and zero-PFR paths verified exactly.',flush=True)
        with np.load(DATA/'inputs.npz') as bank: noise,labels=bank['noise'],bank['labels']
        for arm,cfg in ARMS.items():
            directory=DATA/arm
            write_json(DATA/'status.json',dict(phase='sampling',arm=arm,pids=[w.pid for w in workers]))
            wait_files([directory/f'rank{r}.json' for r in range(4)],workers)
            images={n:np.empty((1000,256,256,3),dtype=np.uint8) for n in VAES}
            latents=np.empty((1000,4,32,32),dtype=np.float32);covered=np.zeros(1000,dtype=int);summaries=[]
            for rank in range(4):
                s=json.loads((directory/f'rank{rank}.json').read_text());summaries.append(s)
                assert s['complete'] and s['request_sha256']==file_sha256(DATA/'request.json')
                assert s['pixel_file_sha256']==file_sha256(directory/f'rank{rank}.npz')
                with np.load(directory/f'rank{rank}.npz') as shard:
                    idx=shard['indices']
                    assert np.array_equal(idx,np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,250,4)]))
                    assert array_sha(noise[idx])==s['noise_sha256'] and array_sha(labels[idx])==s['label_sha256']
                    for name in VAES: images[name][idx]=shard[name]
                    latents[idx]=shard['latents'];covered[idx]+=1
            assert np.all(covered==1) and np.isfinite(latents).all()
            cost=req['counts'][arm]
            assert sum(s['full_sample_calls'] for s in summaries)==1000*cost['full_per_image']
            assert sum(s['prefix_sample_calls'] for s in summaries)==1000*cost['prefix_per_image']
            np.savez(directory/'latents.npz',latents=latents,labels=labels)
            for name in VAES:
                sub=directory/name;sub.mkdir()
                np.savez(sub/'samples.npz',arr_0=images[name])
                write_json(DATA/'status.json',dict(phase='evaluation',arm=arm,vae=name,pids=[w.pid for w in workers]))
                cmd=[sys.executable,str(ROOT/'experiments/evaluate_raev2_official_samples.py'),
                    '--branch',arm+'_'+name+'='+str(sub/'samples.npz'),'--output',str(sub/'fid.csv'),
                    '--batch-size','32','--device','cuda','--feature-cache-dir',str(sub/'features')]
                with (sub/'evaluation.log').open('x') as log:
                    subprocess.run(cmd,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
                metrics=json.loads((sub/'fid.json').read_text())[0]
                assert metrics['sample_sha256']==file_sha256(sub/'samples.npz')
                row=dict(arm=arm,vae=name,metrics=metrics,counts=cost,
                    trajectory_gpu_seconds=sum(s['timings']['trajectory_seconds'] for s in summaries),
                    noise_coupling_gpu_seconds=sum(s['timings']['noise_coupling_seconds'] for s in summaries),
                    decode_gpu_seconds=sum(s['timings']['decode_seconds'][name] for s in summaries))
                results.append(row);write_json(sub/'result.json',row);write_json(DATA/'results.json',results)
                print(json.dumps(row),flush=True)
            write_json(directory/'advance.json',dict(complete=True))
        for w in workers:
            if w.wait()!=0: raise RuntimeError('Worker exit failure')
        for p,d in req['sources'].items(): assert file_sha256(Path(p))==d,p
        write_json(DATA/'status.json',dict(phase='complete',seconds=time.perf_counter()-began,research_goal_achieved=False,results=results))
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
