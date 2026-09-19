"""Freeze and run four selected XL configurations on a genuinely new 5K bank."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_official_sit_pfr_upper_20260909 import sample,CountedModel
from experiments.run_official_sit_native_sde_20260909 import state_sha
from experiments.sample_official_sit_pfr_symmetric_20260909 import array_sha,write_json,CKPT
from experiments.run_internal_guidance_sit_audit import load_model
from experiments.raev2_training_core import file_sha256 as sha

RAW=Path('/home/zhoushunyu/data/eqvae/experiments')
DATA=RAW/'official_sit_xl_selected_5k_20260909'
UPPER=RAW/'official_sit_pfr_upper_20260909'
OUT=ROOT/'docs/data/guidance_goal_20260909'
ARMS={'ordinary250':('ordinary',2.5),'projected200':('projected',2.),
      'ordinary200':('ordinary',2.),'projected150':('projected',1.5)}
N=5000
REFS={arm:UPPER/arm for arm in ('ordinary250','projected200','ordinary200')}
REFS['projected150']=RAW/'official_sit_pfr_symmetric_20260909/projected150'


def batch_seed(batch):
    data=f'official_sit_xl_selected_5k_20260909/confirmation/noise/batch/{batch}'.encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8],'big')%(2**63)


def prepare():
    torch.set_num_threads(4)
    assert json.loads((UPPER/'status.json').read_text())['phase']=='complete'
    old=json.loads((UPPER/'request.json').read_text())
    for p,d in old['sources'].items(): assert sha(Path(p))==d,p
    assert sha(CKPT)==old['checkpoint_sha256']
    assert json.loads((OUT/'official_sit_pfr_extended_audit.json').read_text())['passed']
    assert json.loads((OUT/'official_sit_xl_multimetrics_audit.json').read_text())['passed']
    history_path=ROOT/'experiments/results/terminal_defect_20260908/official_sit_pfr_noise.json'
    history=json.loads(history_path.read_text())
    assert history['cross_bank_overlap']==0 and [b['samples'] for b in history['banks']]==[1000,5000]
    with np.load(UPPER/'inputs.npz') as f: old_noise,old_labels=f['noise'],f['labels']
    assert array_sha(old_noise)==history['banks'][0]['noise_sha256']
    assert [array_sha(z) for z in old_noise]==history['banks'][0]['per_image_sha256']
    for arm in ('pfr','ordinary115'):
        s=json.loads((RAW/'official_sit_pfr_5k_20260908'/arm/'quality/summary.json').read_text())
        assert s['noise_sha256']==history['banks'][1]['noise_sha256']
        assert s['label_sha256']==history['banks'][1]['label_sha256']
    seeds=[batch_seed(b) for b in range(N//4)];assert len(set(seeds))==N//4
    chunks=[]
    for seed in seeds:
        gen=torch.Generator(device='cpu').manual_seed(seed)
        chunks.append(torch.randn(4,4,32,32,generator=gen,dtype=torch.float32).numpy())
    noise=np.concatenate(chunks);labels=np.arange(N,dtype=np.int64)%1000
    individual=[array_sha(z) for z in noise];assert len(set(individual))==N
    assert np.array_equal(np.bincount(labels),np.full(1000,5))
    overlap={str(b['seed']):len(set(individual)&set(b['per_image_sha256'])) for b in history['banks']}
    assert set(overlap.values())=={0}
    DATA.mkdir(parents=True,exist_ok=False)
    np.savez(DATA/'inputs.npz',noise=noise,labels=labels)
    (DATA/'old_inputs.npz').write_bytes((UPPER/'inputs.npz').read_bytes())
    sources=[Path(__file__),ROOT/'docs/OFFICIAL_SIT_XL_SELECTED_5K_PROTOCOL_20260909_ZH.md',
        history_path,OUT/'official_sit_pfr_extended.csv',OUT/'official_sit_pfr_extended_audit.json',
        OUT/'official_sit_xl_multimetrics.csv',OUT/'official_sit_xl_multimetrics_audit.json']+[Path(p) for p in old['sources']]
    sources=list(dict.fromkeys(sources));(DATA/'sources').mkdir()
    for i,p in enumerate(sources): (DATA/'sources'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    references={}
    for arm,path in REFS.items():
        m=json.loads((path/'fid.json').read_text())[0]
        assert sha(path/'samples.npz')==m['sample_sha256']
        references[arm]=dict(directory=str(path),pixel_sha256=m['sample_sha256'])
    write_json(DATA/'request.json',dict(arms=ARMS,samples=N,batch_size=4,batch_seeds=seeds,
        input_file_sha256=sha(DATA/'inputs.npz'),noise_sha256=array_sha(noise),label_sha256=array_sha(labels),
        per_image_noise_sha256=individual,historical_bank_overlap=overlap,historical_noise_audit_sha256=sha(history_path),
        checkpoint_sha256=old['checkpoint_sha256'],vae_state_sha256=old['vae_state_sha256'],
        old_input_file_sha256=sha(DATA/'old_inputs.npz'),sources={str(p):sha(p) for p in sources},references=references,
        primary_contrast=['projected200','ordinary250'],sensitivity_contrast=['projected150','ordinary200'],
        sampling_bank_role='new_5k_fixed_configuration_confirmation',selection_status='Ordinary 1K minimum remains at boundary'))
    write_json(DATA/'status.json',dict(phase='prepared',research_goal_achieved=False,samples=N,arms=ARMS,overlap=overlap))
    print(json.dumps(dict(prepared=True,noise_sha256=array_sha(noise),label_sha256=array_sha(labels),overlap=overlap)),flush=True)


@torch.inference_mode()
def worker(rank):
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    req=json.loads((DATA/'request.json').read_text())
    assert req['arms']=={a:list(v) for a,v in ARMS.items()} and req['samples']==N
    for p,d in req['sources'].items(): assert sha(Path(p))==d,p
    assert sha(DATA/'inputs.npz')==req['input_file_sha256'] and sha(DATA/'old_inputs.npz')==req['old_input_file_sha256']
    with np.load(DATA/'inputs.npz') as f: noise,labels=f['noise'],f['labels']
    with np.load(DATA/'old_inputs.npz') as f: old_noise,old_labels=f['noise'],f['labels']
    assert array_sha(noise)==req['noise_sha256'] and array_sha(labels)==req['label_sha256']
    model,meta=load_model(repo=ROOT/'research_repos/internal_guidance_study/Internal-Guidance/SiT',
        checkpoint_path=CKPT,model_name='SiT-XL/2',encoder_depth=8,state_key='ema',device=torch.device('cuda'))
    from diffusers import AutoencoderKL
    vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse',local_files_only=True).cuda().eval().requires_grad_(False)
    assert state_sha(vae)==req['vae_state_sha256']
    start=rank*4;previews={};records={}
    for arm,(method,scale) in ARMS.items():
        old_pixels,_,_,_=sample(CountedModel(model),vae,torch.from_numpy(old_noise[start:start+4]).cuda(),
                                torch.from_numpy(old_labels[start:start+4]).cuda(),scale,method)
        with np.load(Path(req['references'][arm]['directory'])/'samples.npz') as f:
            assert np.array_equal(old_pixels,f['arr_0'][start:start+4]),(rank,arm)
        previews[arm],_,_,_=sample(CountedModel(model),vae,torch.from_numpy(noise[start:start+4]).cuda(),
                                   torch.from_numpy(labels[start:start+4]).cuda(),scale,method)
        records[arm]=dict(old_pixels_exact=True,old_pixel_array_sha256=array_sha(old_pixels),
                           new_pixel_array_sha256=array_sha(previews[arm]))
    np.savez(DATA/f'preflight_rank{rank}.npz',**previews)
    write_json(DATA/f'parity_rank{rank}.json',dict(passed=True,rank=rank,indices=list(range(start,start+4)),records=records,
        preflight_sha256=sha(DATA/f'preflight_rank{rank}.npz')))
    while not (DATA/'parity_passed.json').exists(): time.sleep(.25)
    for arm,(method,scale) in ARMS.items():
        directory=DATA/arm;directory.mkdir(exist_ok=True)
        images,endpoints,stats,indices=[],[],[],[];counter=CountedModel(model);prefix_calls=0;batches=0
        torch.cuda.synchronize();began=time.perf_counter()
        for batch in range(rank,N//4,4):
            start=batch*4
            pixels,z,diagnostics,partial=sample(counter,vae,torch.from_numpy(noise[start:start+4]).cuda(),
                                              torch.from_numpy(labels[start:start+4]).cuda(),scale,method)
            if batch==rank: assert np.array_equal(pixels,previews[arm]),(rank,arm)
            images.append(pixels);indices.extend(range(start,start+4));prefix_calls+=partial;batches+=1
            if z is not None: endpoints.append(z);stats.append(diagnostics)
            if batches%32==0: print(json.dumps(dict(arm=arm,rank=rank,images=batches*4,seconds=time.perf_counter()-began)),flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-began
        full,partial=(115,0) if method=='ordinary' else (100,50)
        assert counter.sample_calls==len(indices)*full and prefix_calls==batches*partial
        path=directory/f'rank{rank}.npz'
        payload=dict(arr_0=np.concatenate(images),indices=np.array(indices,dtype=np.int64))
        if endpoints: payload.update(latents=np.concatenate(endpoints),diagnostics=np.concatenate(stats))
        np.savez(path,**payload)
        write_json(directory/f'rank{rank}.json',dict(complete=True,arm=arm,rank=rank,method=method,scale=scale,
            samples=len(indices),batch_size=4,full_batch_calls=counter.batch_calls,full_sample_calls=counter.sample_calls,
            prefix_batch_calls=prefix_calls,prefix_sample_calls=prefix_calls*4,inference_seconds=seconds,
            pixel_file_sha256=sha(path),noise_sha256=array_sha(noise[indices]),label_sha256=array_sha(labels[indices]),
            request_sha256=sha(DATA/'request.json'),metadata=meta))
        print(json.dumps(dict(arm=arm,rank=rank,complete=True,seconds=seconds)),flush=True)
        while not (directory/'advance.json').exists(): time.sleep(.25)
    for p,d in req['sources'].items(): assert sha(Path(p))==d,p


def wait_files(paths,workers):
    while not all(p.is_file() for p in paths):
        for w in workers:
            if w.poll() is not None: raise RuntimeError(f'Worker {w.pid} exited at barrier: {w.returncode}')
        time.sleep(1)


def main():
    torch.set_num_threads(4)
    assert json.loads((RAW/'official_sit_sde_pfr_control_20260909/status.json').read_text())['phase']=='complete', 'Finish SDE cohort before starting 5K'
    assert json.loads((DATA/'status.json').read_text())['phase']=='prepared'
    req=json.loads((DATA/'request.json').read_text())
    for p,d in req['sources'].items(): assert sha(Path(p))==d,p
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
        assert all(p['passed'] and set(p['records'])==set(ARMS) for p in parity)
        assert sorted(i for p in parity for i in p['indices'])==list(range(16))
        write_json(DATA/'parity_passed.json',dict(passed=True,unique_inputs_per_bank=16,records=parity))
        print('Old-bank pixel parity and fixed new-bank previews passed for all four configurations.',flush=True)
        with np.load(DATA/'inputs.npz') as f: noise,labels=f['noise'],f['labels']
        for arm,(method,scale) in ARMS.items():
            directory=DATA/arm;write_json(DATA/'status.json',dict(phase='sampling',arm=arm,pids=[w.pid for w in workers]))
            wait_files([directory/f'rank{r}.json' for r in range(4)],workers)
            images=np.empty((N,256,256,3),dtype=np.uint8);covered=np.zeros(N,dtype=int);summaries=[]
            extra={} if method=='ordinary' else dict(latents=np.empty((N,4,32,32),dtype=np.float32),diagnostics=np.empty((N,5,4),dtype=np.float32))
            for rank in range(4):
                s=json.loads((directory/f'rank{rank}.json').read_text());summaries.append(s)
                assert s['method']==method and s['scale']==scale and s['request_sha256']==sha(DATA/'request.json')
                assert s['pixel_file_sha256']==sha(directory/f'rank{rank}.npz')
                with np.load(directory/f'rank{rank}.npz') as f:
                    idx=f['indices'];assert np.array_equal(idx,np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,N//4,4)]))
                    assert array_sha(noise[idx])==s['noise_sha256'] and array_sha(labels[idx])==s['label_sha256']
                    assert f['arr_0'].dtype==np.uint8
                    images[idx]=f['arr_0'];covered[idx]+=1
                    for key,value in extra.items():
                        assert np.isfinite(f[key]).all();value[idx]=f[key]
            assert np.all(covered==1)
            full,partial=(115,0) if method=='ordinary' else (100,50)
            assert sum(s['full_sample_calls'] for s in summaries)==full*N
            assert sum(s['prefix_sample_calls'] for s in summaries)==partial*N
            np.savez(directory/'samples.npz',arr_0=images)
            if extra: np.savez(directory/'diagnostics.npz',**extra,labels=labels)
            write_json(DATA/'status.json',dict(phase='evaluation',arm=arm,pids=[w.pid for w in workers]))
            cmd=[sys.executable,str(ROOT/'experiments/evaluate_raev2_official_samples.py'),
                '--branch',arm+'='+str(directory/'samples.npz'),'--output',str(directory/'fid.csv'),
                '--batch-size','32','--device','cuda','--feature-cache-dir',str(directory/'features')]
            with (directory/'evaluation.log').open('x') as log:
                subprocess.run(cmd,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics=json.loads((directory/'fid.json').read_text())[0]
            assert metrics['sample_sha256']==sha(directory/'samples.npz')
            row=dict(arm=arm,method=method,scale=scale,metrics=metrics,samples=N,full_per_image=full,prefix_per_image=partial,
                gpu_inference_seconds_sum=sum(s['inference_seconds'] for s in summaries),inference_seconds_max=max(s['inference_seconds'] for s in summaries))
            if extra: row['diagnostic_sha256']=sha(directory/'diagnostics.npz')
            results.append(row);write_json(directory/'result.json',row);write_json(DATA/'results.json',results)
            print(json.dumps(row),flush=True);write_json(directory/'advance.json',dict(complete=True))
        for w in workers:
            if w.wait()!=0: raise RuntimeError('Worker exit failure')
        for p,d in req['sources'].items(): assert sha(Path(p))==d,p
        write_json(DATA/'status.json',dict(phase='complete',seconds=time.perf_counter()-began,research_goal_achieved=False,results=results))
    except BaseException as error:
        write_json(DATA/'status.json',dict(phase='failed',error=repr(error)));raise
    finally:
        for w in workers:
            if w.poll() is None: w.terminate()
        for w in workers: w.wait()
        for log in logs: log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-only',action='store_true');group.add_argument('--run-prepared',action='store_true')
    group.add_argument('--rank',type=int,choices=range(4));args=parser.parse_args()
    if args.prepare_only: prepare()
    elif args.run_prepared: main()
    else: worker(args.rank)
