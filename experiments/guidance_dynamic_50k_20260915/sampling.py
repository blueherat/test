import os
import subprocess
import time
import numpy as np
import torch
from . import config as k
from .models import Adapter


def canonical(model,method,tick):
    if tick:return k.point_key(method,tick)
    base=k.ARMS[method]['base']
    if base=='strong':return k.point_key('strong',0)
    if base=='ig':return k.point_key('native',round(k.settings(model)['alpha']*40))
    return k.point_key('cfg_native',round(k.settings(model)['cfg_extra']*40))


def quality_inputs(model):
    root=k.model_root(model)/'quality_inputs'
    if (root/'complete.json').exists():return
    root.mkdir(parents=True,exist_ok=True)
    old=(k.EXPS/'guidance_loss_50k_20260914/sit_small/screen1000/inputs') if model=='sit_small' else (
        k.EXPS/'jit_readout_transfer_20260913/screen_1000/inputs')
    previous=np.load(old/'noise.npy',mmap_mode='r');previous_labels=np.load(old/'labels.npy')
    assert len(previous)==len(previous_labels)==1000
    cfg=k.settings(model);rng=np.random.default_rng(2026091562+(model=='jit'))
    output=np.lib.format.open_memmap(root/'noise.npy',mode='w+',dtype=np.float32,shape=(5000,*cfg['shape']))
    output[:1000]=previous
    for start in range(1000,5000,32):output[start:start+32]=rng.standard_normal((min(32,5000-start),*cfg['shape']),dtype=np.float32)
    output.flush();del output
    labels=np.concatenate([previous_labels,rng.permutation(np.arange(4000)%cfg['classes'])])
    np.save(root/'labels.npy',labels)
    k.atomic(root/'complete.json',dict(complete=True,samples=5000,first_1000_preserved=True,
        request_sha256=k.sha(k.ROOT/'request.json'),files={str(root/f):k.sha(root/f) for f in ('noise.npy','labels.npy')},
        previous_files={str(old/f):k.sha(old/f) for f in ('noise.npy','labels.npy')}))


def field(adapter,head,method,coefficient,z,t,left):
    spec=k.ARMS.get(method,dict(base='ig' if method=='native' else 'cfg' if method=='cfg_native' else 'strong',loss='none'))
    active=left<.5
    factor=(6/7 if left<.25 else 1.) if adapter.name=='sit_small' else 1.
    full,weak=adapter.full(z,t,adapter.labels,native=spec['base']=='ig' and active)
    features=dict(adapter.values)
    base=full
    if weak is not None:
        alpha=coefficient if method=='native' else adapter.cfg['alpha']
        base=full+alpha*factor*(full-weak)
    if spec['base']=='cfg':
        enabled=left<.75 if adapter.name=='sit_small' else (.1<float(t)<1.)
        # Retain the official JiT CFG query schedule, including the inactive boundary queries.
        if enabled or adapter.name=='jit':
            uncond,_=adapter.full(z,t,torch.full_like(adapter.labels,adapter.cfg['classes']))
            beta=coefficient if method=='cfg_native' else adapter.cfg['cfg_extra']
            base=base+(beta if enabled else 0.)*(full-uncond)
    if head is not None and active and coefficient:
        pred=adapter.unpatch(head(features['context'],features['condition'])).float()
        adapter.head_calls+=1
        if spec['loss'] in k.WEAK_LOSSES:base=base+coefficient*factor*(full-pred)
        else:base=base+coefficient*factor*(4. if spec['loss']=='covariance' else 1.)*pred
    return adapter.native_to_velocity(base,z,t)


@torch.inference_mode()
def integrate(adapter,head,method,coefficient,noise,labels):
    adapter.labels=labels;z=noise.clone();before=adapter.counts()
    cfg=k.ARMS.get(method,{}).get('base')=='cfg' or method=='cfg_native'
    steps=64 if adapter.name=='sit_small' else 50 if cfg else 100
    grid=torch.linspace(0,1,steps+1,device='cuda')
    with adapter.autocast():
        for index,(t,u) in enumerate(zip(grid[:-1],grid[1:])):
            left=float(t);v=field(adapter,head,method,coefficient,z,t,left)
            predicted=z+(u-t)*v
            heun=adapter.name=='sit_small' or (cfg and index<steps-1)
            if heun:
                second=field(adapter,head,method,coefficient,predicted,u,left)
                z=z+((u-t)/2)*(v+second)
            else:z=predicted
            if not torch.isfinite(z).all() or z.abs().max()>1e6:raise FloatingPointError(f'Divergent sample at step {index}')
    after=adapter.counts()
    counts={key:after[key]-before[key] for key in ('full','head','native_head')}
    counts['blocks']=[a-b for a,b in zip(after['blocks'],before['blocks'])]
    assert counts['blocks']==[counts['full']]*12,counts
    return z,counts


def point_root(model,point):return k.model_root(model)/'points'/point
def batch_starts(model,n,rank):return tuple(range(0,n,k.settings(model)['sample_batch']))[rank::4]


def verify_batch(path,point,provenance):
    row=k.read(path.with_suffix('.json'))
    assert row['request_sha256']==k.sha(k.ROOT/'request.json') and row['point']==point
    assert row['head_provenance']==provenance and row['sha256']==k.sha(path)
    return row


@torch.inference_mode()
def sample(model,point,n,rank):
    root=point_root(model,point);stage=root/f'n{n}'
    stage.mkdir(parents=True,exist_ok=True)
    method,tick=k.parse_point(point)
    adapter=Adapter(model);head=adapter.loaded_head(method) if method in k.ARMS and tick else None
    provenance={} if head is None else {str(k.model_root(model)/'training'/method/'head.pt'):
        k.sha(k.model_root(model)/'training'/method/'head.pt')}
    bank=k.model_root(model)/'quality_inputs'
    noise=np.load(bank/'noise.npy',mmap_mode='r');labels=np.load(bank/'labels.npy')
    records=[];valid=True;error=None
    try:
        for index,start in enumerate(batch_starts(model,n,rank)):
            k.check_stop();path=root/'batches'/f'{start:05d}.npz'
            if path.exists() and not path.with_suffix('.json').exists():path.unlink()
            if not path.exists():
                batch=k.settings(model)['sample_batch']
                z=torch.from_numpy(np.array(noise[start:start+batch])).cuda();y=torch.from_numpy(labels[start:start+batch]).cuda()
                begin=time.perf_counter();result,counts=integrate(adapter,head,method,tick/40,z,y)
                pixels=adapter.pixels(result);torch.cuda.synchronize()
                path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
                payload=dict(arr_0=pixels,labels=y.cpu().numpy(),start=start,
                    noise_sha256=k.array_sha(noise[start:start+len(y)]),seconds=time.perf_counter()-begin,
                    state_finite=True,state_sha256=k.array_sha(result.cpu().numpy()))
                if model=='sit_small':payload['latents']=result.cpu().numpy()
                with tmp.open('wb') as f:np.savez_compressed(f,**payload)
                tmp.replace(path)
                k.atomic(path.with_suffix('.json'),dict(sha256=k.sha(path),point=point,head_provenance=provenance,
                    request_sha256=k.sha(k.ROOT/'request.json'),counts=counts))
            meta=verify_batch(path,point,provenance);records.append(dict(file=str(path),sha256=meta['sha256']))
            if index%10==0:k.atomic(stage/f'progress_rank{rank}.json',dict(pid=os.getpid(),batches=index+1,total_batches=len(batch_starts(model,n,rank))))
    except FloatingPointError as exc:valid=False;error=str(exc)
    k.atomic(stage/f'rank{rank}.json',dict(complete=True,valid=valid,error=error,point=point,rank=rank,n=n,
        request_sha256=k.sha(k.ROOT/'request.json'),head_provenance=provenance,records=records))


def collect(model,point,n):
    root=point_root(model,point);stage=root/f'n{n}'
    rows=[k.read(stage/f'rank{rank}.json') for rank in range(4)]
    assert all(row['complete'] and row['n']==n and row['point']==point for row in rows)
    valid=all(row['valid'] for row in rows)
    if not valid:
        k.atomic(stage/'summary.json',dict(complete=True,valid=False,point=point,n=n,
            request_sha256=k.sha(k.ROOT/'request.json'),reason='Numerical divergence'))
        return
    provenance=rows[0]['head_provenance'];assert all(r['head_provenance']==provenance for r in rows)
    bank=k.model_root(model)/'quality_inputs'
    labels=np.load(bank/'labels.npy');noise=np.load(bank/'noise.npy',mmap_mode='r')
    pixels=[];records=[];coverage=[];expected_counts=None
    for start in range(0,n,k.settings(model)['sample_batch']):
        path=root/'batches'/f'{start:05d}.npz';meta=verify_batch(path,point,provenance)
        if expected_counts is None:expected_counts=meta['counts']
        assert expected_counts==meta['counts']
        with np.load(path) as value:
            size=len(value['arr_0']);assert int(value['start'])==start
            np.testing.assert_array_equal(value['labels'],labels[start:start+size])
            assert str(value['noise_sha256'])==k.array_sha(noise[start:start+size])
            assert bool(value['state_finite']) and value['arr_0'].dtype==np.uint8
            if 'latents' in value:assert np.isfinite(value['latents']).all()
            coverage.extend(range(start,start+size));pixels.append(value['arr_0'])
        records.append(dict(file=str(path),sha256=meta['sha256']))
    assert coverage==list(range(n))
    path=stage/'samples.npz';tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,arr_0=np.concatenate(pixels))
    tmp.replace(path)
    if n==5000:
        initial=k.read(root/'n1000/summary.json')
        assert initial['records']==records[:len(initial['records'])]
    k.atomic(stage/'summary.json',dict(complete=True,valid=True,point=point,n=n,
        request_sha256=k.sha(k.ROOT/'request.json'),head_provenance=provenance,counts=expected_counts,
        records=records,samples_path=str(path),samples_sha256=k.sha(path),first_1000_reused=n==5000,
        bank_sha256=k.sha(bank/'complete.json')))


def evaluate(model,point,n):
    root=point_root(model,point)/f'n{n}';summary=k.read(root/'summary.json')
    if not summary['valid']:
        k.atomic(root/'metrics.json',dict(summary,fid=None,inception_score=None));return
    assert k.sha(summary['samples_path'])==summary['samples_sha256']
    if model=='sit_small':
        ref=k.EXPS.parent/'imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'
        command=['/data/shared/envs/adm-fid/bin/python',str(k.WORK/'experiments/compute_adm_fid.py'),
            '--reference',str(ref),'--samples',summary['samples_path'],'--output',str(root/'adm.json'),
            '--activations-output',str(root/'activations.npz'),'--batch-size','32']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
            TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1')
    else:
        command=[k.PYTHON,str(k.WORK/'experiments/evaluate_raev2_official_samples.py'),
            '--branch',point+'='+summary['samples_path'],'--output',str(root/'official.csv'),
            '--batch-size','64','--device','cuda','--feature-cache-dir',str(root/'feature_cache')]
        env=dict(os.environ)
    with (root/'evaluation.log').open('a') as f:subprocess.run(command,cwd=k.WORK,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    result=k.read(root/'adm.json') if model=='sit_small' else k.read(root/'official.json')[0]
    assert np.isfinite(result['fid']) and np.isfinite(result['inception_score'])
    k.atomic(root/'metrics.json',dict(summary,fid=result['fid'],inception_score=result['inception_score']))
    print(model,point,n,result['fid'],flush=True)
