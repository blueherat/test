import io
import os
import time
import numpy as np
import torch
from . import config as k
from .models import Adapter
from .sampling import integrate


@torch.inference_mode()
def weak_integrate(adapter,head,noise,labels):
    z=noise.clone();steps=64 if adapter.name=='sit_small' else 100
    grid=torch.linspace(0,1,steps+1,device='cuda')
    def field(state,t):
        features=adapter.features(state,t.expand(len(state)),labels)
        raw=adapter.unpatch(head(features['context'],features['condition'])).float()
        return adapter.native_to_velocity(raw,state,t)
    with adapter.autocast():
        for t,u in zip(grid[:-1],grid[1:]):
            v=field(z,t);predicted=z+(u-t)*v
            z=z+((u-t)/2)*(v+field(predicted,u)) if adapter.name=='sit_small' else predicted
            if not torch.isfinite(z).all() or z.abs().max()>1e6:raise FloatingPointError('Standalone weak endpoint diverged')
    return z


@torch.inference_mode()
def generate(model,source,split,rank):
    root=k.model_root(model)/'endpoints'/source/split;(root/'batches').mkdir(parents=True,exist_ok=True)
    labels=np.load(k.model_root(model)/'data'/f'{split}_labels.npy')
    adapter=Adapter(model);head=adapter.loaded_head('self') if source=='weak' else None
    provenance={} if head is None else {str(k.model_root(model)/'training/self/head.pt'):
        k.sha(k.model_root(model)/'training/self/head.pt')}
    batch=k.settings(model)['endpoint_batch'];records=[]
    for index,start in enumerate(tuple(range(0,len(labels),batch))[rank::4]):
        k.check_stop();path=root/'batches'/f'{start:08d}.npz';meta_path=path.with_suffix('.json')
        if not meta_path.exists():
            size=min(batch,len(labels)-start)
            rng=np.random.default_rng(np.random.SeedSequence([2026091581,k.MODELS.index(model),
                ('strong','weak','ig','cfg').index(source),int(split=='validation'),start]))
            initial=rng.standard_normal((size,*k.settings(model)['shape']),dtype=np.float32)
            z=torch.from_numpy(initial).cuda();y=torch.from_numpy(labels[start:start+size]).cuda()
            begin=time.perf_counter()
            if source=='weak':value=weak_integrate(adapter,head,z,y)
            else:
                method={'strong':'strong','ig':'native','cfg':'cfg_native'}[source]
                coefficient=adapter.cfg['cfg_extra'] if source=='cfg' else adapter.cfg['alpha']
                value,_=integrate(adapter,None,method,coefficient,z,y)
            tmp=path.with_suffix('.tmp');files={}
            if model=='sit_small':
                native=value.float().cpu().numpy()
                with tmp.open('wb') as f:np.savez(f,latents=native,labels=labels[start:start+size],start=start)
            else:
                from PIL import Image
                pixels=adapter.pixels(value);native=pixels.astype(np.float32).transpose(0,3,1,2)/127.5-1
                binary=path.with_suffix('.bin');binary_tmp=binary.with_suffix('.tmpbin')
                offsets=[];sizes=[]
                with binary_tmp.open('wb') as f:
                    for pixel in pixels:
                        stream=io.BytesIO();Image.fromarray(pixel).save(stream,format='PNG',compress_level=1)
                        payload=stream.getvalue();offsets.append(f.tell());sizes.append(len(payload));f.write(payload)
                binary_tmp.replace(binary);files[str(binary)]=k.sha(binary)
                with tmp.open('wb') as f:np.savez(f,offsets=offsets,sizes=sizes,labels=labels[start:start+size],start=start,
                    sum=native.sum(0,dtype=np.float64),square_sum=np.square(native,dtype=np.float64).sum(0))
            tmp.replace(path);files[str(path)]=k.sha(path)
            k.atomic(meta_path,dict(complete=True,start=start,size=size,files=files,head_provenance=provenance,
                request_sha256=k.sha(k.ROOT/'request.json'),initial_noise_sha256=k.array_sha(initial),seconds=time.perf_counter()-begin))
        meta=k.read(meta_path);assert meta['head_provenance']==provenance and meta['request_sha256']==k.sha(k.ROOT/'request.json')
        records.append(dict(file=str(meta_path),sha256=k.sha(meta_path)))
        if index%10==0:k.atomic(root/f'progress_rank{rank}.json',dict(pid=os.getpid(),batches=index+1,
            batches_total=len(tuple(range(0,len(labels),batch))[rank::4]),samples_total=len(labels)))
    k.atomic(root/f'rank{rank}.json',dict(complete=True,rank=rank,records=records,request_sha256=k.sha(k.ROOT/'request.json')))


def collect(model,source,split):
    root=k.model_root(model)/'endpoints'/source/split
    labels=np.load(k.model_root(model)/'data'/f'{split}_labels.npy');cfg=k.settings(model)
    values=np.lib.format.open_memmap(root/'clean.tmp.npy',mode='w+',dtype=np.float32,shape=(len(labels),*cfg['shape'])) if model=='sit_small' else None
    total=np.zeros(cfg['shape'],dtype=np.float64);squares=total.copy();records=[];coverage=[]
    for start in range(0,len(labels),cfg['endpoint_batch']):
        path=root/'batches'/f'{start:08d}.npz';meta=k.read(path.with_suffix('.json'))
        assert meta['request_sha256']==k.sha(k.ROOT/'request.json') and meta['start']==start
        for filename,digest in meta['files'].items():assert k.sha(filename)==digest
        with np.load(path) as value:
            size=len(value['labels']);np.testing.assert_array_equal(value['labels'],labels[start:start+size])
            if model=='sit_small':
                native=value['latents'];assert np.isfinite(native).all();values[start:start+size]=native
                total+=native.sum(0,dtype=np.float64);squares+=np.square(native,dtype=np.float64).sum(0)
            else:total+=value['sum'];squares+=value['square_sum']
            coverage.extend(range(start,start+size))
        records.append(dict(file=str(path.with_suffix('.json')),sha256=k.sha(path.with_suffix('.json'))))
    assert coverage==list(range(len(labels)))
    files={}
    if values is not None:
        values.flush();del values;(root/'clean.tmp.npy').replace(root/'clean.npy');files[str(root/'clean.npy')]=k.sha(root/'clean.npy')
    tau=.5*float(np.sqrt(np.maximum(squares/len(labels)-(total/len(labels))**2,0).mean()))
    k.atomic(root/'complete.json',dict(complete=True,samples=len(labels),source=source,split=split,tau=tau,
        request_sha256=k.sha(k.ROOT/'request.json'),files=files,records=records,
        storage='continuous float32 latent' if model=='sit_small' else 'ordinary uint8 model output, losslessly packed PNG'))
