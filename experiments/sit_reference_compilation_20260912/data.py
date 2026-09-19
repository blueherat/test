from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from . import catalog as c, models
from experiments.sit_measure_guidance_20260912 import data as real_data
from experiments.sit_prefix_coarse_20260912 import core as prefix_core
from experiments.sit_strong_reference_20260912 import train as old_train
from experiments.lifting_scale_sweep_20260909 import Runtime, SMALL_CKPT, SMALL_HEAD, atomic, read, sha


def sources():
    return sorted(set([*Path(__file__).resolve().parent.glob('*.py'),c.PROTOCOL,
        Path(models.heads.__file__),Path(real_data.__file__),Path(prefix_core.__file__),
        Path(prefix_core.models.__file__),Path(prefix_core.catalog.__file__),Path(old_train.__file__),
        c.WORK/'experiments/sit_strong_reference_20260912/run.py']))


def prepare():
    c.ROOT.mkdir(parents=True,exist_ok=True)
    if (c.ROOT/'data_request.json').exists():return verify()
    parent=old_train.verify()
    receipt=read(c.PREFIX/'training/native/complete.json')
    assert receipt['passed'] and receipt['checkpoint_sha256']==sha(c.PREFIX/'training/native/model.pt')
    assets=dict(parent['assets']);assets.update({str(p):sha(p) for p in
        [c.PREFIX/'training/native/model.pt',c.PREFIX/'training/native/complete.json']})
    inputs={str(real_data.ROOT/name):sha(real_data.ROOT/name) for name in
        ('train_moments.npy','validation_moments.npy','class_means.npy','local_partners.npy','random_partners.npy','heat_sigma.npy')}
    source_hashes=dict(parent['sources']);source_hashes.update({str(p):sha(p) for p in sources()})
    request=dict(sources=source_hashes,assets=assets,input_files=inputs,
        old_stop_markers=parent['old_stop_markers'],data_seed=c.DATA_SEED,
        trajectories=c.TRAJECTORIES,train_trajectories=c.TRAIN_TRAJECTORIES,
        trajectory_target='frozen successful native prefix at guided predictor states',
        codewords=c.CODEWORDS,codebook_patches=c.CODEBOOK_PATCHES,kmeans_iterations=c.KMEANS_ITERATIONS,
        created_unix=time.time())
    atomic(c.ROOT/'data_request.json',request)
    snapshots=c.ROOT/'sources';snapshots.mkdir(exist_ok=True)
    for i,path in enumerate(sorted(source_hashes)):(snapshots/f'{i:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    return request


def verify():
    request=read(c.ROOT/'data_request.json')
    for key in ('sources','assets','input_files','old_stop_markers'):
        for path,digest in request[key].items():assert sha(path)==digest,(key,path)
    return request


@torch.no_grad()
def trajectory_data(rank):
    verify();out=c.ROOT/'data';out.mkdir(exist_ok=True)
    path=out/f'rollout_rank{rank}.pt'
    if path.exists():
        assert read(path.with_suffix('.json'))['sha256']==sha(path);return
    rt=Runtime('sit_small');teacher=prefix_core.get_weak(rt,'native')
    model_hash=old_train.state_sha(rt.model)
    generator=torch.Generator(device='cuda').manual_seed(c.DATA_SEED)
    noises=torch.randn((c.TRAJECTORIES,4,32,32),device='cuda',generator=generator)
    records={key:[] for key in ('index','step','z','teacher','strong','f4','f12','context','labels','times')}
    torch.cuda.synchronize();begin=time.perf_counter()
    with models.Capture(rt) as capture:
        for start in range(rank*8,c.TRAJECTORIES,32):
            index=torch.arange(start,min(start+8,c.TRAJECTORIES),device='cuda')
            labels=index%100;rt.labels=labels;z=noises[index].clone()
            offsets=(index//100+3*labels)%8
            for k in range(32):
                t,h=k/64,1/64;ts=z.new_full((len(z),),t)
                amount=.8*(6/7 if t<.25 else 1.)
                strong=rt.field(z,z.new_tensor(t),'full');weak=teacher(z,ts,labels)
                selection=offsets==k%8
                if selection.any():
                    values=dict(index=index,step=torch.full_like(index,k),z=z,teacher=weak,strong=strong,
                        f4=capture.values[4],f12=capture.values[12],context=capture.context,labels=labels,times=ts)
                    for key,value in values.items():records[key].append(value[selection].cpu())
                first=strong+amount*(strong-weak);pred=z+h*first
                strong2=rt.field(pred,pred.new_tensor(t+h),'full');weak2=teacher(pred,ts+h,labels)
                second=strong2+amount*(strong2-weak2)
                z=z+(h/2)*(first+second)
                assert torch.isfinite(z).all()
            if start//32%8==0:print(dict(rank=rank,trajectory_start=start),flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
    assert old_train.state_sha(rt.model)==model_hash and all(p.grad is None for p in rt.model.parameters())
    values={key:torch.cat(value) for key,value in records.items()}
    temp=path.with_suffix('.tmp.pt');torch.save(values,temp);temp.replace(path)
    atomic(path.with_suffix('.json'),dict(passed=True,sha256=sha(path),states=len(values['index']),
        seconds=elapsed,strong_unchanged=True,teacher_inference_only=True,
        data_request_sha256=sha(c.ROOT/'data_request.json')))


@torch.no_grad()
def codebook():
    verify();out=c.ROOT/'data';out.mkdir(exist_ok=True);path=out/'codebook.pt'
    if path.exists():assert read(path.with_suffix('.json'))['sha256']==sha(path);return
    torch.cuda.set_device(0);torch.set_num_threads(4)
    generator=torch.Generator(device='cuda').manual_seed(c.DATA_SEED+1)
    pool=real_data.Pool('train');patches=[]
    for _ in range(c.CODEBOOK_PATCHES//(32*256)):
        clean,_=pool.draw('native',generator,32);patches.append(models.patchify(clean).reshape(-1,16))
    values=torch.cat(patches);assert len(values)==c.CODEBOOK_PATCHES
    perm=torch.randperm(len(values),device='cuda',generator=generator)
    centers=values[perm[:c.CODEWORDS]].clone();history=[]
    prior=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.synchronize();begin=time.perf_counter()
    try:
        for iteration in range(c.KMEANS_ITERATIONS):
            totals=torch.zeros_like(centers);counts=torch.zeros(c.CODEWORDS,device='cuda');loss=0.
            for chunk in values.split(4096):
                distance=chunk.square().sum(1,keepdim=True)+centers.square().sum(1)[None]-2*chunk@centers.T
                smallest,index=distance.min(1);loss+=float(smallest.sum())
                totals.index_add_(0,index,chunk);counts.index_add_(0,index,torch.ones_like(index,dtype=torch.float32))
            present=counts>0;centers[present]=totals[present]/counts[present,None]
            history.append(loss/(len(values)*16))
    finally:torch.backends.cuda.matmul.allow_tf32=prior
    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
    assert torch.isfinite(centers).all()
    torch.save(dict(centers=centers.cpu(),quantization_mse=history,seed=c.DATA_SEED+1),path)
    atomic(path.with_suffix('.json'),dict(passed=True,sha256=sha(path),seconds=elapsed,patches=len(values),
        centers=c.CODEWORDS,iterations=c.KMEANS_ITERATIONS,final_mse=history[-1],
        data_request_sha256=sha(c.ROOT/'data_request.json')))


def assemble():
    verify();out=c.ROOT/'data';records=[];files={}
    for rank in range(4):
        path=out/f'rollout_rank{rank}.pt';receipt=read(path.with_suffix('.json'))
        assert receipt['passed'] and receipt['sha256']==sha(path)
        assert receipt['data_request_sha256']==sha(c.ROOT/'data_request.json')
        records.append(torch.load(path,map_location='cpu',weights_only=False))
        files[str(path)]=sha(path);files[str(path.with_suffix('.json'))]=sha(path.with_suffix('.json'))
    index=torch.cat([r['index'] for r in records]).numpy();step=torch.cat([r['step'] for r in records]).numpy()
    assert np.array_equal(np.bincount(index,minlength=c.TRAJECTORIES),np.full(c.TRAJECTORIES,4))
    assert len(set(zip(index.tolist(),step.tolist())))==4*c.TRAJECTORIES
    train=index<c.TRAIN_TRAJECTORIES
    counts=np.zeros((100,32),dtype=int);np.add.at(counts,(index[train]%100,step[train]),1)
    assert np.all(counts==1)
    for name in ('codebook.pt','codebook.json'):
        path=out/name;files[str(path)]=sha(path)
    assert read(out/'codebook.json')['sha256']==sha(out/'codebook.pt')
    atomic(out/'complete.json',dict(passed=True,states=len(index),train_states=int(train.sum()),
        validation_states=int((~train).sum()),every_train_class_time_once=True,files=files,
        data_request_sha256=sha(c.ROOT/'data_request.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--rank',type=int);g.add_argument('--codebook',action='store_true');g.add_argument('--assemble',action='store_true')
    a=p.parse_args()
    if a.rank is not None:trajectory_data(a.rank)
    elif a.codebook:codebook()
    else:assemble()
