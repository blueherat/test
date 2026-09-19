"""Paired short refinements of the same pretrained depth-four readout."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import catalog
from experiments.sit_measure_guidance_20260912 import data
from experiments import imagenet100_sit_internal_v_head as heads
from experiments.lifting_scale_sweep_20260909 import Runtime,SMALL_CKPT,SMALL_HEAD,atomic,read,sha


def training_sources():
    return [Path(__file__).resolve(),Path(catalog.__file__).resolve(),
        Path(data.__file__).resolve(),Path(heads.__file__).resolve(),catalog.PROTOCOL,
        Path(__file__).resolve().parent/'__init__.py']


def prepare():
    root=catalog.ROOT
    root.mkdir(parents=True,exist_ok=True)
    if (root/'training_request.json').exists():return verify()
    parent=read(catalog.MEASURE/'training_request.json')
    for key in ('sources','assets','input_files'):
        for path,digest in parent[key].items():assert sha(path)==digest,path
    sources=dict(parent['sources'])
    sources.update({str(p):sha(p) for p in training_sources()})
    original=torch.load(SMALL_HEAD,map_location='cpu',weights_only=False)
    assert original['step']==50000 and original['config']['internal_depth']==4
    assert original['config']['source_checkpoint_sha256']==sha(SMALL_CKPT)
    assert original['config']['learning_rate']==catalog.LR
    request=dict(sources=sources,assets={str(SMALL_CKPT):sha(SMALL_CKPT),str(SMALL_HEAD):sha(SMALL_HEAD)},
        input_files=parent['input_files'],parent_training_request_sha256=sha(catalog.MEASURE/'training_request.json'),
        initial_head='depth4 velocity head 50K EMA',strong_model_frozen=True,head_depth=4,
        methods=list(catalog.METHODS),steps=catalog.STEPS,batch=catalog.BATCH,learning_rate=catalog.LR,
        adamw_betas=[.9,.999],weight_decay=0.,ema=catalog.EMA,gradient_clip=1.,
        train_seed=catalog.SEED,time_range=[.01,.99],null_probability=.1,
        training_precision='bf16 autocast, float32 optimizer/head weights',
        selected_by_validation=False,no_fid_used=True,created_unix=time.time())
    atomic(root/'training_request.json',request)
    snapshot=root/'training_sources';snapshot.mkdir(exist_ok=True)
    for i,path in enumerate(sorted(sources)):
        (snapshot/f'{i:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    return request


def verify():
    request=read(catalog.ROOT/'training_request.json')
    for category in ('sources','assets','input_files'):
        for path,digest in request[category].items():assert sha(path)==digest,path
    return request


def prediction(rt,head,z,t,labels):
    with torch.no_grad():
        features,condition=heads.extract_internal_features(rt.model,z,t,labels,internal_depth=4)
    return heads.internal_velocity_from_features(rt.model,head,features,condition,latent_channels=4)


@torch.no_grad()
def validate(rt,head,pool,method):
    generator=torch.Generator(device='cuda').manual_seed(catalog.SEED+100)
    rows=[]
    for tv in (.15,.5,.85):
        total,size=0.,0
        for start in range(0,400,20):
            labels=torch.arange(start,start+20,device='cuda')%100
            clean,labels=pool.draw(method,generator,20,labels=labels)
            noise=torch.randn(clean.shape,device='cuda',generator=generator)
            times=torch.full((len(clean),),tv,device='cuda')
            result=prediction(rt,head,tv*clean+(1-tv)*noise,times,labels)
            total+=float((result.double()-(clean-noise).double()).square().sum())
            size+=clean.numel()
        rows.append(dict(time=tv,mse=total/size,images=400))
    return rows


def train(method):
    verify();root=catalog.ROOT
    folder=root/'training'/method;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'complete.json').exists():
            done=read(folder/'complete.json')
            assert done['request_sha256']==sha(root/'training_request.json')
            assert done['checkpoint_sha256']==sha(folder/'model.pt')
            return
        rt=Runtime('sit_small');torch.set_num_threads(4)
        rt.model.eval().requires_grad_(False)
        head=copy.deepcopy(rt.head.module).eval().requires_grad_(True)
        ema=copy.deepcopy(head).eval().requires_grad_(False)
        assert not any(p.requires_grad for p in rt.model.parameters())
        torch.manual_seed(catalog.SEED)
        pool=data.Pool('train');heldout=data.Pool('validation')
        before=validate(rt,ema,heldout,method)
        optimizer=torch.optim.AdamW(head.parameters(),lr=catalog.LR,betas=(.9,.999),weight_decay=0.)
        generator=torch.Generator(device='cuda').manual_seed(catalog.SEED)
        start_step=0
        if (folder/'resume.pt').exists():
            saved=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
            assert saved['request_sha256']==sha(root/'training_request.json')
            head.load_state_dict(saved['head']);ema.load_state_dict(saved['ema'])
            optimizer.load_state_dict(saved['optimizer']);generator.set_state(saved['generator'])
            start_step=saved['step']
        losses=[];begin=time.perf_counter()
        for step in range(start_step+1,catalog.STEPS+1):
            clean,labels=pool.draw(method,generator,catalog.BATCH)
            noise=torch.randn(clean.shape,device='cuda',generator=generator)
            times=.01+.98*torch.rand(len(clean),device='cuda',generator=generator)
            dropped=torch.rand(len(clean),device='cuda',generator=generator)<.1
            labels=torch.where(dropped,torch.full_like(labels,100),labels)
            z=times[:,None,None,None]*clean+(1-times[:,None,None,None])*noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                value=prediction(rt,head,z,times,labels)
                loss=(value.float()-(clean-noise)).square().mean()
            assert torch.isfinite(loss),(method,step)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for target,source in zip(ema.parameters(),head.parameters()):target.lerp_(source,1-catalog.EMA)
            losses.append(float(loss.detach()))
            if step==1 or step%50==0:
                torch.cuda.synchronize()
                status=dict(phase='training',method=method,step=step,total=catalog.STEPS,
                    loss_mean=float(np.mean(losses[-50:])),gradient_norm=float(norm),
                    elapsed_seconds=time.perf_counter()-begin,pid=os.getpid())
                atomic(folder/'status.json',status);print(json.dumps(status),flush=True)
            if step%250==0 or step==catalog.STEPS or (root/'STOP_AFTER_CURRENT').exists():
                temp=folder/'resume.tmp.pt'
                torch.save(dict(head=head.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                    generator=generator.get_state(),step=step,request_sha256=sha(root/'training_request.json')),temp)
                temp.replace(folder/'resume.pt')
            if (root/'STOP_AFTER_CURRENT').exists():
                atomic(folder/'status.json',dict(phase='stopped_after_step',step=step));return
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        assert all(p.grad is None for p in rt.model.parameters())
        after=validate(rt,ema,heldout,method);native=validate(rt,ema,heldout,'native')
        temp=folder/'model.tmp.pt'
        torch.save(dict(ema={k:v.cpu() for k,v in ema.state_dict().items()},step=catalog.STEPS,
            method=method,request_sha256=sha(root/'training_request.json')),temp)
        temp.replace(folder/'model.pt')
        done=dict(passed=True,method=method,step=catalog.STEPS,request_sha256=sha(root/'training_request.json'),
            checkpoint_sha256=sha(folder/'model.pt'),training_seconds=elapsed,resumed_from_step=start_step,
            target_mse_before=before,target_mse_after=after,native_mse_after=native,
            trainable_parameters=sum(p.numel() for p in head.parameters()),strong_gradients_absent=True,
            selected_by_validation=False,no_fid_used=True,created_unix=time.time())
        atomic(folder/'complete.json',done);atomic(folder/'status.json',dict(phase='complete',**done))
        print(json.dumps(done),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--prepare',action='store_true');a.add_argument('--train',choices=catalog.METHODS)
    args=p.parse_args()
    if args.prepare:prepare()
    else:train(args.train)
