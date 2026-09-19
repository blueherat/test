"""Frozen S/W trajectory source classification, then posterior-gated IG."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import common as c

STAGE='ig_screen_400'
DATA='ig_source_data'
TRAIN_N,VAL_N,TRAIN_STEPS=1000,200,2000
SEED=2026121261
PROTOCOL=c.WORK/'docs/IG_SOURCE_POSTERIOR_PROTOCOL_20260912_ZH.md'
CONFIGS=[dict(arm='native_base',mode='native',factor=1.),
    dict(arm='posterior',mode='posterior',factor=1.),
    dict(arm='time_mean',mode='time_mean',factor=1.),
    dict(arm='permuted',mode='permuted',factor=1.),
    dict(arm='reversed',mode='reversed',factor=1.),
    dict(arm='native_half',mode='native',factor=.5),
    dict(arm='native_double',mode='native',factor=2.)]


class Capture:
    def __init__(self,rt):
        self.value=None
        def hook(module,args,out):
            self.value=out[:,:256].detach().float().mean(1)
        self.handle=rt.model.blocks[3 if rt.name=='sit_small' else 7].register_forward_hook(hook)
    def close(self):self.handle.remove()


class Head(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std)
        self.net=nn.Sequential(nn.Linear(len(mean)+2,128),nn.SiLU(),nn.Linear(128,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,feature,t,substage):
        z=(feature.float()-self.mean)/self.std
        extra=z.new_tensor([float(t),float(substage)]).expand(len(z),2)
        return self.net(torch.cat((z,extra),1)).flatten()


def head_path(model):return c.ROOT/model/DATA/'head.pt'


def prepare():
    sources=c.source_manifest([Path(__file__),PROTOCOL,c.WORK/'experiments/guidance_pasted_20260912/run_ig.py'])
    for model in c.MODELS:
        data=c.prepare_bank(model,DATA,TRAIN_N+VAL_N,SEED)
        quality=c.prepare_bank(model,STAGE,c.SAMPLES,SEED+10)
        request=dict(model=model,train_trajectories_per_source=TRAIN_N,validation_trajectories_per_source=VAL_N,
            sources=sources,assets={str(p):c.sha(p) for p in c.asset_paths(model)},
            inputs={str(p):c.sha(p) for folder in (data,quality) for p in folder.glob('*.npy')},
            source_targets={'full':1,'base':0},feature_depth=4 if model=='sit_small' else 8,
            feature='mean of first 256 spatial tokens; no gap or output norm features',
            fit=dict(steps=TRAIN_STEPS,batch=512,lr=.001,weight_decay=0.,seed=SEED+20,hidden=128),
            actual_sampler_queries=True,clean_endpoint_renoising=False,update_strong=False,update_weak=False,
            final_step_only=True,quality_configs=CONFIGS,primary_branch=0,quality_samples=c.SAMPLES,
            mixture_alpha=.5,normalization='2*(1-r), so r=.5 gives existing native IG',
            permutation='swap posterior between two current states with exactly the same class and solver query',
            time_mean='frozen average posterior on native IG quality paths; no fitting to quality metrics')
        p=c.ROOT/model/DATA/'request.json'
        if p.exists():assert c.read(p)==request
        else:c.atomic(p,request)


@torch.inference_mode()
def source_rollout(rt,capture,noise,labels,kind,save_states=False):
    rt.labels=labels;z=noise.clone();features=[];times=[];stages=[];states=[];before=rt.counts.copy()
    def field(x,t,left,k,substage):
        v=rt.field(x,t,kind)
        features.append(capture.value.clone());times.append(float(t));stages.append(substage)
        if save_states and substage==0 and k%8==0:states.append((k,x[:1].float().cpu().numpy()))
        return v
    with rt.context():
        for k,(t,u) in enumerate(zip(rt.grid[:-1],rt.grid[1:])):
            left=float(t)
            if c.amount(rt,left,'ig')==0:break
            h=u-t;v=field(z,t,left,k,0)
            if rt.name=='sit_small':z=z+(h/2)*(v+field(z+h*v,u,left,k,1))
            else:z=z+h*v
            if not torch.isfinite(z).all():raise FloatingPointError(f'{kind} source path at step {k}')
    return dict(features=torch.stack(features,1).half().cpu().numpy(),times=np.array(times,dtype=np.float32),
        substages=np.array(stages,dtype=np.int64),counts={k:rt.counts[k]-before[k] for k in before},states=states)


def validation(head,features,targets,times,stages):
    probs=[]
    with torch.no_grad():
        for start in range(0,len(features),512):
            x=torch.from_numpy(features[start:start+512].astype(np.float32)).cuda()
            extra=torch.from_numpy(np.column_stack((times[start:start+512],stages[start:start+512])).astype(np.float32)).cuda()
            z=(x-head.mean)/head.std
            probs.append(head.net(torch.cat((z,extra),1)).sigmoid().flatten().cpu().numpy())
    p=np.concatenate(probs);q=np.clip(p,1e-7,1-1e-7);y=targets
    ece=0.
    for lo in np.arange(0,1,.1):
        mask=(p>=lo)&(p<lo+.1)
        if mask.any():ece+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return dict(n=len(y),bce=float(-(y*np.log(q)+(1-y)*np.log(1-q)).mean()),
        brier=float(np.square(p-y).mean()),accuracy=float(((p>=.5)==y).mean()),ece10=float(ece),
        mean_p_source_strong=float(p[y==1].mean()),mean_p_source_weak=float(p[y==0].mean()))


def train_head(rt):
    root=c.ROOT/rt.name/DATA;all_x=[];all_y=[];all_ids=[];t=None;st=None
    for path in sorted((root/'features').glob('*.npz')):
        with np.load(path) as d:
            if t is None:t=d['times'];st=d['substages']
            else:np.testing.assert_array_equal(t,d['times']);np.testing.assert_array_equal(st,d['substages'])
            all_x.append(d['features']);all_y.extend([float(d['source'])]*len(d['features']));all_ids.extend(d['ids'])
    x=np.concatenate(all_x);y=np.array(all_y,dtype=np.float32);ids=np.array(all_ids)
    assert set(ids)==set(range(TRAIN_N+VAL_N)) and len(ids)==2*(TRAIN_N+VAL_N)
    split=ids<TRAIN_N
    def flatten(mask):
        n=int(mask.sum());return x[mask].reshape(-1,x.shape[-1]),np.repeat(y[mask],len(t)),np.tile(t,n),np.tile(st,n)
    train=flatten(split);val=flatten(~split)
    mean=torch.from_numpy(train[0].mean(0,dtype=np.float64).astype(np.float32)).cuda()
    std=torch.from_numpy(train[0].std(0,dtype=np.float64).astype(np.float32)).cuda().clamp_min(1e-4)
    torch.manual_seed(SEED+20);torch.cuda.manual_seed_all(SEED+20)
    head=Head(mean,std).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0.)
    before=validation(head,*val);rng=np.random.default_rng(SEED+21)
    torch.cuda.synchronize();begin=time.perf_counter()
    for step in range(TRAIN_STEPS):
        batch=rng.integers(0,len(train[0]),512)
        f=torch.from_numpy(train[0][batch].astype(np.float32)).cuda()
        extra=torch.from_numpy(np.column_stack((train[2][batch],train[3][batch])).astype(np.float32)).cuda()
        target=torch.from_numpy(train[1][batch]).cuda()
        logits=head.net(torch.cat(((f-head.mean)/head.std,extra),1)).flatten()
        loss=F.binary_cross_entropy_with_logits(logits,target)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    torch.cuda.synchronize();seconds=time.perf_counter()-begin
    after=validation(head,*val)
    assert all(not p.requires_grad and p.grad is None for p in rt.model.parameters())
    torch.save(dict(state=head.state_dict(),feature_dim=len(mean),step=TRAIN_STEPS),head_path(rt.name))
    c.atomic(root/'training_complete.json',dict(complete=True,train_seconds=seconds,before=before,after=after,
        trainable_parameters=sum(p.numel() for p in head.parameters()),strong_gradients_absent=True,
        model_training_mode=rt.model.training,head_sha256=c.sha(head_path(rt.name)),
        request_sha256=c.sha(root/'request.json'),no_denoiser_training=True,selected_by_validation=False))


def source_worker(model,parent_pid):
    request=c.read(c.ROOT/model/DATA/'request.json')
    for group in ('sources','assets','inputs'):
        for p,digest in request[group].items():assert c.sha(p)==digest,p
    rt=c.runtime(model);capture=Capture(rt);first,_,labels=c.bank(model,DATA)
    b=2*c.PAIR_BATCH[model];root=c.ROOT/model/DATA;out=root/'features';out.mkdir(exist_ok=True)
    h=c.sha(root/'request.json');cost=[]
    try:
        for start in range(0,TRAIN_N+VAL_N,b):
            c.check_parent(parent_pid)
            noise,y=c.cuda(first[start:start+b]),c.cuda(labels[start:start+b])
            for kind,label in [('full',1),('base',0)]:
                path=out/f'{start:04d}_{kind}.npz'
                if path.exists():continue
                torch.cuda.synchronize();begin=time.perf_counter()
                r=source_rollout(rt,capture,noise,y,kind,save_states=start==0)
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                np.savez(path,features=r['features'],times=r['times'],substages=r['substages'],source=label,
                    ids=np.arange(start,start+len(noise)),labels=np.array(labels[start:start+b]),
                    seconds=seconds,request_sha256=h,noise_sha256=c.array_sha(first[start:start+b]))
                c.atomic(path.with_suffix('.json'),dict(sha256=c.sha(path),counts=r['counts'],seconds=seconds,request_sha256=h))
                if start==0:
                    np.savez(root/f'{kind}_actual_state_examples.npz',**{f'step{k}':z for k,z in r['states']})
                cost.append(seconds)
            if start%100==0:c.atomic(root/'progress.json',dict(phase='actual_source_paths',complete_seed_pairs=start+len(noise),total=TRAIN_N+VAL_N))
        # The two model-origin labels begin at exactly the same state distribution.
        with np.load(out/'0000_full.npz') as s,np.load(out/'0000_base.npz') as w:
            np.testing.assert_array_equal(s['features'][:,0],w['features'][:,0])
        train_head(rt)
        c.atomic(root/'complete.json',dict(complete=True,source_seconds=sum(float(np.load(p)['seconds']) for p in out.glob('*.npz')),
            feature_files=len(list(out.glob('*.npz'))),same_initial_source_features_exact=True,
            runtime_sources=rt.sources,request_sha256=h))
    finally:capture.close()


def load_head(model):
    state=torch.load(head_path(model),map_location='cpu',weights_only=False)
    head=Head(state['state']['mean'],state['state']['std']);head.load_state_dict(state['state'])
    return head.cuda().eval().requires_grad_(False)


@torch.inference_mode()
def sample(rt,capture,head,first,second,labels,spec,profile=None,override=None,collect=False):
    n=len(first);targets=torch.cat((labels,labels));trace=[]
    def field(z,t,left,step,substage):
        a=c.amount(rt,left,'ig',spec['factor'])
        if not a:
            s=rt.field(z,t,'full')
            if collect:trace.append(z.new_full((len(z),),.5))
            return s
        s,w=rt.pair(z,t);mode=spec['mode']
        if override is not None:p=z.new_full((len(z),),override)
        elif mode in ('posterior','permuted','reversed') or collect:
            # Classifier runs in FP32 even when the frozen RAE trunk uses BF16.
            with torch.autocast('cuda',enabled=False):p=head(capture.value,t,substage).sigmoid()
        else:p=z.new_full((len(z),),.5)
        if collect:trace.append(p.clone())
        if mode=='native':effective=a
        else:
            if mode=='permuted':p=torch.cat((p[n:],p[:n]))
            if mode=='time_mean':
                index=step*2+substage if rt.name=='sit_small' else step
                p=z.new_full((len(z),),float(profile[index]))
            multiplier=2*p if mode=='reversed' else 2*(1-p)
            effective=a*multiplier.reshape((-1,)+(1,)*(z.ndim-1))
        if rt.name=='raev2':return w+(1+effective)*(s-w)
        return s+effective*(s-w)
    with rt.context():z,counts=c.integrate(rt,torch.cat((first,second)),targets,field)
    return z,counts,torch.stack(trace,1).cpu().numpy() if collect else None


def quality_worker(model,rank,world,parent_pid):
    root=c.ROOT/model/STAGE;data=c.ROOT/model/DATA
    rt=c.runtime(model);capture=Capture(rt);head=load_head(model)
    first,second,labels=c.bank(model,STAGE);b=c.PAIR_BATCH[model]
    h=c.sha(data/'request.json');checkpoint=c.sha(head_path(model))
    f,s,y=[c.cuda(a[:b]) for a in (first,second,labels)]
    native,counts,_=sample(rt,capture,head,f,s,y,CONFIGS[0])
    gated,gcounts,_=sample(rt,capture,head,f,s,y,CONFIGS[1],override=.5)
    assert torch.equal(native,gated),(native-gated).abs().max().item()
    assert counts==gcounts and counts['prefix']==0 and counts['full']==(128 if model=='sit_small' else 100)
    c.atomic(root/f'preflight{rank}.json',dict(passed=True,half_posterior_native_exact=True,
        counts=counts,head_sha256=checkpoint,request_sha256=h,runtime_sources=rt.sources))
    try:
        for spec in CONFIGS:
            profile=None
            if spec['mode']=='time_mean':
                while not (root/'time_profile.npy').exists():c.check_parent(parent_pid);time.sleep(3)
                profile=np.load(root/'time_profile.npy')
            out=root/spec['arm']/f'rank{rank}'
            for start in range(rank*b,c.SAMPLES,world*b):
                c.check_parent(parent_pid);path=out/f'batch{start:04d}.npz'
                if path.exists():
                    receipt=c.read(path.with_suffix('.json'));assert receipt['sha256']==c.sha(path) and receipt['request_sha256']==h
                    continue
                f,s,y=[c.cuda(a[start:start+b]) for a in (first,second,labels)]
                torch.cuda.synchronize();begin=time.perf_counter()
                z,count,probabilities=sample(rt,capture,head,f,s,y,spec,profile,collect=spec['arm']=='native_base')
                pixels=rt.decode(z);torch.cuda.synchronize();seconds=time.perf_counter()-begin
                stats=dict(seconds=seconds,full_calls=count['full'],pair_pixels=pixels[b:],second_latents=z[b:].cpu().numpy(),head_sha256=checkpoint)
                if probabilities is not None:stats['source_probabilities']=probabilities
                assert count['prefix']==0
                c.save_batch(path,pixels[:b],z[:b].cpu().numpy(),np.array(labels[start:start+b]),stats,h,start,c.array_sha(first[start:start+b]))
                if start%(50*b)==rank*b:c.atomic(root/f'progress{rank}.json',dict(arm=spec['arm'],start=start,rank=rank))
            c.atomic(out/'complete.json',dict(complete=True,request_sha256=h,head_sha256=checkpoint))
            print(model,rank,spec['arm'],'complete',flush=True)
    finally:capture.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--source',action='store_true')
    p.add_argument('--model',choices=c.MODELS);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--parent-pid',type=int,default=0);a=p.parse_args()
    if a.prepare:prepare()
    elif a.source:source_worker(a.model,a.parent_pid)
    else:quality_worker(a.model,a.rank,a.world,a.parent_pid)
