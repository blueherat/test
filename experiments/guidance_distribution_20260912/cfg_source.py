"""Train only a CFG source discriminator on actual conditional/null trajectories."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit
from experiments.guidance_pasted_20260912 import common as c,ig
from . import mixture as m

N,TRAIN,CAL=800,600,100
STAGE='cfg_source_data'
SEED=2026121321


def prepare(model):
    m.configure();bank=c.prepare_bank(model,STAGE,N,SEED)
    rp=m.ROOT/model/STAGE/'request.json'
    request=dict(model=model,samples_per_source=N,train_ids=[0,599],calibration_ids=[600,699],audit_ids=[700,799],
        seed=SEED,source_targets={'full':1,'null':0},actual_sampler_queries=True,
        feature_map='conditional prefix mean for BOTH origins',null_source_extra_conditional_prefix_offline=True,
        independent_denoiser_training=False,fit=dict(steps=2000,batch=512,lr=.001,weight_decay=0.,seed=SEED+1),
        sources=c.source_manifest([Path(__file__).resolve(),m.PROTOCOL,c.WORK/'experiments/guidance_pasted_20260912/ig.py']),
        assets={str(p):c.sha(p) for p in c.asset_paths(model)},
        inputs={str(p):c.sha(p) for p in bank.glob('*.npy')})
    if rp.exists():assert c.read(rp)==request
    else:c.atomic(rp,request)
    return rp


def verify(rp):
    request=c.read(rp)
    for key in ('sources','assets','inputs'):
        for p,h in request[key].items():assert c.sha(p)==h,p


@torch.inference_mode()
def rollout(rt,capture,noise,labels,kind):
    rt.labels=labels;z=noise.clone();features=[];times=[];stages=[]
    before=rt.counts.copy()
    def field(x,t,substage):
        if kind=='full':v=rt.field(x,t,'full')
        else:
            rt.labels=torch.full_like(labels,c.CLASSES[rt.name])
            try:v=rt.field(x,t,'full')
            finally:rt.labels=labels
            # This is an offline observation of the SAME conditional feature map.
            rt.field(x,t,'base')
        features.append(capture.value.clone());times.append(float(t));stages.append(substage)
        return v
    with rt.context():
        for t,u in zip(rt.grid[:-1],rt.grid[1:]):
            if not c.amount(rt,float(t),'cfg'):break
            h=u-t;v=field(z,t,0)
            if rt.name=='sit_small':z=z+h/2*(v+field(z+h*v,u,1))
            else:z=z+h*v
            assert torch.isfinite(z).all()
    return dict(features=torch.stack(features,1).half().cpu().numpy(),
        times=np.asarray(times,dtype=np.float32),substages=np.asarray(stages,dtype=np.int64),
        counts={k:rt.counts[k]-before[k] for k in before})


@torch.inference_mode()
def worker(model,rank,world,parent=0):
    m.configure();rp=m.ROOT/model/STAGE/'request.json';verify(rp)
    rt=c.runtime(model);capture=ig.Capture(rt);root=rp.parent
    first,_,labels=c.bank(model,STAGE)
    # The chosen representation must be identical whether the suffix is run.
    z=c.cuda(first[:rt.batch]);rt.labels=c.cuda(labels[:rt.batch])
    with rt.context():
        rt.field(z,rt.grid[0],'full');a=capture.value.clone()
        rt.field(z,rt.grid[0],'base');b=capture.value.clone()
    assert torch.equal(a,b)
    c.atomic(root/f'preflight{rank}.json',dict(passed=True,conditional_prefix_full_features_exact=True))
    try:
        for start in range(rank*rt.batch,N,world*rt.batch):
            c.check_parent(parent);z=c.cuda(first[start:start+rt.batch]);y=c.cuda(labels[start:start+rt.batch])
            for kind,label in (('full',1),('null',0)):
                p=root/'features'/f'{start:04d}_{kind}.npz'
                if p.exists():
                    receipt=c.read(p.with_suffix('.json'));assert c.sha(p)==receipt['sha256'];continue
                torch.cuda.synchronize();begin=time.perf_counter()
                value=rollout(rt,capture,z,y,kind)
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez(f,features=value['features'],times=value['times'],
                    substages=value['substages'],ids=np.arange(start,start+len(y)),labels=np.array(y.cpu()),source=label,
                    seconds=seconds,request_sha256=c.sha(rp),noise_sha256=c.array_sha(first[start:start+len(y)]))
                tmp.replace(p)
                c.atomic(p.with_suffix('.json'),dict(sha256=c.sha(p),seconds=seconds,counts=value['counts'],request_sha256=c.sha(rp)))
            if start%(25*rt.batch)==rank*rt.batch:
                c.atomic(root/f'progress{rank}.json',dict(pid=os.getpid(),start=start,total=N))
        c.atomic(root/f'complete{rank}.json',dict(complete=True,rank=rank,world=world))
    finally:capture.close()


def fit(model,device='cpu'):
    m.configure();torch.set_num_threads(4)
    root=m.ROOT/model/STAGE;rp=root/'request.json';verify(rp)
    all_x=[];all_y=[];all_ids=[];times=stages=None;records=[]
    for p in sorted((root/'features').glob('*.npz')):
        receipt=c.read(p.with_suffix('.json'));assert c.sha(p)==receipt['sha256']
        with np.load(p) as d:
            if times is None:times=d['times'];stages=d['substages']
            else:
                np.testing.assert_array_equal(times,d['times']);np.testing.assert_array_equal(stages,d['substages'])
            all_x.append(d['features']);all_y.extend([int(d['source'])]*len(d['ids']));all_ids.extend(d['ids'])
            assert str(d['request_sha256'])==c.sha(rp)
        records.append(dict(file=str(p),sha256=receipt['sha256']))
    x=np.concatenate(all_x);labels=np.asarray(all_y,dtype=np.float32);ids=np.asarray(all_ids)
    assert len(x)==2*N
    for source in (0,1):np.testing.assert_array_equal(np.sort(ids[labels==source]),np.arange(N))
    # Align initial states by seed to check provenance, not just by file order.
    a=x[labels==1][np.argsort(ids[labels==1]),0]
    b=x[labels==0][np.argsort(ids[labels==0]),0]
    np.testing.assert_array_equal(a,b)
    nquery=len(times)
    def flat(mask):
        z=x[mask].reshape(-1,x.shape[-1]).astype(np.float32)
        y=np.repeat(labels[mask],nquery)
        extra=np.column_stack((np.tile(times,int(mask.sum())),np.tile(stages,int(mask.sum())))).astype(np.float32)
        return z,y,extra
    train=flat(ids<TRAIN);cal=flat((ids>=TRAIN)&(ids<TRAIN+CAL));audit=flat(ids>=TRAIN+CAL)
    mean=torch.from_numpy(train[0].mean(0,dtype=np.float64).astype(np.float32)).to(device)
    std=torch.from_numpy(train[0].std(0,dtype=np.float64).astype(np.float32)).to(device).clamp_min(1e-4)
    torch.manual_seed(SEED+1)
    head=ig.Head(mean,std).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0.)
    rng=np.random.default_rng(SEED+2);begin=time.perf_counter()
    for _ in range(2000):
        idx=rng.integers(0,len(train[0]),512)
        z=torch.from_numpy(train[0][idx]).to(device);y=torch.from_numpy(train[1][idx]).to(device)
        ex=torch.from_numpy(train[2][idx]).to(device)
        logits=head.net(torch.cat(((z-head.mean)/head.std,ex),1)).flatten()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    seconds=time.perf_counter()-begin
    def logits_for(values):
        z,_,extra=values;result=[]
        with torch.no_grad():
            for start in range(0,len(z),512):
                a=torch.from_numpy(z[start:start+512]).to(device)
                b=torch.from_numpy(extra[start:start+512]).to(device)
                result.append(head.net(torch.cat(((a-head.mean)/head.std,b),1)).flatten().cpu().numpy())
        return np.concatenate(result).astype(np.float64)
    lc=logits_for(cal);la=logits_for(audit)
    gradient=lambda beta:np.mean((expit(beta*lc)-cal[1])*lc)
    hi=1.
    while gradient(hi)<0 and hi<1024:hi*=2
    beta=0. if gradient(0)>=0 else brentq(gradient,0,hi)
    def metric(l,y,beta):
        q=beta*l
        return dict(n=len(y),bce=float(np.mean(np.logaddexp(0,q)-y*q)),
            brier=float(np.mean((expit(q)-y)**2)),accuracy=float(((q>=0)==y).mean()))
    head=head.cpu()
    raw=dict(state=head.state_dict(),feature_dim=len(mean),step=2000,request_sha256=c.sha(rp))
    torch.save(raw,root/'raw_head.pt')
    with torch.no_grad():
        head.net[-1].weight.mul_(beta);head.net[-1].bias.mul_(beta)
    torch.save(dict(state=head.state_dict(),feature_dim=len(mean),step=2000,
        inverse_temperature=float(beta),request_sha256=c.sha(rp)),root/'head.pt')
    c.atomic(root/'training_complete.json',dict(complete=True,model=model,train_seconds=seconds,
        inverse_temperature=float(beta),audit_raw=metric(la,audit[1],1.),audit_calibrated=metric(la,audit[1],beta),
        calibration_raw=metric(lc,cal[1],1.),calibration_fitted=metric(lc,cal[1],beta),
        initial_source_features_exact=True,head_sha256=c.sha(root/'head.pt'),raw_head_sha256=c.sha(root/'raw_head.pt'),
        request_sha256=c.sha(rp),records=records,trainable_parameters=sum(p.numel() for p in head.parameters()),
        source_seconds=sum(c.read(Path(r['file']).with_suffix('.json'))['seconds'] for r in records),
        class_coverage='800 distinct classes in RAE; source train/cal/audit use disjoint class subsets'))
    print(model,'CFG source head trained',beta,metric(la,audit[1],beta),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--fit',action='store_true');p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world',type=int,default=1);p.add_argument('--parent',type=int,default=0)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.prepare:prepare(a.model)
    elif a.worker:worker(a.model,a.rank,a.world,a.parent)
    elif a.fit:fit(a.model,a.device)
    else:p.error('choose --prepare, --worker or --fit')
