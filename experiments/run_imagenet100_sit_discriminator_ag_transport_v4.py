#!/usr/bin/env python3
"""Long mechanism-and-policy experiment for discriminator-calibrated AutoGuidance.

Hypothesis
----------
Let S be the v800 strong velocity field and W a weak checkpoint. AutoGuidance is

    v = S + gamma * (S - W).

Train a discriminator to distinguish true p_t latents from a one-step strong
transport q_t^S produced from true p_s states:

    real: x_t ~ p_t
    fake: y_t = x_s + (t-s) S(x_s,s),  x_s ~ p_s.

For an optimal binary discriminator, its logit approximates

    d(y;s,t,c) ~= log p_t(y|c) - log q_t^S(y|c).

For g=S-W, a one-step AG perturbation changes y by (t-s)*gamma*g, so the local
log-density-ratio slope is

    (t-s) <grad_y d, g>.

With a quadratic penalty on actual AG action gamma^2 ||g||^2, a principled local
coefficient is proportional to

    (t-s) [ <grad_y d, g> / ||g||^2 ]_+.

This script tests whether that quantity explains known AG temporal utility and
whether the resulting state-dependent coefficient improves paired FID.\n\nV2 engineering fixes: discriminator training defaults to batch 32 and an 8 GiB\nallocator budget; worker failures surface the last 120 log lines in the parent\nterminal, and Phase-1 performs a one-batch CUDA/model/cache preflight.

Phases
------
1. Train two independent class/time-conditional discriminators on ImageNet-100
   train SD-VAE latents, using v800 only. Keep EMA weights.
2. On held-out validation latents, audit v180/v240/v270/v300/v400/v500/v600
   over 8 time bins: discriminator AUC, gap RMS, cos(grad d,S-W), normalized
   projection coefficient, and finite-difference discriminator improvement.
3. Exact-paired FID-1K seed-0 screen on v800/v180:
      baseline; fixed gamma=3.05; early-only gamma=4.5 until t=0.5;
      discriminator state-adaptive policy over a small 1D gain grid.
4. Compare best state-adaptive policy with:
      time-only discriminator profile; sign-gated fixed gamma=4.5.
5. Repeat key policies with sampling seed 1 and summarize.

Evidence boundaries
-------------------
This is a falsifiable mechanism probe, not a proof of globally optimal control.
The discriminator is teacher-forced on true p_s states and one-step transport;
free-running sampling is distribution shifted. FID-1K is screening only.

Run
---
    cd /home/zhoushunyu/eqvae
    python experiments/run_imagenet100_sit_discriminator_ag_transport_v1.py --gpus 1,3

Smoke test
----------
    python experiments/run_imagenet100_sit_discriminator_ag_transport_v1.py \
      --gpus 1,3 --disc-steps 1000 --audit-samples-per-bin 64 --num-samples 200
"""
from __future__ import annotations

import argparse, copy, csv, gc, hashlib, json, math, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES=100
LATENT_SHAPE=(4,32,32)
MOMENT_SHAPE=(8,32,32)
SD_VAE_SCALING_FACTOR=0.18215
EXPECTED_SEED0_NOISE_SHA256='b693d3cc2f28249d84942f74586d1afda2df10879225cd69bbf5d6a2d602c7b8'
EXPECTED_SEED0_LABEL_SHA256='76fcd0fce6808c069a79ee8fd795edf2a1785d73758dc62306e51700c44c0758'
DEFAULT_WEAK_STEPS=(180000,240000,270000,300000,400000,500000,600000)
DEFAULT_TIME_CENTERS=tuple((i+0.5)/8.0 for i in range(8))
DATA_ROOT_CANDIDATES=(Path('/data/users/zhoushunyu/eqvae/imagenet_sit_flow'),Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow'))
DEFAULT_ADM_PYTHON=Path('/data/shared/envs/adm-fid/bin/python')


def detect_repo_root():
    here=Path(__file__).resolve()
    for root in (Path.cwd().resolve(),here.parent,here.parent.parent):
        if (root/'experiments/train_imagenet100_sit_flow.py').is_file(): return root
    raise FileNotFoundError('Cannot find repo root; run from /home/zhoushunyu/eqvae')

def detect_data_root():
    marker=Path('runs/sit-s-2_seed0/checkpoints/step_00800000.pt')
    for root in DATA_ROOT_CANDIDATES:
        if (root/marker).is_file(): return root
    raise FileNotFoundError('Cannot find ImageNet-100 SiT data root')

def parse_gpu_list(s):
    try: out=tuple(int(x.strip()) for x in s.split(',') if x.strip())
    except ValueError as e: raise argparse.ArgumentTypeError('comma-separated GPU ids required') from e
    if not out or len(set(out))!=len(out) or any(x<0 for x in out): raise argparse.ArgumentTypeError('GPU ids must be unique non-negative')
    return out

def parse_float_tuple(s):
    try: out=tuple(float(x.strip()) for x in s.split(',') if x.strip())
    except ValueError as e: raise argparse.ArgumentTypeError('comma-separated floats required') from e
    if not out or any(not math.isfinite(x) for x in out): raise argparse.ArgumentTypeError('finite values required')
    return out

def parse_int_tuple(s):
    try: out=tuple(int(x.strip()) for x in s.split(',') if x.strip())
    except ValueError as e: raise argparse.ArgumentTypeError('comma-separated ints required') from e
    if not out: raise argparse.ArgumentTypeError('non-empty list required')
    return out

def tag_float(x): return f'{float(x):.8g}'.replace('-','m').replace('.','p').replace('+','')

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def read_json(path):
    x=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(x,dict): raise ValueError(f'expected JSON object: {path}')
    return x

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def setup_device(seed,allow_tf32=True):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    d=torch.device('cuda:0'); torch.cuda.set_device(d)
    random.seed(seed); np.random.seed(seed%(2**32)); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=bool(allow_tf32); torch.backends.cudnn.allow_tf32=bool(allow_tf32)
    if hasattr(torch,'set_float32_matmul_precision'): torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
    return d

@dataclass
class RepoContext:
    repo:Path; data:Path; cache:Path; ckpt_dir:Path; strong_checkpoint:Path; reference_stats:Path; official_sit_repo:Path

def build_repo_context(args):
    repo=detect_repo_root(); data=args.data_root.expanduser().resolve() if args.data_root else detect_data_root()
    cache=data/'imagenet100_cmc_sdvae'; ckpt=data/'runs/sit-s-2_seed0/checkpoints'
    strong=ckpt/'step_00800000.pt'; ref=data/'adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'
    official=Path('/home/zhoushunyu/data/research_repos/SiT')
    required=[strong,ref,cache/'train_moments.npy',cache/'train_labels.npy',cache/'validation_moments.npy',cache/'validation_labels.npy']
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError('missing required files:\n  '+'\n  '.join(missing))
    return RepoContext(repo,data,cache,ckpt,strong,ref,official)

def import_repo_modules(repo):
    if str(repo) not in sys.path: sys.path.insert(0,str(repo))
    from experiments.imagenet100_sit_multiscale_models import evaluate_sit_field,load_sit_field_model
    from experiments.sample_imagenet100_sit_fid import configure_cuda_allocator,decode_latents_in_chunks,official_pixel_quantization
    from experiments.sample_imagenet100_sit_flow import integrate_velocity
    from experiments.train_imagenet100_sit_flow import load_official_sit_module
    return locals()

class MomentsCache:
    def __init__(self,cache_dir,split):
        self.moments=np.load(cache_dir/f'{split}_moments.npy',mmap_mode='r',allow_pickle=False)
        self.labels=np.load(cache_dir/f'{split}_labels.npy',mmap_mode='r',allow_pickle=False)
        if self.moments.dtype!=np.float32 or tuple(self.moments.shape[1:])!=MOMENT_SHAPE: raise ValueError('bad moments cache')
        self.class_indices=[np.flatnonzero(self.labels.astype(np.int64)==c) for c in range(NUM_CLASSES)]
        if any(len(x)<2 for x in self.class_indices): raise ValueError('need >=2 examples per class')
    def sample_pairs(self,b,rng):
        labels=rng.integers(0,NUM_CLASSES,size=b,dtype=np.int64); a=np.empty(b,np.int64); z=np.empty(b,np.int64)
        for i,c in enumerate(labels.tolist()):
            pool=self.class_indices[c]; ia=int(rng.integers(0,len(pool))); ib=int(rng.integers(0,len(pool)-1));
            if ib>=ia: ib+=1
            a[i]=int(pool[ia]); z[i]=int(pool[ib])
        return a,z,labels
    def fetch(self,idx,device):
        arr=np.asarray(self.moments[idx],dtype=np.float32); return torch.from_numpy(arr).to(device)

def sample_posterior(m):
    mean,std=m.chunk(2,dim=1); return (mean+std*torch.randn_like(mean))*SD_VAE_SCALING_FACTOR

def linear_state(data,noise,t):
    tt=t.reshape(-1,1,1,1); return (1-tt)*noise+tt*data

def sample_times(b,device,s_max,dt_min,dt_max):
    s=torch.rand(b,device=device)*float(s_max); upper=torch.minimum(torch.full_like(s,float(dt_max)),(1-s).clamp_min(float(dt_min)))
    dt=float(dt_min)+torch.rand(b,device=device)*(upper-float(dt_min)); return s.float(),(s+dt).float(),dt.float()

def load_field(checkpoint,device,ctx,mods,sit=None,source=None):
    if sit is None: sit,source=mods['load_official_sit_module'](ctx.official_sit_repo,verify_source=True)
    model,sem,meta=mods['load_sit_field_model'](checkpoint_path=checkpoint,weights='ema',sit_module=sit,source_metadata=source,device=device)
    return model,sem,meta,sit,source

def eval_field(model,sem,x,t,labels,mods,bf16):
    with torch.no_grad():
        if bf16:
            with torch.autocast('cuda',dtype=torch.bfloat16): y=mods['evaluate_sit_field'](model,sem,x,t,labels)
        else: y=mods['evaluate_sit_field'](model,sem,x,t,labels)
    return y.float()

# discriminator

def fourier(x,bands=16):
    x=x.float().reshape(-1,1); f=2.0**torch.arange(bands,device=x.device,dtype=torch.float32); a=2*math.pi*x*f.reshape(1,-1)
    return torch.cat([x,torch.sin(a),torch.cos(a)],1)

class CondBlock(nn.Module):
    def __init__(self,ci,co,cd,down=False):
        super().__init__(); self.n1=nn.GroupNorm(min(32,ci),ci); self.c1=nn.Conv2d(ci,co,3,padding=1); self.n2=nn.GroupNorm(min(32,co),co); self.c2=nn.Conv2d(co,co,3,padding=1); self.film=nn.Linear(cd,2*co); self.skip=nn.Conv2d(ci,co,1) if ci!=co else nn.Identity(); self.down=down
    def forward(self,x,c):
        h=self.c1(F.silu(self.n1(x))); sc,sh=self.film(c).chunk(2,1); h=self.n2(h); h=h*(1+sc[:,:,None,None])+sh[:,:,None,None]; h=self.c2(F.silu(h)); h=h+self.skip(x)
        return F.avg_pool2d(h,2) if self.down else h

class RatioDisc(nn.Module):
    def __init__(self,base=64,cond=256,class_dim=128,bands=16):
        super().__init__(); self.cfg=dict(base=base,cond=cond,class_dim=class_dim,bands=bands); td=1+2*bands
        self.ce=nn.Embedding(NUM_CLASSES,class_dim); self.cm=nn.Sequential(nn.Linear(class_dim+3*td,cond),nn.SiLU(),nn.Linear(cond,cond),nn.SiLU())
        self.stem=nn.Conv2d(4,base,3,padding=1); self.blocks=nn.ModuleList([CondBlock(base,base,cond),CondBlock(base,2*base,cond,True),CondBlock(2*base,2*base,cond),CondBlock(2*base,4*base,cond,True),CondBlock(4*base,4*base,cond),CondBlock(4*base,4*base,cond,True)])
        self.norm=nn.GroupNorm(min(32,4*base),4*base); self.out=nn.Linear(4*base,1); self.bands=bands
    def forward(self,x,s,t,label):
        c=self.cm(torch.cat([self.ce(label),fourier(s,self.bands),fourier(t,self.bands),fourier((t-s).clamp_min(0),self.bands)],1)); h=self.stem(x.float())
        for b in self.blocks: h=b(h,c)
        return self.out(F.silu(self.norm(h)).mean((2,3))).squeeze(1)

@torch.no_grad()
def ema_update(dst,src,decay):
    sp=dict(src.named_parameters());
    for n,p in dst.named_parameters(): p.mul_(decay).add_(sp[n],alpha=1-decay)
    sb=dict(src.named_buffers());
    for n,b in dst.named_buffers(): b.copy_(sb[n])

def binary_auc(real,fake):
    real=np.asarray(real,float); fake=np.asarray(fake,float); score=np.r_[real,fake]; lab=np.r_[np.ones(len(real),int),np.zeros(len(fake),int)]
    order=np.argsort(score,kind='mergesort'); rank=np.empty(len(score),float); i=0
    while i<len(score):
        j=i+1
        while j<len(score) and score[order[j]]==score[order[i]]: j+=1
        rank[order[i:j]]=0.5*((i+1)+j); i=j
    n1=int(lab.sum()); n0=len(lab)-n1; return float((rank[lab==1].sum()-n1*(n1+1)/2)/(n1*n0))

def avg_rank(v):
    x=np.asarray(v,float); o=np.argsort(x,kind='mergesort'); r=np.empty(len(x),float); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[o[j]]==x[o[i]]: j+=1
        r[o[i:j]]=0.5*((i+1)+j); i=j
    return r

def spearman(a,b):
    if len(a)<2 or len(a)!=len(b): return float('nan')
    ra,rb=avg_rank(a),avg_rank(b)
    return float(np.corrcoef(ra,rb)[0,1]) if ra.std()>0 and rb.std()>0 else float('nan')

def make_disc(args,device): return RatioDisc(args.disc_base_channels,args.disc_cond_dim,args.disc_class_dim,args.disc_time_bands).to(device)

def make_batch(cache,b,rng,device,strong,sem,mods,args,fixed_s=None,fixed_dt=None):
    fi,ri,ln=cache.sample_pairs(b,rng); labels=torch.from_numpy(ln).to(device=device,dtype=torch.long); fd=sample_posterior(cache.fetch(fi,device)); rd=sample_posterior(cache.fetch(ri,device)); fn=torch.randn_like(fd); rn=torch.randn_like(rd)
    if fixed_s is None: s,t,dt=sample_times(b,device,args.disc_s_max,args.disc_dt_min,args.disc_dt_max)
    else:
        dv=float(fixed_dt if fixed_dt is not None else min(args.audit_delta,1-fixed_s-1e-4)); s=torch.full((b,),float(fixed_s),device=device); dt=torch.full((b,),dv,device=device); t=s+dt
    xs=linear_state(fd,fn,s); xr=linear_state(rd,rn,t); sv=eval_field(strong,sem,xs,s,labels,mods,args.strong_bf16); xf=(xs+dt[:,None,None,None]*sv).detach()
    return dict(xs=xs,xr=xr,xf=xf,sv=sv,labels=labels,s=s,t=t,dt=dt)

def disc_cfg(args): return dict(base=args.disc_base_channels,cond=args.disc_cond_dim,class_dim=args.disc_class_dim,bands=args.disc_time_bands)

def save_disc(path,model,ema,opt,step,seed,best,args,strong_sha):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.pt.tmp'); torch.save(dict(format='eqvae_ag_transport_disc_v1',model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=step,seed=seed,best_auc=best,disc_config=disc_cfg(args),strong_sha256=strong_sha),tmp); os.replace(tmp,path)

def load_disc(path,device):
    x=torch.load(path,map_location='cpu',weights_only=False); d=RatioDisc(**x['disc_config']).to(device); d.load_state_dict(x['ema']); d.eval().requires_grad_(False); return d,x

@torch.no_grad()
def eval_disc_auc(d,cache,rng_seed,device,strong,sem,mods,args):
    rng=np.random.default_rng(rng_seed); R=[]; Fk=[]
    for _ in range(args.disc_val_batches):
        b=make_batch(cache,args.disc_batch_size,rng,device,strong,sem,mods,args); R.append(d(b['xr'],b['s'],b['t'],b['labels']).cpu().numpy()); Fk.append(d(b['xf'],b['s'],b['t'],b['labels']).cpu().numpy())
    R=np.concatenate(R); Fk=np.concatenate(Fk); return dict(auc=binary_auc(R,Fk),accuracy_at_zero=float(((R>0).sum()+(Fk<0).sum())/(len(R)+len(Fk))),real_logit_mean=float(R.mean()),fake_logit_mean=float(Fk.mean()),n=len(R))

def worker_train(args,seed):
    ctx=build_repo_context(args); mods=import_repo_modules(ctx.repo); device=setup_device(10000+seed)
    alloc=mods['configure_cuda_allocator'](device,limit_gib=args.cuda_allocator_limit_gib)
    print(f'[D{seed}] CUDA allocator: {alloc}',flush=True)
    print(f'[D{seed}] loading v800 strong...',flush=True)
    strong,sem,meta,_,_=load_field(ctx.strong_checkpoint,device,ctx,mods)
    print(f'[D{seed}] loading latent caches...',flush=True)
    train=MomentsCache(ctx.cache,'train'); val=MomentsCache(ctx.cache,'validation')
    print(f'[D{seed}] creating discriminator batch={args.disc_batch_size} R1 every {args.disc_r1_interval} steps...',flush=True)
    d=make_disc(args,device); ema=copy.deepcopy(d).eval().requires_grad_(False); opt=torch.optim.AdamW(d.parameters(),lr=args.disc_lr,betas=(0.0,0.99),weight_decay=args.disc_weight_decay); rng=np.random.default_rng(70000+seed); out=args.output_root/'discriminators'/f'seed{seed}'; out.mkdir(parents=True,exist_ok=True); logs=[]; best=-1.0; start=time.perf_counter()
    print(f'[D{seed}] preflight one real/fake batch...',flush=True)
    try:
        pb=make_batch(train,args.disc_batch_size,rng,device,strong,sem,mods,args)
        with torch.no_grad():
            _=d(pb['xr'],pb['s'],pb['t'],pb['labels'])
            _=d(pb['xf'],pb['s'],pb['t'],pb['labels'])
        del pb, _
        torch.cuda.empty_cache()
        print(f'[D{seed}] preflight OK; allocated={torch.cuda.memory_allocated(device)/2**30:.2f} GiB reserved={torch.cuda.memory_reserved(device)/2**30:.2f} GiB',flush=True)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f'discriminator preflight CUDA OOM with batch={args.disc_batch_size}, '
            f'allocator_limit={args.cuda_allocator_limit_gib} GiB. '
            'Retry with --disc-batch-size 16 --cuda-allocator-limit-gib 10'
        ) from e
    for step in range(1,args.disc_steps+1):
        try:
            d.train(); b=make_batch(train,args.disc_batch_size,rng,device,strong,sem,mods,args); real=b['xr'].detach(); use_r1=args.disc_r1_gamma>0 and step%args.disc_r1_interval==0
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f'CUDA OOM before discriminator forward at step={step}, '
                f'batch={args.disc_batch_size}, allocator_limit={args.cuda_allocator_limit_gib} GiB'
            ) from e
        if use_r1: real.requires_grad_(True)
        rl=d(real,b['s'],b['t'],b['labels']); fl=d(b['xf'].detach(),b['s'],b['t'],b['labels']); logistic=F.softplus(-rl).mean()+F.softplus(fl).mean(); loss=logistic; r1=torch.zeros((),device=device)
        if use_r1:
            gr=torch.autograd.grad(rl.sum(),real,create_graph=True,retain_graph=True)[0]; r1=gr.float().square().flatten(1).mean(1).mean(); loss=loss+0.5*args.disc_r1_gamma*args.disc_r1_interval*r1
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(d.parameters(),args.disc_grad_clip); opt.step(); ema_update(ema,d,args.disc_ema_decay)
        if step==1 or step%args.disc_log_every==0 or step==args.disc_steps:
            row=dict(step=step,loss=float(loss.detach()),logistic=float(logistic.detach()),r1=float(r1.detach()),acc=float(0.5*((rl>0).float().mean()+(fl<0).float().mean())),elapsed=time.perf_counter()-start); logs.append(row); write_csv(out/'train_log.csv',logs); print(f'[D{seed}] {step}/{args.disc_steps} loss={row["loss"]:.4f} acc={row["acc"]:.3f}',flush=True)
        if step%args.disc_eval_every==0 or step==args.disc_steps:
            m=eval_disc_auc(ema,val,90000+seed+step,device,strong,sem,mods,args); m.update(step=step,seed=seed); atomic_json(out/f'val_{step:06d}.json',m); print(f'[D{seed}] val AUC={m["auc"]:.4f}',flush=True)
            if m['auc']>best: best=m['auc']; save_disc(out/'best.pt',d,ema,opt,step,seed,best,args,meta['checkpoint_sha256']); atomic_json(out/'best_validation.json',m)
        if step%args.disc_save_every==0 or step==args.disc_steps: save_disc(out/'latest.pt',d,ema,opt,step,seed,best,args,meta['checkpoint_sha256'])
    print(f'[D{seed}] complete best AUC={best:.4f}',flush=True)

# ensemble + audit

def load_ensemble(args,device):
    out=[]
    for seed in args.disc_seeds:
        p=args.output_root/'discriminators'/f'seed{seed}'/'best.pt'
        if not p.is_file(): raise FileNotFoundError(p)
        d,_=load_disc(p,device); out.append(d)
    return out

def ens_logits(ens,x,s,t,lab): return torch.stack([d(x,s,t,lab) for d in ens]).mean(0)

def ens_grad(ens,x,s,t,lab):
    with torch.enable_grad():
        y=x.detach().float().requires_grad_(True); l=ens_logits(ens,y,s,t,lab); g=torch.autograd.grad(l.sum(),y)[0]
    return l.detach(),g.detach()

def aggregate_rows(rows):
    out=[]
    for key in sorted({(r['weak'],r['s']) for r in rows}):
        sub=[r for r in rows if (r['weak'],r['s'])==key]; rr=np.array([r['disc_real_logit'] for r in sub]); ff=np.array([r['disc_fake_logit'] for r in sub]); z=dict(weak=key[0],s=float(key[1]),target_t=float(np.mean([r['target_t'] for r in sub])),dt=float(np.mean([r['dt'] for r in sub])),n=len(sub),disc_auc=binary_auc(rr,ff))
        for fld in ('gap_rms','grad_rms','cos_grad_gap','inner_mean','signed_coeff','positive_coeff','finite_delta_plus','finite_delta_minus'):
            v=np.array([r[fld] for r in sub],float); z[fld+'_mean']=float(v.mean()); z[fld+'_std']=float(v.std(ddof=1)) if len(v)>1 else 0.0
        z['positive_alignment_fraction']=float(np.mean([r['inner_mean']>0 for r in sub])); out.append(z)
    return out

def historical_static_fid(ctx):
    for p in (ctx.data/'checkpoint_reference_long_study_v1/summary/static_maturity_best.csv',ctx.repo/'checkpoint_reference_long_study_v1/summary/static_maturity_best.csv'):
        if p.is_file():
            d={}
            with p.open('r',encoding='utf-8',newline='') as f:
                for r in csv.DictReader(f):
                    if r.get('reference'): d[r['reference']]=float(r['fid'])
            if d: return d
    return {}

def worker_audit(args):
    ctx=build_repo_context(args); mods=import_repo_modules(ctx.repo); device=setup_device(123456); mods['configure_cuda_allocator'](device,limit_gib=args.cuda_allocator_limit_gib)
    strong,sem,smeta,sit,source=load_field(ctx.strong_checkpoint,device,ctx,mods); ens=load_ensemble(args,device); cache=MomentsCache(ctx.cache,'validation'); rng=np.random.default_rng(123987); rows=[]; ad=args.output_root/'audit'; ad.mkdir(parents=True,exist_ok=True)
    for step in args.weak_steps:
        name=f'v{step//1000}'; wp=ctx.ckpt_dir/f'step_{step:08d}.pt'
        if not wp.is_file(): print('[audit] skip',wp); continue
        weak,wsem,_=mods['load_sit_field_model'](checkpoint_path=wp,weights='ema',sit_module=sit,source_metadata=source,device=device); print('[audit]',name,flush=True)
        for sv in args.audit_time_centers:
            dt=min(args.audit_delta,1-float(sv)-1e-4)
            if dt<=0: continue
            rem=args.audit_samples_per_bin
            while rem>0:
                bb=min(rem,args.audit_batch_size); b=make_batch(cache,bb,rng,device,strong,sem,mods,args,fixed_s=float(sv),fixed_dt=dt); wv=eval_field(weak,wsem,b['xs'],b['s'],b['labels'],mods,args.strong_bf16); gap=(b['sv']-wv).float(); fl,grad=ens_grad(ens,b['xf'],b['s'],b['t'],b['labels'])
                with torch.no_grad():
                    rl=ens_logits(ens,b['xr'],b['s'],b['t'],b['labels']); pr=args.audit_probe_gamma; plus=b['xf']+b['dt'][:,None,None,None]*pr*gap; minus=b['xf']-b['dt'][:,None,None,None]*pr*gap; pl=ens_logits(ens,plus,b['s'],b['t'],b['labels']); ml=ens_logits(ens,minus,b['s'],b['t'],b['labels'])
                inner=(grad*gap).mean((1,2,3)); gm=gap.square().mean((1,2,3)).clamp_min(1e-12); dm=grad.square().mean((1,2,3)).clamp_min(1e-12); cos=inner/torch.sqrt(gm*dm); signed=b['dt']*inner/gm; pos=b['dt']*F.relu(inner)/gm
                for i in range(bb): rows.append(dict(weak=name,weak_step=step,s=float(b['s'][i]),target_t=float(b['t'][i]),dt=float(b['dt'][i]),disc_real_logit=float(rl[i]),disc_fake_logit=float(fl[i]),gap_rms=float(torch.sqrt(gm[i])),grad_rms=float(torch.sqrt(dm[i])),cos_grad_gap=float(cos[i]),inner_mean=float(inner[i]),signed_coeff=float(signed[i]),positive_coeff=float(pos[i]),finite_delta_plus=float(pl[i]-fl[i]),finite_delta_minus=float(ml[i]-fl[i])))
                rem-=bb
        del weak; gc.collect(); torch.cuda.empty_cache()
    write_csv(ad/'samples.csv',rows); agg=aggregate_rows(rows); write_csv(ad/'aggregate.csv',agg); hist=historical_static_fid(ctx); by={}
    for name in sorted({r['weak'] for r in agg}):
        sub=[r for r in agg if r['weak']==name]; early=[r for r in sub if r['s']<0.5]; late=[r for r in sub if r['s']>=0.5]
        mean=lambda items,f: float(np.mean([x[f] for x in items])) if items else float('nan')
        by[name]=dict(positive_coeff_full=mean(sub,'positive_coeff_mean'),positive_coeff_early=mean(early,'positive_coeff_mean'),positive_coeff_late=mean(late,'positive_coeff_mean'),cos_full=mean(sub,'cos_grad_gap_mean'),cos_early=mean(early,'cos_grad_gap_mean'),cos_late=mean(late,'cos_grad_gap_mean'),finite_plus_full=mean(sub,'finite_delta_plus_mean'),finite_plus_early=mean(early,'finite_delta_plus_mean'),finite_plus_late=mean(late,'finite_delta_plus_mean'),historical_static_fid=hist.get(name))
    matched=[(n,z) for n,z in by.items() if z['historical_static_fid'] is not None]; corr={}
    if len(matched)>=3:
        for f in ('positive_coeff_full','positive_coeff_early','finite_plus_full','finite_plus_early'): corr[f]=spearman([z[f] for _,z in matched],[-z['historical_static_fid'] for _,z in matched])
    dv={}
    for seed in args.disc_seeds:
        p=args.output_root/'discriminators'/f'seed{seed}'/'best_validation.json';
        if p.is_file(): dv[str(seed)]=read_json(p)
    ext={}
    for p in (ctx.data/'external_v180_temporal_utility_fid1k_v1/summary/summary.json',ctx.repo/'external_v180_temporal_utility_fid1k_v1/summary/summary.json'):
        if p.is_file(): ext=read_json(p); break
    v180=by.get('v180',{}); supported=bool(v180 and v180['positive_coeff_early']>v180['positive_coeff_late'])
    summary=dict(format='eqvae_discriminator_ag_transport_audit_v1',disc_validation=dv,time_centers=list(args.audit_time_centers),audit_delta=args.audit_delta,weak_summary=by,spearman_vs_historical_static_quality=corr,v180_early_positive_coeff=v180.get('positive_coeff_early'),v180_late_positive_coeff=v180.get('positive_coeff_late'),v180_mechanism_direction_consistent_with_observed_high_noise_dominance=supported,external_v180_temporal_summary=ext,sample_csv=str(ad/'samples.csv'),aggregate_csv=str(ad/'aggregate.csv'))
    atomic_json(ad/'summary.json',summary)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        for n in ('v180','v270','v400','v600'):
            q=[r for r in agg if r['weak']==n]
            if q: plt.plot([r['s'] for r in q],[r['positive_coeff_mean'] for r in q],marker='o',label=n)
        plt.xlabel('SiT time s (0=noise, 1=data)'); plt.ylabel('mean positive projection coefficient'); plt.legend(); plt.tight_layout(); plt.savefig(ad/'positive_coeff_by_time.png',dpi=180); plt.close()
        plt.figure(figsize=(8,5))
        for n in ('v180','v270','v400','v600'):
            q=[r for r in agg if r['weak']==n]
            if q: plt.plot([r['s'] for r in q],[r['cos_grad_gap_mean'] for r in q],marker='o',label=n)
        plt.xlabel('SiT time s (0=noise, 1=data)'); plt.ylabel('mean cos(grad log-ratio, S-W)'); plt.legend(); plt.tight_layout(); plt.savefig(ad/'cosine_by_time.png',dpi=180); plt.close()
    except Exception as e: print('[audit] plotting skipped',e)
    print('\n=== AUDIT ===');
    for n,z in by.items(): print(n,'coeff early/late',z['positive_coeff_early'],z['positive_coeff_late'],'cos',z['cos_early'],z['cos_late'])
    print('v180 temporal direction supported:',supported)

# sampling policies

class Field:
    def __init__(self,mode,strong,sem,weak,wsem,mods,labels,args,c,ens=None,profile=None):
        self.mode=mode; self.strong=strong; self.sem=sem; self.weak=weak; self.wsem=wsem; self.mods=mods; self.labels=labels; self.args=args; self.c=c; self.ens=ens; self.profile=profile; self.nfe=0; self.sf=0; self.wf=0; self.dg=0; self.gs=0.; self.gc=0; self.gp=0; self.gclip=0
    def S(self,x,t): self.sf+=1; return eval_field(self.strong,self.sem,x,t,self.labels,self.mods,self.args.strong_bf16)
    def W(self,x,t): self.wf+=1; return eval_field(self.weak,self.wsem,x,t,self.labels,self.mods,self.args.strong_bf16)
    def rec(self,g):
        z=g.detach().float().reshape(-1); self.gs+=float(z.sum()); self.gc+=len(z); self.gp+=int((z>0).sum()); mx=float(self.c.get('gamma_max',self.args.policy_gamma_max)); self.gclip+=int((z>=mx-1e-7).sum())
    def adaptive(self,tv,x,sv,gap,times,gate=False):
        dt=min(float(self.c.get('lookahead',self.args.policy_lookahead)),1-tv-1e-4)
        if dt<=0: return torch.zeros(len(x),1,1,1,device=x.device)
        target=torch.full_like(times,tv+dt); y=x.detach().float()+dt*sv.detach().float(); _,gr=ens_grad(self.ens,y,times,target,self.labels); inn=(gr*gap).mean((1,2,3))
        if gate: gamma=float(self.c['gamma'])*(inn>0).float()
        else:
            ms=gap.square().mean((1,2,3)).clamp_min(self.args.policy_gap_eps); gamma=float(self.c['gain'])*dt*F.relu(inn)/ms
        gamma=gamma.clamp(0,float(self.c.get('gamma_max',self.args.policy_gamma_max))); self.dg+=1; return gamma[:,None,None,None]
    def __call__(self,t,x):
        self.nfe+=1; tv=float(t.detach().float().item()); times=torch.full((len(x),),tv,device=x.device); sv=self.S(x,times)
        if self.mode=='baseline': return sv.detach()
        wv=self.W(x,times); gap=(sv-wv).float()
        if self.mode=='fixed': g=torch.full((len(x),1,1,1),float(self.c['gamma']),device=x.device)
        elif self.mode=='early': g=torch.full((len(x),1,1,1),float(self.c['gamma']) if tv<float(self.c['cutoff']) else 0.,device=x.device)
        elif self.mode=='disc_state': g=self.adaptive(tv,x,sv,gap,times,False)
        elif self.mode=='disc_gate': g=self.adaptive(tv,x,sv,gap,times,True)
        elif self.mode=='disc_time':
            centers,coef=self.profile; val=float(self.c['gain'])*max(0.,float(np.interp(tv,centers,coef))); val=min(val,float(self.c.get('gamma_max',self.args.policy_gamma_max))); g=torch.full((len(x),1,1,1),val,device=x.device)
        else: raise ValueError(self.mode)
        self.rec(g); return (sv+g*gap).detach()
    def summary(self):
        return dict(nfe=self.nfe,strong_forwards=self.sf,weak_forwards=self.wf,disc_grad_evals=self.dg,gamma_mean=self.gs/self.gc if self.gc else 0.,gamma_positive_fraction=self.gp/self.gc if self.gc else 0.,gamma_clipped_fraction=self.gclip/self.gc if self.gc else 0.)

def time_profile(args):
    p=args.output_root/'audit'/'aggregate.csv'; rows=[]
    with p.open('r',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['weak']=='v180': rows.append(r)
    rows.sort(key=lambda r:float(r['s'])); return np.array([float(r['s']) for r in rows]),np.array([float(r['positive_coeff_mean']) for r in rows])

def adm_python(args):
    p=(args.adm_python if args.adm_python else DEFAULT_ADM_PYTHON).expanduser().absolute()
    if not p.is_file(): raise FileNotFoundError(p)
    probe=subprocess.run([str(p),'-c','import tensorflow.compat.v1 as tf; print(tf.__version__)'],capture_output=True,text=True)
    if probe.returncode!=0: raise RuntimeError(f'ADM Python invalid: {p}\n{probe.stderr}')
    return p

def worker_sample(args,cp):
    ctx=build_repo_context(args); mods=import_repo_modules(ctx.repo); c=read_json(cp); seed=int(c['seed']); mode=c['mode']; device=setup_device(222000+seed); mods['configure_cuda_allocator'](device,limit_gib=args.cuda_allocator_limit_gib)
    strong,sem,smeta,sit,source=load_field(ctx.strong_checkpoint,device,ctx,mods); weak=wsem=wmeta=None
    if mode!='baseline': weak,wsem,wmeta=mods['load_sit_field_model'](checkpoint_path=ctx.ckpt_dir/'step_00180000.pt',weights='ema',sit_module=sit,source_metadata=source,device=device)
    ens=load_ensemble(args,device) if mode in {'disc_state','disc_gate'} else None; profile=time_profile(args) if mode=='disc_time' else None
    from diffusers.models import AutoencoderKL
    vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse',local_files_only=True); vae.to(device).eval().requires_grad_(False)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    out=cp.parent; images=np.empty((args.num_samples,256,256,3),np.uint8); labs=np.empty(args.num_samples,np.int16); nd=hashlib.sha256(); ld=hashlib.sha256(); cur=0; totals=dict(nfe=0,strong_forwards=0,weak_forwards=0,disc_grad_evals=0,gs=0.,gc=0,gp=0.,gclip=0.); start=time.perf_counter()
    while cur<args.num_samples:
        b=min(args.sample_batch_size,args.num_samples-cur); noise=torch.randn(b,*LATENT_SHAPE,device=device); labels=torch.randint(0,NUM_CLASSES,(b,),device=device); f=Field(mode,strong,sem,weak,wsem,mods,labels,args,c,ens,profile); endpoint=mods['integrate_velocity'](noise,f,num_output_points=args.num_output_points,atol=args.atol,rtol=args.rtol)
        if not torch.isfinite(endpoint).all(): raise FloatingPointError(c['name'])
        with torch.no_grad(): dec=mods['decode_latents_in_chunks'](vae,endpoint,scaling_factor=SD_VAE_SCALING_FACTOR,chunk_size=args.vae_decode_batch_size)
        stop=cur+b; images[cur:stop]=mods['official_pixel_quantization'](dec); labs[cur:stop]=labels.cpu().numpy().astype(np.int16,copy=False); nd.update(noise.detach().cpu().contiguous().numpy().tobytes()); ld.update(labels.detach().cpu().contiguous().numpy().tobytes()); s=f.summary(); totals['nfe']+=s['nfe']; totals['strong_forwards']+=s['strong_forwards']; totals['weak_forwards']+=s['weak_forwards']; totals['disc_grad_evals']+=s['disc_grad_evals']; cnt=s['nfe']*b; totals['gs']+=s['gamma_mean']*cnt; totals['gc']+=cnt; totals['gp']+=s['gamma_positive_fraction']*cnt; totals['gclip']+=s['gamma_clipped_fraction']*cnt; cur=stop
        if cur==b or cur==args.num_samples or cur%256==0: print(f'[{c["name"]}] {cur}/{args.num_samples} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    sp=out/f'samples_n{args.num_samples}.npz'; np.savez(sp,arr_0=images); np.save(out/f'labels_n{args.num_samples}.npy',labs,allow_pickle=False); nsh,lsh=nd.hexdigest(),ld.hexdigest()
    if seed==0 and args.num_samples==1000 and args.sample_batch_size==8:
        if nsh!=EXPECTED_SEED0_NOISE_SHA256 or lsh!=EXPECTED_SEED0_LABEL_SHA256: raise RuntimeError(f'pairing fingerprint mismatch noise={nsh} labels={lsh}')
    den=max(1,totals['gc']); manifest=dict(format='eqvae_discriminator_ag_transport_samples_v1',condition=c,strong=smeta,weak=wmeta,sampling=dict(num_samples=args.num_samples,batch_size=args.sample_batch_size,seed=seed,num_output_points=args.num_output_points,atol=args.atol,rtol=args.rtol,integrator='dopri5',time_convention='t=0 noise -> t=1 data'),noise_sha256=nsh,label_sha256=lsh,field_stats=dict(nfe=totals['nfe'],strong_forwards=totals['strong_forwards'],weak_forwards=totals['weak_forwards'],disc_grad_evals=totals['disc_grad_evals'],gamma_mean=totals['gs']/den,gamma_positive_fraction=totals['gp']/den,gamma_clipped_fraction=totals['gclip']/den),samples=str(sp),elapsed_seconds=time.perf_counter()-start); atomic_json(out/'sampling_manifest.json',manifest)
    # Release sampling models before the ADM evaluator subprocess claims GPU memory.
    del vae, strong
    if weak is not None: del weak
    if ens is not None: del ens
    gc.collect(); torch.cuda.empty_cache()
    ap=adm_python(args); mp=out/'adm_metrics.json'; subprocess.run([str(ap),str(ctx.repo/'experiments/compute_adm_fid.py'),'--reference',str(ctx.reference_stats),'--samples',str(sp),'--batch-size',str(args.fid_batch_size),'--gpu-memory-fraction',str(args.fid_gpu_memory_fraction),'--output',str(mp)],cwd=ctx.repo,env=os.environ.copy(),check=True); metrics=read_json(mp); atomic_json(out/'condition_result.json',dict(format='eqvae_discriminator_ag_transport_condition_result_v1',condition=c,sampling_manifest=manifest,metrics=metrics))
    if not args.keep_samples: sp.unlink(missing_ok=True)
    print(f'[complete] {c["name"]}: FID={float(metrics["fid"]):.4f} gamma_mean={manifest["field_stats"]["gamma_mean"]:.4f}',flush=True)

# orchestration

def mkcond(name,mode,seed,gamma=None,gain=None,cutoff=None,gamma_max=None,lookahead=None):
    c=dict(format='eqvae_discriminator_ag_transport_condition_v1',name=name,mode=mode,seed=int(seed),formula='S + gamma(x,t)*(S-W)')
    if gamma is not None: c['gamma']=float(gamma)
    if gain is not None: c['gain']=float(gain)
    if cutoff is not None: c['cutoff']=float(cutoff)
    if gamma_max is not None: c['gamma_max']=float(gamma_max)
    if lookahead is not None: c['lookahead']=float(lookahead)
    return c

def _tail_text(path, n=120):
    try:
        lines=path.read_text(encoding='utf-8',errors='replace').splitlines()
        return '\n'.join(lines[-int(n):])
    except Exception as e:
        return f'<could not read log tail: {e}>'

def run_logged(cmd,gpu,log,cwd):
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env.setdefault('OMP_NUM_THREADS','1'); env.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True'); log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\n=== launch ===\n'+' '.join(map(str,cmd))+'\n'); f.flush()
        p=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
    if p.returncode!=0:
        tail=_tail_text(log,120)
        print(f'\n[FAILED GPU {gpu}] {log}\n----- log tail -----\n{tail}\n----- end log tail -----\n',flush=True)
        raise subprocess.CalledProcessError(p.returncode,cmd)

def child_args(args):
    out=['--output-root',str(args.output_root),'--disc-seeds',','.join(map(str,args.disc_seeds)),'--weak-steps',','.join(map(str,args.weak_steps)),'--disc-steps',str(args.disc_steps),'--disc-batch-size',str(args.disc_batch_size),'--disc-lr',repr(args.disc_lr),'--disc-weight-decay',repr(args.disc_weight_decay),'--disc-ema-decay',repr(args.disc_ema_decay),'--disc-r1-gamma',repr(args.disc_r1_gamma),'--disc-r1-interval',str(args.disc_r1_interval),'--disc-grad-clip',repr(args.disc_grad_clip),'--disc-log-every',str(args.disc_log_every),'--disc-eval-every',str(args.disc_eval_every),'--disc-save-every',str(args.disc_save_every),'--disc-val-batches',str(args.disc_val_batches),'--disc-base-channels',str(args.disc_base_channels),'--disc-cond-dim',str(args.disc_cond_dim),'--disc-class-dim',str(args.disc_class_dim),'--disc-time-bands',str(args.disc_time_bands),'--disc-dt-min',repr(args.disc_dt_min),'--disc-dt-max',repr(args.disc_dt_max),'--disc-s-max',repr(args.disc_s_max),'--audit-time-centers',','.join(map(str,args.audit_time_centers)),'--audit-delta',repr(args.audit_delta),'--audit-probe-gamma',repr(args.audit_probe_gamma),'--audit-samples-per-bin',str(args.audit_samples_per_bin),'--audit-batch-size',str(args.audit_batch_size),'--policy-lookahead',repr(args.policy_lookahead),'--policy-gamma-max',repr(args.policy_gamma_max),'--policy-gap-eps',repr(args.policy_gap_eps),'--num-samples',str(args.num_samples),'--sample-batch-size',str(args.sample_batch_size),'--vae-decode-batch-size',str(args.vae_decode_batch_size),'--num-output-points',str(args.num_output_points),'--atol',repr(args.atol),'--rtol',repr(args.rtol),'--fid-batch-size',str(args.fid_batch_size),'--fid-gpu-memory-fraction',repr(args.fid_gpu_memory_fraction),'--cuda-allocator-limit-gib',repr(args.cuda_allocator_limit_gib),('--strong-bf16' if args.strong_bf16 else '--no-strong-bf16')]
    if args.keep_samples: out+=['--keep-samples']
    if args.data_root: out+=['--data-root',str(args.data_root)]
    if args.adm_python: out+=['--adm-python',str(args.adm_python)]
    return out

def train_parallel(args,ctx):
    if len(args.disc_seeds)>len(args.gpus): raise ValueError('need >=1 GPU per discriminator seed')
    jobs=[]; script=Path(__file__).resolve()
    for seed,gpu in zip(args.disc_seeds,args.gpus):
        best=args.output_root/'discriminators'/f'seed{seed}'/'best.pt'
        if best.is_file() and not args.force_retrain_discriminator: print('[reuse] disc',seed); continue
        jobs.append((gpu,[sys.executable,str(script),'--worker','train-disc','--disc-seed',str(seed),*child_args(args)],args.output_root/'logs'/f'train_disc_seed{seed}.log'))
    with ThreadPoolExecutor(max_workers=max(1,len(jobs))) as pool:
        fs=[pool.submit(run_logged,cmd,gpu,log,ctx.repo) for gpu,cmd,log in jobs]
        for f in fs: f.result()

def run_audit(args,ctx):
    p=args.output_root/'audit'/'summary.json'
    if p.is_file() and not args.force_audit: print('[reuse] audit'); return
    run_logged([sys.executable,str(Path(__file__).resolve()),'--worker','audit',*child_args(args)],args.gpus[0],args.output_root/'logs'/'audit.log',ctx.repo)

def write_condition(root,phase,c):
    od=root/phase/c['name']; od.mkdir(parents=True,exist_ok=True); p=od/'condition.json'; atomic_json(p,c); return p

def valid_result(p,n):
    if not p.is_file(): return False
    try:
        x=read_json(p); return int(x['sampling_manifest']['sampling']['num_samples'])==n and all(isinstance(x['metrics'].get(k),(int,float)) for k in ('fid','sfid','inception_score'))
    except Exception: return False

def run_conditions(args,ctx,phase,conds):
    paths=[write_condition(args.output_root,phase,c) for c in conds]; lanes={g:[] for g in args.gpus}
    for i,p in enumerate(paths): lanes[args.gpus[i%len(args.gpus)]].append(p)
    def lane(gpu,ps):
        for cp in ps:
            rp=cp.parent/'condition_result.json'
            if valid_result(rp,args.num_samples): print('[reuse]',phase,cp.parent.name); continue
            print('[launch GPU',gpu,']',phase,cp.parent.name,flush=True); run_logged([sys.executable,str(Path(__file__).resolve()),'--worker','sample','--condition-json',str(cp),*child_args(args)],gpu,args.output_root/'logs'/f'{phase}__{cp.parent.name}.log',ctx.repo)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        fs=[pool.submit(lane,g,ps) for g,ps in lanes.items() if ps]
        for f in fs: f.result()
    return [read_json(p.parent/'condition_result.json') for p in paths]

def row(x):
    c=x['condition']; q=x['metrics']; m=x['sampling_manifest']; return dict(name=c['name'],mode=c['mode'],seed=c['seed'],gamma=c.get('gamma',''),gain=c.get('gain',''),cutoff=c.get('cutoff',''),fid=float(q['fid']),sfid=float(q['sfid']),inception_score=float(q['inception_score']),gamma_mean=float(m['field_stats']['gamma_mean']),gamma_positive_fraction=float(m['field_stats']['gamma_positive_fraction']),gamma_clipped_fraction=float(m['field_stats']['gamma_clipped_fraction']),noise_sha256=m['noise_sha256'],label_sha256=m['label_sha256'])
def gain_grid(args):
    a=read_json(args.output_root/'audit'/'summary.json'); e=float(a['v180_early_positive_coeff'])
    if not math.isfinite(e) or e<=1e-12: return tuple(args.policy_fallback_gains)
    center=min(max(args.policy_reference_gamma/e,1e-4),1e5); return tuple(sorted({round(center*m,10) for m in args.policy_gain_multipliers if center*m>0}))

def orchestrate(args):
    ctx=build_repo_context(args); args.output_root.mkdir(parents=True,exist_ok=True); atomic_json(args.output_root/'request.json',dict(format='eqvae_discriminator_ag_transport_request_v1',gpus=list(args.gpus),disc_seeds=list(args.disc_seeds),disc_steps=args.disc_steps,weak_steps=list(args.weak_steps),audit_time_centers=list(args.audit_time_centers),num_samples=args.num_samples,hypothesis='AG usefulness follows alignment of S-W with discriminator-estimated marginal transport residual'))
    print('\n=== PHASE 1 train discriminators ==='); train_parallel(args,ctx)
    print('\n=== PHASE 2 mechanism audit ==='); run_audit(args,ctx); aud=read_json(args.output_root/'audit'/'summary.json'); print('v180 coeff early/late:',aud.get('v180_early_positive_coeff'),aud.get('v180_late_positive_coeff')); print('temporal direction supported:',aud.get('v180_mechanism_direction_consistent_with_observed_high_noise_dominance'))
    print('\n=== PHASE 3 seed0 policy screen ==='); gains=gain_grid(args); print('gain grid:',gains); c0=[mkcond('seed0_baseline','baseline',0),mkcond('seed0_fixed_v180_g3p05','fixed',0,gamma=3.05),mkcond('seed0_early_v180_g4p5_t0p5','early',0,gamma=4.5,cutoff=.5)]+[mkcond(f'seed0_disc_state_k{tag_float(k)}','disc_state',0,gain=k,gamma_max=args.policy_gamma_max,lookahead=args.policy_lookahead) for k in gains]; r0=run_conditions(args,ctx,'03_seed0_screen',c0); rows0=[row(x) for x in r0]; write_csv(args.output_root/'summary'/'seed0_screen.csv',rows0); states=[r for r in rows0 if r['mode']=='disc_state']; best=min(states,key=lambda r:r['fid']); kg=float(best['gain']); print('best state gain',kg,'FID',best['fid'])
    print('\n=== PHASE 4 controls ==='); cc=[mkcond(f'seed0_disc_time_k{tag_float(kg)}','disc_time',0,gain=kg,gamma_max=args.policy_gamma_max),mkcond('seed0_disc_gate_fixed_g4p5','disc_gate',0,gamma=4.5,gamma_max=args.policy_gamma_max,lookahead=args.policy_lookahead)]; rc=run_conditions(args,ctx,'04_seed0_controls',cc)
    print('\n=== PHASE 5 seed1 confirm ==='); c1=[mkcond('seed1_baseline','baseline',1),mkcond('seed1_fixed_v180_g3p05','fixed',1,gamma=3.05),mkcond('seed1_early_v180_g4p5_t0p5','early',1,gamma=4.5,cutoff=.5),mkcond(f'seed1_disc_state_k{tag_float(kg)}','disc_state',1,gain=kg,gamma_max=args.policy_gamma_max,lookahead=args.policy_lookahead),mkcond(f'seed1_disc_time_k{tag_float(kg)}','disc_time',1,gain=kg,gamma_max=args.policy_gamma_max),mkcond('seed1_disc_gate_fixed_g4p5','disc_gate',1,gamma=4.5,gamma_max=args.policy_gamma_max,lookahead=args.policy_lookahead)]; r1=run_conditions(args,ctx,'05_seed1_confirm',c1)
    allr=[row(x) for x in r0+rc+r1]; write_csv(args.output_root/'summary'/'all_policy_results.csv',allr); by={r['name']:r for r in allr}; pairs={'baseline':['seed0_baseline','seed1_baseline'],'fixed_g3p05':['seed0_fixed_v180_g3p05','seed1_fixed_v180_g3p05'],'early_g4p5_t0p5':['seed0_early_v180_g4p5_t0p5','seed1_early_v180_g4p5_t0p5'],'disc_state_best':[best['name'],f'seed1_disc_state_k{tag_float(kg)}'],'disc_time_best':[f'seed0_disc_time_k{tag_float(kg)}',f'seed1_disc_time_k{tag_float(kg)}'],'disc_gate_fixed':['seed0_disc_gate_fixed_g4p5','seed1_disc_gate_fixed_g4p5']}; means={}
    for n,names in pairs.items():
        q=[by[x] for x in names if x in by]; means[n]=dict(fid_mean=float(np.mean([x['fid'] for x in q])),fid_values=[x['fid'] for x in q],sfid_mean=float(np.mean([x['sfid'] for x in q])),is_mean=float(np.mean([x['inception_score'] for x in q])),gamma_mean=float(np.mean([x['gamma_mean'] for x in q])))
    paircheck={}
    for seed in (0,1):
        q=[r for r in allr if r['seed']==seed]; paircheck[str(seed)]=dict(noise=sorted({r['noise_sha256'] for r in q}),labels=sorted({r['label_sha256'] for r in q}),exact=len({r['noise_sha256'] for r in q})==1 and len({r['label_sha256'] for r in q})==1)
    concl=dict(mechanism_temporal_direction_supported=aud.get('v180_mechanism_direction_consistent_with_observed_high_noise_dominance'),state_policy_beats_fixed=means['disc_state_best']['fid_mean']<means['fixed_g3p05']['fid_mean'],state_policy_beats_early_cutoff=means['disc_state_best']['fid_mean']<means['early_g4p5_t0p5']['fid_mean'],state_policy_beats_time_only=means['disc_state_best']['fid_mean']<means['disc_time_best']['fid_mean'])
    final=dict(format='eqvae_discriminator_ag_transport_final_summary_v1',best_seed0_gain=kg,audit_summary=aud,pairing=paircheck,two_seed_means=means,conclusion_flags=concl,all_policy_results_csv=str(args.output_root/'summary'/'all_policy_results.csv')); atomic_json(args.output_root/'summary'/'final_summary.json',final)
    try:
        import matplotlib.pyplot as plt
        order=['baseline','fixed_g3p05','early_g4p5_t0p5','disc_time_best','disc_gate_fixed','disc_state_best']; vals=[means[n]['fid_mean'] for n in order]; plt.figure(figsize=(9,5)); plt.bar(np.arange(len(vals)),vals); plt.xticks(np.arange(len(vals)),order,rotation=20,ha='right'); plt.ylabel('paired FID-1K mean'); plt.tight_layout(); plt.savefig(args.output_root/'summary'/'policy_fid_comparison.png',dpi=180); plt.close()
    except Exception as e: print('[summary] plot skipped',e)
    print('\n=== FINAL ===');
    for n,z in means.items(): print(f'{n:22s} FID={z["fid_mean"]:.4f} values={z["fid_values"]} gamma_mean={z["gamma_mean"]:.4f}')
    print('flags:',json.dumps(concl,indent=2)); print('summary:',args.output_root/'summary'/'final_summary.json')

# CLI

def self_test():
    assert abs(binary_auc([2,3,4],[-1,0,1])-1)<1e-12; assert abs(spearman([1,2,3],[4,5,6])-1)<1e-12; assert abs(spearman([1,2,3],[6,5,4])+1)<1e-12; assert all(0<x<1 for x in DEFAULT_TIME_CENTERS); print('SELF_TEST_OK')

def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--worker',choices=('train-disc','audit','sample')); p.add_argument('--disc-seed',type=int); p.add_argument('--condition-json',type=Path); p.add_argument('--self-test',action='store_true'); p.add_argument('--gpus',type=parse_gpu_list,default=parse_gpu_list('1,3')); p.add_argument('--data-root',type=Path); p.add_argument('--output-root',type=Path); p.add_argument('--adm-python',type=Path)
    p.add_argument('--disc-seeds',type=parse_int_tuple,default=(0,1)); p.add_argument('--disc-steps',type=int,default=15000); p.add_argument('--disc-batch-size',type=int,default=64); p.add_argument('--disc-lr',type=float,default=2e-4); p.add_argument('--disc-weight-decay',type=float,default=1e-4); p.add_argument('--disc-ema-decay',type=float,default=.999); p.add_argument('--disc-r1-gamma',type=float,default=1.); p.add_argument('--disc-r1-interval',type=int,default=16); p.add_argument('--disc-grad-clip',type=float,default=10.); p.add_argument('--disc-log-every',type=int,default=100); p.add_argument('--disc-eval-every',type=int,default=1000); p.add_argument('--disc-save-every',type=int,default=1000); p.add_argument('--disc-val-batches',type=int,default=16); p.add_argument('--disc-base-channels',type=int,default=64); p.add_argument('--disc-cond-dim',type=int,default=256); p.add_argument('--disc-class-dim',type=int,default=128); p.add_argument('--disc-time-bands',type=int,default=16); p.add_argument('--disc-dt-min',type=float,default=.025); p.add_argument('--disc-dt-max',type=float,default=.125); p.add_argument('--disc-s-max',type=float,default=.95); p.add_argument('--force-retrain-discriminator',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--weak-steps',type=parse_int_tuple,default=DEFAULT_WEAK_STEPS); p.add_argument('--audit-time-centers',type=parse_float_tuple,default=DEFAULT_TIME_CENTERS); p.add_argument('--audit-delta',type=float,default=.05); p.add_argument('--audit-probe-gamma',type=float,default=1.); p.add_argument('--audit-samples-per-bin',type=int,default=256); p.add_argument('--audit-batch-size',type=int,default=64); p.add_argument('--force-audit',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--policy-lookahead',type=float,default=.05); p.add_argument('--policy-gamma-max',type=float,default=8.); p.add_argument('--policy-gap-eps',type=float,default=1e-8); p.add_argument('--policy-reference-gamma',type=float,default=3.); p.add_argument('--policy-gain-multipliers',type=parse_float_tuple,default=(.25,.5,1.,2.,4.)); p.add_argument('--policy-fallback-gains',type=parse_float_tuple,default=(.25,.5,1.,2.,4.,8.,16.))
    p.add_argument('--num-samples',type=int,default=1000); p.add_argument('--sample-batch-size',type=int,default=8); p.add_argument('--vae-decode-batch-size',type=int,default=2); p.add_argument('--num-output-points',type=int,default=250); p.add_argument('--atol',type=float,default=1e-6); p.add_argument('--rtol',type=float,default=1e-3); p.add_argument('--fid-batch-size',type=int,default=8); p.add_argument('--fid-gpu-memory-fraction',type=float,default=.25); p.add_argument('--cuda-allocator-limit-gib',type=float,default=15.); p.add_argument('--strong-bf16',action=argparse.BooleanOptionalAction,default=False); p.add_argument('--keep-samples',action='store_true'); return p

def validate(a):
    if a.disc_steps<1 or a.disc_batch_size<2: raise ValueError('bad discriminator size')
    if not (0<a.disc_dt_min<=a.disc_dt_max<1): raise ValueError('bad dt range')
    if not (0<a.disc_s_max<=1-a.disc_dt_min): raise ValueError('bad s_max')
    if any(not(0<x<1) for x in a.audit_time_centers): raise ValueError('audit times must lie in (0,1)')
    if any(x<=0 or x>=800000 for x in a.weak_steps): raise ValueError('weak steps must lie in (0,800000)')

def main():
    a=parser().parse_args();
    if a.self_test: self_test(); return
    validate(a); ctx=build_repo_context(a); a.output_root=(a.output_root.expanduser().resolve() if a.output_root else ctx.data/'discriminator_ag_transport_v1'); a.output_root.mkdir(parents=True,exist_ok=True)
    if a.worker=='train-disc':
        if a.disc_seed is None: raise ValueError('--disc-seed required')
        worker_train(a,a.disc_seed); return
    if a.worker=='audit': worker_audit(a); return
    if a.worker=='sample':
        if a.condition_json is None: raise ValueError('--condition-json required')
        worker_sample(a,a.condition_json.expanduser().resolve()); return
    orchestrate(a)

if __name__=='__main__': main()