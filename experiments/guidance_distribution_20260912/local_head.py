"""An input-local readout and an architecture/data-matched contextual control."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch import nn
from experiments.guidance_pasted_20260912 import common as c

ROOT=c.EXPS/'guidance_distribution_20260912'
PROTOCOL=c.WORK/'docs/IG_INPUT_LOCAL_PROTOCOL_20260912_ZH.md'
TRAIN_SEED=2026121301
STEPS=3000
STAGE='input_local_screen_400'
N=400
KINDS=('local_base','local_half','context_base','context_half',
       'native_base','native_half','native_double','strong')
DEPTH={'sit_small':4,'raev2':8}
BATCH={'sit_small':32,'raev2':8}
DATA=c.EXPS.parent


def configure():
    c.ROOT=ROOT


def data_paths(model):
    if model=='sit_small':
        r=c.EXPS/'sit_measure_guidance_20260912'
        return [r/(name+'.npy') for name in ('train_moments','validation_moments',
                 'class_means','local_partners','random_partners','heat_sigma')]
    r=c.EXPS/'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1'
    return [r/'summary.json']+[r/s/f for s in ('train','validation') for f in ('latents.npy','metadata.npz')]


def shape(rt):
    if rt.name=='sit_small':
        return int(rt.model.pos_embed.shape[-1]),int(rt.model.x_embedder.patch_size[0]),4
    return rt.model.enc_hidden_size,rt.model.s_patch_size,rt.model.in_channels


def position(side,device):
    q=(torch.arange(side,device=device,dtype=torch.float32)+.5)/side
    yy,xx=torch.meshgrid(q,q,indexing='ij')
    return torch.stack([fun(freq*torch.pi*v) for v in (yy,xx)
                        for freq in (1.,2.) for fun in (torch.sin,torch.cos)],-1).reshape(1,side*side,8)


class Head(nn.Module):
    def __init__(self,width,outdim,side):
        super().__init__()
        self.token=nn.Linear(width,width)
        self.condition=nn.Linear(width,width,bias=False)
        self.position=nn.Linear(8,width,bias=False)
        self.output=nn.Linear(width,outdim)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        for name in ('token_mean','condition_mean'):
            self.register_buffer(name,torch.zeros(width))
        for name in ('token_std','condition_std'):
            self.register_buffer(name,torch.ones(width))
        self.register_buffer('positions',position(side,'cpu'))

    def forward(self,tokens,condition):
        x=(tokens.float()-self.token_mean)/self.token_std
        y=(condition.float()-self.condition_mean)/self.condition_std
        value=self.token(x)+self.condition(y)[:,None]+self.position(self.positions)
        return self.output(torch.nn.functional.silu(value))


def make_head(rt):
    width,patch,channels=shape(rt)
    side=c.LATENTS[rt.name][-1]//patch
    return Head(width,patch*patch*channels,side).cuda()


def unpatchify(rt,tokens):
    if rt.name=='raev2':return rt.model.unpatchify(tokens,rt.model.s_patch_size)
    from experiments.imagenet100_sit_internal_v_head import unpatchify_channels
    return unpatchify_channels(rt.model,tokens,channels=4)


def patchify(rt,x):
    _,p,_=shape(rt);b,ch,h,w=x.shape
    return x.reshape(b,ch,h//p,p,w//p,p).permute(0,2,4,3,5,1).reshape(b,h*w//p**2,ch*p*p)


@torch.no_grad()
def features(rt,x,t,y,context=True):
    if rt.name=='sit_small':
        from experiments.imagenet100_sit_internal_v_head import embed_sit_inputs
        seq,cond=embed_sit_inputs(rt.model,x,t,y);local=seq
        if context:
            for block in rt.model.blocks[:DEPTH[rt.name]]:seq=block(seq,cond)
    else:
        kwargs=dict(context=y,attn_mask=None)
        seq,_=rt.model._build_sequence(x,t,kwargs)
        n=rt.model.s_embedder.num_patches
        local=seq[:,:n];cond=seq[:,n:].mean(1)
        if context:
            mask=rt.model._build_attn_mask(seq,kwargs)
            for block in rt.model.blocks[:DEPTH[rt.name]]:
                seq=block(seq,rt.model.enc_rope,mask)
        seq=seq[:,:n]
    return dict(local=local,context=seq,condition=cond)


class Capture:
    def __init__(self,rt):
        self.rt=rt;self.values={};self.handles=[]
        def before(module,args):
            seq=args[0]
            if rt.name=='sit_small':self.values.update(local=seq,condition=args[1])
            else:
                n=rt.model.s_embedder.num_patches
                self.values.update(local=seq[:,:n],condition=seq[:,n:].mean(1))
        def after(module,args,value):
            self.values['context']=value if rt.name=='sit_small' else value[:,:rt.model.s_embedder.num_patches]
        self.handles.append(rt.model.blocks[0].register_forward_pre_hook(before))
        self.handles.append(rt.model.blocks[DEPTH[rt.name]-1].register_forward_hook(after))
    def close(self):
        for h in self.handles:h.remove()
        self.values.clear()


class RealData:
    def __init__(self,model):
        self.model=model
        if model=='sit_small':
            from experiments.sit_measure_guidance_20260912.data import Pool
            self.banks={s:Pool(s) for s in ('train','validation')}
        else:
            from experiments.train_raev2_observable_potential import load_banks
            self.banks,self.record=load_banks(data_paths(model)[0].parent)
    def draw(self,split,g,count):
        if self.model=='sit_small':return self.banks[split].draw('native',g,count)
        from experiments.train_raev2_observable_potential import batch_from_bank
        bank=self.banks[split]
        ids=torch.randint(len(bank[0]),(count,),generator=g,device='cuda').cpu().numpy()
        return batch_from_bank(bank,ids,'cuda')


def training_batch(rt,data,split,g,count):
    clean,y=data.draw(split,g,count)
    t=.01+.98*torch.rand(count,device='cuda',generator=g)
    eps=torch.randn(clean.shape,device='cuda',generator=g)
    a=t[:,None,None,None]
    if rt.name=='sit_small':z=a*clean+(1-a)*eps;target=clean-eps
    else:z=(1-a)*clean+a*eps;target=clean
    with rt.context():value=features(rt,z,t,y)
    return value,patchify(rt,target),z,t,y


@torch.inference_mode()
def checks(rt,heads):
    g=torch.Generator(device='cuda').manual_seed(TRAIN_SEED+3)
    z=torch.randn((2,*c.LATENTS[rt.name]),device='cuda',generator=g)
    t=torch.full((2,),.47,device='cuda');y=torch.tensor([0,1],device='cuda')
    rt.labels=y
    with rt.context():full0,weak0=rt.pair(z,t[0])
    capture=Capture(rt)
    with rt.context():
        full1,weak1=rt.pair(z,t[0])
        direct=features(rt,z,t,y)
    assert torch.equal(full0,full1) and torch.equal(weak0,weak1)
    equality={k:bool(torch.equal(direct[k],capture.values[k])) for k in direct}
    assert all(equality.values()),equality
    capture.close()
    p=shape(rt)[1]
    changed=torch.randn(z.shape,device='cuda',generator=g)*2
    changed[:,:,:p,:p]=z[:,:,:p,:p]
    with rt.context():
        other=features(rt,changed,t,y)
        pred1=heads['local'](direct['local'],direct['condition'])
        pred2=heads['local'](other['local'],other['condition'])
    assert torch.equal(direct['local'][:,0],other['local'][:,0])
    assert torch.equal(pred1[:,0],pred2[:,0])
    context_change=float((direct['context'][:,0].float()-other['context'][:,0].float()).abs().max())
    assert context_change>0
    # This checks the zero-strength sampling branch, without evaluating a head.
    capture=Capture(rt);before=rt.counts.copy()
    with rt.context():zero=field(rt,heads,capture,'strong',z,t[0],float(t[0]))
    capture.close()
    with rt.context():strong=rt.field(z,t[0],'full')
    assert torch.equal(zero,strong)
    assert all(not q.requires_grad for q in rt.model.parameters())
    return dict(passed=True,hook_native_outputs_exact=True,captured_features_exact=equality,
                outside_patch_prediction_exact=True,context_feature_max_change=context_change,
                zero_strength_exact=True,strong_parameters_frozen=True)


def verify_request(path):
    request=c.read(path)
    for group in ('sources','assets','data','inputs','heads'):
        for p,h in request.get(group,{}).items():assert c.sha(p)==h,p
    return request


def prepare_training(model):
    root=ROOT/model/'input_local_training'
    sources=c.source_manifest([Path(__file__).resolve(),PROTOCOL,
        c.WORK/'experiments/sit_measure_guidance_20260912/data.py',
        c.WORK/'experiments/train_raev2_observable_potential.py'])
    request=dict(model=model,steps=STEPS,batch=BATCH[model],seed=TRAIN_SEED,lr=.0003,
        weight_decay=.0001,ema=.995,normalization_batches=32,head_names=['local','context'],
        sources=sources,assets={str(p):c.sha(p) for p in c.asset_paths(model)},
        data={str(p):c.sha(p) for p in data_paths(model)})
    path=root/'request.json'
    if path.exists():assert c.read(path)==request
    else:c.atomic(path,request)
    return path


def train(model):
    configure();request_path=prepare_training(model);verify_request(request_path)
    root=request_path.parent
    if (root/'summary.json').exists():return
    torch.manual_seed(TRAIN_SEED)
    rt=c.runtime(model);data=RealData(model)
    heads={'local':make_head(rt)}
    heads['context']=copy.deepcopy(heads['local'])
    g=torch.Generator(device='cuda').manual_seed(TRAIN_SEED)
    c.atomic(root/'checks_before.json',checks(rt,heads))
    sums={k:torch.zeros(shape(rt)[0],device='cuda',dtype=torch.float64) for k in ('local','context','condition')}
    squares={k:v.clone() for k,v in sums.items()};counts={k:0 for k in sums}
    with torch.no_grad():
        for _ in range(32):
            feats,_,_,_,_=training_batch(rt,data,'train',g,BATCH[model])
            for k,x in feats.items():
                x=x.reshape(-1,x.shape[-1]).double()
                sums[k]+=x.sum(0);squares[k]+=x.square().sum(0);counts[k]+=len(x)
        for k,head in heads.items():
            for key,prefix in ((k,'token'),('condition','condition')):
                mean=sums[key]/counts[key];std=(squares[key]/counts[key]-mean.square()).clamp_min(1e-8).sqrt()
                getattr(head,prefix+'_mean').copy_(mean.float())
                getattr(head,prefix+'_std').copy_(std.float())
    ema={k:copy.deepcopy(h).eval().requires_grad_(False) for k,h in heads.items()}
    opts={k:torch.optim.AdamW(h.parameters(),lr=.0003,weight_decay=.0001) for k,h in heads.items()}
    history=[];torch.cuda.synchronize();start_time=time.perf_counter()
    for step in range(1,STEPS+1):
        feats,target,_,_,_=training_batch(rt,data,'train',g,BATCH[model])
        losses={}
        for k,h in heads.items():
            opts[k].zero_grad(set_to_none=True)
            with rt.context():pred=h(feats[k],feats['condition'])
            loss=(pred.float()-target).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            for p in h.parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
            opts[k].step()
            with torch.no_grad():
                for p,q in zip(ema[k].parameters(),h.parameters()):p.lerp_(q,.005)
            losses[k]=float(loss.detach())
        if step==1 or step%100==0:
            row=dict(step=step,**losses);history.append(row)
            c.atomic(root/'progress.json',dict(pid=os.getpid(),**row))
            print(model,'train',row,flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-start_time
    validation={k:[] for k in heads}
    vg=torch.Generator(device='cuda').manual_seed(TRAIN_SEED+7)
    with torch.no_grad():
        for _ in range(32):
            feats,target,_,_,_=training_batch(rt,data,'validation',vg,BATCH[model])
            with rt.context():
                for k,h in ema.items():
                    validation[k].append(float((h(feats[k],feats['condition']).float()-target).square().mean()))
    after=checks(rt,ema);c.atomic(root/'checks_after.json',after)
    path=root/'head.pt';tmp=path.with_suffix('.tmp')
    torch.save(dict(ema={k:h.state_dict() for k,h in ema.items()},step=STEPS,model=model,
        request_sha256=c.sha(request_path)),tmp);tmp.replace(path)
    summary=dict(complete=True,model=model,steps=STEPS,batch=BATCH[model],training_seconds=seconds,
        validation_mse={k:float(np.mean(v)) for k,v in validation.items()},history=history,
        parameters_per_head=sum(p.numel() for p in ema['local'].parameters()),
        shape=shape(rt),head_sha256=c.sha(path),request_sha256=c.sha(request_path),checks=after)
    c.atomic(root/'summary.json',summary)
    print(model,'training complete',summary['validation_mse'],seconds,flush=True)


def load_heads(rt):
    root=ROOT/rt.name/'input_local_training'
    summary=c.read(root/'summary.json')
    assert summary['complete'] and c.sha(root/'head.pt')==summary['head_sha256']
    state=torch.load(root/'head.pt',map_location='cpu',weights_only=True)
    assert state['step']==STEPS and state['request_sha256']==c.sha(root/'request.json')
    heads={k:make_head(rt).eval().requires_grad_(False) for k in ('local','context')}
    for k,h in heads.items():h.load_state_dict(state['ema'][k],strict=True)
    return heads


def field(rt,heads,capture,kind,z,t,left):
    factor=.5 if kind.endswith('_half') else 2. if kind.endswith('_double') else 1.
    amount=c.amount(rt,left,'ig',factor) if kind!='strong' else 0.
    if not amount:return rt.field(z,t,'full')
    full,weak=rt.pair(z,t)
    if not kind.startswith('native'):
        key=kind.split('_')[0]
        pred=unpatchify(rt,heads[key](capture.values[key],capture.values['condition'])).float()
        if rt.name=='raev2':
            pred=rt.native.clean_to_velocity(pred,z,rt.times(z,t),denominator_floor=float(rt.cfg.transport.t_eps))
        weak=pred
    return full+amount*(full-weak)


def prepare_screen(model):
    configure();bank=c.prepare_bank(model,STAGE,N,2026121307)
    tr=ROOT/model/'input_local_training'
    verify_request(tr/'request.json')
    request=dict(model=model,samples=N,seed=2026121307,kinds=list(KINDS),generated_paths_per_sample=1,
        sources=c.source_manifest([Path(__file__).resolve(),PROTOCOL]),
        assets={str(p):c.sha(p) for p in c.asset_paths(model)},
        inputs={str(p):c.sha(p) for p in bank.glob('*.npy')},
        heads={str(tr/f):c.sha(tr/f) for f in ('head.pt','request.json','summary.json')},
        rule='local FID <= every nonlocal control - 2 and IS >= .9*native_base; then fresh 1k')
    path=ROOT/model/STAGE/'request.json'
    if path.exists():assert c.read(path)==request
    else:c.atomic(path,request)
    return path


@torch.inference_mode()
def sample(model,kinds,parent=0):
    configure();rp=prepare_screen(model);verify_request(rp);rh=c.sha(rp)
    rt=c.runtime(model);heads=load_heads(rt);capture=Capture(rt)
    noise,_,labels=c.bank(model,STAGE)
    for kind in kinds:
        root=ROOT/model/STAGE/kind
        for start in range(0,N,rt.batch):
            c.check_parent(parent)
            path=root/'batches'/f'{start:04d}.npz'
            if path.exists():continue
            z=c.cuda(noise[start:start+rt.batch]);y=c.cuda(labels[start:start+rt.batch])
            torch.cuda.synchronize();begin=time.perf_counter()
            with rt.context():
                latent,counts=c.integrate(rt,z,y,lambda x,t,left,i,j:field(rt,heads,capture,kind,x,t,left))
            pixels=rt.decode(latent);torch.cuda.synchronize();seconds=time.perf_counter()-begin
            assert counts==dict(full=128 if model=='sit_small' else 100,prefix=0),counts
            c.save_batch(path,pixels,latent.float().cpu().numpy(),np.array(y.cpu()),dict(seconds=seconds,
                full_calls=counts['full'],prefix_calls=counts['prefix']),rh,start,c.array_sha(noise[start:start+len(y)]))
            if start%100==0:c.atomic(root/'progress.json',dict(pid=os.getpid(),complete=start+len(y),total=N))
        collect(model,kind)
        print(model,kind,'sampling complete',flush=True)
    capture.close()


def collect(model,kind):
    root=ROOT/model/STAGE/kind;noise,_,labels=c.bank(model,STAGE)
    files=sorted((root/'batches').glob('*.npz'));images=[];records=[];coverage=[];seconds=0.
    for p in files:
        meta=c.read(p.with_suffix('.json'));assert c.sha(p)==meta['sha256']
        with np.load(p) as d:
            start=int(d['start']);n=len(d['arr_0']);coverage.extend(range(start,start+n))
            np.testing.assert_array_equal(d['labels'],labels[start:start+n])
            assert str(d['noise_sha256'])==c.array_sha(noise[start:start+n])
            assert str(d['request_sha256'])==c.sha(ROOT/model/STAGE/'request.json')
            assert np.isfinite(d['latents']).all()
            assert int(d['prefix_calls'])==0
            assert int(d['full_calls'])==(128 if model=='sit_small' else 100)
            images.append(d['arr_0']);seconds+=float(d['seconds'])
        records.append(dict(file=str(p),sha256=meta['sha256']))
    assert coverage==list(range(N))
    np.savez(root/'samples.npz',arr_0=np.concatenate(images))
    c.atomic(root/'summary.json',dict(complete=True,model=model,stage=STAGE,arm=kind,
        primary_samples=N,generated_paths=N,seconds=seconds,full_calls_per_output=128 if model=='sit_small' else 100,
        prefix_calls_at_inference=0,samples_sha256=c.sha(root/'samples.npz'),
        request_sha256=c.sha(ROOT/model/STAGE/'request.json'),records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True)
    p.add_argument('--train',action='store_true');p.add_argument('--sample',nargs='+',choices=KINDS)
    p.add_argument('--parent',type=int,default=0);a=p.parse_args()
    if a.train:train(a.model)
    elif a.sample:sample(a.model,a.sample,a.parent)
    else:p.error('choose --train or --sample')
