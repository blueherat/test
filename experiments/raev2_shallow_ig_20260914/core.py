"""Original DDTFinalLayer and repository Context MLP, attached at encoder depth4."""
import copy,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as old

ROOT=c.EXPS/'raev2_shallow_ig_20260914'
DEPTH=4
TRAIN=ROOT/'training'
KINDS=('native','mlp')


def state_hash(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def make_heads(rt):
    # Same class, dimensions and initialization as the published base_final_layer.
    native=copy.deepcopy(rt.model.base_final_layer).train().requires_grad_(True)
    with torch.no_grad():
        native.norm.weight.fill_(1)
        for module in (native.linear,native.adaln_modulation[-1]):
            module.weight.zero_();module.bias.zero_()
    mlp=old.make_head(rt).train().requires_grad_(True)
    return dict(native=native,mlp=mlp)


@torch.no_grad()
def features(rt,z,t,y,depth=DEPTH):
    kwargs=dict(context=y,attn_mask=None)
    seq,temb=rt.model._build_sequence(z,t,kwargs)
    n=rt.model.s_embedder.num_patches;condition=seq[:,n:].mean(1)
    mask=rt.model._build_attn_mask(seq,kwargs)
    for block in rt.model.blocks[:depth]:seq=block(seq,rt.model.enc_rope,mask)
    return dict(tokens=seq[:,:n],condition=condition,temb=temb)


def predict(kind,head,feats):
    if kind=='native':
        x=torch.nn.functional.silu(feats['temb']+feats['tokens'])
        return head(x,x)
    return head(feats['tokens'],feats['condition'])


class Capture:
    def __init__(self,rt,depth=DEPTH):
        self.values={};n=rt.model.s_embedder.num_patches
        def time_hook(module,args,value):self.values['temb']=value[0]
        def before(module,args):self.values['condition']=args[0][:,n:].mean(1)
        def after(module,args,value):self.values['tokens']=value[:,:n]
        self.handles=[rt.model.t_embedder.register_forward_hook(time_hook),
            rt.model.blocks[0].register_forward_pre_hook(before),
            rt.model.blocks[depth-1].register_forward_hook(after)]
    def close(self):
        for h in self.handles:h.remove()
        self.values.clear()


class Counts:
    def __init__(self,rt,heads):
        self.blocks=[0]*30;self.heads={k:0 for k in heads};self.handles=[]
        for i,b in enumerate(rt.model.blocks):
            def hook(module,args,index=i):self.blocks[index]+=1
            self.handles.append(b.register_forward_pre_hook(hook))
        for k,h in heads.items():
            def hook(module,args,key=k):self.heads[key]+=1
            self.handles.append(h.register_forward_pre_hook(hook))
    def reset(self):
        self.blocks[:]=[0]*30
        for k in self.heads:self.heads[k]=0
    def close(self):
        for h in self.handles:h.remove()


@torch.inference_mode()
def sample(rt,heads,capture,noise,labels,kind,alpha):
    assert kind in ('incumbent',*KINDS) and np.isfinite(alpha) and alpha>=0
    def field(z,t,left,*_):
        if not (.1<=left<=1) or not alpha:return rt.field(z,t,'full')
        full,weak=rt.pair(z,t)
        if kind!='incumbent':
            clean=old.unpatchify(rt,predict(kind,heads[kind],capture.values)).float()
            weak=rt.native.clean_to_velocity(clean,z,rt.times(z,t),denominator_floor=float(rt.cfg.transport.t_eps))
        return full+alpha*(full-weak)
    with rt.context():return c.integrate(rt,noise,labels,field)


@torch.inference_mode()
def check(rt,heads):
    assert rt.model.base_model_depth==8 and rt.model.num_enc_blocks==28 and rt.model.num_dec_blocks==2
    g=torch.Generator(device='cuda').manual_seed(2026091430)
    z=torch.randn((2,1024,16,16),device='cuda',generator=g);t=torch.full((2,),.47,device='cuda');y=torch.tensor([2,61],device='cuda')
    with rt.context():full0,base0=rt.model(z,t,context=y,attn_mask=None)
    capture=Capture(rt)
    with rt.context():
        full1,base1=rt.model(z,t,context=y,attn_mask=None)
        shared={k:v.clone() for k,v in capture.values.items()}
        direct=features(rt,z,t,y)
        assert all(torch.equal(shared[k],direct[k]) for k in direct)
        eight=features(rt,z,t,y,depth=8)
        base2=old.unpatchify(rt,predict('native',rt.model.base_final_layer,eight))
    assert torch.equal(full0,full1) and torch.equal(base0,base1) and torch.equal(base0,base2)
    capture.close()
    assert type(heads['native']) is type(rt.model.base_final_layer)
    assert {k:tuple(v.shape) for k,v in heads['native'].state_dict().items()}=={k:tuple(v.shape) for k,v in rt.model.base_final_layer.state_dict().items()}
    assert all(not p.requires_grad and p.grad is None for p in rt.model.parameters())
    return dict(passed=True,native_ig_depth=8,new_depth=4,encoder_blocks=28,decoder_blocks=2,
        native_readout_class_and_shapes_exact=True,depth8_native_readout_bitwise=True,
        shared_and_direct_depth4_features_bitwise=True,capture_preserves_original_outputs=True,
        strong_frozen=True,parameters={k:sum(p.numel() for p in h.parameters()) for k,h in heads.items()})


def load(rt):
    summary=c.read(TRAIN/'summary.json');assert summary['complete'] and c.sha(TRAIN/'head.pt')==summary['head_sha256']
    state=torch.load(TRAIN/'head.pt',map_location='cpu',weights_only=True)
    assert state['step']==50000 and state['request_sha256']==c.sha(TRAIN/'request.json')
    heads=make_heads(rt)
    for k,h in heads.items():h.load_state_dict(state['ema'][k]);h.eval().requires_grad_(False)
    return heads
