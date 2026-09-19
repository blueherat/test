from __future__ import annotations
import numpy as np
import torch
from . import catalog
from experiments import sit_guidance_fusion_20260910 as previous

FAMILIES = catalog.FAMILIES
configurations = catalog.configurations
install = previous.install
limiting_checks = previous.limiting_checks


def modulate(x,shift,scale):
    return x*(1+scale[:,None])+shift[:,None]


def qkv(attention,x):
    batch,tokens,width=x.shape
    q,k,v=attention.qkv(x).reshape(batch,tokens,3,attention.num_heads,
        width//attention.num_heads).permute(2,0,3,1,4).unbind(0)
    return attention.q_norm(q),attention.k_norm(k),v


def route(attention,conditional_input,null_input,amount,key):
    q,k,v=qkv(attention,conditional_input)
    logits=(q*attention.scale)@k.transpose(-2,-1)
    if key in ('routing_odds','routing_linear'):
        uq,uk,_=qkv(attention,null_input)
        negative=(uq*attention.scale)@uk.transpose(-2,-1)
        if key=='routing_odds':
            probability=torch.softmax(logits+amount*(logits-negative),dim=-1)
        else:
            cp,up=torch.softmax(logits,dim=-1),torch.softmax(negative,dim=-1)
            probability=cp+amount*(cp-up)
    elif key=='routing_temperature':
        probability=torch.softmax((1+amount)*logits,dim=-1)
    else:
        assert key=='manual_kernel'
        probability=torch.softmax(logits,dim=-1)
    # The value dictionary is conditional in every case. Only the weights change.
    value=(probability@v).transpose(1,2).reshape_as(conditional_input)
    return attention.proj_drop(attention.proj(value))


def forward(rt,x,t,labels,amount,key,*,force_manual=False):
    if amount==0 and not force_manual:
        rt.labels=labels
        return rt.field(x,x.new_tensor(t),'full')
    model=rt.model
    assert rt.name=='sit_small' and not model.training and rt.semantics.prediction_target=='velocity'
    rt.counts['full']+=1
    ts=x.new_full((len(x),),t)
    timestep=model.t_embedder(ts)
    ec=model.y_embedder(labels,False)
    eu=model.y_embedder(torch.full_like(labels,100),False)
    c,u=timestep+ec,timestep+eu
    h=model.x_embedder(x)+model.pos_embed
    if key=='embedding_extrapolation':
        c=timestep+ec+amount*(ec-eu)
        for block in model.blocks:
            h=block(h,c)
    else:
        for block in model.blocks:
            shift,scale,gate,mlp_shift,mlp_scale,mlp_gate=block.adaLN_modulation(c).chunk(6,dim=1)
            normalized=block.norm1(h)
            conditional_input=modulate(normalized,shift,scale)
            null_input=None
            if key in ('routing_odds','routing_linear'):
                ns,nc,*_=block.adaLN_modulation(u).chunk(6,dim=1)
                null_input=modulate(normalized,ns,nc)
            output=route(block.attn,conditional_input,null_input,amount,key)
            h=h+gate[:,None]*output
            h=h+mlp_gate[:,None]*block.mlp(modulate(block.norm2(h),mlp_shift,mlp_scale))
    value=model.unpatchify(model.final_layer(h,c))
    if model.learn_sigma:
        value,_=value.chunk(2,dim=1)
    assert value.shape==x.shape
    return value


@torch.inference_mode()
def sample(rt,noise,labels,config,*,zero=False):
    if config['parameters'].get('inherited_exact'):
        return previous.sample(rt,noise,labels,config,zero=zero)
    if zero:
        anchor=next(c for c in configurations() if c['family']=='strong')
        return previous.sample(rt,noise,labels,anchor)
    rt.labels=labels
    z=noise.clone()
    before=rt.counts.copy()
    active=0
    for k in range(64):
        t,h=k/64,1/64
        amount=config['strength'] if t<config['cutoff'] else 0.
        force=config['key']=='manual_kernel' and t<config['cutoff']
        first=forward(rt,z,t,labels,amount,config['key'],force_manual=force)
        second=forward(rt,z+h*first,t+h,labels,amount,config['key'],force_manual=force)
        z=z+h*(first+second)/2
        active+=2*int(amount!=0 or force)
        if not torch.isfinite(z).all() or z.abs().max()>1e6:
            raise FloatingPointError(f'{config["arm"]}: invalid state at step {k}')
    full=rt.counts['full']-before['full']
    prefix=rt.counts['prefix']-before['prefix']
    assert full==128 and prefix==0,(full,prefix)
    extra=active*len(rt.model.blocks) if config['key'] in ('routing_odds','routing_linear') else 0
    return z,dict(full_calls=full,prefix_calls=prefix,auxiliary_full_calls=0,
        diagnostic_queries=0,strang_active_steps=0,diagnostics=np.zeros((len(z),5)),
        internal_extra_qkv_calls=extra,internal_extra_logits_products=extra,
        internal_manual_attention_calls=0 if config['key']=='embedding_extrapolation' else active*len(rt.model.blocks),
        guidance_uses_external_semantics=False)
