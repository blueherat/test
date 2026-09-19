import copy
import hashlib
from contextlib import nullcontext
import numpy as np
import torch
from . import config as k
from experiments.guidance_distribution_20260912 import local_head as local
from experiments.guidance_loss_50k_20260914.components import SourceHead


def fingerprint(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Adapter:
    def __init__(self,model):
        self.name=model;self.cfg=k.settings(model);self.full_calls=0;self.head_calls=0;self.native_calls=0
        self.values={};self.blocks=[0]*12;self.handles=[]
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        if model=='sit_small':
            from experiments.guidance_pasted_20260912 import common as c
            self.rt=c.runtime(model);self.model=self.rt.model;self.native=self.rt.head.module
        else:
            from experiments import jit_internal_guidance as jig
            self.jig=jig;self.model=jig.load_source('cuda')
            heads=jig.Readouts().cuda()
            state=torch.load(k.EXPS/'jit_internal_readouts_20260908/last.pt',map_location='cpu',weights_only=False)
            assert state['step']==50000
            heads.load_state_dict(state['ema']);self.native=heads.layers['4'].eval().requires_grad_(False)
        for index,block in enumerate(self.model.blocks):
            def mark(module,args,index=index):self.blocks[index]+=1
            self.handles.append(block.register_forward_pre_hook(mark))
        def before(module,args):self.values['condition']=args[1]
        def after(module,args,value):
            if self.name=='jit' and 3>=self.model.in_context_start:value=value[:,self.model.in_context_len:]
            self.values['context']=value
        self.handles.append(self.model.blocks[0].register_forward_pre_hook(before))
        self.handles.append(self.model.blocks[3].register_forward_hook(after))
    def autocast(self,training=False):
        return torch.autocast('cuda',dtype=torch.bfloat16) if training or self.name=='jit' else nullcontext()
    def make_head(self):
        return local.Head(self.cfg['width'],int(np.prod(self.cfg['shape'][:1]))*self.cfg['patch']**2,16).cuda()
    def patch(self,value):
        p=self.cfg['patch'];b,ch,h,w=value.shape
        return value.reshape(b,ch,h//p,p,w//p,p).permute(0,2,4,3,5,1).reshape(b,h*w//p**2,ch*p*p)
    def unpatch(self,value):
        return local.unpatchify(self.rt,value) if self.name=='sit_small' else self.jig.unpatchify(value)
    @torch.no_grad()
    def features(self,z,t,labels):
        if self.name=='sit_small':return local.features(self.rt,z,t,labels)
        value,condition=self.jig.features(self.model,z,t,labels,depths=(4,))
        return dict(context=value['4'],condition=condition)
    def full(self,z,t,labels,native=False):
        if t.ndim==0:t=t.expand(len(z))
        self.full_calls+=1
        if self.name=='sit_small':
            self.rt.labels=labels
            if native:
                self.native_calls+=1
                return self.rt.pair(z,t)
            return self.rt.field(z,t,'full'),None
        strong=self.model(z,t,labels)
        weak=None
        if native:
            self.native_calls+=1
            weak=self.unpatch(self.native(self.values['context'],self.values['condition']))
        return strong.float(),None if weak is None else weak.float()
    def native_to_velocity(self,raw,z,t):
        if self.name=='sit_small':return raw.float()
        if t.ndim==0:t=t.expand(len(z))
        return self.jig.velocity(raw,z,t)
    def correction_to_velocity(self,raw,t):
        if self.name=='sit_small':return raw.float()
        return raw.float()/(1-t).clamp_min(.05)
    def ig_amount(self,t):
        if self.name=='sit_small':return self.cfg['alpha']*torch.where(t<.25,6/7,1.)*(t<.5)
        return self.cfg['alpha']*(t<.5)
    def base(self,z,t,labels,kind):
        full,weak=self.full(z,t,labels,native=kind=='ig')
        if kind=='ig':full=full+self.ig_amount(t)[:,None,None,None]*(full-weak)
        if kind=='cfg':
            uncond,_=self.full(z,t,torch.full_like(labels,self.cfg['classes']))
            amount=self.cfg['cfg_extra']*((t<.75) if self.name=='sit_small' else ((t>.1)&(t<1.)))
            full=full+amount[:,None,None,None]*(full-uncond)
        return full
    def loaded_head(self,arm):
        root=k.model_root(self.name)/'training'/arm
        receipt=k.read(root/'complete.json');assert receipt['steps']==k.STEPS
        assert k.sha(root/'head.pt')==receipt['head_sha256']
        state=torch.load(root/'head.pt',map_location='cpu',weights_only=False)
        assert state['request_sha256']==k.sha(k.ROOT/'request.json')
        head=self.make_head();head.load_state_dict(state['ema'])
        return head.eval().requires_grad_(False)
    def counts(self):return dict(full=self.full_calls,head=self.head_calls,native_head=self.native_calls,blocks=list(self.blocks))
    def pixels(self,z):
        if self.name=='sit_small':return self.rt.decode(z)
        return np.round(np.clip((z.float().cpu().numpy().transpose(0,2,3,1)+1)*127.5,0,255)).astype(np.uint8)
