from __future__ import annotations
import copy
import torch
from torch import nn
from experiments import imagenet100_sit_internal_v_head as heads


def patchify(z):
    b,c,h,w=z.shape
    return z.reshape(b,c,h//2,2,w//2,2).permute(0,2,4,3,5,1).reshape(b,h*w//4,4*c)


def unpatchify(rt, tokens):
    return heads.unpatchify_channels(rt.model,tokens,channels=4)


def make_head(rt, method, codewords=256):
    # Deep and shallow teacher heads have identical parameter shapes. Deep starts
    # from the strong readout's velocity channels, shallow from the native IG head.
    module=copy.deepcopy(rt.head.module if method=='ig_shallow' else rt.model.final_layer)
    for layer in module.modules():
        layer._forward_hooks.clear();layer._forward_pre_hooks.clear()
    if method=='ig_shallow':return module.eval()
    old=module.linear
    output=codewords if method=='cfg_categorical' else 16
    module.linear=nn.Linear(old.in_features,output,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        if method=='cfg_categorical':
            module.linear.weight.zero_();module.linear.bias.zero_()
        else:
            # Strong tokens interleave velocity/variance channels within each pixel.
            ids=torch.arange(32,device=old.weight.device).reshape(4,8)[:,:4].reshape(-1)
            module.linear.weight.copy_(old.weight[ids]);module.linear.bias.copy_(old.bias[ids])
    return module.eval()


class Capture:
    def __init__(self,rt,depths=(4,12)):
        self.values={};self.handles=[]
        for depth in depths:
            def callback(module,args,value,d=depth):
                self.values[d]=value;self.context=args[1]
            self.handles.append(rt.model.blocks[depth-1].register_forward_hook(callback))
    def close(self):
        for h in self.handles:h.remove()
        self.values.clear()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def categorical_delta(cond_logits,null_logits,codebook,amount,kind):
    pc=cond_logits.softmax(-1);pu=null_logits.softmax(-1)
    mc=pc@codebook
    if kind=='mean':return amount*(mc-pu@codebook)
    pg=((1+amount)*cond_logits-amount*null_logits).softmax(-1)
    return pg@codebook-mc
