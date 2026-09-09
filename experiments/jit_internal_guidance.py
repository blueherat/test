"""Official JiT model with independently trained internal clean-output readouts."""
import os
os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
import sys
from pathlib import Path
import torch
import torch.nn as nn
REPO=Path('/data/users/zhoushunyu/research_repos/JiT')
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
import model_jit
CHECKPOINT=Path('/home/zhoushunyu/data/eqvae/models/JiT/jit-b-16/checkpoint-last.pth')

def load_source(device):
    model=model_jit.JiT_models['JiT-B/16'](input_size=256,in_channels=3,num_classes=1000,attn_drop=0.,proj_drop=0.)
    payload=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=True)
    weights={k.removeprefix('net.'):v for k,v in payload['model_ema1'].items()}
    model.load_state_dict(weights,strict=True)
    model=model.to(device).eval().requires_grad_(False)
    # Official rotary tables are plain tensors, not registered buffers.
    for rope in [model.feat_rope,model.feat_rope_incontext]:
        rope.freqs_cos=rope.freqs_cos.to(device);rope.freqs_sin=rope.freqs_sin.to(device)
    return model


def features(model,z,t,labels,depths=(4,8)):
    c=model.t_embedder(t)+model.y_embedder(labels)
    y=model.y_embedder(labels)
    x=model.x_embedder(z)
    x+=model.pos_embed  # Preserve official in-place BF16 rounding.
    outputs={}
    for i,block in enumerate(model.blocks):
        if i==model.in_context_start:
            context=y.unsqueeze(1).repeat(1,model.in_context_len,1)+model.in_context_posemb
            x=torch.cat([context,x],dim=1)
        x=block(x,c,model.feat_rope if i<model.in_context_start else model.feat_rope_incontext)
        if i+1 in depths:
            outputs[str(i+1)]=x[:,model.in_context_len:] if i>=model.in_context_start else x
        if i+1==max(depths):break
    return outputs,c


class Readouts(nn.Module):
    def __init__(self,depths=(4,8)):
        super().__init__();self.layers=nn.ModuleDict({str(d):model_jit.FinalLayer(768,16,3) for d in depths})
        for h in self.layers.values():
            nn.init.zeros_(h.linear.weight);nn.init.zeros_(h.linear.bias)
            nn.init.zeros_(h.adaLN_modulation[-1].weight);nn.init.zeros_(h.adaLN_modulation[-1].bias)
    def forward(self,feats,c):
        return {d:unpatchify(self.layers[d](feats[d],c)) for d in self.layers}


def unpatchify(x):
    return torch.einsum('nhwpqc->nchpwq',x.reshape(len(x),16,16,16,16,3)).reshape(len(x),3,256,256)


def velocity(clean,z,t):return (clean.float()-z)/(1-t[:,None,None,None]).clamp_min(.05)
