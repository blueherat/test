from __future__ import annotations
import copy
import torch
from torch import nn
from experiments import imagenet100_sit_internal_v_head as heads


class WeakPrefix(nn.Module):
    """An independent four-block weak model; the source runtime is untouched."""
    def __init__(self,rt):
        super().__init__()
        self.x_embedder=copy.deepcopy(rt.model.x_embedder)
        self.t_embedder=copy.deepcopy(rt.model.t_embedder)
        self.y_embedder=copy.deepcopy(rt.model.y_embedder)
        self.register_buffer('pos_embed',rt.model.pos_embed.detach().clone())
        self.blocks=nn.ModuleList([copy.deepcopy(block) for block in rt.model.blocks[:4]])
        self.readout=copy.deepcopy(rt.head.module)
        for layer in self.modules():
            layer._forward_hooks.clear();layer._forward_pre_hooks.clear()

    def forward(self,z,t,labels):
        features,condition=heads.extract_internal_features(self,z,t,labels,internal_depth=4)
        return heads.internal_velocity_from_features(self,self.readout,features,condition,latent_channels=4)
