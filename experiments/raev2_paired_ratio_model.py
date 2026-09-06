"""Small conditional noisy-latent density-ratio critic, with d=(1-t)f."""
import math
import torch
from torch import nn


class PairedRatioCritic(nn.Module):
    def __init__(self, class_features, width=128, layers=2, heads=4):
        super().__init__()
        features = class_features.float()
        features = features / features.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        self.register_buffer('class_features', features)
        self.input = nn.Linear(1024, width)
        self.position = nn.Parameter(torch.zeros(1, 256, width))
        self.condition = nn.Linear(features.shape[1], width)
        self.time = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width,
                                       dropout=0., activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        nn.init.normal_(self.position, std=.02)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, state, t, labels):
        a = 1 - t
        temporal = torch.stack((t, a, t.square(), a.square(),
                                (math.pi * t).sin(), (math.pi * t).cos()), -1)
        h = self.input(state.flatten(2).transpose(1, 2)) + self.position
        h = h + (self.condition(self.class_features[labels]) + self.time(temporal))[:, None]
        for block in self.blocks:
            h = block(h)
        return self.output(h.mean(1)).squeeze(-1)

    def clean_correction(self, state, t, labels):
        # ∇_z logit = (1-t)∇f, so the clean conversion cancels its
        # apparent 1/(1-t) singularity analytically before evaluation.
        with torch.enable_grad():
            z = state.detach().float().requires_grad_(True)
            value = self(z, t, labels)
            gradient, = torch.autograd.grad(value.sum(), z)
        return t.reshape(-1, 1, 1, 1).square() * gradient.detach()
