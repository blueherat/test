"""Two capacity controls for diffusion-only training of SiT depth4 readouts."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.guidance_distribution_20260912.local_head import Head, position
from .heads import DeeperMLPHead


class WideMLPHead(nn.Module):
    def __init__(self, width=384, outdim=16, side=16, hidden=768, extra_layers=2):
        super().__init__()
        self.token = nn.Linear(width, hidden)
        self.condition = nn.Linear(width, hidden, bias=False)
        self.position = nn.Linear(8, hidden, bias=False)
        self.hidden_residuals = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(extra_layers))
        self.output = nn.Linear(hidden, outdim)
        for layer in (*self.hidden_residuals, self.output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for name in ('token_mean', 'condition_mean'):
            self.register_buffer(name, torch.zeros(width))
        for name in ('token_std', 'condition_std'):
            self.register_buffer(name, torch.ones(width))
        self.register_buffer('positions', position(side, 'cpu'))

    def forward(self, tokens, condition):
        x = (tokens.float() - self.token_mean) / self.token_std
        c = (condition.float() - self.condition_mean) / self.condition_std
        h = F.silu(self.token(x) + self.condition(c)[:, None] + self.position(self.positions))
        for layer in self.hidden_residuals:
            h = h + F.silu(layer(h))
        return self.output(h)


def make(variant, model=None):
    if variant in ('linear', 'block1', 'block2'):
        from .sit_transformer_heads import SiTAdapterHead
        if model is None: raise ValueError('Native adapters require the frozen model for initialization')
        return SiTAdapterHead(model, {'linear':0, 'block1':1, 'block2':2}[variant])
    return {'moderate': DeeperMLPHead, 'large': WideMLPHead, 'shallow': Head}[variant](384, 16, 16)


def describe(head):
    if hasattr(head, 'architecture'): return head.architecture()
    return dict(input_width=384, hidden_width=head.token.out_features,
                extra_layers=len(head.hidden_residuals) if isinstance(head, WideMLPHead)
                else int(isinstance(head, DeeperMLPHead)),
                parameters=sum(p.numel() for p in head.parameters()), depth=4,
                output='16 values per patch, unpatchified to 4x32x32 velocity')
