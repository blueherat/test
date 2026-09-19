"""Small class-conditioned residual UNet; no corruption-time input."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, channels_in, channels_out, context_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.context = nn.Linear(context_dim, channels_out)
        self.norm2 = nn.GroupNorm(8, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.skip = nn.Identity() if channels_in == channels_out else nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, x, context):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.context(context)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.skip(x) + hidden


class Refiner(nn.Module):
    def __init__(self, num_classes=100, width=32, context_dim=64):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError('width must be a positive multiple of 8')
        self.num_classes = int(num_classes)
        self.context_dim = int(context_dim)
        self.label_embedding = nn.Embedding(num_classes, context_dim)
        self.context = nn.Sequential(nn.Linear(context_dim, context_dim), nn.SiLU(),
                                     nn.Linear(context_dim, context_dim))
        self.input = nn.Conv2d(4, width, 3, padding=1)
        self.enc0 = Block(width, width, context_dim)
        self.down0 = nn.Conv2d(width, 2*width, 3, stride=2, padding=1)
        self.enc1 = Block(2*width, 2*width, context_dim)
        self.down1 = nn.Conv2d(2*width, 3*width, 3, stride=2, padding=1)
        self.middle = Block(3*width, 3*width, context_dim)
        self.up1 = nn.Conv2d(3*width, 2*width, 3, padding=1)
        self.dec1 = Block(4*width, 2*width, context_dim)
        self.up0 = nn.Conv2d(2*width, width, 3, padding=1)
        self.dec0 = Block(2*width, width, context_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
                                    nn.Conv2d(width, 4, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, z, labels=None):
        if z.ndim != 4 or z.shape[1] != 4:
            raise ValueError('Expected latent batch [B,4,H,W]')
        if labels is None:
            context = z.new_zeros((len(z), self.context_dim))
        else:
            if labels.shape != (len(z),):
                raise ValueError('labels must have shape [B]')
            context = self.context(self.label_embedding(labels))
        h0 = self.enc0(self.input(z), context)
        h1 = self.enc1(self.down0(h0), context)
        hidden = self.middle(self.down1(h1), context)
        hidden = self.up1(F.interpolate(hidden, size=h1.shape[-2:], mode='nearest'))
        hidden = self.dec1(torch.cat((hidden, h1), dim=1), context)
        hidden = self.up0(F.interpolate(hidden, size=h0.shape[-2:], mode='nearest'))
        hidden = self.dec0(torch.cat((hidden, h0), dim=1), context)
        return z + self.output(hidden)
