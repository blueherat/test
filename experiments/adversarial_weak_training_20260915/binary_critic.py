"""A single real/fake logit, conditioned on the requested image class."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


class BinaryCritic(nn.Module):
    def __init__(self, dimension=2048, hidden=256, classes=100):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(dimension))
        self.register_buffer('feature_std', torch.ones(dimension))
        self.first = spectral_norm(nn.Linear(dimension, hidden))
        self.second = spectral_norm(nn.Linear(hidden, hidden))
        self.output = spectral_norm(nn.Linear(hidden, 1))
        self.condition = nn.Embedding(classes, hidden)
        nn.init.normal_(self.condition.weight, std=.02)
        self.hidden = hidden

    def forward(self, features, labels):
        values = (features - self.feature_mean) / self.feature_std
        values = F.leaky_relu(self.first(values), .2)
        values = F.leaky_relu(self.second(values), .2)
        return self.output(values).squeeze(-1) + (self.condition(labels) * values).sum(-1) / math.sqrt(self.hidden)


def discriminator_loss(real_logits, fake_logits):
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def weak_loss(fake_logits):
    return F.softplus(-fake_logits).mean()
