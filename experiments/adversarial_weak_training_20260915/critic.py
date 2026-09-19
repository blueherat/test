"""Learned nonlinear, class-conditional four-source image-feature critic."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


class SourceCritic(nn.Module):
    def __init__(self, dimension=2048, hidden=256, classes=100):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(dimension))
        self.register_buffer("feature_std", torch.ones(dimension))
        self.first = spectral_norm(nn.Linear(dimension, hidden))
        self.second = spectral_norm(nn.Linear(hidden, hidden))
        self.output = spectral_norm(nn.Linear(hidden, 4))
        self.condition = nn.Embedding(classes, 4 * hidden)
        nn.init.normal_(self.condition.weight, std=.02)
        self.hidden = hidden

    def forward(self, features, labels):
        values = (features - self.feature_mean) / self.feature_std
        values = F.leaky_relu(self.first(values), .2)
        values = F.leaky_relu(self.second(values), .2)
        condition = self.condition(labels).view(len(labels), 4, self.hidden)
        return self.output(values) + (condition * values[:, None]).sum(-1) / math.sqrt(self.hidden)
