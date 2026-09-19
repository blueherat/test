"""Fixed training measures; no generated-image metric enters construction."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

WORK = Path('/home/zhoushunyu/eqvae')
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/sit_measure_guidance_20260912')
DATA = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae')
PROTOCOL = WORK/'docs/SIT_MEASURE_GUIDANCE_TRAINING_20260912_ZH.md'
METHODS = ('native', 'clt', 'factor', 'local_label',
           'defensive', 'heat', 'plain_mix', 'random_label')
TRAIN_PER_CLASS, VALID_PER_CLASS = 256, 32
SEED, TRAIN_SEED = 2026120912, 2026120913
STEPS, BATCH, LR, EMA = 1500, 32, 0.00005, 0.995


def transform(parents, labels, means, method, coin, heat_noise, heat_sigma):
    """parents are four conditionally independent draws, shape [4,B,C,H,W]."""
    first, second = parents[:2]
    center = means[labels]
    if method in ('native', 'local_label', 'random_label'):
        return first
    if method == 'clt':
        return center+(parents.sum(0)-4*center)/2
    if method == 'plain_mix':
        return parents.mean(0)
    if method == 'defensive':
        coarse = transform(parents, labels, means, 'clt', coin, heat_noise, heat_sigma)
        return torch.where(coin[:, None, None, None], first, coarse)
    if method == 'heat':
        return first+heat_sigma*heat_noise
    if method == 'factor':
        value = first.clone()
        value[:, :, :16, 16:] = parents[1, :, :, :16, 16:]
        value[:, :, 16:, :16] = parents[2, :, :, 16:, :16]
        value[:, :, 16:, 16:] = parents[3, :, :, 16:, 16:]
        return value
    raise ValueError(method)


def pair_classes(embedding):
    """Disjoint nearest pairs: Q=(I+P)/2 is exactly doubly stochastic."""
    values = np.asarray(embedding, dtype=np.float64)
    values = values/np.linalg.norm(values, axis=1, keepdims=True).clip(1e-12)
    distance = 1-values @ values.T
    pairs = [(float(distance[i, j]), i, j) for i in range(100) for j in range(i+1, 100)]
    partner = np.full(100, -1, dtype=np.int64)
    for _, i, j in sorted(pairs):
        if partner[i] < 0 and partner[j] < 0:
            partner[i], partner[j] = j, i
    assert np.all(partner >= 0) and np.array_equal(partner[partner], np.arange(100))
    return partner


class Pool:
    def __init__(self, split, device='cuda'):
        self.device = device
        self.moments = torch.from_numpy(np.load(ROOT/f'{split}_moments.npy')).to(device)
        self.means = torch.from_numpy(np.load(ROOT/'class_means.npy')).to(device)
        self.partners = {key: torch.from_numpy(np.load(ROOT/f'{key}_partners.npy')).to(device)
                         for key in ('local', 'random')}
        self.heat_sigma = float(np.load(ROOT/'heat_sigma.npy'))

    def draw(self, method, generator, count=BATCH, labels=None):
        device = self.device
        labels = (torch.randint(100, (count,), device=device, generator=generator)
                  if labels is None else labels)
        source = labels.clone()
        coin = torch.rand(count, device=device, generator=generator) < .5
        if method in ('local_label', 'random_label'):
            alternate = self.partners[method.split('_')[0]][labels]
            source = torch.where(coin, source, alternate)
        indices = torch.randint(self.moments.shape[1], (4, count), device=device, generator=generator)
        moments = self.moments[source[None], indices].float()
        posterior_noise = torch.randn((4, count, 4, 32, 32), device=device, generator=generator)
        parents = (moments[:, :, :4]+moments[:, :, 4:]*posterior_noise)*.18215
        heat_noise = torch.randn((count, 4, 32, 32), device=device, generator=generator)
        value = transform(parents, labels, self.means, method, coin, heat_noise, self.heat_sigma)
        return value, labels
