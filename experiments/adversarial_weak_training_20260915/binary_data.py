"""Real training endpoints and fresh initial noise; no generated sample bank."""

import numpy as np
import torch
from experiments.guidance_dynamic_50k_20260915 import config as k
from experiments.guidance_dynamic_50k_20260915.data import RealDataset
from experiments import train_imagenet100_sit_flow as base


class RealEndpointData:
    def __init__(self, seed):
        self.real = RealDataset('sit_small', 'train')
        labels = np.load(k.model_root('sit_small') / 'data/train_labels.npy')
        self.by_class = [np.flatnonzero(labels == label) for label in range(100)]
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device='cuda').manual_seed(seed + 17)
        assert len(self.real) == 126689

    def draw(self, count):
        classes = self.rng.integers(100, size=count)
        ids = [int(self.rng.choice(self.by_class[label])) for label in classes]
        moments = torch.stack([self.real[index][0] for index in ids]).cuda()
        posterior = torch.randn((count, 4, 32, 32), generator=self.generator, device='cuda')
        real = base.sample_sdvae_posterior(moments, posterior)
        noise = torch.randn(real.shape, generator=self.generator, device='cuda')
        labels = torch.as_tensor(classes, dtype=torch.long, device='cuda')
        return real, noise, labels

    def state_dict(self):
        return dict(numpy=self.rng.bit_generator.state, torch=self.generator.get_state().cpu())

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['numpy']
        self.generator.set_state(state['torch'].cpu())
