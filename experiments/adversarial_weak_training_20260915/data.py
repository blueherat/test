"""Full real training set and full strong bank; refresh real posterior every draw."""

import numpy as np
import torch
from experiments.guidance_dynamic_50k_20260915 import config as k
from experiments.guidance_dynamic_50k_20260915.data import RealDataset, GeneratedBank
from experiments import train_imagenet100_sit_flow as base


class EndpointData:
    def __init__(self, seed):
        self.real = RealDataset("sit_small", "train")
        self.strong = GeneratedBank("sit_small", "strong", "train")
        labels = np.load(k.model_root("sit_small") / "data/train_labels.npy")
        self.by_class = [np.flatnonzero(labels == label) for label in range(100)]
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device="cuda").manual_seed(seed + 17)
        assert len(self.real) == self.strong.receipt["samples"] == 126689

    def draw(self, count):
        labels_np = self.rng.integers(100, size=count)
        return self._draw_labels(labels_np)

    def draw_pairs(self, count):
        """Adjacent, conditionally independent pairs, with uniform class prior."""
        assert count % 2 == 0
        labels_np = np.repeat(self.rng.integers(100, size=count // 2), 2)
        return self._draw_labels(labels_np)

    def _draw_labels(self, labels_np):
        count = len(labels_np)
        ids = [int(self.rng.choice(self.by_class[label])) for label in labels_np]
        moments = torch.stack([self.real[index][0] for index in ids]).cuda()
        posterior = torch.randn((count, 4, 32, 32), generator=self.generator, device="cuda")
        real = base.sample_sdvae_posterior(moments, posterior)
        labels = torch.as_tensor(labels_np, dtype=torch.long, device="cuda")
        strong, _ = self.strong.draw(labels_np, self.generator)
        noise = torch.randn(real.shape, generator=self.generator, device="cuda")
        return real, strong, noise, labels

    def state_dict(self):
        return dict(numpy=self.rng.bit_generator.state, torch=self.generator.get_state().cpu())

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["numpy"]
        self.generator.set_state(state["torch"].cpu())


def decoded(adapter, latents):
    # All four sources use this identical continuous decode, including real
    # posterior samples. The target is decoded real-latent distribution in SiT.
    return (adapter.rt.vae.decode(latents.float() / base.SD_VAE_SCALING_FACTOR).sample + 1.) / 2.
