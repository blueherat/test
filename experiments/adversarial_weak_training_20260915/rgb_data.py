"""Actual ImageNet-100 training RGB images and fresh SiT initial noise.

The packed dataset is in parquet row order, whereas the latent cache is in
ImageFolder order. Select the population by audited original class IDs; never
interpret latent-cache row indices as packed image indices.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.guidance_dynamic_50k_20260915 import config as k
from experiments.raev2_training_core import DeterministicImageNetPacked


class RawImageEndpointData:
    def __init__(self, seed, device='cuda'):
        self.device = device
        self.index_manifest = k.SIT_DATA.parent / 'imagenet100_cmc/manifest.json'
        self.packed_manifest = k.JIT_DATA / 'manifest.json'
        index = json.loads(self.index_manifest.read_text())
        packed = json.loads(self.packed_manifest.read_text())
        assert index['format'] == 'eqvae_imagenet100_cmc_index_v1'
        assert packed['split'] == 'train'
        assert Path(index['source']['imagenet_parquet_root']).resolve() == Path(packed['source_root']).resolve()
        classes = sorted(index['classes'], key=lambda row: row['label'])
        assert [row['label'] for row in classes] == list(range(100))
        self.original_labels = [row['original_imagenet_label'] for row in classes]
        assert len(set(self.original_labels)) == 100
        self.real = DeterministicImageNetPacked(k.JIT_DATA, image_size=256, horizontal_flip=False)
        labels = np.concatenate(self.real._labels)
        self.by_class = [np.flatnonzero(labels == label) for label in self.original_labels]
        assert all(len(pool) == row['train_count'] for pool, row in zip(self.by_class, classes))
        assert sum(map(len, self.by_class)) == index['splits']['train']['count'] == 126689
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device=device).manual_seed(seed + 17)
        selected = np.concatenate(self.by_class).astype(np.int64)
        self.provenance = dict(
            images=126689, classes=100, source_split='train', real_target='actual RGB, no VAE reconstruction',
            preprocessing='existing ADM-style center crop 256, no horizontal flip, float RGB / 255',
            index_manifest=str(self.index_manifest), index_manifest_sha256=k.sha(self.index_manifest),
            packed_manifest=str(self.packed_manifest), packed_manifest_sha256=k.sha(self.packed_manifest),
            packed_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
            selected_class_ordered_packed_indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
            original_imagenet_labels=self.original_labels,
            class_counts=list(map(len, self.by_class)),
            same_training_population_as_latent_cache=True,
            image_order='packed parquet rows; not assumed to match latent-cache row order',
        )

    def draw(self, count):
        classes = self.rng.integers(100, size=count)
        ids = [int(self.rng.choice(self.by_class[label])) for label in classes]
        images = []
        for index, label in zip(ids, classes):
            image, original_label, returned_index = self.real[index]
            assert original_label == self.original_labels[label] and returned_index == index
            images.append(image)
        real = torch.stack(images).to(self.device)
        noise = torch.randn((count, 4, 32, 32), generator=self.generator, device=self.device)
        labels = torch.as_tensor(classes, dtype=torch.long, device=self.device)
        return real, noise, labels

    def state_dict(self):
        return dict(numpy=self.rng.bit_generator.state, torch=self.generator.get_state().cpu())

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['numpy']
        self.generator.set_state(state['torch'].cpu())
