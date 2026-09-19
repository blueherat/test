"""Fresh real training RGB and noise, retaining the SiT class mapping protocol."""
import numpy as np
import torch
from experiments.adversarial_weak_training_20260915.rgb_data import RawImageEndpointData


class ImageNetData(RawImageEndpointData):
    def __init__(self, model, seed):
        if model == 'sit_small':
            super().__init__(seed)
            self.shape = (4, 32, 32)
            return
        from experiments.raev2_training_core import DeterministicImageNetPacked
        from experiments.guidance_dynamic_50k_20260915 import config as k
        self.device = 'cuda'
        self.shape = (3, 256, 256) if model == 'jit' else (1024, 16, 16)
        self.real = DeterministicImageNetPacked(k.JIT_DATA, image_size=256, horizontal_flip=False)
        packed = k.read(k.JIT_DATA/'manifest.json')
        if packed['split'] != 'train':
            raise ValueError('Only the training split may fit the critic')
        labels = np.concatenate(self.real._labels)
        order = np.argsort(labels, kind='stable')
        counts = np.bincount(labels, minlength=1000)
        assert len(counts) == 1000 and counts.min() > 0
        self.by_class = np.split(order, np.cumsum(counts)[:-1])
        self.original_labels = list(range(1000))
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device='cuda').manual_seed(seed+17)
        self.provenance = dict(images=len(labels), classes=1000, source_split='train',
            real_target='actual RGB, no reconstruction', preprocessing='ADM center crop 256; no horizontal flip',
            manifest=str(k.JIT_DATA/'manifest.json'), manifest_sha256=k.sha(k.JIT_DATA/'manifest.json'))

    def draw(self, count):
        classes = self.rng.integers(len(self.by_class), size=count)
        ids = [int(self.rng.choice(self.by_class[y])) for y in classes]
        images=[]
        for i,y in zip(ids,classes):
            image,label,index = self.real[i]
            assert label == self.original_labels[y] and index == i
            images.append(image)
        return (torch.stack(images).cuda(),
                torch.randn((count,*self.shape),device='cuda',generator=self.generator),
                torch.as_tensor(classes,device='cuda',dtype=torch.long))
