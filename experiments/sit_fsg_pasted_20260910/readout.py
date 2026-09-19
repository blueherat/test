"""Independent category readout, separate from the optimized ConvNeXt."""
from __future__ import annotations
import json
import numpy as np
import torch
from experiments.sit_fsg_pasted_20260910 import core


class IndependentReadout:
    def __init__(self):
        from torchvision.models import resnet18, ResNet18_Weights
        self.model = resnet18(weights=None).cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(core.INDEPENDENT_WEIGHTS, map_location='cpu', weights_only=True))
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        rows = sorted(json.loads(core.MANIFEST.read_text())['classes'], key=lambda row: row['label'])
        self.original = torch.tensor([row['original_imagenet_label'] for row in rows], device='cuda')

    @torch.inference_mode()
    def images(self, pixels, labels):
        results = []
        for begin in range(0, len(pixels), 16):
            image = torch.from_numpy(np.array(pixels[begin:begin+16])).cuda().permute(0, 3, 1, 2).float()/255
            target = self.original[torch.as_tensor(np.array(labels[begin:begin+16]), device='cuda')]
            with core.old.exact_matmul():
                logits = self.model(self.transform(image))
            probability = logits.softmax(-1)
            p = probability.gather(1, target[:, None]).squeeze(1)
            success = logits.argmax(-1) == target
            results.append(torch.stack((p, success.float(), logits.argmax(-1).float()), 1).cpu().numpy())
        return np.concatenate(results)

    def latents(self, rt, z, labels):
        return self.images(rt.decode(z), labels.detach().cpu().numpy())
