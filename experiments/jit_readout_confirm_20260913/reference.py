"""Fixed JiT-B/16 MLP internal guidance with one backbone and one weak readout.

Run with CUDA_VISIBLE_DEVICES set before importing torch. This entry point retains
the existing Euler100 / alpha=.3 / first50 configuration and does not load the
original or equal-training comparison heads.
"""
from pathlib import Path

import numpy as np
import torch

from . import core as m
from experiments.jit_readout_transfer_20260913 import common as c
from experiments.jit_readout_transfer_20260913 import sample as base


class Reference:
    def __init__(self):
        c.verify(c.TRAIN / 'request.json')
        training = c.read(c.TRAIN / 'summary.json')
        path = c.TRAIN / 'head.pt'
        if not training['complete'] or c.sha(path) != training['head_sha256']:
            raise ValueError('Retained MLP checkpoint provenance failed')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['steps'] != 3000 or state['request_sha256'] != c.sha(c.TRAIN / 'request.json'):
            raise ValueError('Unexpected retained training request or step')
        c.setup()
        self.model = c.jig.load_source('cuda')
        self.head = c.Head(768, 768, 16).cuda().eval().requires_grad_(False)
        self.head.load_state_dict(state['ema']['mlp'], strict=True)
        self.selected = False
        self.weak = None
        self.full_calls = self.head_calls = 0
        self.block_calls = [0] * len(self.model.blocks)
        self.handles = [self.model.register_forward_pre_hook(self._count)]
        for i, block in enumerate(self.model.blocks):
            def count_block(module, args, index=i):
                self.block_calls[index] += 1
            self.handles.append(block.register_forward_pre_hook(count_block))
        self.handles.append(self.model.blocks[3].register_forward_hook(self._readout))
        self.provenance = dict(checkpoint=str(path), checkpoint_sha256=c.sha(path), depth=4,
            parameters=sum(p.numel() for p in self.head.parameters()),
            original_ig_parameters=training['native_parameters'],
            backbone_parameters=sum(p.numel() for p in self.model.parameters()),
            loaded_backbones=1, loaded_weak_heads=1, extra_prefix_calls=0)

    def _count(self, module, args):
        self.full_calls += 1

    def _readout(self, module, args, output):
        if self.selected:
            self.weak = c.jig.unpatchify(self.head(output, args[1]))
            self.head_calls += 1

    def query(self, z, t, labels, head=None):
        if head not in (None, 'mlp'):
            raise ValueError('This runtime contains only the retained MLP readout')
        self.selected, self.weak = head == 'mlp', None
        times = t.expand(len(z))
        try:
            strong = self.model(z, times, labels)
            weak = self.weak
        finally:
            self.selected = False
        return c.jig.velocity(strong, z, times), None if weak is None else c.jig.velocity(weak, z, times)

    def counts(self):
        return np.array([self.full_calls, self.head_calls, *self.block_calls], dtype=np.int64)

    @torch.inference_mode()
    def sample(self, noise, labels):
        """Return final FP32 states and measured counts using the frozen sampler."""
        return base.sample(self, noise, labels, 'mlp')

    @staticmethod
    def pixels(states):
        """Convert returned states to the exact uint8 NHWC evaluation format."""
        return c.pixels(states)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def check():
    c.verify(m.ROOT / m.STAGE / 'request.json')
    rt = Reference()
    noise = np.load(m.ROOT / m.STAGE / 'inputs/noise.npy', mmap_mode='r')
    labels = np.load(m.ROOT / m.STAGE / 'inputs/labels.npy')
    initial_hash = c.state_sha(rt.model)
    records = []
    for start in (0, 4):
        rank = (start // m.BATCH) % 3
        path = m.ROOT / m.STAGE / f'mlp/rank{rank}/batch{start:05d}.npz'
        meta = c.read(path.with_suffix('.json'))
        assert meta['sha256'] == c.sha(path)
        x = torch.from_numpy(noise[start:start + 4].copy()).cuda()
        y = torch.from_numpy(labels[start:start + 4].copy()).cuda()
        output, counts = rt.sample(x, y)
        pixels = c.pixels(output)
        with np.load(path) as data:
            np.testing.assert_array_equal(output.cpu().numpy(), data['latents'])
            np.testing.assert_array_equal(pixels, data['arr_0'])
            np.testing.assert_array_equal(y.cpu().numpy(), data['labels'])
        assert counts == dict(full=100, prefix=0, head=50, blocks=[100] * 12)
        records.append(dict(start=start, file=str(path), sha256=c.sha(path), states_and_pixels_exact=True, counts=counts))
    x = torch.from_numpy(noise[:4].copy()).cuda()
    y = torch.from_numpy(labels[:4].copy()).cuda()
    strong, strong_counts = base.sample(rt, x, y, 'strong')
    zero, zero_counts = base.sample(rt, x, y, 'mlp', zero=True)
    assert torch.equal(strong, zero) and strong_counts == zero_counts
    assert initial_hash == c.state_sha(rt.model)
    assert not any(p.requires_grad or p.grad is not None for p in rt.model.parameters())
    assert not any(p.requires_grad or p.grad is not None for p in rt.head.parameters())
    result = dict(passed=True, samples_replayed=8, new_quality_samples=0, records=records,
        zero_guidance_exact=True, strong_weights_unchanged=True, strong_state_sha256=initial_hash,
        provenance=rt.provenance, request_sha256=c.sha(m.ROOT / m.STAGE / 'request.json'),
        sources={str(p): c.sha(p) for p in (Path(__file__), Path(base.__file__), Path(c.__file__))})
    c.atomic(m.ROOT / 'replacement_api_verification.json', result)
    rt.close()
    print('Single-readout JiT entry point reproduced 8 formal samples exactly', rt.provenance, flush=True)


if __name__ == '__main__':
    check()
