from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
import numpy as np
import torch
from torch import nn
from experiments import jit_internal_guidance as jig

WORK = Path('/home/zhoushunyu/eqvae')
EXPS = Path('/home/zhoushunyu/data/eqvae/experiments')
ROOT = EXPS / 'jit_prefix_transfer_20260912'
DATA = Path('/data/shared/imagenet-1k/random_access_v1')
OLD_HEAD_ROOT = EXPS / 'jit_internal_readouts_20260908'
HEAD = OLD_HEAD_ROOT / 'last.pt'
PROTOCOL = WORK / 'docs/JIT_PREFIX_TRANSFER_PROTOCOL_20260912_ZH.md'
STEPS, BATCH, SEED = 1500, 32, 2026120913
SAMPLES, SAMPLE_BATCH, NOISE_SEED, LABEL_SEED = 5000, 4, 2026091217, 2026091218
METHODS = ('native_prefix', 'head_only')
ARMS = ('original_ig', 'head_only_ig', 'native_prefix', 'cfg_reference')
OLD_STOP = [EXPS / name / 'STOP_AFTER_CURRENT' for name in (
    'sit_refined_priority_20260911', 'sit_control_output_50ideas_20260910',
    'sit_apg_mechanism_extension_20260911', 'sit_broad_resume_20260912')]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def state_sha(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def atomic_torch(path, value):
    temp = path.with_suffix('.tmp.pt')
    torch.save(value, temp)
    temp.replace(path)


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)


def load_models():
    strong = jig.load_source('cuda')
    heads = jig.Readouts().cuda().eval().requires_grad_(False)
    state = torch.load(HEAD, map_location='cpu', weights_only=False)
    assert state['step'] == 50000
    heads.load_state_dict(state['ema'], strict=True)
    return strong, heads.layers['4']


class WeakPrefix(nn.Module):
    def __init__(self, strong, readout):
        super().__init__()
        assert strong.in_context_start == 4
        for name in ('x_embedder', 't_embedder', 'y_embedder', 'feat_rope', 'feat_rope_incontext'):
            setattr(self, name, copy.deepcopy(getattr(strong, name)))
        self.register_buffer('pos_embed', strong.pos_embed.detach().clone())
        self.register_buffer('in_context_posemb', strong.in_context_posemb.detach().clone())
        self.in_context_start, self.in_context_len = strong.in_context_start, strong.in_context_len
        self.blocks = nn.ModuleList([copy.deepcopy(block) for block in strong.blocks[:4]])
        self.readout = copy.deepcopy(readout)
        self.eval().requires_grad_(False)

    def configure_training(self, method):
        assert method in METHODS
        self.requires_grad_(method == 'native_prefix')
        self.readout.requires_grad_(True)
        return self

    def forward(self, z, t, labels):
        features, c = jig.features(self, z, t, labels, depths=(4,))
        return jig.unpatchify(self.readout(features['4'], c))


def verify_request(path):
    request = read(path)
    for category in ('sources', 'assets', 'inputs', 'old_stop_markers'):
        for name, digest in request.get(category, {}).items():
            assert sha(name) == digest, (category, name)
    return request
