import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
from pathlib import Path
import numpy as np
import torch
from experiments import jit_internal_guidance as jig
from experiments.jit_prefix_transfer_20260912 import common as utilities
from experiments.guidance_distribution_20260912.local_head import Head

WORK, EXPS, DATA = utilities.WORK, utilities.EXPS, utilities.DATA
ROOT = EXPS / 'jit_readout_transfer_20260913'
PROTOCOL = WORK / 'docs/JIT_READOUT_TRANSFER_PROTOCOL_20260913_ZH.md'
TRAIN = ROOT / 'training'
OLD = EXPS / 'jit_internal_readouts_20260908'
STEPS, BATCH, SEED = 3000, 32, 2026121371
STAGE, N, SAMPLE_BATCH, SAMPLE_SEED = 'screen_1000', 1000, 4, 2026121381
ARMS = ('mlp', 'native_fresh', 'native_base', 'adg', 'strong', 'cfg_reference')
sha, array_sha, state_sha = utilities.sha, utilities.array_sha, utilities.state_sha
read, atomic, atomic_torch = utilities.read, utilities.atomic, utilities.atomic_torch


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)


def make_heads():
    return dict(mlp=Head(768, 768, 16).cuda(),
                native_fresh=jig.Readouts(depths=(4,)).cuda().layers['4'])


def original_head():
    heads = jig.Readouts().cuda()
    state = torch.load(OLD / 'last.pt', map_location='cpu', weights_only=False)
    assert state['step'] == 50000
    heads.load_state_dict(state['ema'], strict=True)
    return heads.layers['4'].eval().requires_grad_(False)


def verify(path):
    request = read(path)
    for group in ('sources', 'assets', 'inputs', 'heads', 'data'):
        for p, digest in request.get(group, {}).items():
            assert sha(p) == digest, (group, p)
    return request


def training_sources(train_file):
    from experiments.guidance_distribution_20260912 import local_head
    return [Path(__file__), Path(train_file), PROTOCOL, Path(jig.__file__),
            Path(utilities.__file__), Path(local_head.__file__),
            jig.REPO / 'model_jit.py', jig.REPO / 'util/model_util.py',
            jig.REPO / 'denoiser.py', WORK / 'experiments/raev2_training_core.py']


def prepare_bank():
    bank = ROOT / STAGE / 'inputs'
    if bank.exists():
        assert all((bank / f).exists() for f in ('noise.npy', 'labels.npy'))
        return bank
    bank.mkdir(parents=True)
    rng = np.random.default_rng(SAMPLE_SEED)
    noise = np.lib.format.open_memmap(bank / 'noise.npy', mode='w+', dtype=np.float32,
                                      shape=(N, 3, 256, 256))
    for start in range(0, N, SAMPLE_BATCH):
        noise[start:start + SAMPLE_BATCH] = rng.standard_normal(
            (min(SAMPLE_BATCH, N - start), 3, 256, 256), dtype=np.float32)
    noise.flush()
    np.save(bank / 'labels.npy', rng.permutation(np.arange(N) % 1000))
    return bank


def pixels(x):
    return np.round(np.clip(((x + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1) * 255, 0, 255)).astype(np.uint8)
