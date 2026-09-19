import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import common as c
from experiments.small_sit_guidance_tuning_20260910 import adg_field
import experiments.small_sit_guidance_tuning_20260910 as angular
from experiments.guidance_pasted_20260912.audit import REFS

EVAL_ROOT = c.EXPS.parent / 'external_sources/nanogen-evals'
REFERENCE = REFS['raev2']


def prepare():
    c.verify(c.TRAIN / 'request.json')
    summary = c.read(c.TRAIN / 'summary.json')
    assert summary['complete'] and summary['head_sha256'] == c.sha(c.TRAIN / 'head.pt')
    bank = c.prepare_bank()
    sources = [Path(__file__), Path(c.__file__), c.PROTOCOL, Path(c.jig.__file__),
               c.jig.REPO / 'model_jit.py', c.jig.REPO / 'util/model_util.py',
               c.jig.REPO / 'denoiser.py', Path(angular.__file__),
               c.WORK / 'experiments/evaluate_raev2_official_samples.py']
    sources += list((EVAL_ROOT / 'fd_evaluator/fd_evaluator').glob('*.py'))
    sources += [EVAL_ROOT / 'fd_evaluator/fd_evaluator/catalogue.yaml']
    weights = [Path('/home/zhoushunyu/.cache/torch/hub/checkpoints') / name for name in
               ('weights-inception-2015-12-05-6726825d.pth', 'pt_inception-2015-12-05-6726825d.pth')]
    request = dict(model='JiT-B/16', stage=c.STAGE, samples=c.N, batch=c.SAMPLE_BATCH,
        sample_seed=c.SAMPLE_SEED, arms=list(c.ARMS), depth=4, ig_alpha=.3,
        ig_steps=100, ig_active_steps=50, cfg_scale=3., cfg_steps=50,
        sources={str(p.resolve()): c.sha(p) for p in sources},
        assets={str(p): c.sha(p) for p in (c.jig.CHECKPOINT, c.OLD / 'last.pt', REFERENCE, *weights)},
        inputs={str(p): c.sha(p) for p in bank.glob('*.npy')},
        heads={str(c.TRAIN / f): c.sha(c.TRAIN / f) for f in ('head.pt', 'summary.json', 'request.json')})
    path = c.ROOT / c.STAGE / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)
    return path


class Runtime:
    def __init__(self):
        self.model = c.jig.load_source('cuda')
        self.heads = c.make_heads()
        state = torch.load(c.TRAIN / 'head.pt', map_location='cpu', weights_only=True)
        assert state['request_sha256'] == c.sha(c.TRAIN / 'request.json')
        for key, head in self.heads.items():
            head.load_state_dict(state['ema'][key], strict=True)
            head.eval().requires_grad_(False)
        self.heads['native_base'] = c.original_head()
        self.selected = None
        self.weak = None
        self.full_calls = self.head_calls = 0
        self.block_calls = [0] * len(self.model.blocks)
        self.handles = [self.model.register_forward_pre_hook(self.count)]
        for i, block in enumerate(self.model.blocks):
            def count_block(module, args, index=i):
                self.block_calls[index] += 1
            self.handles.append(block.register_forward_pre_hook(count_block))
        self.handles.append(self.model.blocks[3].register_forward_hook(self.readout))

    def count(self, module, args):
        self.full_calls += 1

    def readout(self, module, args, output):
        if self.selected is not None:
            self.weak = c.jig.unpatchify(self.heads[self.selected](output, args[1]))
            self.head_calls += 1

    def query(self, z, t, labels, head=None):
        self.selected, self.weak = head, None
        times = t.expand(len(z))
        try:
            strong = self.model(z, times, labels)
            weak = self.weak
        finally:
            self.selected = None
        return c.jig.velocity(strong, z, times), None if weak is None else c.jig.velocity(weak, z, times)

    def counts(self):
        return np.array([self.full_calls, self.head_calls, *self.block_calls], dtype=np.int64)


def field(rt, z, t, labels, arm, active, zero=False):
    head = ('native_base' if arm == 'adg' else arm) if arm not in ('strong', 'cfg_reference') and active and not zero else None
    strong, weak = rt.query(z, t, labels, head)
    if arm == 'cfg_reference':
        uncond, _ = rt.query(z, t, torch.full_like(labels, 1000))
        scale = torch.where((t.expand(len(z)) < 1.) & (t.expand(len(z)) > .1), 3., 1.)[:, None, None, None]
        return uncond + scale * (strong - uncond)
    if weak is None:
        return strong
    if arm == 'adg':
        return adg_field(z, strong, weak, t, .3)[0]
    return strong + .3 * (strong - weak)


@torch.inference_mode()
def sample(rt, noise, labels, arm, zero=False):
    before = rt.counts()
    z = noise.clone()
    steps = 50 if arm == 'cfg_reference' else 100
    grid = torch.linspace(0, 1, steps + 1, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for i, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
            v = field(rt, z, t, labels, arm, i < 50, zero)
            prediction = z + (u - t) * v
            if arm == 'cfg_reference' and i < steps - 1:
                second = field(rt, prediction, u, labels, arm, False, zero)
                z = z + (u - t) * (.5 * (v + second))
            else:
                z = prediction
    assert torch.isfinite(z).all()
    counts = rt.counts() - before
    full = 198 if arm == 'cfg_reference' else 100
    heads = 0 if arm in ('strong', 'cfg_reference') or zero else 50
    assert counts[0] == full and counts[1] == heads and np.all(counts[2:] == full), counts
    return z, dict(full=full, prefix=0, head=heads, blocks=counts[2:].tolist())


@torch.inference_mode()
def preflight(rt, noise, labels):
    z, y = torch.from_numpy(noise[:2].copy()).cuda(), torch.from_numpy(labels[:2].copy()).cuda()
    errors = {}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for value in (0., .37, .8):
            t = torch.tensor(value, device='cuda')
            strong, weak = rt.query(z, t, y, 'native_base')
            full = c.jig.velocity(rt.model(z, t.expand(2), y), z, t.expand(2))
            feat, cond = c.jig.features(rt.model, z, t.expand(2), y, depths=(4,))
            expected = c.jig.velocity(c.jig.unpatchify(rt.heads['native_base'](feat['4'], cond)), z, t.expand(2))
            assert torch.equal(full, strong) and torch.equal(expected, weak)
            errors[str(value)] = dict(strong=0., weak=0.)
        expected = z.clone()
        grid = torch.linspace(0, 1, 101, device='cuda')
        for i, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
            if i < 50:
                feat, cond = c.jig.features(rt.model, expected, t.expand(2), y, depths=(4, 12))
                full = c.jig.velocity(c.jig.unpatchify(rt.model.final_layer(feat['12'], cond)), expected, t.expand(2))
                weak = c.jig.velocity(c.jig.unpatchify(rt.heads['native_base'](feat['4'], cond)), expected, t.expand(2))
                v = full + .3 * (full - weak)
            else:
                v = c.jig.velocity(rt.model(expected, t.expand(2), y), expected, t.expand(2))
            expected = expected + (u - t) * v
        actual, _ = sample(rt, z, y, 'native_base')
        assert torch.equal(expected, actual), float((expected - actual).abs().max())
        strong, _ = sample(rt, z, y, 'strong')
        zero, _ = sample(rt, z, y, 'mlp', zero=True)
        assert torch.equal(strong, zero)
        from denoiser import Denoiser
        official = Denoiser.__new__(Denoiser)
        torch.nn.Module.__init__(official)
        official.net = rt.model
        official.num_classes, official.cfg_scale, official.cfg_interval, official.t_eps = 1000, 3., (.1, 1.), .05
        expected = z.clone()
        grid = torch.linspace(0, 1, 51, device='cuda')
        for i, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
            t, u = t.expand(2, 1, 1, 1), u.expand(2, 1, 1, 1)
            expected = official._heun_step(expected, t, u, y) if i < 49 else official._euler_step(expected, t, u, y)
        actual, _ = sample(rt, z, y, 'cfg_reference')
        assert torch.equal(expected, actual), float((expected - actual).abs().max())
    return dict(passed=True, direct_output_errors=errors, shared_full_exact=True, old_weak_exact=True,
                original_ig_full_trajectory_exact=True, zero_guidance_exact=True, official_cfg_full_trajectory_exact=True)


def worker(rank, world, parent):
    path = c.ROOT / c.STAGE / 'request.json'
    c.verify(path)
    request_hash = c.sha(path)
    c.setup()
    rt = Runtime()
    bank = c.ROOT / c.STAGE / 'inputs'
    noise, labels = np.load(bank / 'noise.npy', mmap_mode='r'), np.load(bank / 'labels.npy')
    c.atomic(c.ROOT / c.STAGE / f'checks{rank}.json', preflight(rt, noise, labels))
    for arm in c.ARMS:
        root = c.ROOT / c.STAGE / arm / f'rank{rank}'
        root.mkdir(parents=True, exist_ok=True)
        for start in range(rank * c.SAMPLE_BATCH, c.N, world * c.SAMPLE_BATCH):
            assert not parent or os.getppid() == parent, 'Controller ended'
            path = root / f'batch{start:04d}.npz'
            if path.exists():
                meta = c.read(path.with_suffix('.json'))
                assert meta['request_sha256'] == request_hash and meta['sha256'] == c.sha(path)
                continue
            x, y = torch.from_numpy(noise[start:start + c.SAMPLE_BATCH].copy()).cuda(), torch.from_numpy(labels[start:start + c.SAMPLE_BATCH].copy()).cuda()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z, counts = sample(rt, x, y, arm)
            pixels = c.pixels(z)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez(f, arr_0=pixels, latents=z.cpu().numpy(), labels=y.cpu().numpy(), start=start,
                    seconds=seconds, full_calls=counts['full'], prefix_calls=0, head_calls=counts['head'],
                    block_calls=counts['blocks'], request_sha256=request_hash,
                    noise_sha256=c.array_sha(noise[start:start + len(y)]))
            tmp.replace(path)
            c.atomic(path.with_suffix('.json'), dict(sha256=c.sha(path), request_sha256=request_hash, start=start))
            if (start // c.SAMPLE_BATCH) % (world * 10) == rank:
                row = dict(pid=os.getpid(), arm=arm, start=start, phase='sampling')
                c.atomic(c.ROOT / c.STAGE / f'progress{rank}.json', row)
                print(row, flush=True)
        c.atomic(root / 'complete.json', dict(complete=True, request_sha256=request_hash))
    c.atomic(c.ROOT / c.STAGE / f'worker{rank}_complete.json', dict(complete=True))


@torch.inference_mode()
def benchmark():
    c.setup()
    rt = Runtime()
    bank = c.ROOT / c.STAGE / 'inputs'
    noise = torch.from_numpy(np.load(bank / 'noise.npy', mmap_mode='r')[:4].copy()).cuda()
    labels = torch.from_numpy(np.load(bank / 'labels.npy')[:4].copy()).cuda()
    seconds = {a: [] for a in c.ARMS}
    for repeat in range(4):
        for arm in c.ARMS[repeat:] + c.ARMS[:repeat]:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z, counts = sample(rt, noise, labels, arm)
            c.pixels(z)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if repeat:
                seconds[arm].append(elapsed)
    c.atomic(c.ROOT / 'inference_benchmark.json', dict(complete=True, batch=4, repeats=3,
        includes_pixel_quantization=True, seconds=seconds, medians={a: float(np.median(v)) for a, v in seconds.items()},
        parameters={a: sum(p.numel() for p in h.parameters()) for a, h in rt.heads.items()},
        instrumentation='Same model and block call counters for all arms'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world', type=int, default=3)
    p.add_argument('--parent', type=int, default=0)
    p.add_argument('--benchmark', action='store_true')
    a = p.parse_args()
    benchmark() if a.benchmark else worker(a.rank, a.world, a.parent)
