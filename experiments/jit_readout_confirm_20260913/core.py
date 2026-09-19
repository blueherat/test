import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.jit_readout_transfer_20260913 import common as c
from experiments.jit_readout_transfer_20260913 import sample as base

ROOT = c.EXPS / 'jit_readout_confirm_20260913'
PROTOCOL = c.WORK / 'docs/JIT_READOUT_CONFIRM_PROTOCOL_20260913_ZH.md'
STAGE, N, SEED, BATCH = 'confirm_5000', 5000, 2026121391, 4
ARMS = ('mlp', 'native_base', 'adg', 'cfg_heun25')


def prepare():
    prior_path = c.ROOT / c.STAGE / 'request.json'
    prior = c.verify(prior_path)
    c.verify(c.TRAIN / 'request.json')
    assert c.read(c.ROOT / 'decision.json')['passes_transfer_gate']
    bank = ROOT / STAGE / 'inputs'
    if not bank.exists():
        bank.mkdir(parents=True)
        rng = np.random.default_rng(SEED)
        values = np.lib.format.open_memmap(bank / 'noise.npy', mode='w+', dtype=np.float32,
                                           shape=(N, 3, 256, 256))
        for start in range(0, N, BATCH):
            values[start:start + BATCH] = rng.standard_normal((min(BATCH, N - start), 3, 256, 256), dtype=np.float32)
        values.flush()
        np.save(bank / 'labels.npy', rng.permutation(np.arange(N) % 1000))
    assert all((bank / f).exists() for f in ('noise.npy', 'labels.npy'))
    request = dict(model='JiT-B/16', stage=STAGE, samples=N, seed=SEED, batch=BATCH, arms=list(ARMS),
        ig_steps=100, ig_alpha=.3, ig_active_steps=50, cfg_steps=25, cfg_scale=3.,
        cfg_interval=[.1, 1.], cfg_last_step_euler=True,
        sources={**prior['sources'], **{str(p.resolve()): c.sha(p) for p in (Path(__file__), PROTOCOL)}},
        assets=prior['assets'], heads=prior['heads'],
        inputs={str(p): c.sha(p) for p in bank.glob('*.npy')},
        data={str(p): c.sha(p) for p in (prior_path, c.ROOT / 'decision.json', c.ROOT / 'completion_verification.json')})
    path = ROOT / STAGE / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)
    return path


@torch.inference_mode()
def sample(rt, noise, labels, arm):
    if arm != 'cfg_heun25':
        return base.sample(rt, noise, labels, arm)
    before, z = rt.counts(), noise.clone()
    grid = torch.linspace(0, 1, 26, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for i, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
            first = base.field(rt, z, t, labels, 'cfg_reference', False)
            prediction = z + (u - t) * first
            if i < 24:
                second = base.field(rt, prediction, u, labels, 'cfg_reference', False)
                z = z + (u - t) * (.5 * (first + second))
            else:
                z = prediction
    counts = rt.counts() - before
    assert counts[0] == 98 and counts[1] == 0 and np.all(counts[2:] == 98), counts
    assert torch.isfinite(z).all()
    return z, dict(full=98, prefix=0, head=0, blocks=counts[2:].tolist())


@torch.inference_mode()
def preflight(rt, noise, labels):
    original_checks = base.preflight(rt, noise, labels)
    z, y = torch.from_numpy(noise[:2].copy()).cuda(), torch.from_numpy(labels[:2].copy()).cuda()
    from denoiser import Denoiser
    official = Denoiser.__new__(Denoiser)
    torch.nn.Module.__init__(official)
    official.net = rt.model
    official.num_classes, official.cfg_scale, official.cfg_interval, official.t_eps = 1000, 3., (.1, 1.), .05
    expected = z.clone()
    grid = torch.linspace(0, 1, 26, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for i, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
            t, u = t.expand(2, 1, 1, 1), u.expand(2, 1, 1, 1)
            expected = official._heun_step(expected, t, u, y) if i < 24 else official._euler_step(expected, t, u, y)
    actual, counts = sample(rt, z, y, 'cfg_heun25')
    assert torch.equal(actual, expected), float((actual - expected).abs().max())
    return dict(passed=True, retained_checks=original_checks, cfg25_official_full_trajectory_exact=True,
                cfg25_full_calls=counts['full'], prefix_calls=0)


def worker(rank, world, parent):
    path = ROOT / STAGE / 'request.json'
    c.verify(path)
    request_hash = c.sha(path)
    c.setup()
    rt = base.Runtime()
    noise = np.load(ROOT / STAGE / 'inputs/noise.npy', mmap_mode='r')
    labels = np.load(ROOT / STAGE / 'inputs/labels.npy')
    c.atomic(ROOT / STAGE / f'checks{rank}.json', preflight(rt, noise, labels))
    for arm in ARMS:
        root = ROOT / STAGE / arm / f'rank{rank}'
        root.mkdir(parents=True, exist_ok=True)
        for start in range(rank * BATCH, N, world * BATCH):
            assert not parent or os.getppid() == parent, 'Controller ended'
            path = root / f'batch{start:05d}.npz'
            if path.exists():
                meta = c.read(path.with_suffix('.json'))
                assert c.sha(path) == meta['sha256'] and meta['request_sha256'] == request_hash
                continue
            x = torch.from_numpy(noise[start:start + BATCH].copy()).cuda()
            y = torch.from_numpy(labels[start:start + BATCH].copy()).cuda()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z, counts = sample(rt, x, y, arm)
            pixels = c.pixels(z)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, arr_0=pixels, latents=z.cpu().numpy(), labels=y.cpu().numpy(), start=start,
                    seconds=seconds, full_calls=counts['full'], prefix_calls=0, head_calls=counts['head'],
                    block_calls=counts['blocks'], request_sha256=request_hash,
                    noise_sha256=c.array_sha(noise[start:start + len(y)]))
            temporary.replace(path)
            c.atomic(path.with_suffix('.json'), dict(sha256=c.sha(path), request_sha256=request_hash, start=start))
            if (start // BATCH) % (world * 25) == rank:
                row = dict(pid=os.getpid(), arm=arm, start=start, phase='sampling')
                c.atomic(ROOT / STAGE / f'progress{rank}.json', row)
                print(row, flush=True)
        c.atomic(root / 'complete.json', dict(complete=True, request_sha256=request_hash))
    c.atomic(ROOT / STAGE / f'worker{rank}_complete.json', dict(complete=True))


@torch.inference_mode()
def benchmark():
    c.setup()
    rt = base.Runtime()
    bank = ROOT / STAGE / 'inputs'
    noise = torch.from_numpy(np.load(bank / 'noise.npy', mmap_mode='r')[:BATCH].copy()).cuda()
    labels = torch.from_numpy(np.load(bank / 'labels.npy')[:BATCH].copy()).cuda()
    seconds = {a: [] for a in ARMS}
    for repeat in range(4):
        for arm in ARMS[repeat:] + ARMS[:repeat]:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z, counts = sample(rt, noise, labels, arm)
            c.pixels(z)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if repeat:
                seconds[arm].append(elapsed)
    c.atomic(ROOT / 'inference_benchmark.json', dict(complete=True, batch=BATCH, repeats=3,
        includes_pixel_quantization=True, seconds=seconds, medians={a: float(np.median(v)) for a, v in seconds.items()},
        full_calls={a: 98 if a == 'cfg_heun25' else 100 for a in ARMS}, prefix_calls=0,
        parameters={a: sum(p.numel() for p in h.parameters()) for a, h in rt.heads.items()},
        instrumentation='Identical model and block call counters on every arm'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world', type=int, default=3)
    p.add_argument('--parent', type=int, default=0)
    p.add_argument('--benchmark', action='store_true')
    args = p.parse_args()
    benchmark() if args.benchmark else worker(args.rank, args.world, args.parent)
