import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.jit_readout_transfer_20260913 import common as c
from experiments.jit_readout_transfer_20260913 import sample as base

ROOT = c.EXPS / 'jit_readout_cfg_application_20260913'
CONFIRM = c.EXPS / 'jit_readout_confirm_20260913'
PROTOCOL = c.WORK / 'docs/JIT_READOUT_CFG_APPLICATION_PROTOCOL_20260913_ZH.md'
STAGE, N, SEED, BATCH = 'screen_1000', 1000, 2026121401, 4
ARMS = ('cfg_reference', 'cfg_more', 'cfg_native', 'cfg_mlp')


def prepare():
    assert c.read(CONFIRM / 'status.json')['phase'] == 'complete'
    assert c.read(CONFIRM / 'decision.json')['passes_ig_5k_gate'], 'No application test after a failed 5K IG confirmation'
    prior_path = CONFIRM / 'confirm_5000/request.json'
    prior = c.verify(prior_path)
    c.verify(c.TRAIN / 'request.json')
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
    request = dict(model='JiT-B/16', stage=STAGE, samples=N, seed=SEED, batch=BATCH, arms=list(ARMS),
        steps=50, solver='Heun with last-step Euler', ig_alpha=.3, ig_active_steps=25,
        cfg_scale=3., cfg_interval=[.1, 1.], full_calls=198,
        sources={**prior['sources'], **{str(p.resolve()): c.sha(p) for p in (Path(__file__), PROTOCOL)}},
        assets=prior['assets'], heads=prior['heads'],
        inputs={str(p): c.sha(p) for p in bank.glob('*.npy')},
        data={str(p): c.sha(p) for p in (prior_path, CONFIRM / 'decision.json', CONFIRM / 'status.json')})
    path = ROOT / STAGE / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)
    return path


def field(rt, z, t, labels, arm, active, zero=False):
    if arm == 'cfg_reference' or not active or zero:
        return base.field(rt, z, t, labels, 'cfg_reference', False)
    head = None if arm == 'cfg_more' else 'mlp' if arm == 'cfg_mlp' else 'native_base'
    strong, weak = rt.query(z, t, labels, head)
    uncond, _ = rt.query(z, t, torch.full_like(labels, 1000))
    scale = torch.where((t.expand(len(z)) < 1.) & (t.expand(len(z)) > .1), 3., 1.)[:, None, None, None]
    return uncond + scale * (strong - uncond) + .3 * (strong - (uncond if arm == 'cfg_more' else weak))


@torch.inference_mode()
def sample(rt, noise, labels, arm, zero=False):
    before, z = rt.counts(), noise.clone()
    grid = torch.linspace(0, 1, 51, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for i, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
            first = field(rt, z, t, labels, arm, i < 25, zero)
            prediction = z + (u - t) * first
            if i < 49:
                second = field(rt, prediction, u, labels, arm, i < 25, zero)
                z = z + (u - t) * (.5 * (first + second))
            else:
                z = prediction
    counts = rt.counts() - before
    heads = 0 if arm in ('cfg_reference', 'cfg_more') or zero else 50
    assert counts[0] == 198 and counts[1] == heads and np.all(counts[2:] == 198), counts
    assert torch.isfinite(z).all()
    return z, dict(full=198, prefix=0, head=heads, blocks=counts[2:].tolist())


@torch.inference_mode()
def preflight(rt, noise, labels):
    original_checks = base.preflight(rt, noise, labels)
    x, y = torch.from_numpy(noise[:2].copy()).cuda(), torch.from_numpy(labels[:2].copy()).cuda()
    official, _ = base.sample(rt, x, y, 'cfg_reference')
    actual, _ = sample(rt, x, y, 'cfg_reference')
    assert torch.equal(official, actual)
    errors = {}
    for arm in ('cfg_mlp', 'cfg_native', 'cfg_more'):
        zero, _ = sample(rt, x, y, arm, zero=True)
        assert torch.equal(zero, official)
        head = None if arm == 'cfg_more' else rt.heads['mlp' if arm == 'cfg_mlp' else 'native_base']
        def reference_field(z, t, active):
            times = t.expand(len(z))
            if active and head is not None:
                features, condition = c.jig.features(rt.model, z, times, y, depths=(4, 12))
                strong = c.jig.velocity(c.jig.unpatchify(rt.model.final_layer(features['12'], condition)), z, times)
                weak = c.jig.velocity(c.jig.unpatchify(head(features['4'], condition)), z, times)
            else:
                strong = c.jig.velocity(rt.model(z, times, y), z, times)
            uncond = c.jig.velocity(rt.model(z, times, torch.full_like(y, 1000)), z, times)
            if arm == 'cfg_more':
                weak = uncond
            scale = torch.where((times < 1.) & (times > .1), 3., 1.)[:, None, None, None]
            cfg = uncond + scale * (strong - uncond)
            return cfg + .3 * (strong - weak) if active else cfg
        expected = x.clone()
        grid = torch.linspace(0, 1, 51, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for i, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
                first = reference_field(expected, t, i < 25)
                predictor = expected + (u - t) * first
                if i < 49:
                    second = reference_field(predictor, u, i < 25)
                    expected = expected + (u - t) * (.5 * (first + second))
                else:
                    expected = predictor
        actual, _ = sample(rt, x, y, arm)
        error = float((actual - expected).abs().max())
        assert torch.equal(actual, expected), (arm, error)
        errors[arm] = error
    return dict(passed=True, retained_checks=original_checks, official_cfg_full_trajectory_exact=True,
                zero_guidance_exact_for=['cfg_mlp', 'cfg_native', 'cfg_more'], reference_full_trajectory_errors=errors, prefix_calls=0)


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
            z, _ = sample(rt, noise, labels, arm)
            c.pixels(z)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if repeat:
                seconds[arm].append(elapsed)
    c.atomic(ROOT / 'inference_benchmark.json', dict(complete=True, batch=BATCH, repeats=3,
        includes_pixel_quantization=True, seconds=seconds, medians={a: float(np.median(v)) for a, v in seconds.items()},
        full_calls={a: 198 for a in ARMS}, prefix_calls=0,
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
