"""Replace only the null readout; keep both CFG backbone evaluations and fixed controls."""
import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local
from experiments.small_sit_guidance_tuning_20260910 import adg_field
import experiments.small_sit_guidance_tuning_20260910 as angular
from experiments.sit_guidance_portfolio_20260910 import operators as prior
from . import train as tr

ROOT, PROTOCOL = tr.ROOT, tr.PROTOCOL
STAGE = 'screen_400'
N, SEED = 400, 2026121361
ARMS = (*tr.MODES, 'native_base', 'native_half', 'adg', 'apg')


def configure():
    c.ROOT = ROOT


def verify():
    path = ROOT / 'sit_small' / STAGE / 'request.json'
    return path, local.verify_request(path)


def prepare():
    configure()
    local.verify_request(tr.TRAIN / 'request.json')
    summary = c.read(tr.TRAIN / 'summary.json')
    assert summary['complete'] and summary['head_sha256'] == c.sha(tr.TRAIN / 'head.pt')
    bank = c.prepare_bank('sit_small', STAGE, N, SEED)
    request = dict(model='sit_small', stage=STAGE, samples=N, seed=SEED, arms=list(ARMS),
        full_calls_per_output=224, prefix_calls=0, single_path=True,
        sources=c.source_manifest([Path(__file__), Path(tr.__file__), Path(local.__file__),
            Path(angular.__file__), Path(prior.__file__), PROTOCOL]),
        assets={str(p): c.sha(p) for p in c.asset_paths('sit_small')},
        inputs={str(p): c.sha(p) for p in bank.glob('*.npy')},
        heads={str(tr.TRAIN / f): c.sha(tr.TRAIN / f) for f in ('head.pt', 'summary.json', 'request.json')})
    path = ROOT / 'sit_small' / STAGE / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)


def branches(rt, heads, arm, z, t):
    original = rt.model.final_layer
    labels = rt.labels
    strong = rt.field(z, t, 'full')
    try:
        rt.labels = torch.full_like(labels, 100)
        if arm in tr.MODES:
            rt.model.final_layer = heads[arm]
        weak = rt.field(z, t, 'full')
    finally:
        rt.model.final_layer = original
        rt.labels = labels
    return strong, weak


def sample(rt, heads, noise, labels, arm, zero=False):
    history, pending = {}, {}
    original = rt.model.final_layer
    def field(z, t, left, step, substage):
        factor = .5 if arm == 'native_half' else 1.
        amount = 0. if zero else c.amount(rt, left, 'cfg', factor)
        if not amount:
            return rt.field(z, t, 'full')
        strong, weak = branches(rt, heads, arm, z, t)
        if arm == 'adg':
            return adg_field(z, strong, weak, t, amount)[0]
        gap = strong - weak
        if arm == 'apg':
            momentum = gap - .5 * history.get('momentum', torch.zeros_like(gap))
            bounded = prior.cap(momentum, 2 * prior.norm(gap))
            direction = bounded - prior.projection(bounded, z + (1 - t) * strong)
            if substage == 0:
                pending['momentum'] = momentum.detach()
            else:
                history['momentum'] = pending['momentum']
            return strong + 2. * direction
        return strong + amount * gap
    with rt.context():
        output, counts = c.integrate(rt, noise, labels, field)
    assert rt.model.final_layer is original and torch.equal(rt.labels, labels)
    return output, counts


@torch.inference_mode()
def preflight(rt, heads):
    noise, _, labels = c.bank('sit_small', STAGE)
    x, y = c.cuda(noise[:2]), c.cuda(labels[:2])
    rt.labels = y
    t = torch.tensor(.3, device='cuda')
    original = rt.model.final_layer
    native = rt.field(x, t, 'full')
    errors = {}
    for arm in tr.MODES:
        strong, weak = branches(rt, heads, arm, x, t)
        assert torch.equal(native, strong)
        tokens, condition = tr.features(rt, x, rt.times(x, t))
        expected = tr.project(rt, heads[arm], tokens, condition).float()
        errors[arm] = float((expected - weak).abs().max())
        assert torch.equal(expected, weak), errors
    zero = [sample(rt, heads, x, y, arm, zero=True)[0] for arm in ('mlp', 'native_base')]
    assert torch.equal(zero[0], zero[1])
    # Full-trajectory comparison with the established APG implementation.
    rt.portfolio_assets = {}
    rt.capture_weak = SimpleNamespace(value=x.new_zeros(1))
    config = dict(arm='apg_check', key='cfg_apg_momentum', theta=-.5, strength=2.,
        source='cfg', cutoff=.75, solver='heun64')
    expected, old_cost = prior.sample(rt, x, y, config, batch_seed=SEED)
    actual, cost = sample(rt, heads, x, y, 'apg')
    assert torch.equal(expected, actual), float((expected - actual).abs().max())
    assert old_cost['full_calls'] == 224 and cost == dict(full=224, prefix=0)
    del rt.portfolio_assets, rt.capture_weak
    assert rt.model.final_layer is original
    return dict(passed=True, conditional_outputs_exact=True, null_readout_errors=errors,
        zero_guidance_exact=True, apg_full_trajectory_exact=True, apg_commits_first_heun_stage=True)


@torch.inference_mode()
def worker(rank, world, parent):
    configure()
    request_path, _ = verify()
    request_hash = c.sha(request_path)
    rt = c.runtime('sit_small')
    heads = tr.load_heads(rt)
    c.atomic(ROOT / 'sit_small' / STAGE / f'checks{rank}.json', preflight(rt, heads))
    noise, _, labels = c.bank('sit_small', STAGE)
    for arm in ARMS:
        root = ROOT / 'sit_small' / STAGE / arm / f'rank{rank}'
        for start in range(rank * rt.batch, N, world * rt.batch):
            c.check_parent(parent)
            path = root / f'batch{start:04d}.npz'
            if path.exists():
                meta = c.read(path.with_suffix('.json'))
                assert meta['sha256'] == c.sha(path) and meta['request_sha256'] == request_hash
                continue
            x, y = c.cuda(noise[start:start + rt.batch]), c.cuda(labels[start:start + rt.batch])
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z, counts = sample(rt, heads, x, y, arm)
            pixels = rt.decode(z)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            assert counts == dict(full=224, prefix=0)
            c.save_batch(path, pixels, z.cpu().numpy(), y.cpu().numpy(),
                dict(seconds=seconds, full_calls=224, prefix_calls=0), request_hash, start,
                c.array_sha(noise[start:start + len(y)]))
            if start % 200 == rank * rt.batch:
                c.atomic(ROOT / 'sit_small' / STAGE / f'progress{rank}.json', dict(pid=os.getpid(), arm=arm, start=start))
        c.atomic(root / 'complete.json', dict(complete=True, request_sha256=request_hash))
        print(rank, arm, 'complete', flush=True)
    if rank == 0:
        seconds = {a: [] for a in ARMS}
        x, y = c.cuda(noise[:rt.batch]), c.cuda(labels[:rt.batch])
        for repeat in range(4):
            for arm in ARMS[repeat:] + ARMS[:repeat]:
                torch.cuda.synchronize()
                begin = time.perf_counter()
                z, counts = sample(rt, heads, x, y, arm)
                rt.decode(z)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                assert counts == dict(full=224, prefix=0)
                if repeat:
                    seconds[arm].append(elapsed)
        c.atomic(ROOT / 'sit_small/inference_benchmark.json', dict(complete=True, seconds=seconds,
            medians={a: float(np.median(v)) for a, v in seconds.items()}, batch=rt.batch, repeats=3,
            includes_decode=True, full_calls=224, prefix_calls=0,
            additional_parameters={k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()}))


def collect(arm):
    root = ROOT / 'sit_small' / STAGE / arm
    if (root / 'summary.json').exists():
        return c.read(root / 'summary.json')
    if not all((root / f'rank{rank}/complete.json').exists() for rank in range(2)):
        return None
    noise, _, labels = c.bank('sit_small', STAGE)
    request_hash = c.sha(root.parent / 'request.json')
    images, records, coverage = [], [], []
    seconds = 0.
    files = sorted(root.glob('rank*/batch*.npz'), key=lambda p: int(c.read(p.with_suffix('.json'))['start']))
    for path in files:
        meta = c.read(path.with_suffix('.json'))
        assert c.sha(path) == meta['sha256'] and meta['request_sha256'] == request_hash
        with np.load(path) as d:
            start, count = int(d['start']), len(d['labels'])
            coverage.extend(range(start, start + count))
            np.testing.assert_array_equal(d['labels'], labels[start:start + count])
            assert str(d['noise_sha256']) == c.array_sha(noise[start:start + count])
            assert str(d['request_sha256']) == request_hash and np.isfinite(d['latents']).all()
            assert int(d['full_calls']) == 224 and int(d['prefix_calls']) == 0
            images.append(d['arr_0'])
            seconds += float(d['seconds'])
        records.append(dict(file=str(path), sha256=meta['sha256']))
    assert coverage == list(range(N))
    np.savez(root / 'samples.npz', arr_0=np.concatenate(images))
    summary = dict(complete=True, model='sit_small', stage=STAGE, arm=arm, primary_samples=N, generated_paths=N,
        full_calls_per_output=224, prefix_calls_at_inference=0, seconds=seconds,
        samples_sha256=c.sha(root / 'samples.npz'), records=records)
    c.atomic(root / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=2)
    parser.add_argument('--parent', type=int, default=0)
    args = parser.parse_args()
    worker(args.rank, args.world, args.parent)
