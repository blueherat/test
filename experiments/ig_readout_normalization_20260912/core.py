"""Independent paired generation for the normalization information hypothesis."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local
from experiments.small_sit_guidance_tuning_20260910 import adg_field
import experiments.small_sit_guidance_tuning_20260910 as angular
from . import train as tr

ROOT = tr.ROOT
PROTOCOL = tr.PROTOCOL
STAGE = 'check_1000'
N = 1000
SEED = 2026121341
ARMS = (*tr.MODES, 'native_base', 'adg')


def configure():
    c.ROOT = ROOT


def verify(model='sit_small'):
    path = ROOT / model / STAGE / 'request.json'
    return path, local.verify_request(path)


def prepare():
    configure()
    local.verify_request(tr.TRAIN / 'request.json')
    summary = c.read(tr.TRAIN / 'summary.json')
    assert summary['complete'] and summary['steps'] == 3000
    assert c.sha(tr.TRAIN / 'head.pt') == summary['head_sha256']
    bank = c.prepare_bank('sit_small', STAGE, N, SEED)
    request = dict(model='sit_small', samples=N, seed=SEED, arms=list(ARMS), single_path=True,
        full_calls_per_output=128, additional_prefix_calls=0,
        sources=c.source_manifest([Path(__file__), Path(tr.__file__), Path(local.__file__), Path(angular.__file__), PROTOCOL]),
        assets={str(p): c.sha(p) for p in c.asset_paths('sit_small')},
        inputs={str(p): c.sha(p) for p in bank.glob('*.npy')},
        heads={str(tr.TRAIN / f): c.sha(tr.TRAIN / f) for f in ('head.pt', 'request.json', 'summary.json')})
    path = ROOT / 'sit_small' / STAGE / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)


def field(rt, heads, capture, arm, z, t, left):
    amount = c.amount(rt, left, 'ig')
    if not amount:
        return rt.field(z, t, 'full')
    full, weak = rt.pair(z, t)
    if arm in tr.MODES:
        weak = local.unpatchify(rt, heads[arm](capture.values['context'], capture.values['condition'])).float()
    if arm == 'adg':
        return adg_field(z, full, weak, t, amount)[0]
    return full + amount * (full - weak)


def sample(rt, heads, capture, noise, labels, arm):
    with rt.context():
        return c.integrate(rt, noise, labels, lambda z, t, left, i, j: field(rt, heads, capture, arm, z, t, left))


@torch.inference_mode()
def worker(parent):
    configure()
    path, _ = verify()
    request_hash = c.sha(path)
    rt = c.runtime('sit_small')
    heads = tr.load_heads(rt)
    g = torch.Generator(device='cuda').manual_seed(SEED + 8)
    z = torch.randn((2, *c.LATENTS['sit_small']), generator=g, device='cuda')
    t = torch.tensor(.3, device='cuda')
    rt.labels = torch.tensor([0, 1], device='cuda')
    full0, weak0 = rt.pair(z, t)
    capture = local.Capture(rt)
    full1, weak1 = rt.pair(z, t)
    direct = local.features(rt, z, rt.times(z, t), rt.labels)
    assert torch.equal(full0, full1) and torch.equal(weak0, weak1)
    assert torch.equal(direct['context'], capture.values['context'])
    errors = {}
    for arm in ARMS:
        actual = field(rt, heads, capture, arm, z, t, float(t))
        weak = weak1 if arm not in tr.MODES else local.unpatchify(rt, heads[arm](direct['context'], direct['condition'])).float()
        amount = c.amount(rt, float(t), 'ig')
        expected = adg_field(z, full1, weak, t, amount)[0] if arm == 'adg' else full1 + amount * (full1 - weak)
        errors[arm] = float((actual - expected).abs().max())
        assert torch.equal(actual, expected), (arm, errors[arm])
    capture.close()
    c.atomic(ROOT / 'sit_small' / STAGE / 'checks.json', dict(passed=True, captured_features_exact=True,
        hook_native_outputs_exact=True, direct_field_errors=errors, strong_parameters_frozen=True))
    noise, _, labels = c.bank('sit_small', STAGE)
    for arm in ARMS:
        capture = local.Capture(rt) if arm in tr.MODES else None
        root = ROOT / 'sit_small' / STAGE / arm / 'rank0'
        for start in range(0, N, rt.batch):
            c.check_parent(parent)
            path = root / f'batch{start:04d}.npz'
            if path.exists():
                metadata = c.read(path.with_suffix('.json'))
                assert c.sha(path) == metadata['sha256'] and metadata['request_sha256'] == request_hash
                continue
            x, y = c.cuda(noise[start:start + rt.batch]), c.cuda(labels[start:start + rt.batch])
            torch.cuda.synchronize()
            begin = time.perf_counter()
            output, counts = sample(rt, heads, capture, x, y, arm)
            pixels = rt.decode(output)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            assert counts == dict(full=128, prefix=0)
            c.save_batch(path, pixels, output.float().cpu().numpy(), y.cpu().numpy(),
                dict(seconds=seconds, full_calls=128, prefix_calls=0), request_hash,
                start, c.array_sha(noise[start:start + len(y)]))
            if start % 200 == 0:
                c.atomic(ROOT / 'sit_small' / STAGE / 'progress.json', dict(pid=os.getpid(), arm=arm, start=start))
        if capture is not None:
            capture.close()
        c.atomic(root / 'complete.json', dict(complete=True, request_sha256=request_hash))
        print(arm, 'sampling complete', flush=True)
    seconds = {a: [] for a in ARMS}
    noise, labels = c.cuda(noise[:rt.batch]), c.cuda(labels[:rt.batch])
    for repeat in range(4):
        for arm in ARMS[repeat:] + ARMS[:repeat]:
            capture = local.Capture(rt) if arm in tr.MODES else None
            torch.cuda.synchronize()
            begin = time.perf_counter()
            output, counts = sample(rt, heads, capture, noise, labels, arm)
            rt.decode(output)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            assert counts == dict(full=128, prefix=0)
            if capture is not None:
                capture.close()
            if repeat:
                seconds[arm].append(elapsed)
    medians = {a: float(np.median(v)) for a, v in seconds.items()}
    c.atomic(ROOT / 'sit_small/inference_benchmark.json', dict(complete=True, seconds=seconds, medians=medians,
        batch=rt.batch, repeats=3, includes_decode=True, full_calls=128, additional_prefix_calls=0,
        parameters={k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()},
        method_sha256=c.sha(Path(__file__))))
    c.atomic(ROOT / 'sit_small' / STAGE / 'worker_complete.json', dict(complete=True))


def collect(model, arm):
    root = ROOT / model / STAGE / arm
    if (root / 'summary.json').exists():
        return c.read(root / 'summary.json')
    if not (root / 'rank0/complete.json').exists():
        return None
    files = sorted(root.glob('rank*/batch*.npz'))
    noise, _, labels = c.bank(model, STAGE)
    images, coverage, records = [], [], []
    seconds = 0.
    request_hash = c.sha(ROOT / model / STAGE / 'request.json')
    for p in files:
        metadata = c.read(p.with_suffix('.json'))
        assert c.sha(p) == metadata['sha256'] and metadata['request_sha256'] == request_hash
        with np.load(p) as d:
            start, n = int(d['start']), len(d['labels'])
            coverage.extend(range(start, start + n))
            np.testing.assert_array_equal(d['labels'], labels[start:start + n])
            assert str(d['noise_sha256']) == c.array_sha(noise[start:start + n])
            assert str(d['request_sha256']) == request_hash and np.isfinite(d['latents']).all()
            assert int(d['full_calls']) == 128 and int(d['prefix_calls']) == 0
            images.append(d['arr_0'])
            seconds += float(d['seconds'])
        records.append(dict(file=str(p), sha256=metadata['sha256']))
    assert coverage == list(range(N))
    np.savez(root / 'samples.npz', arr_0=np.concatenate(images))
    summary = dict(complete=True, model=model, stage=STAGE, arm=arm, primary_samples=N, generated_paths=N,
        full_calls_per_output=128, prefix_calls_at_inference=0, seconds=seconds,
        samples_sha256=c.sha(root / 'samples.npz'), records=records)
    c.atomic(root / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=int, default=0)
    args = p.parse_args()
    worker(args.parent)
