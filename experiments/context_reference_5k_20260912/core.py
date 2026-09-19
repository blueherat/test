"""Validate the retained contextual head on 5K, including a higher-NFE native baseline."""
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

ROOT = c.EXPS / 'context_reference_5k_20260912'
PROTOCOL = c.WORK / 'docs/CONTEXT_REFERENCE_5K_PROTOCOL_20260912_ZH.md'
STAGE = 'validate_5000'
N = 5000
SEED = 2026121331
ARMS = ('context_base', 'native_base', 'adg', 'native_66')


def configure():
    c.ROOT = ROOT


def training_root(model):
    return local.ROOT / model / 'input_local_training'


def verify(model):
    path = ROOT / model / STAGE / 'request.json'
    return path, local.verify_request(path)


def prepare():
    configure()
    for model in ('sit_small',):
        training = training_root(model)
        local.verify_request(training / 'request.json')
        summary = c.read(training / 'summary.json')
        assert summary['complete'] and summary['steps'] == 3000
        assert c.sha(training / 'head.pt') == summary['head_sha256']
        assert not (local.ROOT / model / local.STAGE).exists()
        selection = c.EXPS / 'context_reference_confirm_20260912' / model / 'confirm_1000/results.json'
        results = c.read(selection)
        assert len(results) == 6
        by = {r['arm']: r for r in results}
        controls = [by[a] for a in ('native_base', 'native_half', 'native_double', 'adg', 'strong')]
        assert by['context_base']['fid'] <= min(r['fid'] for r in controls) - 1
        assert by['context_base']['inception_score'] >= .9 * by['native_base']['inception_score']
        bank = c.prepare_bank(model, STAGE, N, SEED)
        request = dict(model=model, stage=STAGE, arms=list(ARMS), samples=N, seed=SEED,
            single_path=True, extra_prefix=False, old_training_data_reverified=True,
            sources=c.source_manifest([Path(__file__).resolve(), Path(local.__file__).resolve(), local.PROTOCOL, PROTOCOL, Path(angular.__file__).resolve(), selection]),
            assets={str(p): c.sha(p) for p in c.asset_paths(model)},
            inputs={str(p): c.sha(p) for p in bank.glob('*.npy')},
            heads={str(training / f): c.sha(training / f) for f in ('head.pt', 'request.json', 'summary.json', 'checks_after.json', 'generation_deferred.json')})
        path = ROOT / model / STAGE / 'request.json'
        if path.exists():
            assert c.read(path) == request
        else:
            c.atomic(path, request)


def field(rt, heads, capture, arm, z, t, left):
    if arm != 'adg':
        return local.field(rt, heads, capture, arm, z, t, left)
    strength = c.amount(rt, left, 'ig')
    if not strength:
        return rt.field(z, t, 'full')
    strong, weak = rt.pair(z, t)
    return adg_field(z, strong, weak, t, strength)[0]


def sample(rt, heads, capture, noise, labels, arm):
    rt.grid = torch.linspace(0, 1, 67 if arm == 'native_66' else 65, device='cuda')
    with rt.context():
        return c.integrate(rt, noise, labels, lambda z, t, left, i, j: field(rt, heads, capture, arm, z, t, left))


@torch.inference_mode()
def worker(model, rank, world, parent):
    configure()
    request_path, request = verify(model)
    request_hash = c.sha(request_path)
    rt = c.runtime(model)
    heads = local.load_heads(rt)
    checks = local.checks(rt, heads)
    g = torch.Generator(device='cuda').manual_seed(SEED + 8)
    x = torch.randn((2, *c.LATENTS[model]), generator=g, device='cuda')
    t = torch.tensor(.3, device='cuda')
    rt.labels = torch.tensor([0, 1], device='cuda')
    with rt.context():
        actual = field(rt, heads, None, 'adg', x, t, float(t))
        strong, weak = rt.pair(x, t)
        expected = adg_field(x, strong, weak, t, c.amount(rt, float(t), 'ig'))[0]
    assert torch.equal(actual, expected)
    checks['adg_entry_exact'] = True
    c.atomic(ROOT / model / STAGE / f'checks{rank}.json', checks)
    noise, _, labels = c.bank(model, STAGE)
    for arm in ARMS:
        capture = local.Capture(rt) if arm.startswith('context') else None
        root = ROOT / model / STAGE / arm / f'rank{rank}'
        for start in range(rank * rt.batch, N, world * rt.batch):
            c.check_parent(parent)
            path = root / f'batch{start:04d}.npz'
            if path.exists():
                meta = c.read(path.with_suffix('.json'))
                assert c.sha(path) == meta['sha256'] and meta['request_sha256'] == request_hash
                continue
            x = c.cuda(noise[start:start + rt.batch])
            y = c.cuda(labels[start:start + rt.batch])
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z, counts = sample(rt, heads, capture, x, y, arm)
            pixels = rt.decode(z)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            assert counts == dict(full=132 if arm == 'native_66' else 128, prefix=0)
            c.save_batch(path, pixels, z.float().cpu().numpy(), y.cpu().numpy(),
                         dict(seconds=seconds, full_calls=counts['full'], prefix_calls=0),
                         request_hash, start, c.array_sha(noise[start:start + len(y)]))
            if start % (25 * rt.batch) == rank * rt.batch:
                c.atomic(ROOT / model / STAGE / f'progress{rank}.json',
                         dict(pid=os.getpid(), arm=arm, start=start, world=world))
        c.atomic(root / 'complete.json', dict(complete=True, request_sha256=request_hash))
        print(model, rank, arm, 'complete', flush=True)
        if capture is not None:
            capture.close()
    c.atomic(ROOT / model / STAGE / f'worker{rank}_complete.json', dict(complete=True))


def collect(model, arm):
    root = ROOT / model / STAGE / arm
    if (root / 'summary.json').exists():
        return c.read(root / 'summary.json')
    files = sorted(root.glob('rank*/batch*.npz'), key=lambda p: int(c.read(p.with_suffix('.json'))['start']))
    sample_count = 0
    for path in files:
        with np.load(path) as data:
            sample_count += len(data['labels'])
    if sample_count != N:
        return None
    noise, _, labels = c.bank(model, STAGE)
    images, records, coverage = [], [], []
    seconds = 0.
    for path in files:
        meta = c.read(path.with_suffix('.json'))
        assert c.sha(path) == meta['sha256']
        with np.load(path) as data:
            start = int(data['start'])
            count = len(data['labels'])
            coverage.extend(range(start, start + count))
            np.testing.assert_array_equal(data['labels'], labels[start:start + count])
            assert str(data['noise_sha256']) == c.array_sha(noise[start:start + count])
            assert str(data['request_sha256']) == c.sha(ROOT / model / STAGE / 'request.json')
            assert np.isfinite(data['latents']).all()
            assert int(data['prefix_calls']) == 0 and int(data['full_calls']) == (132 if arm == 'native_66' else 128)
            images.append(data['arr_0'])
            seconds += float(data['seconds'])
        records.append(dict(file=str(path), sha256=meta['sha256']))
    assert coverage == list(range(N))
    np.savez(root / 'samples.npz', arr_0=np.concatenate(images))
    summary = dict(complete=True, model=model, stage=STAGE, arm=arm, primary_samples=N,
        generated_paths=N, full_calls_per_output=132 if arm == 'native_66' else 128,
        prefix_calls_at_inference=0, seconds=seconds, samples_sha256=c.sha(root / 'samples.npz'), records=records)
    c.atomic(root / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--model', choices=c.MODELS)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--parent', type=int, default=0)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        worker(args.model, args.rank, args.world, args.parent)
