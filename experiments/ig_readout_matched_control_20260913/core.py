"""One original-architecture head trained on exactly the retained MLP's data stream."""
import argparse
import copy
from dataclasses import replace
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local

ROOT = c.EXPS / 'ig_readout_matched_control_20260913'
TRAIN = ROOT / 'sit_small/training'
PROTOCOL = c.WORK / 'docs/IG_READOUT_MATCHED_CONTROL_PROTOCOL_20260913_ZH.md'
STAGE = 'matched_1000'
SEED = 2026121341
N = 1000
ARM = 'native_fresh'
ORIGINAL = c.EXPS / 'ig_readout_normalization_20260912/sit_small/check_1000'


def configure():
    c.ROOT = ROOT


def prepare_training():
    old = local.ROOT / 'sit_small/input_local_training'
    local.verify_request(old / 'request.json')
    request = dict(model='sit_small', steps=3000, batch=32, seed=local.TRAIN_SEED,
        lr=.0003, weight_decay=.0001, ema=.995, normalization_batches=32,
        sources=c.source_manifest([Path(__file__), Path(local.__file__), PROTOCOL,
            c.WORK / 'experiments/sit_measure_guidance_20260912/data.py']),
        assets={str(p): c.sha(p) for p in c.asset_paths('sit_small')},
        data={str(p): c.sha(p) for p in local.data_paths('sit_small')},
        heads={str(old / f): c.sha(old / f) for f in ('head.pt', 'request.json', 'summary.json')})
    path = TRAIN / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)
    return path


def train(parent=0):
    path = prepare_training()
    local.verify_request(path)
    if (TRAIN / 'summary.json').exists():
        return
    torch.manual_seed(local.TRAIN_SEED)
    rt = c.runtime('sit_small')
    data = local.RealData('sit_small')
    raw = local.make_head(rt)
    native = copy.deepcopy(rt.head.module).train().requires_grad_(True)
    assert native.norm_final.elementwise_affine is False and native.norm_final.eps == 1e-6
    # The official factory zero-initializes its two parameter-bearing Linear layers.
    with torch.no_grad():
        for p in native.parameters():
            p.zero_()
    heads = dict(raw=raw, native_fresh=native)
    g = torch.Generator(device='cuda').manual_seed(local.TRAIN_SEED)
    sums = {k: torch.zeros(384, device='cuda', dtype=torch.float64) for k in ('context', 'condition')}
    squares = {k: v.clone() for k, v in sums.items()}
    counts = {k: 0 for k in sums}
    with torch.no_grad():
        for _ in range(32):
            feats, _, _, _, _ = local.training_batch(rt, data, 'train', g, 32)
            for k in sums:
                value = feats[k].reshape(-1, 384).double()
                sums[k] += value.sum(0)
                squares[k] += value.square().sum(0)
                counts[k] += len(value)
        for key, prefix in (('context', 'token'), ('condition', 'condition')):
            mean = sums[key] / counts[key]
            std = (squares[key] / counts[key] - mean.square()).clamp_min(1e-8).sqrt()
            getattr(raw, prefix + '_mean').copy_(mean.float())
            getattr(raw, prefix + '_std').copy_(std.float())
    ema = {k: copy.deepcopy(h).eval().requires_grad_(False) for k, h in heads.items()}
    optimizers = {k: torch.optim.AdamW(h.parameters(), lr=.0003, weight_decay=.0001) for k, h in heads.items()}
    history = []
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for step in range(1, 3001):
        feats, target, _, _, _ = local.training_batch(rt, data, 'train', g, 32)
        losses = {}
        for mode, head in heads.items():
            optimizers[mode].zero_grad(set_to_none=True)
            with rt.context():
                prediction = head(feats['context'], feats['condition'])
            loss = (prediction.float() - target).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizers[mode].step()
            with torch.no_grad():
                for p, q in zip(ema[mode].parameters(), head.parameters()):
                    p.lerp_(q, .005)
            losses[mode] = float(loss.detach())
        if step == 1 or step % 100 == 0:
            c.check_parent(parent)
            row = dict(step=step, **losses)
            history.append(row)
            c.atomic(TRAIN / 'progress.json', dict(pid=os.getpid(), **row))
            print(row, flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    validation = {k: [] for k in heads}
    vg = torch.Generator(device='cuda').manual_seed(local.TRAIN_SEED + 7)
    with torch.no_grad():
        for _ in range(32):
            feats, target, _, _, _ = local.training_batch(rt, data, 'validation', vg, 32)
            for mode, head in ema.items():
                validation[mode].append(float((head(feats['context'], feats['condition']).float() - target).square().mean()))
    old = torch.load(local.ROOT / 'sit_small/input_local_training/head.pt', map_location='cpu', weights_only=True)['ema']['context']
    differences = {k: float((ema['raw'].state_dict()[k].cpu() - v).abs().max()) for k, v in old.items()}
    assert max(differences.values()) == 0, differences
    assert all(not p.requires_grad and p.grad is None for p in rt.model.parameters())
    checkpoint = TRAIN / 'head.pt'
    temporary = checkpoint.with_suffix('.tmp')
    torch.save(dict(ema={k: h.state_dict() for k, h in ema.items()}, steps=3000, request_sha256=c.sha(path)), temporary)
    temporary.replace(checkpoint)
    summary = dict(complete=True, steps=3000, batch=32, shared_training_seconds=seconds,
        history=history, validation_mse={k: float(np.mean(v)) for k, v in validation.items()},
        parameters={k: sum(p.numel() for p in h.parameters()) for k, h in ema.items()},
        raw_replay_exact=True, raw_replay_max_difference=0., strong_frozen=True,
        request_sha256=c.sha(path), head_sha256=c.sha(checkpoint))
    c.atomic(TRAIN / 'summary.json', summary)
    print('Matched training complete', summary['validation_mse'], flush=True)


def prepare_sampling():
    configure()
    local.verify_request(TRAIN / 'request.json')
    summary = c.read(TRAIN / 'summary.json')
    assert summary['complete'] and summary['raw_replay_exact']
    assert c.sha(TRAIN / 'head.pt') == summary['head_sha256']
    local.verify_request(ORIGINAL / 'request.json')
    bank = c.prepare_bank('sit_small', STAGE, N, SEED)
    for p in bank.glob('*.npy'):
        assert c.sha(p) == c.sha(ORIGINAL / 'inputs' / p.name)
    request = dict(model='sit_small', stage=STAGE, samples=N, seed=SEED, arm=ARM,
        retrospective_matched_control=True, source_bank_identical=True,
        sources=c.source_manifest([Path(__file__), Path(local.__file__), PROTOCOL]),
        assets={str(p): c.sha(p) for p in c.asset_paths('sit_small')},
        inputs={str(p): c.sha(p) for p in bank.glob('*.npy')},
        heads={str(TRAIN / f): c.sha(TRAIN / f) for f in ('head.pt', 'request.json', 'summary.json')},
        comparisons={str(ORIGINAL / arm / f): c.sha(ORIGINAL / arm / f)
            for arm in ('raw', 'native_base', 'adg') for f in ('metrics.json', 'samples.npz')})
    path = ROOT / 'sit_small' / STAGE / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)
    return path


def sample(rt, noise, labels):
    with rt.context():
        return c.integrate(rt, noise, labels, lambda z, t, left, i, j: rt.guided(z, t, c.amount(rt, left, 'ig')))


@torch.inference_mode()
def worker(parent=0):
    path = prepare_sampling()
    local.verify_request(path)
    request_hash = c.sha(path)
    rt = c.runtime('sit_small')
    original_spec = rt.head
    head = copy.deepcopy(rt.head.module)
    state = torch.load(TRAIN / 'head.pt', map_location='cpu', weights_only=True)
    assert state['request_sha256'] == c.sha(TRAIN / 'request.json')
    head.load_state_dict(state['ema'][ARM], strict=True)
    head.eval().requires_grad_(False)
    new_spec = replace(rt.head, module=head, checkpoint=str(TRAIN / 'head.pt'), checkpoint_sha256=c.sha(TRAIN / 'head.pt'))
    rt.head = new_spec
    noise, _, labels = c.bank('sit_small', STAGE)
    x, y = c.cuda(noise[:2]), c.cuda(labels[:2])
    rt.labels = y
    t = torch.tensor(.3, device='cuda')
    full, weak = rt.pair(x, t)
    feats = local.features(rt, x, rt.times(x, t), y)
    expected = local.unpatchify(rt, head(feats['context'], feats['condition'])).float()
    assert torch.equal(weak, expected)
    c.atomic(ROOT / 'sit_small' / STAGE / 'checks.json', dict(passed=True, direct_weak_readout_exact=True, shared_forward=True))
    root = ROOT / 'sit_small' / STAGE / ARM / 'rank0'
    for start in range(0, N, rt.batch):
        c.check_parent(parent)
        output = root / f'batch{start:04d}.npz'
        if output.exists():
            meta = c.read(output.with_suffix('.json'))
            assert meta['request_sha256'] == request_hash and meta['sha256'] == c.sha(output)
            continue
        x, y = c.cuda(noise[start:start + rt.batch]), c.cuda(labels[start:start + rt.batch])
        torch.cuda.synchronize()
        begin = time.perf_counter()
        z, counts = sample(rt, x, y)
        pixels = rt.decode(z)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        assert counts == dict(full=128, prefix=0)
        c.save_batch(output, pixels, z.cpu().numpy(), y.cpu().numpy(),
            dict(seconds=seconds, full_calls=128, prefix_calls=0), request_hash, start, c.array_sha(noise[start:start + len(y)]))
        if start % 200 == 0:
            c.atomic(ROOT / 'sit_small' / STAGE / 'progress.json', dict(pid=os.getpid(), start=start))
    c.atomic(root / 'complete.json', dict(complete=True))
    seconds = {k: [] for k in ('native_original', ARM)}
    x, y = c.cuda(noise[:rt.batch]), c.cuda(labels[:rt.batch])
    for repeat in range(4):
        order = ('native_original', ARM) if repeat % 2 == 0 else (ARM, 'native_original')
        for kind in order:
            rt.head = original_spec if kind == 'native_original' else new_spec
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z, counts = sample(rt, x, y)
            rt.decode(z)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            assert counts == dict(full=128, prefix=0)
            if repeat:
                seconds[kind].append(elapsed)
    c.atomic(ROOT / 'sit_small/inference_benchmark.json', dict(complete=True, seconds=seconds,
        medians={k: float(np.median(v)) for k, v in seconds.items()}, batch=rt.batch, repeats=3,
        includes_decode=True, full_calls=128, prefix_calls=0, parameters=301840, net_added_parameters=0))


def collect():
    configure()
    root = ROOT / 'sit_small' / STAGE / ARM
    if (root / 'summary.json').exists():
        return c.read(root / 'summary.json')
    assert (root / 'rank0/complete.json').exists()
    noise, _, labels = c.bank('sit_small', STAGE)
    images, records, coverage = [], [], []
    seconds = 0.
    request_hash = c.sha(root.parent / 'request.json')
    for path in sorted((root / 'rank0').glob('batch*.npz')):
        meta = c.read(path.with_suffix('.json'))
        assert c.sha(path) == meta['sha256'] and meta['request_sha256'] == request_hash
        with np.load(path) as d:
            start, count = int(d['start']), len(d['labels'])
            coverage.extend(range(start, start + count))
            np.testing.assert_array_equal(d['labels'], labels[start:start + count])
            assert str(d['noise_sha256']) == c.array_sha(noise[start:start + count])
            assert str(d['request_sha256']) == request_hash
            assert np.isfinite(d['latents']).all() and int(d['full_calls']) == 128 and int(d['prefix_calls']) == 0
            images.append(d['arr_0'])
            seconds += float(d['seconds'])
        records.append(dict(file=str(path), sha256=meta['sha256']))
    assert coverage == list(range(N))
    np.savez(root / 'samples.npz', arr_0=np.concatenate(images))
    summary = dict(complete=True, model='sit_small', stage=STAGE, arm=ARM, primary_samples=N, generated_paths=N,
        full_calls_per_output=128, prefix_calls_at_inference=0, seconds=seconds,
        samples_sha256=c.sha(root / 'samples.npz'), records=records)
    c.atomic(root / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--parent', type=int, default=0)
    args = parser.parse_args()
    if args.train:
        train(args.parent)
    else:
        worker(args.parent)
