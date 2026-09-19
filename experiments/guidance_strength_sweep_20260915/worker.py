"""One coefficient point, exact anchor reuse, or the current idea's training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

from . import config as k
from . import planning
from experiments.guidance_loss_50k_20260914 import components as x
from experiments.guidance_loss_50k_20260914.sampling import Counter, head_provenance


def field(rt, head, capture, point, z, t, left):
    method, tick = k.parse(point)
    coefficient = k.value(tick)
    a = x.c.amount(rt, left, 'ig')
    if method == 'strong':
        return rt.field(z, t, 'full')
    if method == 'native':
        a *= coefficient / k.old.ALPHA
        if not a:
            return rt.field(z, t, 'full')
        full, weak = rt.pair(z, t)
        return full + a * (full - weak)
    if method == 'cfg_native':
        full = rt.field(z, t, 'full')
        b = coefficient if left < .75 else 0.
        return full + b * (full - x.unconditional(rt, z, t)) if b else full
    spec = k.old.spec(method)
    need = head is not None and a != 0 and coefficient != 0
    if spec['base'] == 'ig' and a:
        full, weak = rt.pair(z, t)
        base = full + a * (full - weak)
    else:
        full = rt.field(z, t, 'full')
        base = full
    features = dict(capture.values) if need else None
    if spec['base'] == 'cfg':
        b = x.c.amount(rt, left, 'cfg')
        if b:
            base = full + b * (full - x.unconditional(rt, z, t))
    if not need:
        return base
    prediction = x.local.unpatchify(rt, head(features['context'], features['condition'])).float()
    if spec['loss'] in k.old.WEAK_LOSSES:
        return base + (a * (coefficient / k.old.ALPHA)) * (full - prediction)
    factor = 4. if spec['loss'] == 'covariance' else 1.
    return base + ((a / k.old.ALPHA) * factor * coefficient) * prediction


def load_head(rt, point):
    method, tick = k.parse(point)
    if method in ('strong', 'native', 'cfg_native') or not tick:
        return None, {}
    return x.legacy.trained_head(rt, method), head_provenance(method)


@torch.inference_mode()
def integrate(rt, head, capture, point, noise, labels):
    counter = Counter(rt, head)
    try:
        result, actual = x.c.integrate(rt, noise, labels,
            lambda z, t, left, i, j: field(rt, head, capture, point, z, t, left))
        expected = k.counts(point)
        assert actual == dict(full=expected['full'], prefix=0)
        assert counter.blocks == [expected['full']] * 12
        assert counter.head == expected['head'] and counter.native_head == expected['native_head']
        return result, expected
    finally:
        counter.close()


@torch.inference_mode()
def preflight():
    rt = x.legacy.runtime()
    capture = x.local.Capture(rt)
    noise = np.load(k.BANK / 'noise.npy', mmap_mode='r')[:8].copy()
    labels = np.load(k.BANK / 'labels.npy')[:8].copy()
    z, y = x.c.cuda(noise), x.c.cuda(labels)
    records = []
    # Reproduce a full batch, including its original batch dimension and TF32 rounding.
    for method, tick in (('guided_weak', 32), ('gaussian', 16), ('contrast_s', 40), ('covariance', 40), ('cfg_native', 50)):
        source = k.anchor(k.key(method, tick))
        old_path = k.old.ROOT / k.old.MODEL / 'screen1000' / source / 'batches/00000.npz'
        point = k.key(method, tick)
        head, _ = load_head(rt, point)
        actual, counts = integrate(rt, head, capture, point, z, y)
        with np.load(old_path) as previous:
            np.testing.assert_array_equal(actual.cpu().numpy(), previous['latents'])
            np.testing.assert_array_equal(rt.decode(actual), previous['arr_0'])
        records.append(dict(point=point, source=str(old_path), source_sha256=k.sha(old_path),
                            full_trajectory_and_pixels_exact=True, counts=counts))
        del head
    strong, counts = integrate(rt, None, capture, k.key('strong', 0), z, y)
    head, _ = load_head(rt, k.key('guided_weak', 32))
    zero, zero_counts = integrate(rt, head, capture, k.key('guided_weak', 0), z, y)
    assert torch.equal(strong, zero) and counts == zero_counts
    assert all(p.grad is None and not p.requires_grad for p in rt.model.parameters())
    capture.close()
    k.atomic(k.ROOT / 'gpu_preflight.json', dict(passed=True, request_sha256=k.sha(k.ROOT / 'request.json'),
        anchors=records, zero_guidance_exact=True, zero_head_calls=0, prefix_calls=0,
        no_new_quality_samples=True, strong_frozen=True))
    print('Strength sweep GPU checks passed: anchors, zero coefficient, and actual call counts', flush=True)


def verify_batch(path, point, provenance):
    row = k.read(path.with_suffix('.json'))
    assert row['sha256'] == k.sha(path)
    assert row['request_sha256'] == k.sha(k.ROOT / 'request.json')
    assert row['point'] == point and row['head_provenance'] == provenance
    return row


@torch.inference_mode()
def sample(point):
    root = k.ROOT / 'points' / point
    job = dict(action='sweep_sample', arm=point, engine='sweep')
    if planning.verify_output(job):
        return
    rt = x.legacy.runtime()
    head, provenance = load_head(rt, point)
    capture = x.local.Capture(rt)
    noise = np.load(k.BANK / 'noise.npy', mmap_mode='r')
    labels = np.load(k.BANK / 'labels.npy')
    rh = k.sha(k.ROOT / 'request.json')
    try:
        for start in range(0, 1000, rt.batch):
            k.check_stop()
            path = root / 'batches' / f'{start:05d}.npz'
            if path.exists() and not path.with_suffix('.json').exists():
                path.unlink()
            if path.exists():
                verify_batch(path, point, provenance)
                continue
            z, y = x.c.cuda(noise[start:start + rt.batch]), x.c.cuda(labels[start:start + rt.batch])
            begin = time.perf_counter()
            result, counts = integrate(rt, head, capture, point, z, y)
            pixels = rt.decode(result)
            torch.cuda.synchronize()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, arr_0=pixels, latents=result.float().cpu().numpy(), labels=y.cpu().numpy(),
                    start=start, seconds=time.perf_counter() - begin, request_sha256=rh,
                    point=point, noise_sha256=k.array_sha(noise[start:start + len(y)]))
            temporary.replace(path)
            k.atomic(path.with_suffix('.json'), dict(sha256=k.sha(path), request_sha256=rh,
                point=point, head_provenance=provenance, counts=counts))
            if start % 80 == 0:
                k.atomic(root / 'progress.json', dict(pid=os.getpid(), point=point, completed=start + len(y), total=1000))
        images, records, coverage, seconds = [], [], [], 0.
        for path in sorted((root / 'batches').glob('*.npz')):
            meta = verify_batch(path, point, provenance)
            assert meta['counts'] == k.counts(point)
            with np.load(path) as batch:
                start, n = int(batch['start']), len(batch['labels'])
                coverage.extend(range(start, start + n))
                np.testing.assert_array_equal(batch['labels'], labels[start:start + n])
                assert str(batch['noise_sha256']) == k.array_sha(noise[start:start + n])
                assert str(batch['point']) == point and str(batch['request_sha256']) == rh
                assert batch['arr_0'].shape == (n, 256, 256, 3) and batch['arr_0'].dtype == np.uint8
                assert np.isfinite(batch['latents']).all()
                images.append(batch['arr_0'])
                seconds += float(batch['seconds'])
            records.append(dict(file=str(path), sha256=meta['sha256']))
        assert coverage == list(range(1000))
        output = root / 'samples.npz'
        temporary = output.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, arr_0=np.concatenate(images))
        temporary.replace(output)
        k.atomic(root / 'summary.json', dict(complete=True, valid=True, point=point, primary_samples=1000,
            counts=k.counts(point), head_provenance=provenance, seconds=seconds, request_sha256=rh,
            samples_path=str(output), samples_sha256=k.sha(output), records=records,
            bank_sha256=k.sha(k.BANK / 'complete.json'), reused=False))
        k.atomic(root / 'progress.json', dict(pid=os.getpid(), point=point, completed=1000, total=1000))
    except FloatingPointError as error:
        k.atomic(root / 'summary.json', dict(complete=True, valid=False, point=point, request_sha256=rh,
            reason='Numerical divergence during sampling', error=str(error), head_provenance=provenance))
        print('Invalid coefficient', point, error, flush=True)
    finally:
        capture.close()


def evaluate(point):
    root = k.ROOT / 'points' / point
    if planning.verify_output(dict(action='sweep_evaluate', arm=point, engine='sweep')):
        return
    assert planning.verify_output(dict(action='sweep_sample', arm=point, engine='sweep'))
    summary = k.read(root / 'summary.json')
    if not summary.get('valid', True):
        k.atomic(root / 'metrics.json', dict(summary, fid=None, inception_score=None))
        return
    reference = k.EXPS.parent / 'imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'
    command = ['/data/shared/envs/adm-fid/bin/python', str(k.WORK / 'experiments/compute_adm_fid.py'),
        '--reference', str(reference), '--samples', summary['samples_path'], '--output', str(root / 'adm.json'),
        '--activations-output', str(root / 'activations.npz'), '--batch-size', '32']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        MKL_NUM_THREADS='4', TF_NUM_INTRAOP_THREADS='4', TF_NUM_INTEROP_THREADS='1')
    with (root / 'evaluation.log').open('a') as stream:
        subprocess.run(command, cwd=k.WORK, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = k.read(root / 'adm.json')
    assert result['sample_count'] == 1000 and np.isfinite(result['fid']) and np.isfinite(result['inception_score'])
    k.atomic(root / 'metrics.json', dict(summary, fid=result['fid'], inception_score=result['inception_score']))
    print(point, result['fid'], result['inception_score'], flush=True)


def reuse(point):
    source = k.anchor(point)
    assert source is not None
    job = dict(action='evaluate', arm=source, stage='screen1000')
    assert planning.original.verify_output(job)
    root = planning.original.output(job).parent
    metric = k.read(root / 'metrics.json')
    expected = k.counts(point)
    assert expected == dict(full=metric['full_calls_per_output'], prefix=metric['prefix_calls_at_inference'],
        head=metric['head_calls_per_output'], native_head=metric.get('native_head_calls_per_output', 0))
    assert metric['primary_samples'] == 1000 and metric['bank_sha256'] == k.sha(k.BANK / 'complete.json')
    value = dict(complete=True, valid=True, reused=True, point=point, primary_samples=1000,
        counts=expected, request_sha256=k.sha(k.ROOT / 'request.json'),
        samples_path=str(root / 'samples.npz'), samples_sha256=metric['samples_sha256'],
        fid=metric['fid'], inception_score=metric['inception_score'],
        head_provenance=metric.get('head_provenance', {}),
        dependencies={str(root / name): k.sha(root / name) for name in ('summary.json', 'metrics.json')},
        original_request_sha256=metric['request_sha256'], bank_sha256=metric['bank_sha256'])
    k.atomic(k.ROOT / 'points' / point / 'summary.json', value)
    k.atomic(k.ROOT / 'points' / point / 'metrics.json', value)
    print('Reused identical existing point', point, source, value['fid'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    for name in ('arm', 'source', 'fold', 'stage'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    k.verify()
    k.old.verify()
    k.check_stop()
    if args.action == 'sweep_preflight':
        preflight()
    elif args.action == 'sweep_sample':
        sample(args.arm)
    elif args.action == 'sweep_evaluate':
        evaluate(args.arm)
    elif args.action == 'reuse':
        reuse(args.arm)
    else:
        # Keep original CLI, code, scientific hashes and checkpoint recovery intact.
        from experiments.guidance_loss_50k_20260914 import worker as original
        original.main()


if __name__ == '__main__':
    try:
        main()
    except k.RequestedStop as error:
        print(error, flush=True)
        raise SystemExit(75)
