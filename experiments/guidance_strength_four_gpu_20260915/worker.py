import argparse
import fcntl
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from . import config as k, planning
from experiments.guidance_strength_sweep_20260915 import worker as previous

x = previous.x


@torch.inference_mode()
def write_batch(rt, head, capture, point, start, noise, labels, path, provenance):
    assert rt.batch == k.SAMPLE_BATCH and start % rt.batch == 0
    if path.exists():
        assert path.with_suffix('.json').exists(), path
        previous.verify_batch(path, point, provenance)
        return
    z = x.c.cuda(noise[start:start + rt.batch])
    y = x.c.cuda(labels[start:start + rt.batch])
    begin = time.perf_counter()
    result, counts = previous.integrate(rt, head, capture, point, z, y)
    pixels = rt.decode(result)
    torch.cuda.synchronize()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, arr_0=pixels, latents=result.float().cpu().numpy(), labels=y.cpu().numpy(),
            start=start, seconds=time.perf_counter() - begin,
            request_sha256=k.sha(k.ROOT / 'request.json'), point=point,
            noise_sha256=k.array_sha(noise[start:start + len(y)]))
    temporary.replace(path)
    k.atomic(path.with_suffix('.json'), dict(sha256=k.sha(path),
        request_sha256=k.sha(k.ROOT / 'request.json'), point=point,
        head_provenance=provenance, counts=counts))


def sample_shard(point, rank):
    job = dict(action='sweep_shard', engine='sweep', arm=point, fold=rank)
    if planning.verify_output(job):
        return
    root = k.ROOT / 'points' / point
    (root / 'shards').mkdir(parents=True, exist_ok=True)
    with (root / 'shards' / f'{rank}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rt = x.legacy.runtime()
        head, provenance = previous.load_head(rt, point)
        capture = x.local.Capture(rt)
        noise = np.load(k.BANK / 'noise.npy', mmap_mode='r')
        labels = np.load(k.BANK / 'labels.npy')
        records, files = [], {}
        try:
            for index, start in enumerate(k.batch_starts(rank)):
                k.check_stop()
                path = root / 'batches' / f'{start:05d}.npz'
                if path.exists() and not path.with_suffix('.json').exists():
                    # The previous worker stopped between the two atomic writes.
                    path.unlink()
                write_batch(rt, head, capture, point, start, noise, labels, path, provenance)
                meta = previous.verify_batch(path, point, provenance)
                files.update({str(path): meta['sha256'], str(path.with_suffix('.json')): k.sha(path.with_suffix('.json'))})
                records.append(dict(file=str(path), sha256=meta['sha256']))
                k.atomic(root / 'shards' / f'{rank}_progress.json', dict(pid=os.getpid(),
                    rank=rank, point=point, batches_done=index + 1, batches_total=len(k.batch_starts(rank))))
            result = dict(valid=True, batch_starts=list(k.batch_starts(rank)), records=records, files=files)
        except FloatingPointError as error:
            result = dict(valid=False, reason='Numerical divergence during sampling', error=str(error))
        finally:
            capture.close()
        k.atomic(planning.output(job), dict(complete=True, point=point, rank=rank,
            request_sha256=k.sha(k.ROOT / 'request.json'), dispatch_sha256=k.sha(k.AMENDMENT),
            head_provenance=provenance, **result))
        print('Finished shard', point, rank, result['valid'], flush=True)


def assemble(point):
    if planning.verify_output(dict(action='sweep_assemble', arm=point, engine='sweep')):
        return
    root = k.ROOT / 'points' / point
    shards = []
    for rank in range(k.SHARDS):
        job = dict(action='sweep_shard', arm=point, fold=rank, engine='sweep')
        assert planning.verify_output(job)
        shards.append(k.read(planning.output(job)))
    provenance = shards[0]['head_provenance']
    assert all(row['head_provenance'] == provenance for row in shards)
    rh = k.sha(k.ROOT / 'request.json')
    invalid = [row for row in shards if not row['valid']]
    if invalid:
        k.atomic(root / 'summary.json', dict(complete=True, valid=False, point=point,
            request_sha256=rh, dispatch_sha256=k.sha(k.AMENDMENT),
            reason='Numerical divergence during sampling', head_provenance=provenance,
            invalid_shards=[row['rank'] for row in invalid]))
        return
    noise = np.load(k.BANK / 'noise.npy', mmap_mode='r')
    labels = np.load(k.BANK / 'labels.npy')
    images, records, coverage, seconds = [], [], [], 0.
    paths = sorted((root / 'batches').glob('*.npz'))
    assert [int(path.stem) for path in paths] == list(range(0, 1000, k.SAMPLE_BATCH))
    for path in paths:
        meta = previous.verify_batch(path, point, provenance)
        assert meta['counts'] == k.counts(point)
        with np.load(path) as batch:
            start, n = int(batch['start']), len(batch['labels'])
            assert start == int(path.stem) and n == min(k.SAMPLE_BATCH, 1000 - start)
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
        bank_sha256=k.sha(k.BANK / 'complete.json'), reused=False,
        dispatch_sha256=k.sha(k.AMENDMENT), shards=k.SHARDS))
    k.atomic(root / 'progress.json', dict(pid=os.getpid(), point=point, completed=1000, total=1000))
    print('Assembled ordered 1000 samples', point, flush=True)


def check(rank):
    # Replay existing images, rather than adding samples to the quality budget.
    rt = x.legacy.runtime()
    point = k.key('guided_weak', 48)
    head, provenance = previous.load_head(rt, point)
    capture = x.local.Capture(rt)
    start = k.batch_starts(rank)[0]
    noise = np.load(k.BANK / 'noise.npy', mmap_mode='r')
    labels = np.load(k.BANK / 'labels.npy')
    source = k.ROOT / 'points' / point / 'batches' / f'{start:05d}.npz'
    try:
        with tempfile.TemporaryDirectory(prefix='eqvae_shard_check_') as temp:
            path = Path(temp) / f'{start:05d}.npz'
            write_batch(rt, head, capture, point, start, noise, labels, path, provenance)
            with np.load(path) as result, np.load(source) as reference:
                for name in ('latents', 'arr_0', 'labels', 'start', 'noise_sha256'):
                    np.testing.assert_array_equal(result[name], reference[name])
            before = k.sha(path)
            write_batch(rt, head, capture, point, start, noise, labels, path, provenance)
            assert k.sha(path) == before, 'Resume must retain the original batch'
    finally:
        capture.close()
    k.atomic(planning.output(dict(action='four_gpu_check', fold=rank)), dict(passed=True, rank=rank,
        request_sha256=k.sha(k.ROOT / 'request.json'), dispatch_sha256=k.sha(k.AMENDMENT),
        replay_source=str(source), source_sha256=k.sha(source),
        pixels_and_latents_exact=True, resume_preserves_existing_batch=True,
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), new_quality_samples=0))
    print('GPU shard replay and resume passed', rank, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    for name in ('arm', 'source', 'fold', 'stage'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    k.verify()
    k.old.verify()
    k.check_stop()
    if args.action == 'sweep_shard':
        sample_shard(args.arm, int(args.fold))
    elif args.action == 'sweep_assemble':
        assemble(args.arm)
    elif args.action == 'four_gpu_check':
        check(int(args.fold))
    else:
        previous.main()


if __name__ == '__main__':
    try:
        main()
    except k.RequestedStop as error:
        print(error, flush=True)
        raise SystemExit(75)
