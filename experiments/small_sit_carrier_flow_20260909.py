"""Three paired small-SiT quality controls for a finite, fixed-time gap flow."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

from experiments.lifting_scale_sweep_20260909 import (
    EXPS, SMALL_DATA, WORK, Runtime, array_sha, atomic, read, sha, source_paths,
)

ROOT = EXPS / 'small_sit_carrier_flow_20260909'
BANK_ROOT = EXPS / 'lifting_wide_scale_20260909/sit_small'
REFERENCE = SMALL_DATA / 'adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'
ARMS = ('ig_restarted', 'write_before', 'write_after')
GRID = (0., .125, .25, .375, .5, 1.)
ALPHAS = (.6, .6, .7, .7, 0.)
HEUN_STEPS = 4
SAMPLES, BATCH, RANKS = 1000, 8, 4


def verify_request():
    request = read(ROOT / 'request.json')
    for category in ('sources', 'assets'):
        for p, h in request[category].items():
            assert sha(p) == h, (category, p)
    for name, h in request['bank_files'].items():
        assert sha(BANK_ROOT / name) == h, name
    assert sha(REFERENCE) == request['reference_sha256']
    return request, sha(ROOT / 'request.json')


def prepare():
    from huggingface_hub import hf_hub_download
    from experiments.train_imagenet100_sit_flow import DEFAULT_OFFICIAL_SIT_REPO, load_official_sit_module

    assert len(GRID) == len(ALPHAS) + 1 and GRID[0] == 0 and GRID[-1] == 1
    assert all(s > t for t, s in zip(GRID[:-1], GRID[1:]))
    assert sum(a > 0 for a in ALPHAS) * 2 * HEUN_STEPS == 32
    old = read(EXPS / 'lifting_wide_scale_20260909/request.json')
    spec = old['models']['sit_small']
    sources = source_paths('sit_small') + [Path(__file__),
        WORK / 'docs/SMALL_SIT_CARRIER_FLOW_PROTOCOL_20260909_ZH.md',
        WORK / 'experiments/compute_adm_fid.py']
    for name in ('train_imagenet100_sit_flow', 'imagenet100_sit_static_pair',
                 'imagenet100_sit_multiscale_guidance', 'sample_imagenet100_sit_fid',
                 'sample_imagenet100_sit_static_pair_fid', 'foresight_fixed_point_flow'):
        sources.append(WORK / 'experiments' / (name + '.py'))
    module, source_metadata = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO, verify_source=True)
    sources.append(Path(inspect.getfile(module)))
    assets = spec['assets'].copy()
    for filename in ('config.json', 'diffusion_pytorch_model.safetensors'):
        p = hf_hub_download('stabilityai/sd-vae-ft-mse', filename, local_files_only=True)
        assets[p] = sha(p)
    for p, h in assets.items():
        assert sha(p) == h, p
    noise = np.load(BANK_ROOT / 'noise.npy', mmap_mode='r')
    labels = np.load(BANK_ROOT / 'labels.npy', mmap_mode='r')
    assert noise.shape == (SAMPLES, 4, 32, 32) and noise.dtype == np.float32
    assert labels.shape == (SAMPLES,) and labels.dtype == np.int64
    assert array_sha(noise) == spec['bank']['noise_sha256']
    assert array_sha(labels) == spec['bank']['label_sha256']
    bank_files = {'noise.npy': spec['bank']['noise_file_sha256'],
                  'labels.npy': spec['bank']['labels_file_sha256']}
    for name, h in bank_files.items():
        assert sha(BANK_ROOT / name) == h, name
    ROOT.mkdir(parents=True, exist_ok=False)
    snapshot = ROOT / 'sources'
    snapshot.mkdir()
    source_hashes = {}
    for i, p in enumerate(sorted(set(sources))):
        source_hashes[str(p)] = sha(p)
        (snapshot / f'{i:02d}_{p.name}').write_bytes(p.read_bytes())
    request = dict(arms=ARMS, grid=GRID, alphas=ALPHAS, heun_steps=HEUN_STEPS,
        main_solver=dict(method='dopri5', rtol=.001, atol=1e-6), samples=SAMPLES,
        batch=BATCH, ranks=RANKS, bank=spec['bank'], bank_root=str(BANK_ROOT),
        bank_files=bank_files, sources=source_hashes, assets=assets,
        source_metadata=source_metadata, reference=str(REFERENCE), reference_sha256=sha(REFERENCE),
        python=sys.executable, versions={name: importlib.metadata.version(name)
            for name in ('torch', 'torchdiffeq', 'diffusers', 'numpy')},
        independent_confirmation=False, research_goal_achieved=False)
    atomic(ROOT / 'request.json', request)
    atomic(ROOT / 'status.json', dict(phase='prepared', research_goal_achieved=False))
    print(json.dumps(dict(prepared=True, root=str(ROOT), request_sha256=sha(ROOT / 'request.json'))), flush=True)


def gap_flow(rt, z, t, amount):
    """Integrate dxi/du = S_t(xi) - W_t(xi), holding original time t fixed."""
    if amount == 0:
        return z
    du = amount / HEUN_STEPS
    for _ in range(HEUN_STEPS):
        strong, weak = rt.pair(z, t)
        k1 = strong - weak
        strong, weak = rt.pair(z + du * k1, t)
        z = z + (.5 * du) * (k1 + strong - weak)
    return z


@torch.inference_mode()
def sample(rt, noise, labels, arm, *, zero=False):
    assert arm in ARMS
    rt.labels = labels
    z = noise.clone()
    grid = torch.tensor(GRID, device=z.device, dtype=z.dtype)
    begin = rt.counts.copy()
    block_nfe, write_rms, state_rms = [], [], []
    rms = lambda x: x.double().flatten(1).square().mean(1).sqrt()
    for k, alpha in enumerate(ALPHAS):
        a = 0. if zero else alpha
        t, s = grid[k], grid[k + 1]
        nfe = rt.counts['full']
        written = torch.zeros(len(z), device=z.device, dtype=torch.float64)
        if arm == 'write_before' and a:
            prior = z
            z = gap_flow(rt, z, t, a * (GRID[k + 1] - GRID[k]))
            written = rms(z - prior)
        function = (lambda u, x: rt.guided(x, u, a)) if arm == 'ig_restarted' else (
            lambda u, x: rt.field(x, u, 'full'))
        z = odeint(function, z, grid[k:k + 2], method='dopri5', rtol=.001, atol=1e-6)[-1]
        if arm == 'write_after' and a:
            prior = z
            z = gap_flow(rt, z, s, a * (GRID[k + 1] - GRID[k]))
            written = rms(z - prior)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'{arm}: nonfinite state at block {k}')
        block_nfe.append(rt.counts['full'] - nfe)
        write_rms.append(written.cpu().numpy())
        state_rms.append(rms(z).cpu().numpy())
    counts = {key: rt.counts[key] - begin[key] for key in begin}
    auxiliary = 0 if zero or arm == 'ig_restarted' else 32
    assert counts['prefix'] == 0 and counts['full'] == sum(block_nfe)
    assert counts['full'] >= auxiliary
    return z, dict(full_calls=counts['full'], prefix_calls=counts['prefix'],
                   auxiliary_full_calls=auxiliary, block_full_calls=np.asarray(block_nfe),
                   block_write_rms=np.asarray(write_rms), block_state_rms=np.asarray(state_rms))


@torch.inference_mode()
def preflight(rt, noise, labels, rank, request):
    rt.labels = labels
    checks = []
    for value in (0., .125, .5, .875):
        t = noise.new_tensor(value)
        full, weak = rt.pair(noise, t)
        torch.testing.assert_close(full, rt.field(noise, t, 'full'), rtol=0, atol=0)
        torch.testing.assert_close(weak, rt.field(noise, t, 'base'), rtol=0, atol=0)
        checks.append(value)
    zero = [sample(rt, noise, labels, arm, zero=True)[0] for arm in ARMS]
    assert all(torch.equal(zero[0], value) for value in zero[1:])
    for p, h in rt.sources.items():
        assert request['sources'].get(p) == h, p
    return dict(passed=True, rank=rank, pair_prefix_exact_at=checks,
        zero_all_three_exact=True, zero_endpoint_sha256=array_sha(zero[0].cpu().numpy()),
        runtime_sources=rt.sources, metadata=rt.metadata, source_semantics=str(rt.semantics),
        cuda_device=torch.cuda.get_device_name(), tf32=torch.backends.cuda.matmul.allow_tf32)


@torch.inference_mode()
def worker(rank):
    request, request_hash = verify_request()
    noise = np.load(BANK_ROOT / 'noise.npy', mmap_mode='r')
    labels = np.load(BANK_ROOT / 'labels.npy', mmap_mode='r')
    rt = Runtime('sit_small')
    first = rank * BATCH
    as_cuda = lambda value: torch.from_numpy(np.array(value)).cuda()
    atomic(ROOT / f'preflight_rank{rank}.json',
        preflight(rt, as_cuda(noise[first:first + BATCH]), as_cuda(labels[first:first + BATCH]), rank, request))
    while not (ROOT / 'preflight_passed.json').exists():
        time.sleep(.25)
    for arm in ARMS:
        out = ROOT / arm / f'rank{rank}'
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for start in range(first, SAMPLES, RANKS * BATCH):
            if (ROOT / arm / 'numerical_failure.json').exists():
                break
            path = out / f'batch{start:04d}.npz'
            assert not path.exists(), 'Do not overwrite or implicitly resume an existing batch'
            n, y = as_cuda(noise[start:start + BATCH]), as_cuda(labels[start:start + BATCH])
            torch.cuda.synchronize()
            begin = time.perf_counter()
            try:
                z, stats = sample(rt, n, y, arm)
            except (FloatingPointError, AssertionError) as error:
                if not isinstance(error, FloatingPointError) and 'underflow in dt' not in str(error):
                    raise
                failure = dict(rank=rank, arm=arm, start=start, error=repr(error), request_sha256=request_hash)
                atomic(out / 'numerical_failure.json', failure)
                try:
                    os.link(out / 'numerical_failure.json', ROOT / arm / 'numerical_failure.json')
                except FileExistsError:
                    pass
                break
            torch.cuda.synchronize()
            trajectory = time.perf_counter() - begin
            begin = time.perf_counter()
            pixels = rt.decode(z)
            torch.cuda.synchronize()
            decode = time.perf_counter() - begin
            assert pixels.shape == (BATCH, 256, 256, 3) and pixels.dtype == np.uint8
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez(stream, arr_0=pixels, latents=z.cpu().numpy(), labels=np.array(labels[start:start + BATCH]),
                    request_sha256=request_hash, noise_sha256=array_sha(noise[start:start + BATCH]),
                    trajectory_seconds=trajectory, decode_seconds=decode, **stats)
            temp.replace(path)
            files.append(dict(start=start, file=path.name, sha256=sha(path)))
            progress = dict(arm=arm, rank=rank, images=len(files) * BATCH)
            atomic(out / 'progress.json', progress)
            if len(files) == 1 or len(files) % 8 == 0:
                print(json.dumps(progress), flush=True)
        failed = (ROOT / arm / 'numerical_failure.json').exists()
        atomic(out / 'summary.json', dict(complete=not failed, rank=rank, files=files,
            request_sha256=request_hash, noise_sha256=request['bank']['noise_sha256'],
            label_sha256=request['bank']['label_sha256']))
        while not (ROOT / arm / 'advance.json').exists():
            time.sleep(.25)


def wait_files(paths, workers):
    while not all(p.exists() for p in paths):
        codes = [p.poll() for p in workers]
        if any(code is not None for code in codes):
            raise RuntimeError(f'Worker exited before required files: {codes}')
        time.sleep(.5)


def evaluate(arm, request, request_hash):
    out = ROOT / arm
    if (out / 'numerical_failure.json').exists():
        result = dict(arm=arm, complete=False, fid=None, status='numerical_failure',
            failure=read(out / 'numerical_failure.json'), request_sha256=request_hash)
        atomic(out / 'result.json', result)
        return result
    labels = np.load(BANK_ROOT / 'labels.npy', mmap_mode='r')
    noise = np.load(BANK_ROOT / 'noise.npy', mmap_mode='r')
    images = np.empty((SAMPLES, 256, 256, 3), np.uint8)
    latents = np.empty_like(noise)
    seen, full, prefix, auxiliary = set(), 0, 0, 0
    trajectory = decode = 0.
    for rank in range(RANKS):
        d = out / f'rank{rank}'
        summary = read(d / 'summary.json')
        assert summary['complete'] and summary['request_sha256'] == request_hash
        assert summary['noise_sha256'] == request['bank']['noise_sha256']
        assert summary['label_sha256'] == request['bank']['label_sha256']
        for rec in summary['files']:
            start, path = rec['start'], d / rec['file']
            assert start not in seen and (start // BATCH) % RANKS == rank
            seen.add(start)
            assert sha(path) == rec['sha256'], path
            with np.load(path) as batch:
                assert str(batch['request_sha256']) == request_hash
                assert str(batch['noise_sha256']) == array_sha(noise[start:start + BATCH])
                np.testing.assert_array_equal(batch['labels'], labels[start:start + BATCH])
                assert batch['latents'].shape == (BATCH, 4, 32, 32) and np.isfinite(batch['latents']).all()
                images[start:start + BATCH] = batch['arr_0']
                latents[start:start + BATCH] = batch['latents']
                trajectory += float(batch['trajectory_seconds'])
                decode += float(batch['decode_seconds'])
                full += int(batch['full_calls'])
                prefix += int(batch['prefix_calls'])
                auxiliary += int(batch['auxiliary_full_calls'])
    assert seen == set(range(0, SAMPLES, BATCH))
    np.savez(out / 'samples.npz', arr_0=images)
    np.save(out / 'latents.npy', latents)
    command = ['/data/shared/envs/adm-fid/bin/python', 'experiments/compute_adm_fid.py',
        '--reference', str(REFERENCE), '--samples', str(out / 'samples.npz'),
        '--output', str(out / 'fid.json'), '--activations-output', str(out / 'activations.npz'),
        '--batch-size', '8', '--gpu-memory-fraction', '.30']
    with (out / 'evaluation.log').open('w') as stream:
        subprocess.run(command, cwd=WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
            TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    metrics = read(out / 'fid.json')
    result = dict(arm=arm, complete=True, samples=SAMPLES, fid=metrics['fid'], metrics=metrics,
        noise_sha256=request['bank']['noise_sha256'], label_sha256=request['bank']['label_sha256'],
        sample_path=str(out / 'samples.npz'), sample_sha256=sha(out / 'samples.npz'),
        latents_sha256=sha(out / 'latents.npy'), activations_sha256=sha(out / 'activations.npz'),
        trajectory_gpu_seconds=trajectory, decode_gpu_seconds=decode,
        sum_batch_gpu_seconds=trajectory + decode, full_calls_per_image=full / (SAMPLES / BATCH),
        prefix_calls_per_image=prefix / (SAMPLES / BATCH), auxiliary_full_calls_per_image=auxiliary / (SAMPLES / BATCH),
        request_sha256=request_hash, coverage_verified=True)
    atomic(out / 'result.json', result)
    return result


def run():
    lock = (ROOT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request, request_hash = verify_request()
    assert read(ROOT / 'status.json')['phase'] == 'prepared'
    workers, streams, results = [], [], []
    begin = time.perf_counter()
    def interrupted(signum, frame):
        raise RuntimeError(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for rank in range(RANKS):
            stream = (ROOT / f'worker{rank}.log').open('w')
            streams.append(stream)
            workers.append(subprocess.Popen([sys.executable, '-u', '-m',
                'experiments.small_sit_carrier_flow_20260909', '--rank', str(rank)], cwd=WORK,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT))
        base_status = dict(controller_pid=os.getpid(), worker_pids=[p.pid for p in workers], research_goal_achieved=False)
        atomic(ROOT / 'status.json', dict(phase='preflight', **base_status))
        wait_files([ROOT / f'preflight_rank{r}.json' for r in range(RANKS)], workers)
        checks = [read(ROOT / f'preflight_rank{r}.json') for r in range(RANKS)]
        assert all(c['passed'] for c in checks)
        assert all(c['runtime_sources'] == checks[0]['runtime_sources'] for c in checks)
        atomic(ROOT / 'preflight_passed.json', dict(passed=True, checks=checks, request_sha256=request_hash))
        for arm in ARMS:
            atomic(ROOT / 'status.json', dict(phase='sampling', arm=arm, **base_status))
            wait_files([ROOT / arm / f'rank{r}/summary.json' for r in range(RANKS)], workers)
            atomic(ROOT / 'status.json', dict(phase='evaluating', arm=arm, **base_status))
            result = evaluate(arm, request, request_hash)
            results.append(result)
            atomic(ROOT / 'results.json', results)
            print(json.dumps(result), flush=True)
            atomic(ROOT / arm / 'advance.json', dict(complete=True, request_sha256=request_hash))
        codes = [p.wait() for p in workers]
        assert codes == [0] * RANKS, codes
        verify_request()
        atomic(ROOT / 'status.json', dict(phase='complete', wall_seconds=time.perf_counter() - begin,
            results=len(results), numerical_failures=sum(not r['complete'] for r in results),
            worker_exit_codes=codes, **base_status))
    except BaseException as error:
        atomic(ROOT / 'status.json', dict(phase='failed', error=repr(error), controller_pid=os.getpid(),
            worker_pids=[p.pid for p in workers], research_goal_achieved=False))
        raise
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            p.wait()
        for stream in streams:
            stream.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--rank', type=int, choices=range(RANKS))
    args = parser.parse_args()
    assert int(args.prepare) + int(args.run_prepared) + int(args.rank is not None) == 1
    if args.prepare:
        prepare()
    elif args.run_prepared:
        run()
    else:
        worker(args.rank)
