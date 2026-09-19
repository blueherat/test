"""Fresh-noise 5K comparison of the fixed residual direction against native IG."""
from __future__ import annotations

import argparse
import fcntl
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

from experiments import small_sit_predictable_gap_20260909 as pilot
from experiments import small_sit_carrier_flow_20260909 as infrastructure
from experiments.lifting_scale_sweep_20260909 import EXPS, WORK, array_sha, atomic, read, sha

ROOT = EXPS / 'small_sit_predictable_gap_confirmation_20260909'
BANK_ROOT = ROOT / 'inputs'
PILOT_BANK_ROOT = infrastructure.BANK_ROOT
PROTOCOL = WORK / 'docs/SMALL_SIT_PREDICTABLE_GAP_CONFIRMATION_PROTOCOL_20260909_ZH.md'
ARMS = ('ig_restarted', 'residual_norm')
GRID, ALPHAS = pilot.GRID, pilot.ALPHAS
SAMPLES, BATCH, RANKS, SEED = 5000, 8, 4, 202609981
COMMON_VERIFY = infrastructure.verify_request


def verify_request():
    request, request_hash = COMMON_VERIFY()
    for p, h in request['preflight_reference_files'].items():
        assert sha(p) == h, p
    return request, request_hash


def prepare():
    pilot.install_infrastructure()
    parent, parent_hash = infrastructure.verify_request()
    assert read(pilot.ROOT / 'status.json')['phase'] == 'complete'
    assert len(read(pilot.ROOT / 'results.json')) == 3
    pilot.verify_fit()
    assert not ROOT.exists()
    ROOT.mkdir(parents=True)
    BANK_ROOT.mkdir()
    noise = np.lib.format.open_memmap(BANK_ROOT / 'noise.npy', mode='w+', dtype=np.float32,
                                    shape=(SAMPLES, 4, 32, 32))
    generator = torch.Generator(device='cuda').manual_seed(SEED)
    for start in range(0, SAMPLES, BATCH):
        noise[start:start + BATCH] = torch.randn((BATCH, 4, 32, 32), generator=generator, device='cuda').cpu().numpy()
    noise.flush()
    labels = np.repeat(np.arange(100, dtype=np.int64), SAMPLES // 100)
    labels = np.random.default_rng(SEED + 1).permutation(labels)
    assert np.all(np.bincount(labels, minlength=100) == 50)
    np.save(BANK_ROOT / 'labels.npy', labels)
    bank_files = {name: sha(BANK_ROOT / name) for name in ('noise.npy', 'labels.npy')}
    sources = dict(parent['sources'])
    for p in (Path(__file__), PROTOCOL):
        sources[str(p)] = sha(p)
    snapshot = ROOT / 'sources'
    snapshot.mkdir()
    for i, p in enumerate(sorted(sources)):
        (snapshot / f'{i:02d}_{Path(p).name}').write_bytes(Path(p).read_bytes())
    references = {}
    for p in (PILOT_BANK_ROOT / 'noise.npy', PILOT_BANK_ROOT / 'labels.npy'):
        references[str(p)] = sha(p)
    for rank in range(RANKS):
        for directory, arm in ((pilot.OLD_ROOT, 'ig_restarted'), (pilot.ROOT, 'residual_norm')):
            p = directory / arm / f'rank{rank}/batch{rank*BATCH:04d}.npz'
            references[str(p)] = sha(p)
    request = dict(parent, arms=ARMS, samples=SAMPLES, batch=BATCH, ranks=RANKS,
        bank_root=str(BANK_ROOT), bank_files=bank_files, sources=sources,
        bank=dict(samples=SAMPLES, batch=BATCH, shape=[SAMPLES, 4, 32, 32], noise_seed=SEED,
            label_seed=SEED+1, noise_sha256=array_sha(noise), label_sha256=array_sha(labels),
            noise_file_sha256=bank_files['noise.npy'], labels_file_sha256=bank_files['labels.npy'],
            noise_generation='one continuous CUDA generator, sequential B8 draws', classes_balanced=True),
        pilot_request_sha256=parent_hash, preflight_reference_files=references,
        new_noise_confirmation=True, independent_training=False, independent_reference=False)
    request.pop('baseline')
    atomic(ROOT / 'request.json', request)
    atomic(ROOT / 'status.json', dict(phase='prepared', research_goal_achieved=False))
    print(json.dumps(dict(prepared=True, request_sha256=sha(ROOT / 'request.json'), bank=request['bank'])), flush=True)


@torch.inference_mode()
def sample(rt, noise, labels, arm):
    if arm == 'residual_norm':
        return pilot.sample(rt, noise, labels, arm)
    assert arm == 'ig_restarted'
    rt.labels = labels
    z = noise.clone()
    grid = z.new_tensor(GRID)
    before = rt.counts.copy()
    calls = []
    for k, alpha in enumerate(ALPHAS):
        nfe = rt.counts['full']
        z = odeint(lambda t, x: rt.guided(x, t, alpha), z, grid[k:k + 2],
                   method='dopri5', rtol=.001, atol=1e-6)[-1]
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'native IG: nonfinite block {k}')
        calls.append(rt.counts['full'] - nfe)
    assert rt.counts['prefix'] == before['prefix']
    return z, dict(full_calls=rt.counts['full']-before['full'], prefix_calls=0, auxiliary_full_calls=0,
        reader_calls=0, block_reader_calls=np.zeros(len(ALPHAS), dtype=np.int64),
        block_full_calls=np.asarray(calls), block_reader_diagnostics=np.zeros((len(ALPHAS),len(z),5)))


@torch.inference_mode()
def preflight(rt, noise, labels, rank, request):
    old_noise = np.load(PILOT_BANK_ROOT / 'noise.npy', mmap_mode='r')
    old_labels = np.load(PILOT_BANK_ROOT / 'labels.npy')
    begin = rank * BATCH
    n = torch.from_numpy(np.array(old_noise[begin:begin+BATCH])).cuda()
    y = torch.from_numpy(np.array(old_labels[begin:begin+BATCH])).cuda()
    parent = pilot.preflight(rt, n, y, rank, request)
    golden = []
    for arm, directory in (('ig_restarted', pilot.OLD_ROOT), ('residual_norm', pilot.ROOT)):
        z, _ = sample(rt, n, y, arm)
        path = directory / arm / f'rank{rank}/batch{begin:04d}.npz'
        with np.load(path) as previous:
            np.testing.assert_array_equal(z.cpu().numpy(), previous['latents'])
        golden.append(dict(arm=arm, path=str(path), sha256=sha(path), exact=True))
    rt.labels = labels
    s, w = rt.pair(noise, noise.new_tensor(.25))
    c = pilot.cache_source.unpatchify(pilot.project(rt.capture_weak.value, rt.projection))
    d = s - w
    _, normalized, _, _ = pilot.directions(d, c)
    torch.testing.assert_close(normalized.flatten(1).norm(dim=1), d.flatten(1).norm(dim=1), rtol=2e-6, atol=2e-6)
    return dict(parent, new_noise_norm_identity=True, both_old_endpoints_exact=golden,
        fresh_noise_sha256=array_sha(noise.cpu().numpy()), fresh_labels_sha256=array_sha(labels.cpu().numpy()))


def install_infrastructure():
    infrastructure.ROOT, infrastructure.BANK_ROOT = ROOT, BANK_ROOT
    infrastructure.ARMS, infrastructure.SAMPLES = ARMS, SAMPLES
    infrastructure.Runtime, infrastructure.sample, infrastructure.preflight = pilot.make_runtime, sample, preflight
    infrastructure.verify_request = verify_request


def run():
    lock = (ROOT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request, request_hash = verify_request()
    assert read(ROOT / 'status.json')['phase'] == 'prepared'
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb') == request['inception_graph_sha256']
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
                'experiments.small_sit_predictable_gap_confirmation_20260909', '--rank', str(rank)], cwd=WORK,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT))
        status = dict(controller_pid=os.getpid(), worker_pids=[p.pid for p in workers], research_goal_achieved=False)
        atomic(ROOT / 'status.json', dict(phase='preflight', **status))
        infrastructure.wait_files([ROOT / f'preflight_rank{r}.json' for r in range(RANKS)], workers)
        checks = [read(ROOT / f'preflight_rank{r}.json') for r in range(RANKS)]
        assert all(c['passed'] for c in checks)
        assert all(c['runtime_sources'] == checks[0]['runtime_sources'] for c in checks)
        atomic(ROOT / 'preflight_passed.json', dict(passed=True, checks=checks, request_sha256=request_hash))
        for arm in ARMS:
            atomic(ROOT / 'status.json', dict(phase='sampling', arm=arm, **status))
            infrastructure.wait_files([ROOT / arm / f'rank{r}/summary.json' for r in range(RANKS)], workers)
            atomic(ROOT / 'status.json', dict(phase='evaluating', arm=arm, **status))
            result = infrastructure.evaluate(arm, request, request_hash)
            results.append(result)
            atomic(ROOT / 'results.json', results)
            print(json.dumps(result), flush=True)
            atomic(ROOT / arm / 'advance.json', dict(complete=True, request_sha256=request_hash))
        codes = [p.wait() for p in workers]
        assert codes == [0] * RANKS, codes
        verify_request()
        atomic(ROOT / 'status.json', dict(phase='complete', results=len(results),
            wall_seconds=time.perf_counter()-begin, numerical_failures=sum(not r['complete'] for r in results),
            worker_exit_codes=codes, **status))
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
    assert sum((args.prepare, args.run_prepared, args.rank is not None)) == 1
    if args.prepare:
        prepare()
    else:
        install_infrastructure()
        run() if args.run_prepared else infrastructure.worker(args.rank)
