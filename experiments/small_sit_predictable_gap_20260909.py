"""Residualize the internal gap against a cheap shallow-feature readout."""
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

from experiments import small_sit_carrier_flow_20260909 as infrastructure
from experiments import small_sit_feedback_reader_20260909 as cache_source
from experiments.lifting_scale_sweep_20260909 import EXPS, WORK, Runtime, array_sha, atomic, read, sha

ROOT = EXPS / 'small_sit_predictable_gap_20260909'
OLD_ROOT = infrastructure.ROOT
CACHE_ROOT = cache_source.ROOT
PROTOCOL = WORK / 'docs/SMALL_SIT_PREDICTABLE_GAP_PROTOCOL_20260909_ZH.md'
ARMS = ('residual_raw', 'residual_norm', 'parallel_norm')
GRID, ALPHAS = infrastructure.GRID, infrastructure.ALPHAS
BATCH, SAMPLES, RANKS = infrastructure.BATCH, infrastructure.SAMPLES, infrastructure.RANKS
RIDGE, STD_FLOOR, NORM_FLOOR = .001, 1e-6, 1e-12


def verify_fit():
    request = read(ROOT / 'fit_request.json')
    amendment_path = ROOT / 'amendments/01_projection_precision/amendment.json'
    amendment = read(amendment_path) if amendment_path.exists() else None
    for category in ('sources', 'assets', 'cache_files'):
        for p, h in request[category].items():
            if amendment and category == 'sources' and p == amendment['source_path']:
                assert h == amendment['source_before_sha256']
                assert sha(amendment_path.parent / 'source_before_fix.py') == h
                assert sha(p) == amendment['source_after_sha256']
                assert sha(ROOT / 'projection.pt') == amendment['unchanged_projection_sha256']
            else:
                assert sha(p) == h, (category, p)
    return request


def prepare_fit():
    parent = read(OLD_ROOT / 'request.json')
    training = cache_source.verify_training()
    collection = read(CACHE_ROOT / 'collection.json')
    assert collection['training_request_sha256'] == sha(CACHE_ROOT / 'training_request.json')
    for category in ('sources', 'assets'):
        for p, h in parent[category].items():
            assert sha(p) == h, p
    cache_files = {r['path']: r['sha256'] for r in collection['records']}
    for name in ('collection.json', 'training_request.json', 'train_indices.npy', 'validation_indices.npy'):
        p = CACHE_ROOT / name
        cache_files[str(p)] = sha(p)
    for p, h in cache_files.items():
        assert sha(p) == h, p
    a, b = (np.load(CACHE_ROOT / (n + '_indices.npy')) for n in ('train', 'validation'))
    assert not set(a.tolist()).intersection(b.tolist())
    sources = dict(parent['sources'], **training['sources'])
    for p in (Path(__file__), PROTOCOL):
        sources[str(p)] = sha(p)
    ROOT.mkdir(parents=True, exist_ok=False)
    snapshot = ROOT / 'sources'
    snapshot.mkdir()
    for i, p in enumerate(sorted(sources)):
        (snapshot / f'{i:02d}_{Path(p).name}').write_bytes(Path(p).read_bytes())
    request = dict(sources=sources, assets=parent['assets'], cache_files=cache_files,
        parent_request_sha256=sha(OLD_ROOT / 'request.json'), collection=collection,
        ridge=RIDGE, std_floor=STD_FLOOR, norm_floor=NORM_FLOOR,
        feature='original conditioned shallow pre-linear feature only', target='native strong minus weak velocity',
        no_deep_features_as_input=True, no_fid_used_for_fit=True,
        inception_graph_sha256=sha('/data/shared/adm_refs/classify_image_graph_def.pb'))
    atomic(ROOT / 'fit_request.json', request)
    atomic(ROOT / 'status.json', dict(phase='fit_prepared', research_goal_achieved=False))
    print(json.dumps(dict(fit_prepared=True, request_sha256=sha(ROOT / 'fit_request.json'))), flush=True)


@torch.no_grad()
def fit():
    verify_fit()
    assert read(ROOT / 'status.json')['phase'] == 'fit_prepared'
    atomic(ROOT / 'status.json', dict(phase='fitting', controller_pid=os.getpid()))
    data = torch.load(CACHE_ROOT / 'train_cache.pt', map_location='cpu', weights_only=True)
    h = data['hw'].cuda().double()
    d = (data['s'] - data['w']).cuda().double()
    del data
    torch.cuda.synchronize()
    begin = time.perf_counter()
    mean, std = h.mean(0), h.std(0, correction=0).clamp_min(STD_FLOOR)
    x = torch.cat(((h - mean) / std, torch.ones((len(h), 1), dtype=h.dtype, device=h.device)), -1)
    del h
    gram, target = x.T @ x / len(x), x.T @ d / len(x)
    penalty = torch.eye(x.shape[1], dtype=x.dtype, device=x.device) * RIDGE
    penalty[-1, -1] = 0
    regularized = gram + penalty
    coefficient = torch.linalg.solve(regularized, target)
    equation_error = torch.linalg.vector_norm(regularized @ coefficient - target) / torch.linalg.vector_norm(target)
    assert equation_error < 1e-10 and torch.isfinite(coefficient).all()
    state = dict(mean=mean.float().cpu(), std=std.float().cpu(), coefficient=coefficient.float().cpu(),
        fit_request_sha256=sha(ROOT / 'fit_request.json'))
    torch.save(state, ROOT / 'projection.pt')
    np.savez(ROOT / 'fit_equations.npz', gram=gram.cpu().numpy(), penalty=penalty.cpu().numpy(),
        target=target.cpu().numpy(), coefficient=coefficient.cpu().numpy(),
        mean=mean.cpu().numpy(), std=std.cpu().numpy())
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    metrics = []
    for split in ('train', 'validation'):
        data = torch.load(CACHE_ROOT / f'{split}_cache.pt', map_location='cpu', weights_only=True)
        hs = data['hw'].cuda()
        delta = (data['s'] - data['w']).cuda()
        s = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in state.items()}
        pred = project(hs, s)
        residual = delta - pred
        energy = delta.double().square().sum()
        metrics.append(dict(split=split, tokens=len(hs), native_gap_mean_square=float(energy / delta.numel()),
            residual_mean_square=float(residual.double().square().mean()),
            residual_energy_ratio=float(residual.double().square().sum() / energy),
            prediction_energy_ratio=float(pred.double().square().sum() / energy),
            residual_prediction_inner_product=float((residual.double() * pred.double()).mean())))
        del data, hs, delta, pred, residual
    result = dict(complete=True, seconds=seconds, parameters=int(coefficient.numel()),
        fitted_tokens=len(x), relative_normal_equation_error=float(equation_error),
        checkpoint=str(ROOT / 'projection.pt'), sha256=sha(ROOT / 'projection.pt'),
        equation_sha256=sha(ROOT / 'fit_equations.npz'), metrics=metrics,
        fit_request_sha256=sha(ROOT / 'fit_request.json'))
    atomic(ROOT / 'fit.json', result)
    atomic(ROOT / 'status.json', dict(phase='fitted', research_goal_achieved=False))
    print(json.dumps(result), flush=True)


def project(h, state):
    normalized = (h - state['mean']) / state['std']
    # Preserve the backbone's TF32 choice; the tiny new readout uses full FP32.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return normalized @ state['coefficient'][:-1] + state['coefficient'][-1]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


class CaptureWeak:
    def __init__(self, rt):
        self.value = None
        def capture(module, inputs):
            self.value = inputs[0].detach()
        self.handle = rt.head.module.linear.register_forward_pre_hook(capture)


def make_runtime(name):
    rt = Runtime(name)
    rt.capture_weak = CaptureWeak(rt)
    rt.projection = torch.load(ROOT / 'projection.pt', map_location='cuda', weights_only=True)
    return rt


def directions(d, c):
    r = d - c
    dims = tuple(range(1, d.ndim))
    dn = d.square().sum(dims, keepdim=True).sqrt()
    rn = r.square().sum(dims, keepdim=True).sqrt()
    fallback = (dn < NORM_FLOOR) | (rn < NORM_FLOOR)
    normalized = torch.where(fallback, d, r * (dn / rn.clamp_min(NORM_FLOOR)))
    coefficient = (normalized * d).sum(dims, keepdim=True) / dn.square().clamp_min(NORM_FLOOR ** 2)
    parallel = torch.where(fallback, d, coefficient * d)
    return r, normalized, parallel, fallback


@torch.inference_mode()
def field(rt, z, t, alpha, arm, *, disable=False):
    strong, weak = rt.pair(z, t)
    if disable:
        return strong + alpha * (strong - weak), None
    d = strong - weak
    c = cache_source.unpatchify(project(rt.capture_weak.value, rt.projection))
    raw, normalized, parallel, fallback = directions(d, c)
    selected = dict(residual_raw=raw, residual_norm=normalized, parallel_norm=parallel)[arm]
    dims = (1, 2, 3)
    energy = d.square().sum(dims).clamp_min(NORM_FLOOR ** 2)
    rn = raw.square().sum(dims).sqrt()
    cn = c.square().sum(dims).sqrt()
    dn = energy.sqrt()
    cosine = (raw * d).sum(dims) / (rn * dn).clamp_min(NORM_FLOOR ** 2)
    orthogonal = (normalized - parallel).square().sum(dims) / energy
    diag = torch.stack((rn / dn, cn / dn, cosine, orthogonal,
                        fallback.flatten(1).any(1).float()), -1)
    return strong + alpha * selected, diag


@torch.inference_mode()
def sample(rt, noise, labels, arm, *, zero=False, disable=False):
    assert arm in ARMS
    rt.labels = labels
    z = noise.clone()
    grid = z.new_tensor(GRID)
    before = rt.counts.copy()
    block_calls, block_reader, diagnostics = [], [], []
    for k, amount in enumerate(ALPHAS):
        alpha = 0. if zero else amount
        nfe = rt.counts['full']
        local = []
        def rhs(t, x):
            if alpha == 0:
                return rt.field(x, t, 'full')
            value, diag = field(rt, x, t, alpha, arm, disable=disable)
            if diag is not None:
                local.append(diag)
            return value
        z = odeint(rhs, z, grid[k:k + 2], method='dopri5', rtol=.001, atol=1e-6)[-1]
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'{arm}: nonfinite state at block {k}')
        block_calls.append(rt.counts['full'] - nfe)
        block_reader.append(len(local))
        diagnostics.append(torch.stack(local).double().mean(0).cpu().numpy() if local else np.zeros((len(z), 5)))
    full = rt.counts['full'] - before['full']
    assert rt.counts['prefix'] == before['prefix'] and full == sum(block_calls)
    return z, dict(full_calls=full, prefix_calls=0, auxiliary_full_calls=0,
        reader_calls=sum(block_reader), block_reader_calls=np.asarray(block_reader),
        block_full_calls=np.asarray(block_calls), block_reader_diagnostics=np.asarray(diagnostics))


@torch.inference_mode()
def preflight(rt, noise, labels, rank, request):
    rt.labels = labels
    checks = []
    for tvalue in (0., .125, .375, .875):
        t = noise.new_tensor(tvalue)
        strong, weak = rt.pair(noise, t)
        h = rt.capture_weak.value.clone()
        assert torch.equal(cache_source.unpatchify(rt.head.module.linear(h)), weak)
        assert torch.equal(strong, rt.field(noise, t, 'full'))
        assert torch.equal(weak, rt.field(noise, t, 'base'))
        predicted = project(h, rt.projection)
        cpu = {k: v.cpu().numpy() for k, v in rt.projection.items() if isinstance(v, torch.Tensor)}
        independent = ((h.cpu().numpy().astype(float) - cpu['mean']) / cpu['std']) @ cpu['coefficient'][:-1].astype(float) + cpu['coefficient'][-1]
        error = float(np.max(np.abs(predicted.cpu().numpy() - independent)))
        assert error < 3e-4, error
        d, c = strong - weak, cache_source.unpatchify(predicted)
        raw, normalized, parallel, fallback = directions(d, c)
        for mode in directions(d, torch.zeros_like(d))[:3]:
            torch.testing.assert_close(mode, d, rtol=2e-6, atol=2e-6)
        dn, nn = d.flatten(1).norm(dim=1), normalized.flatten(1).norm(dim=1)
        torch.testing.assert_close(nn, dn, rtol=2e-6, atol=2e-6)
        inner = ((normalized - parallel) * d).flatten(1).sum(1)
        assert (inner.abs() <= 2e-6 * dn.square().clamp_min(1e-12)).all()
        checks.append(dict(t=tvalue, projection_numpy_max_error=error, fallback_count=int(fallback.sum())))
    z, _ = sample(rt, noise, labels, ARMS[0], disable=True)
    golden = OLD_ROOT / 'ig_restarted' / f'rank{rank}' / f'batch{rank * BATCH:04d}.npz'
    with np.load(golden) as data:
        np.testing.assert_array_equal(z.cpu().numpy(), data['latents'])
    zeros = [sample(rt, noise, labels, arm, zero=True)[0] for arm in ARMS]
    assert all(torch.equal(zeros[0], v) for v in zeros[1:])
    timings = {}
    t = noise.new_tensor(.25)
    for mode in ('native', *ARMS):
        for _ in range(5):
            rt.guided(noise, t, .6) if mode == 'native' else field(rt, noise, t, .6, mode)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(30):
            rt.guided(noise, t, .6) if mode == 'native' else field(rt, noise, t, .6, mode)
        torch.cuda.synchronize()
        timings[mode] = (time.perf_counter() - begin) / 30
    for p, h in rt.sources.items():
        assert request['sources'].get(p) == h, p
    return dict(passed=True, rank=rank, checks=checks, ordinary_ig_latents_exact=True,
        zero_alpha_exact=True, norm_and_projection_identities=True, original_weak_readout_exact=True,
        golden_sha256=sha(golden), per_query_seconds=timings, runtime_sources=rt.sources)


def install_infrastructure():
    infrastructure.ROOT, infrastructure.ARMS = ROOT, ARMS
    infrastructure.Runtime, infrastructure.sample, infrastructure.preflight = make_runtime, sample, preflight


def prepare():
    fit_request = verify_fit()
    assert read(ROOT / 'status.json')['phase'] == 'fitted'
    fit_record = read(ROOT / 'fit.json')
    assert fit_record['complete'] and sha(fit_record['checkpoint']) == fit_record['sha256']
    parent = read(OLD_ROOT / 'request.json')
    baseline = read(OLD_ROOT / 'ig_restarted/result.json')
    assert baseline['complete'] and sha(baseline['sample_path']) == baseline['sample_sha256']
    assets = dict(parent['assets'])
    assets[fit_record['checkpoint']] = fit_record['sha256']
    sources = dict(fit_request['sources'])
    amendment_path = ROOT / 'amendments/01_projection_precision/amendment.json'
    if amendment_path.exists():
        amendment = read(amendment_path)
        sources[amendment['source_path']] = amendment['source_after_sha256']
        sources[str(amendment_path)] = sha(amendment_path)
        sources[str(amendment_path.parent / 'AMENDMENT.md')] = sha(amendment_path.parent / 'AMENDMENT.md')
    request = dict(parent, arms=ARMS, sources=sources, assets=assets,
        baseline=baseline, parent_request_sha256=sha(OLD_ROOT / 'request.json'),
        fit_request_sha256=sha(ROOT / 'fit_request.json'), fit=fit_record,
        backbone_evaluations_per_rhs=1, inception_graph_sha256=fit_request['inception_graph_sha256'])
    request.pop('heun_steps')
    atomic(ROOT / 'request.json', request)
    atomic(ROOT / 'status.json', dict(phase='prepared', research_goal_achieved=False))
    print(json.dumps(dict(prepared=True, request_sha256=sha(ROOT / 'request.json'))), flush=True)


def run():
    lock = (ROOT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request, request_hash = infrastructure.verify_request()
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
                'experiments.small_sit_predictable_gap_20260909', '--rank', str(rank)], cwd=WORK,
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
        infrastructure.verify_request()
        atomic(ROOT / 'status.json', dict(phase='complete', results=len(results),
            wall_seconds=time.perf_counter() - begin, numerical_failures=sum(not r['complete'] for r in results),
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
    parser.add_argument('--prepare-fit', action='store_true')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--rank', type=int, choices=range(RANKS))
    args = parser.parse_args()
    assert sum((args.prepare_fit, args.fit, args.prepare, args.run_prepared, args.rank is not None)) == 1
    if args.prepare_fit:
        prepare_fit()
    elif args.fit:
        fit()
    elif args.prepare:
        prepare()
    else:
        install_infrastructure()
        run() if args.run_prepared else infrastructure.worker(args.rank)
