"""Low-dimensional posterior-precision balance: two paired real-SiT controls."""
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
from experiments.lifting_scale_sweep_20260909 import EXPS, WORK, Runtime, array_sha, atomic, read, sha

OLD_ROOT = infrastructure.ROOT
ROOT = EXPS / 'small_sit_precision_balance_20260909'
ARMS = ('precision_rank1', 'precision_rank2')
GRID, ALPHAS = infrastructure.GRID, infrastructure.ALPHAS
BATCH, SAMPLES, RANKS = infrastructure.BATCH, infrastructure.SAMPLES, infrastructure.RANKS
FD_STEP = .01
EIG_FLOOR, COND_MAX, DIRECTION_FLOOR = 1e-6, 1000., 1e-6


def dot(x, y):
    return (x.double() * y.double()).flatten(1).mean(1)


def rms(x):
    return dot(x, x).sqrt()


def expand(x):
    return x.to(torch.float32).reshape(-1, 1, 1, 1)


def positive(matrix):
    finite = torch.isfinite(matrix).all(dim=(-1, -2))
    safe = torch.nan_to_num(matrix, nan=0., posinf=0., neginf=0.)
    eig = torch.linalg.eigvalsh(safe)
    valid = finite & (eig[:, 0] > EIG_FLOOR) & (eig[:, -1] < COND_MAX * eig[:, 0])
    return valid, safe


@torch.inference_mode()
def precision_field(rt, z, t, alpha, rank, *, fd_step=FD_STEP):
    full, weak = rt.pair(z, t)
    gap = full - weak
    ordinary = full + alpha * gap
    # Diagnostics: selected rank, effective alpha, orthogonal/ordinary correction,
    #             strong/weak projected antisymmetry, active guidance query.
    diag = torch.zeros(len(z), 6, device=z.device, dtype=torch.float64)
    diag[:, 1] = alpha
    diag[:, 5] = 1.
    if float(t) == 0.:
        return ordinary, diag
    sigma = 1. - t
    clean_gap = sigma * gap
    norm1 = rms(clean_gap)
    u1 = clean_gap / expand(norm1.clamp_min(1e-12))
    step = fd_step * rms(z).clamp_min(1e-6)

    def response(u):
        plus, minus = z + expand(step) * u, z - expand(step) * u
        ps, pw = rt.pair(plus, t)
        ms, mw = rt.pair(minus, t)
        return ((plus + sigma * ps) - (minus + sigma * ms)) / expand(2 * step), (
            (plus + sigma * pw) - (minus + sigma * mw)) / expand(2 * step)

    s1, w1 = response(u1)
    cs, cw = dot(u1, s1), dot(u1, w1)
    denominator = (1. + alpha) * cw - alpha * cs
    valid1 = (norm1 > DIRECTION_FLOOR) & (cs > EIG_FLOOR) & (cw > EIG_FLOOR) & (denominator > EIG_FLOOR)
    valid1 &= torch.isfinite(cs) & torch.isfinite(cw) & torch.isfinite(denominator)
    # Scalar matrices have condition number one. The absolute floor remains relevant.
    gain = alpha * cs / torch.where(valid1, denominator, torch.ones_like(denominator))
    gain = torch.where(valid1, gain, torch.full_like(gain, alpha))
    result = full + expand(gain) * gap
    diag[:, 0] = valid1.double()
    if rank == 2:
        remainder = s1 - w1
        remainder = remainder - expand(dot(u1, remainder)) * u1
        norm2 = rms(remainder)
        u2 = remainder / expand(norm2.clamp_min(1e-12))
        s2, w2 = response(u2)
        raw_s = torch.stack((dot(u1, s1), dot(u1, s2), dot(u2, s1), dot(u2, s2)), -1).reshape(-1, 2, 2)
        raw_w = torch.stack((dot(u1, w1), dot(u1, w2), dot(u2, w1), dot(u2, w2)), -1).reshape(-1, 2, 2)
        symmetric_s = (raw_s + raw_s.transpose(-1, -2)) / 2
        symmetric_w = (raw_w + raw_w.transpose(-1, -2)) / 2
        vs, ks = positive(symmetric_s)
        vw, kw = positive(symmetric_w)
        vc, kc = positive((1. + alpha) * kw - alpha * ks)
        valid2 = (norm1 > DIRECTION_FLOOR) & (norm2 > DIRECTION_FLOOR) & vs & vw & vc
        identity = torch.eye(2, device=z.device, dtype=torch.float64).expand(len(z), -1, -1)
        solve_matrix = torch.where(valid2[:, None, None], kc, identity)
        rhs = torch.stack((norm1, torch.zeros_like(norm1)), -1)
        coeff = alpha * (ks @ torch.linalg.solve(solve_matrix, rhs).unsqueeze(-1)).squeeze(-1)
        delta = expand(coeff[:, 0]) * u1 + expand(coeff[:, 1]) * u2
        result = torch.where(valid2[:, None, None, None], full + delta / sigma, result)
        diag[:, 0] = torch.where(valid2, torch.full_like(gain, 2.), diag[:, 0])
        fro = lambda x: x.square().sum(dim=(-1, -2)).sqrt()
        diag[:, 3] = fro(raw_s - raw_s.transpose(-1, -2)) / fro(symmetric_s).clamp_min(1e-12)
        diag[:, 4] = fro(raw_w - raw_w.transpose(-1, -2)) / fro(symmetric_w).clamp_min(1e-12)
    correction = result - full
    coefficient = dot(correction, gap) / dot(gap, gap).clamp_min(1e-24)
    diag[:, 1] = coefficient
    diag[:, 2] = rms(correction - expand(coefficient) * gap) / (alpha * rms(gap)).clamp_min(1e-12)
    return result, diag


@torch.inference_mode()
def sample(rt, noise, labels, arm, *, zero=False):
    assert arm in ARMS
    rank = int(arm[-1])
    rt.labels = labels
    z = noise.clone()
    grid = z.new_tensor(GRID)
    before = rt.counts.copy()
    all_diag, block_calls, state_rms, mode_counts = [], [], [], []
    rhs_total = 0
    for k, amount in enumerate(ALPHAS):
        alpha = 0. if zero else amount
        local_diag = []
        calls = rt.counts['full']
        def field(t, x):
            nonlocal rhs_total
            rhs_total += 1
            if alpha == 0:
                return rt.field(x, t, 'full')
            value, diag = precision_field(rt, x, t, alpha, rank)
            local_diag.append(diag)
            return value
        z = odeint(field, z, grid[k:k + 2], method='dopri5', rtol=.001, atol=1e-6)[-1]
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'{arm}: nonfinite state at block {k}')
        block_calls.append(rt.counts['full'] - calls)
        state_rms.append(rms(z).cpu().numpy())
        if local_diag:
            values = torch.stack(local_diag)
            all_diag.append(values.mean(0).cpu().numpy())
            mode_counts.append(torch.stack([(values[:, :, 0] == mode).sum(0) for mode in (0, 1, 2)], -1).cpu().numpy())
        else:
            all_diag.append(np.zeros((len(z), 6)))
            mode_counts.append(np.zeros((len(z), 3), dtype=np.int64))
    full = rt.counts['full'] - before['full']
    prefix = rt.counts['prefix'] - before['prefix']
    assert prefix == 0 and full == sum(block_calls)
    return z, dict(full_calls=full, prefix_calls=prefix, auxiliary_full_calls=full - rhs_total,
        rhs_calls=rhs_total, block_full_calls=np.asarray(block_calls),
        block_state_rms=np.asarray(state_rms), block_response_means=np.asarray(all_diag),
        block_mode_counts=np.asarray(mode_counts))


@torch.inference_mode()
def preflight(rt, noise, labels, rank, request):
    rt.labels = labels
    t = noise.new_tensor(.375)
    strong, weak = rt.pair(noise, t)
    torch.testing.assert_close(strong, rt.field(noise, t, 'full'), rtol=0, atol=0)
    torch.testing.assert_close(weak, rt.field(noise, t, 'base'), rtol=0, atol=0)
    z = noise.clone()
    grid = z.new_tensor(GRID)
    finite_difference_checks = []
    for k, alpha in enumerate(ALPHAS):
        z = odeint(lambda u, x: rt.guided(x, u, alpha), z, grid[k:k + 2],
                   method='dopri5', rtol=.001, atol=1e-6)[-1]
        if k in (0, 2):
            v1, d1 = precision_field(rt, z, grid[k + 1], alpha, 2, fd_step=FD_STEP)
            v2, d2 = precision_field(rt, z, grid[k + 1], alpha, 2, fd_step=FD_STEP / 2)
            finite_difference_checks.append(dict(t=float(grid[k + 1]),
                relative_output_difference=(rms(v1 - v2) / rms(v1).clamp_min(1e-12)).cpu().tolist(),
                selected_rank_at_fd1=d1[:, 0].cpu().tolist(), selected_rank_at_fd_half=d2[:, 0].cpu().tolist()))
    golden = OLD_ROOT / 'ig_restarted' / f'rank{rank}' / f'batch{rank * BATCH:04d}.npz'
    with np.load(golden) as old:
        np.testing.assert_array_equal(z.cpu().numpy(), old['latents'])
    zeros = [sample(rt, noise, labels, arm, zero=True)[0] for arm in ARMS]
    assert torch.equal(zeros[0], zeros[1])
    for p, h in rt.sources.items():
        assert request['sources'].get(p) == h, p
    return dict(passed=True, rank=rank, pair_prefix_exact=True, zero_exact=True,
        ordinary_ig_latents_exact=True, golden=str(golden), golden_sha256=sha(golden),
        finite_difference_checks=finite_difference_checks, runtime_sources=rt.sources, metadata=rt.metadata)


def install_infrastructure():
    # Reuse the audited batch/evaluation lifecycle with explicit process-local routing.
    # This does not modify any previous experiment's files or frozen source.
    infrastructure.ROOT = ROOT
    infrastructure.ARMS = ARMS
    infrastructure.sample = sample
    infrastructure.preflight = preflight


def prepare():
    parent = read(OLD_ROOT / 'request.json')
    for category in ('sources', 'assets'):
        for p, h in parent[category].items():
            assert sha(p) == h, p
    baseline = read(OLD_ROOT / 'ig_restarted/result.json')
    assert baseline['complete'] and baseline['coverage_verified']
    assert sha(baseline['sample_path']) == baseline['sample_sha256']
    ROOT.mkdir(parents=True, exist_ok=False)
    snapshot = ROOT / 'sources'
    snapshot.mkdir()
    sources = parent['sources'].copy()
    for p in (Path(__file__), WORK / 'docs/SMALL_SIT_PRECISION_BALANCE_PROTOCOL_20260909_ZH.md',
              WORK / 'train_gen/evaluator.py'):
        sources[str(p)] = sha(p)
    for i, p in enumerate(sorted(sources)):
        (snapshot / f'{i:02d}_{Path(p).name}').write_bytes(Path(p).read_bytes())
    request = dict(parent, arms=ARMS, sources=sources, parent_request_sha256=sha(OLD_ROOT / 'request.json'),
        baseline=baseline, finite_difference_step=FD_STEP, eigenvalue_floor=EIG_FLOOR,
        condition_number_max=COND_MAX, direction_rms_floor=DIRECTION_FLOOR,
        inception_graph_sha256=sha('/data/shared/adm_refs/classify_image_graph_def.pb'))
    request.pop('heun_steps')
    atomic(ROOT / 'request.json', request)
    atomic(ROOT / 'status.json', dict(phase='prepared', research_goal_achieved=False))
    print(json.dumps(dict(prepared=True, request_sha256=sha(ROOT / 'request.json'))), flush=True)


def run():
    lock = (ROOT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request, request_hash = infrastructure.verify_request()
    assert read(ROOT / 'status.json')['phase'] == 'prepared'
    assert sha(request['baseline']['sample_path']) == request['baseline']['sample_sha256']
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
                'experiments.small_sit_precision_balance_20260909', '--rank', str(rank)], cwd=WORK,
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
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--rank', type=int, choices=range(RANKS))
    args = parser.parse_args()
    assert int(args.prepare) + int(args.run_prepared) + int(args.rank is not None) == 1
    install_infrastructure()
    if args.prepare:
        prepare()
    elif args.run_prepared:
        run()
    else:
        infrastructure.worker(args.rank)
