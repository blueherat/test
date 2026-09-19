"""Real-network reverse-mode derivative check, separate from frozen quality trials."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torchdiffeq import odeint

from experiments import small_sit_precision_balance_20260909 as study
from experiments.lifting_scale_sweep_20260909 import Runtime, atomic, read, sha


def summaries(a):
    a = np.asarray(a)
    return dict(median=float(np.median(a)), mean=float(np.mean(a)), maximum=float(np.max(a)))


def main():
    study.install_infrastructure()
    request, request_hash = study.infrastructure.verify_request()
    status = read(study.ROOT / 'status.json')
    assert status['phase'] == 'complete' or (
        status['phase'] == 'paused_after_numerical_review' and status['owned_processes_confirmed_gone']
    ), 'Do not compete with quality sampling for GPU time'
    rt = Runtime('sit_small')
    rt.model.requires_grad_(False)
    rt.head.module.requires_grad_(False)
    noise = np.load(study.infrastructure.BANK_ROOT / 'noise.npy', mmap_mode='r')
    labels = np.load(study.infrastructure.BANK_ROOT / 'labels.npy')
    rows = []
    begin = time.perf_counter()
    for start in range(0, 32, study.BATCH):
        rt.labels = torch.from_numpy(labels[start:start + study.BATCH].copy()).cuda()
        z = torch.from_numpy(np.array(noise[start:start + study.BATCH])).cuda()
        grid = z.new_tensor(study.GRID)
        for k, alpha in enumerate(study.ALPHAS[:3]):
            with torch.no_grad():
                z = odeint(lambda t, x: rt.guided(x, t, alpha), z, grid[k:k + 2],
                           method='dopri5', rtol=.001, atol=1e-6)[-1]
            if k not in (0, 2):
                continue
            t, sigma = grid[k + 1], 1. - grid[k + 1]
            with torch.no_grad():
                s, w = rt.pair(z, t)
                gap = sigma * (s - w)
                u1 = gap / study.expand(study.rms(gap).clamp_min(1e-12))
                def response(u, step_factor):
                    h = step_factor * study.rms(z).clamp_min(1e-6)
                    zp, zm = z + study.expand(h) * u, z - study.expand(h) * u
                    sp, wp = rt.pair(zp, t)
                    sm, wm = rt.pair(zm, t)
                    return ((zp + sigma * sp) - (zm + sigma * sm)) / study.expand(2 * h), (
                        (zp + sigma * wp) - (zm + sigma * wm)) / study.expand(2 * h)
                s1, w1 = response(u1, study.FD_STEP)
                rem = s1 - w1
                rem -= study.expand(study.dot(u1, rem)) * u1
                u2 = rem / study.expand(study.rms(rem).clamp_min(1e-12))
                basis = (u1, u2)
                fd_matrices = []
                for radius in (study.FD_STEP, study.FD_STEP / 2):
                    responses = [response(u, radius) for u in basis]
                    pair = []
                    for head in (0, 1):
                        pair.append(torch.stack([study.dot(u, responses[j][head])
                            for u in basis for j in (0, 1)], -1).reshape(-1, 2, 2))
                    fd_matrices.append(pair)
            state = z.detach().clone().requires_grad_(True)
            vs, vw = rt.pair(state, t)
            means = (state + sigma * vs, state + sigma * vw)
            ad_matrices = []
            for head, mean in enumerate(means):
                entries = []
                for i, u in enumerate(basis):
                    grad = torch.autograd.grad((mean * u).sum(), state,
                        retain_graph=not (head == 1 and i == 1), create_graph=False)[0]
                    entries.extend(study.dot(grad, v) for v in basis)
                ad_matrices.append(torch.stack(entries, -1).reshape(-1, 2, 2).detach())
            with torch.no_grad():
                fro = lambda x: x.square().sum(dim=(-1, -2)).sqrt()
                for local in range(len(z)):
                    item = dict(input_index=start + local, t=float(t), alpha=alpha, matrices={})
                    for head, name in enumerate(('strong', 'weak')):
                        ad = ad_matrices[head][local]
                        fd, half = fd_matrices[0][head][local], fd_matrices[1][head][local]
                        item['matrices'][name] = dict(autodiff=ad.cpu().tolist(), fd1=fd.cpu().tolist(),
                            fd_half=half.cpu().tolist(),
                            fd1_relative_error=float(fro(fd - ad) / fro(ad).clamp_min(1e-12)),
                            fd_half_relative_error=float(fro(half - ad) / fro(ad).clamp_min(1e-12)))
                    item['balance'] = {}
                    for name, matrices in [('autodiff', ad_matrices), ('fd1', fd_matrices[0]), ('fd_half', fd_matrices[1])]:
                        s, w = matrices[0][local], matrices[1][local]
                        s, w = (s + s.T) / 2, (w + w.T) / 2
                        c = (1 + alpha) * w - alpha * s
                        e = torch.linalg.eigvalsh(c)
                        item['balance'][name] = dict(strong_eigenvalues=torch.linalg.eigvalsh(s).cpu().tolist(),
                            weak_eigenvalues=torch.linalg.eigvalsh(w).cpu().tolist(),
                            combination_eigenvalues=e.cpu().tolist(),
                            rank1_denominator=float(c[0, 0]),
                            rank1_effective_alpha=float(alpha * s[0, 0] / c[0, 0]) if abs(float(c[0, 0])) > 1e-12 else None)
                    rows.append(item)
            del state, vs, vw, means, ad_matrices, fd_matrices
    result = dict(request_sha256=request_hash, code_sha256=sha(__file__), samples=32, states=len(rows),
        state_selection='first 32 inputs, native IG states at .125 and .375',
        basis='shared basis fixed from 1% finite difference; derivative estimators do not change basis',
        precision='FP32/TF32, original network, first-order reverse-mode AD',
        wall_seconds=time.perf_counter() - begin, rows=rows,
        summary={head: {radius: summaries([x['matrices'][head][radius + '_relative_error'] for x in rows])
            for radius in ('fd1', 'fd_half')} for head in ('strong', 'weak')},
        interpretation='Numerical derivative comparison only; AD of the network is not a certified posterior covariance.')
    out = study.ROOT / 'derivative_audit.json'
    atomic(out, result)
    portable = study.WORK / 'docs/data/small_sit_precision_balance_20260909/derivative_audit.json'
    atomic(portable, result)
    print(json.dumps(dict(output=str(out), wall_seconds=result['wall_seconds'], summary=result['summary'])), flush=True)


if __name__ == '__main__':
    main()
