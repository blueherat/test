#!/usr/bin/env python3
"""CPU-only, held-out heat-scale diagnostic for existing SiT weak fields.

See docs/AG_IG_SMOOTHING_DIAGNOSTIC_PROTOCOL_20260911_ZH.md.
Does not modify checkpoints, launch GPU work, or run a quality sweep.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.imagenet100_sit_multiscale_models import (
    evaluate_sit_field, evaluate_source_with_heads,
    load_internal_head_for_source, load_sit_field_model,
)
from experiments.train_imagenet100_sit_flow import (
    DEFAULT_CACHE_DIR, DEFAULT_OFFICIAL_SIT_REPO,
    load_official_sit_module, sample_sdvae_posterior,
)

REPO = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow')
OUT = REPO / 'docs/data/ag_ig_smoothing_20260911'
RAW = DATA / 'ag_ig_smoothing_cpu_20260911'
TIMES = [.15, .30, .50, .70, .90]
DELTAS = [-.75, -.5, -.25, -.1, -.03, -.01, 0., .01, .03, .1, .25, .5, 1., 2., 4.]
DERIVATIVE_STEPS = [.002, .001]
SEED = 20260911


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def score_from_velocity(z, t, v):
    """Score in additive-noise y=z/t coordinates, not in z coordinates."""
    return float(t) * (float(t) * v - z) / (1. - float(t))


def mean_square(x):
    return np.square(x.astype(np.float64)).reshape(len(x), -1).mean(1)


def per_dot(x, y):
    return (x.astype(np.float64) * y.astype(np.float64)).reshape(len(x), -1).mean(1)


def neural():
    if torch.cuda.is_initialized():
        raise RuntimeError('This diagnostic must not initialize CUDA')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    device = torch.device('cpu')
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    if (RAW / 'manifest.json').exists():
        raise FileExistsError('Existing run: use summarize; never overwrite raw observations')
    rng = np.random.default_rng(SEED)
    moments = np.load(DEFAULT_CACHE_DIR / 'validation_moments.npy', mmap_mode='r')
    labels_all = np.load(DEFAULT_CACHE_DIR / 'validation_labels.npy', mmap_mode='r')
    ids = rng.choice(len(moments), 32, replace=False)
    labels = torch.from_numpy(np.array(labels_all[ids], dtype=np.int64))
    moments_batch = torch.from_numpy(np.array(moments[ids]))
    clean = sample_sdvae_posterior(moments_batch, torch.randn(32, 4, 32, 32))
    noise = torch.randn_like(clean)
    np.savez_compressed(RAW / 'inputs.npz', ids=ids, labels=labels.numpy(), clean=clean.numpy(), noise=noise.numpy())
    manifest = dict(protocol='ag_ig_smoothing_cpu_v1', seed=SEED, ids=ids.tolist(),
                    split=['fit'] * 16 + ['test'] * 16, times=TIMES, deltas=DELTAS,
                    derivative_steps=DERIVATIVE_STEPS, device='cpu', dtype='float32',
                    torch_version=torch.__version__, threads=4, contexts=['teacher'],
                    script_sha256=digest(__file__), protocol_sha256=digest(REPO / 'docs/AG_IG_SMOOTHING_DIAGNOSTIC_PROTOCOL_20260911_ZH.md'),
                    inputs_sha256=digest(RAW / 'inputs.npz'),
                    data_manifest_sha256=digest(DEFAULT_CACHE_DIR / 'manifest.json'),
                    completed=False)
    write_json(RAW / 'manifest.json', manifest)
    start = time.monotonic()
    module, source = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO, verify_source=True)
    strong_path = DATA / 'runs/sit-s-2_seed0/checkpoints/step_00800000.pt'
    strong, semantics, strong_meta = load_sit_field_model(
        checkpoint_path=strong_path, weights='ema', sit_module=module,
        source_metadata=source, device=device)
    weak, weak_semantics, weak_meta = load_sit_field_model(
        checkpoint_path=DATA / 'runs/sit-s-2_seed0/checkpoints/step_00500000.pt',
        weights='ema', sit_module=module, source_metadata=source, device=device)
    heads = {}
    for depth in [4, 6, 8, 10]:
        path = (DATA / 'runs/sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/checkpoints/step_00050000.pt'
                if depth == 8 else DATA / f'multiscale_guidance_study_v1/runs/depth{depth}_v/checkpoints/step_00050000.pt')
        name = f'depth{depth}'
        heads[name] = load_internal_head_for_source(
            checkpoint_path=path, name=name, head_weights='ema', model=strong,
            sit_module=module, source_checkpoint_path=strong_path,
            source_metadata=source, device=device)
    manifest.update(strong=strong_meta, external_weak=weak_meta, official_sit=source,
                    heads={k: dict(depth=v.depth, checkpoint=v.checkpoint,
                                   checkpoint_sha256=v.checkpoint_sha256,
                                   source_checkpoint_sha256=v.source_checkpoint_sha256) for k, v in heads.items()})
    write_json(RAW / 'manifest.json', manifest)
    print(f'Loaded verified models in {time.monotonic()-start:.1f}s', flush=True)
    model_calls = 0
    with torch.inference_mode():
        for ti, t in enumerate(TIMES):
            tstart = time.monotonic()
            z = (1.-t)*noise + t*clean
            y = z / t
            q = ((1.-t)/t)**2
            expanded = torch.full((len(z),), t)
            full, trained, _ = evaluate_source_with_heads(strong, z, expanded, labels, heads=heads, source_semantics=semantics)
            direct = evaluate_sit_field(strong, semantics, z[:2], expanded[:2], labels[:2])
            parity = float((full[:2] - direct).abs().max())
            if not torch.allclose(full[:2], direct, atol=2e-5, rtol=2e-5):
                raise AssertionError(f'Shared-head/full-forward parity failure: {parity}')
            trained['external_v500'] = evaluate_sit_field(weak, weak_semantics, z, expanded, labels)
            model_calls += 3
            saved = dict(strong=score_from_velocity(z, t, full).numpy(), z=z.numpy(), q=np.array(q), t=np.array(t), parity_max=np.array(parity))
            saved.update({f'weak_{name}': score_from_velocity(z, t, velocity).numpy() for name, velocity in trained.items()})
            for delta in sorted(set(DELTAS + [s*h for h in DERIVATIVE_STEPS for s in [-1, 1]])):
                if delta == 0:
                    score = saved['strong']
                else:
                    shifted_t = 1. / (1. + np.sqrt(q*(1.+delta)))
                    shifted_z = y * shifted_t
                    velocity = evaluate_sit_field(strong, semantics, shifted_z, torch.full((len(z),), shifted_t), labels)
                    score = score_from_velocity(shifted_z, shifted_t, velocity).numpy()
                    model_calls += 1
                saved[f'shift_{delta:g}'] = score
            np.savez_compressed(RAW / f'time_{ti}.npz', **saved)
            print(f't={t:.2f}: {time.monotonic()-tstart:.1f}s, max parity error={parity:.3g}', flush=True)
    manifest.update(completed=True, wall_seconds=time.monotonic()-start, model_calls=model_calls,
                    cuda_initialized=torch.cuda.is_initialized(),
                    observations={p.name:digest(p) for p in RAW.glob('time_*.npz')})
    write_json(RAW / 'manifest.json', manifest)
    write_json(OUT / 'neural_manifest.json', manifest)
    summarize()


def summarize():
    manifest = json.loads((RAW / 'manifest.json').read_text())
    if not manifest['completed']:
        raise RuntimeError('Raw evaluation is incomplete')
    fit, test = slice(0, 16), slice(16, 32)
    bootstrap = np.random.default_rng(SEED + 1).integers(0, 16, (2000, 16))
    curves, summaries, per_image, derivatives = [], [], [], []
    def interval(numerator, denominator):
        ratios = numerator[bootstrap].sum(1) / denominator[bootstrap].sum(1)
        return np.quantile(ratios, [.025, .975]).tolist()
    for ti, t in enumerate(TIMES):
        obs = np.load(RAW / f'time_{ti}.npz')
        q = float(obs['q'])
        s = obs['strong']
        eps = DERIVATIVE_STEPS[-1]
        deriv = (obs[f'shift_{eps:g}'].astype(np.float64) - obs[f'shift_{-eps:g}'])/(2*eps*q)
        eps2 = DERIVATIVE_STEPS[0]
        deriv2 = (obs[f'shift_{eps2:g}'].astype(np.float64) - obs[f'shift_{-eps2:g}'])/(2*eps2*q)
        deriv_convergence = float(np.sqrt(mean_square(deriv-deriv2).sum()/mean_square(deriv).sum()))
        for name in [*manifest['heads'], 'external_v500']:
            w = obs[f'weak_{name}']
            target = w.astype(np.float64)-s
            gap2 = mean_square(target)
            errors = np.array([mean_square(w.astype(np.float64)-obs[f'shift_{delta:g}']) for delta in DELTAS])
            for j, delta in enumerate(DELTAS):
                curves.append(dict(weak=name, t=t, q=q, delta=delta, tau=q*delta,
                                   fit_ratio=float(errors[j,fit].sum()/gap2[fit].sum()),
                                   test_ratio=float(errors[j,test].sum()/gap2[test].sum())))
            for mode in ['nonnegative', 'signed']:
                allowed = [j for j, d in enumerate(DELTAS) if mode == 'signed' or d >= 0]
                j = min(allowed, key=lambda k: errors[k,fit].sum())
                e = errors[j]
                ci = interval(e[test], gap2[test])
                summaries.append(dict(weak=name, t=t, q=q, fit_type=f'finite_{mode}',
                                      delta=DELTAS[j], tau=q*DELTAS[j],
                                      fit_ratio=float(e[fit].sum()/gap2[fit].sum()),
                                      test_ratio=float(e[test].sum()/gap2[test].sum()),
                                      test_ratio_ci_low=ci[0], test_ratio_ci_high=ci[1],
                                      gap_rms=float(np.sqrt(gap2[test].mean()))))
                for i, (error, g2) in enumerate(zip(e, gap2)):
                    per_image.append(dict(weak=name, t=t, id=manifest['ids'][i], split=manifest['split'][i],
                                          fit_type=f'finite_{mode}', delta=DELTAS[j], tau=q*DELTAS[j],
                                          squared_residual=float(error), squared_gap=float(g2)))
            tau = float(per_dot(target, deriv)[fit].sum()/mean_square(deriv)[fit].sum())
            cosine = float(per_dot(target, deriv)[test].sum()/np.sqrt(gap2[test].sum()*mean_square(deriv)[test].sum()))
            derivatives.append(dict(weak=name, t=t, q=q, test_cosine=cosine,
                                    derivative_relative_rms_disagreement=deriv_convergence, fit_tau=tau,
                                    test_tau_oracle=float(per_dot(target,deriv)[test].sum()/mean_square(deriv)[test].sum())))
            for mode in ['nonnegative', 'signed']:
                effective_tau = max(tau, 0) if mode == 'nonnegative' else tau
                e = mean_square(target-effective_tau*deriv)
                ci = interval(e[test], gap2[test])
                summaries.append(dict(weak=name, t=t, q=q, fit_type=f'derivative_{mode}',
                                      delta=effective_tau/q, tau=effective_tau,
                                      fit_ratio=float(e[fit].sum()/gap2[fit].sum()),
                                      test_ratio=float(e[test].sum()/gap2[test].sum()),
                                      test_ratio_ci_low=ci[0], test_ratio_ci_high=ci[1],
                                      gap_rms=float(np.sqrt(gap2[test].mean()))))
                for i, (error, g2) in enumerate(zip(e,gap2)):
                    per_image.append(dict(weak=name,t=t,id=manifest['ids'][i],split=manifest['split'][i],
                                          fit_type=f'derivative_{mode}',delta=effective_tau/q,tau=effective_tau,
                                          squared_residual=float(error),squared_gap=float(g2)))
    write_csv(OUT / 'neural_fit_curves.csv', curves)
    write_csv(OUT / 'neural_summary.csv', summaries)
    write_csv(OUT / 'neural_per_image.csv', per_image)
    write_csv(OUT / 'neural_derivatives.csv', derivatives)
    print(json.dumps([r for r in summaries if r['fit_type']=='finite_nonnegative'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['neural', 'summarize'])
    args = parser.parse_args()
    if args.mode == 'neural':
        neural()
    else:
        summarize()
