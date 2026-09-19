"""CPU diagnostics for the two independent CFG / IG research lines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'docs/data/cfg_ig_research_20260912'
RAW = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/ag_ig_smoothing_cpu_20260911')
CFG_RAW = RAW.parent / 'cfg_bayes_compatibility_cpu_20260912'
PY_SEED = 20260912


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jsave(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def csave(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def energy(x):
    return np.square(x).reshape(len(x), -1).mean(1)


def nuisance():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((RAW / 'manifest.json').read_text())
    with (REPO / 'docs/data/ag_ig_smoothing_20260911/refined_summary.csv').open() as f:
        heat = {(r['weak'], float(r['t'])): r for r in csv.DictReader(f)
                if r['fit_type'] == 'finite_nonnegative'}
    boot = np.random.default_rng(PY_SEED).integers(0, 16, size=(2000, 16))
    summaries, details = [], []
    for ti, t in enumerate(old['times']):
        obs = np.load(RAW / f'time_{ti}.npz')
        strong = obs['strong'].astype(np.float64)
        y = obs['z'].astype(np.float64) / t
        for weak in [*old['heads'], 'external_v500']:
            gap = obs[f'weak_{weak}'].astype(np.float64) - strong
            denom = energy(gap)
            for model, basis in [('score_scale', [strong]), ('radial', [y]),
                                 ('score_scale_radial', [strong, y])]:
                A = np.stack([b[:16].reshape(-1) for b in basis], axis=1)
                scales = np.linalg.norm(A, axis=0)
                A = A / scales
                coeff, _, rank, singular = np.linalg.lstsq(A, gap[:16].reshape(-1), rcond=1e-6)
                coeff = coeff / scales
                prediction = sum(c * b for c, b in zip(coeff, basis))
                residual = energy(gap - prediction)
                ratios = residual[16:][boot].sum(1) / denom[16:][boot].sum(1)
                lo, hi = np.quantile(ratios, [.025, .975])
                summaries.append(dict(weak=weak, t=t, model=model,
                    coefficient_score=float(coeff[0]) if model != 'radial' else 0.,
                    coefficient_radial=float(coeff[-1]) if model != 'score_scale' else 0.,
                    rank=int(rank), normalized_design_condition=float(singular[0] / singular[-1]),
                    fit_residual_ratio=float(residual[:16].sum() / denom[:16].sum()),
                    test_residual_ratio=float(residual[16:].sum() / denom[16:].sum()),
                    test_ci_low=float(lo), test_ci_high=float(hi),
                    heat_test_residual_ratio=float(heat[weak, t]['test_ratio'])))
                for i in range(32):
                    details.append(dict(weak=weak, t=t, model=model, id=old['ids'][i],
                        split=old['split'][i], squared_residual=float(residual[i]), squared_gap=float(denom[i])))
    csave(OUT / 'ig_nuisance_summary.csv', summaries)
    csave(OUT / 'ig_nuisance_per_image.csv', details)
    sources = [RAW / 'manifest.json', RAW / 'inputs.npz',
               *[RAW / f'time_{i}.npz' for i in range(5)],
               REPO / 'docs/data/ag_ig_smoothing_20260911/refined_summary.csv']
    jsave(OUT / 'ig_nuisance_manifest.json', dict(completed=True, observations_reused=True,
        model_queries=0, evaluation_is_exploratory=True, bootstrap_seed=PY_SEED,
        bootstrap_replicates=2000, script_sha256=sha(__file__),
        sources={str(p): sha(p) for p in sources}, rows=len(summaries), per_image_rows=len(details)))
    print(json.dumps([r for r in summaries if r['model'] == 'score_scale_radial'], indent=2), flush=True)


def geometry(fields, null, label):
    """All K fields are required. Ratios are invariant to common score coordinates."""
    fields = np.asarray(fields, dtype=np.float64).reshape(len(fields), -1)
    null = np.asarray(null, dtype=np.float64).reshape(-1)
    gaps = fields - null
    target_e = float(gaps[label] @ gaps[label])
    center = fields.mean(0)
    centered = fields - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    result = {}
    for cutoff in [1e-6, 1e-5, 1e-4]:
        rank = int((singular > cutoff * singular[0]).sum()) if singular[0] > 0 else 0
        common = center - null
        if rank:
            common = common - vh[:rank].T @ (vh[:rank] @ common)
        result[f'affine_rank_{cutoff:g}'] = rank
        result[f'affine_gap_energy_ratio_{cutoff:g}'] = float(common @ common / max(target_e, 1e-30))
        if cutoff == 1e-5:
            result['orthogonality_relative'] = float(np.max(np.abs(centered @ common)) /
                max(np.linalg.norm(centered, axis=1).max() * np.linalg.norm(common), 1e-30))
    scale = max(float(np.mean(np.sum(gaps * gaps, axis=1))), 1e-30)
    Q = (gaps @ gaps.T) / scale
    n = len(fields)
    sol = minimize(lambda w: .5 * float(w @ Q @ w), np.ones(n) / n,
        jac=lambda w: Q @ w, bounds=[(0., 1.)] * n,
        constraints=[dict(type='eq', fun=lambda w: float(w.sum() - 1), jac=lambda w: np.ones(n))],
        method='SLSQP', options=dict(ftol=1e-12, maxiter=2000))
    w = sol.x
    grad = Q @ w
    r = w @ gaps
    result.update(hull_gap_energy_ratio=float(r @ r / max(target_e, 1e-30)),
        hull_solver_success=bool(sol.success), hull_iterations=int(sol.nit),
        hull_sum_error=float(abs(w.sum()-1)), hull_min_weight=float(w.min()),
        hull_fw_gap=float(w @ grad - grad.min()),
        target_gap_rms=float(np.sqrt(target_e / gaps.shape[1])),
        class_spread_rms=float(np.sqrt(np.mean(centered * centered))),
        smallest_nonzero_singular_relative=float(singular[-2]/singular[0]) if singular[0] else 0.)
    return result, w


def analytic():
    OUT.mkdir(parents=True, exist_ok=True)
    means = np.array([[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    point = np.array([.2, -.1, .3])
    v = .7
    fields = (means-point) / v
    logp = -.5 * np.sum((means-point)**2, axis=1)/v
    posterior = np.exp(logp-logp.max()); posterior /= posterior.sum()
    null = posterior @ fields
    exact, _ = geometry(fields, null, 0)
    biased, _ = geometry(fields, null + np.array([0., 0., .4]), 0)
    subset, _ = geometry(fields[:2], null, 0)
    assert exact['hull_gap_energy_ratio'] < 1e-10
    assert biased['affine_gap_energy_ratio_1e-05'] > .01
    assert subset['hull_gap_energy_ratio'] > .001
    bifurcations = []
    for q in [2., 1., .6, .2, .01]:
        for x in [0., .1, .5, 1.]:
            v = .09 + q
            s = -x/v + np.tanh(x/v)/v
            k = -1/v + (1 - np.tanh(x/v)**2)/v**2
            sw = -x/(1+v)
            kw = -1/(1+v)
            bifurcations.append(dict(q=q, x=x, strong_score=float(s), weak_score=float(sw),
                strong_curvature=float(k), weak_curvature=float(kw),
                disagreement_gate=bool(k > 0 >= kw)))
    csave(OUT / 'mode_resolution_analytic.csv', bifurcations)
    jsave(OUT / 'analytic_checks.json', dict(exact_mixture=exact, biased_null=biased,
        incomplete_label_set_false_positive=subset, posterior=posterior.tolist(), passed=True))


def cfg():
    import torch
    from experiments.imagenet100_sit_multiscale_models import load_sit_field_model, evaluate_sit_field
    from experiments.train_imagenet100_sit_flow import load_official_sit_module, DEFAULT_OFFICIAL_SIT_REPO, NUM_CLASSES
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    CFG_RAW.mkdir(parents=True, exist_ok=True)
    if (CFG_RAW / 'manifest.json').exists():
        raise FileExistsError('Do not overwrite a previous CFG diagnostic')
    positions, times = [0, 1, 16, 17], [.15, .5, .9]
    old = json.loads((RAW / 'manifest.json').read_text())
    inputs = np.load(RAW / 'inputs.npz')
    start = time.monotonic()
    manifest = dict(completed=False, positions=positions, ids=inputs['ids'][positions].tolist(),
        times=times, num_classes=NUM_CLASSES, device='cpu', dtype='float32', threads=4,
        input_manifest_sha256=sha(RAW/'manifest.json'), input_sha256=sha(RAW/'inputs.npz'),
        script_sha256=sha(__file__),
        protocol_sha256=sha(REPO/'docs/CFG_IG_RESEARCH_DIAGNOSTIC_PROTOCOL_20260912_ZH.md'))
    jsave(CFG_RAW/'manifest.json', manifest)
    module, source = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO, verify_source=True)
    model, semantics, metadata = load_sit_field_model(
        checkpoint_path=Path(old['strong']['checkpoint']), weights='ema', sit_module=module,
        source_metadata=source, device=torch.device('cpu'))
    assert model.y_embedder.embedding_table.num_embeddings == NUM_CLASSES + 1
    rows, weight_rows = [], []
    model_calls = 0
    with torch.inference_mode():
        for ti, t in enumerate(times):
            for pos in positions:
                z = torch.from_numpy((1-t)*inputs['noise'][pos:pos+1] + t*inputs['clean'][pos:pos+1])
                values = []
                for first in range(0, NUM_CLASSES+1, 32):
                    labels = torch.arange(first, min(first+32, NUM_CLASSES+1), dtype=torch.long)
                    b = len(labels)
                    values.append(evaluate_sit_field(model, semantics, z.expand(b,-1,-1,-1),
                        torch.full((b,), t), labels).numpy())
                    model_calls += 1
                values = np.concatenate(values)
                label = int(inputs['labels'][pos]); ident = int(inputs['ids'][pos])
                filename = f'time_{ti}_position_{pos}.npz'
                np.savez_compressed(CFG_RAW / filename, fields=values, z=z.numpy(), t=t, label=label, id=ident)
                row, weights = geometry(values[:NUM_CLASSES], values[NUM_CLASSES], label)
                rows.append(dict(t=t, position=pos, id=ident, label=label, **row))
                weight_rows.extend(dict(t=t, id=ident, class_index=k, weight=float(w)) for k,w in enumerate(weights))
                print(json.dumps(rows[-1]), flush=True)
    manifest.update(completed=True, wall_seconds=time.monotonic()-start, model_calls=model_calls,
        equivalent_single_sample_forwards=len(times)*len(positions)*(NUM_CLASSES+1), model=metadata,
        cuda_initialized=torch.cuda.is_initialized(),
        observations={p.name:sha(p) for p in CFG_RAW.glob('time_*.npz')})
    jsave(CFG_RAW/'manifest.json', manifest); jsave(OUT/'cfg_manifest.json', manifest)
    csave(OUT/'cfg_bayes_geometry.csv', rows); csave(OUT/'cfg_hull_weights.csv', weight_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['nuisance', 'analytic', 'cfg'])
    args = parser.parse_args()
    globals()[args.task]()
