"""Frozen CPU reuse for a single noise-endpoint velocity-risk witness.

No CUDA noise replay. Integrate only the zero-field risk analytically under the
standard Gaussian bridge law; this is not the original per-noise paired difference.
"""
from __future__ import annotations
import time
WALL_START, CPU_START = time.perf_counter(), time.process_time()
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import numpy as np

R = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
OUT = Path(__file__).resolve().parent
PROTOCOL = 'raev2_noise_endpoint_zero_witness_v1'
DIMENSION = 1024 * 16 * 16


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def rec(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': sha(path)}


def verify(record):
    assert sha(record['path']) == record['sha256'], record['path']


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def cost():
    return {'wall_seconds': time.perf_counter() - WALL_START,
            'cpu_seconds': time.process_time() - CPU_START,
            'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'timing_scope': 'From import time, excluding final JSON serialization, interpreter exit and parent shell.',
            'gpu_calls': 0, 'model_calls': 0, 'fid_calls': 0, 'noise_draws': 0}


def prepare():
    if (OUT / 'request.json').exists():
        raise FileExistsError(OUT / 'request.json')
    bank_path = R / 'potential_clean_bank_fp32_v1/summary.json'
    old_request_path = R / 'observable_potential_validation_v1/shard0/request.json'
    old_summary_path = old_request_path.with_name('summary.json')
    omitted_request_path = R / 'potential_omitted_witness_v1/shard0/request.json'
    omitted_summary_path = omitted_request_path.with_name('summary.json')
    bank, old, old_summary, omitted, omitted_summary = [json.loads(p.read_text()) for p in (
        bank_path, old_request_path, old_summary_path, omitted_request_path, omitted_summary_path)]
    assert bank['complete'] and old_summary['complete'] and omitted_summary['complete']
    assert old['time_grid'][0] == omitted['time_grid'][0] == 1.
    assert old['batch_size'] == omitted['batch_size'] == 32
    assert old['seeds']['validation_noise'] == omitted['noise_seed'] == 202609094
    assert old_summary['baseline_checkpoint_step'] == 100080
    assert old['baseline_checkpoint']['sha256'] == omitted['baseline']['sha256']
    assert old['baseline_formula'] == 'native head dtype B + 1.78*(F-B), official interval [.1,1], then float32'
    assert omitted_summary['old_residual_mse_bitwise_parity']
    sources = {'runner': rec(__file__), 'bank_summary': rec(bank_path),
               'validation_request': rec(old_request_path), 'validation_summary': rec(old_summary_path),
               'omitted_request': rec(omitted_request_path), 'omitted_summary': rec(omitted_summary_path),
               'validation_latents': rec(bank['banks']['validation']['latents']['path']),
               'validation_metadata': rec(bank['banks']['validation']['metadata']['path']),
               'train_metadata': rec(bank['banks']['train']['metadata']['path']),
               'legacy_step': rec(R / 'observable_potential_validation_v1/shard0/step000.npz'),
               'fp64_step': rec(R / 'potential_omitted_witness_v1/shard0/step000.npz'),
               'validation_runner': rec(old['sources']['train_raev2_observable_potential.py']['path']),
               'omitted_runner': rec(R / 'potential_omitted_witness_v1/shard0/runner_source.py'),
               'predicted_clean_request_read_for_scope_only': rec(R / 'predicted_clean_interaction_n5000x2_v1/request.json')}
    assert sources['bank_summary']['sha256'] == old['bank']['sha256']
    assert sources['validation_request']['sha256'] == old_summary['request']['sha256']
    assert sources['omitted_request']['sha256'] == omitted_summary['request']['sha256']
    assert sources['validation_latents']['sha256'] == bank['banks']['validation']['latents']['sha256']
    assert sources['validation_metadata']['sha256'] == bank['banks']['validation']['metadata']['sha256'] == omitted['validation_metadata']['sha256']
    assert sources['train_metadata']['sha256'] == bank['banks']['train']['metadata']['sha256']
    assert sources['legacy_step']['sha256'] == omitted['old_step_artifacts'][0]['sha256']
    assert sources['fp64_step']['sha256'] == omitted_summary['step_artifacts'][0]['sha256']
    assert sources['validation_runner']['sha256'] == old['sources']['train_raev2_observable_potential.py']['sha256']
    assert sources['omitted_runner']['sha256'] == omitted['runner']['sha256']
    with np.load(sources['validation_metadata']['path'], allow_pickle=False) as metadata:
        ids, labels, rows = (metadata[k].copy() for k in ('ids', 'labels', 'rows'))
    with np.load(sources['train_metadata']['path'], allow_pickle=False) as metadata:
        assert np.intersect1d(rows, metadata['rows']).size == 0
    assert np.array_equal(ids, np.arange(1000)) and np.unique(rows).size == 1000
    classes, counts = np.unique(labels, return_counts=True)
    assert np.array_equal(classes, np.arange(1000)) and np.all(counts == 1)
    np.savez(OUT / 'cohort_identity.npz', ids=ids, labels=labels, source_rows=rows)
    sources['cohort_identity'] = rec(OUT / 'cohort_identity.npz')
    request = {'protocol': PROTOCOL, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'phase': 'Frozen before evaluating residual means, latent norms or endpoint-risk differences.',
        'time': 1., 'step': 0, 'samples': 1000, 'dimension': DIMENSION,
        'cohort': 'All original validation IDs, exactly one per each of 1000 classes. Labels come from metadata, never ID modulo.',
        'baseline': old['baseline_checkpoint'], 'baseline_identity_scope': 'Historical archived checkpoint SHA and generation requests; current checkpoint not loaded or rehashed because no model is executed.',
        'formula': old['baseline_formula'], 'bridge': 'z=(1-t)X+t*epsilon; at t=1 z=epsilon independently drawn from N(0,I) using CUDA Generator seed 202609094, B32, fixed source metadata.',
        'zero_field': 'dataward velocity b=0; at t=1 its equivalent clean prediction is z, not clean zero.',
        'estimator': 'mean_i [r2_i - 1 - ||X_i||^2 / D], with r2 from FP64 reduction of stored FP32 residuals in the matched omitted-witness replay.',
        'identity': 'E_epsilon ||epsilon-X||^2 / D = 1+||X||^2/D; averaging X conditional on class recovers true mean-velocity risk difference.',
        'primary': 'A positive risk difference supports, and a negative risk difference opposes, the initial-velocity-worse-than-zero mechanism at exactly t=1.',
        'statistics': 'Equal-weighted all-class mean, sample SD/sqrt(1000) only as descriptive class SEM, mean +/- 1.96 SEM, count and range; no class exclusion, bootstrap, seed selection, or time scan.',
        'precision': 'FP16 cached X promoted directly to FP64 for norms. r2 uses FP32 subtraction then FP64 square/reduction from existing cache; legacy FP32 square/reduction retained as a numerical crosscheck.',
        'limits': ['The exact original per-noise paired difference is unavailable: no full z or ||z-X||^2 saved in the inspected matched caches.',
                   'This analytically integrated zero-field risk is an unbiased alternative under the standard Gaussian bridge law, not the realized original paired difference.',
                   'No variance reduction claim: covariance with r2 changes when only one risk term is integrated.',
                   'PRNG floating-point normal samples and cached residual rounding are numerical approximations to the ideal Gaussian / real-arithmetic identity.',
                   'One fixed noise cohort and one source per class; descriptive SEM is not a precise conditional sampling CI.',
                   'This diagnoses only the t=1 local velocity-risk mechanism, not finite Euler endpoint quality, FID, convergence of all parts of the model, or usefulness of every possible zero window.'],
        'sources': sources, 'device': 'CPU', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_calls': 0, 'model_calls': 0, 'fid_calls': 0, 'noise_draws': 0}
    save(OUT / 'request.json', request)
    save(OUT / 'prepare_cost.json', cost())
    print(json.dumps({'request': rec(OUT / 'request.json')}), flush=True)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    mean, sem = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
    return {'mean': mean, 'descriptive_class_sem': sem,
            'descriptive_mean_plus_minus_1p96_sem': [mean - 1.96 * sem, mean + 1.96 * sem],
            'minimum': float(values.min()), 'maximum': float(values.max()),
            'negative_classes': int(np.sum(values < 0)), 'positive_classes': int(np.sum(values > 0))}


def compute():
    if (OUT / 'summary.json').exists():
        raise FileExistsError(OUT / 'summary.json')
    request = json.loads((OUT / 'request.json').read_text())
    assert request['protocol'] == PROTOCOL
    for source in request['sources'].values():
        verify(source)
    with np.load(OUT / 'cohort_identity.npz', allow_pickle=False) as stored:
        ids, labels, rows = (stored[k].copy() for k in ('ids', 'labels', 'source_rows'))
    with np.load(request['sources']['legacy_step']['path'], allow_pickle=False) as old:
        assert np.array_equal(old['ids'], ids) and np.array_equal(old['labels'], labels)
        legacy = old['residual_mse'].copy()
    with np.load(request['sources']['fp64_step']['path'], allow_pickle=False) as current:
        assert current['time'].item() == 1. and current['step'].item() == 0
        assert np.array_equal(current['ids'], ids) and np.array_equal(current['labels'], labels)
        assert np.array_equal(current['residual_mse_fp32_parity'], legacy)
        r2 = current['residual_energy_fp64'].copy()
    latent = np.load(request['sources']['validation_latents']['path'], mmap_mode='r', allow_pickle=False)
    assert latent.shape == (1000, 1024, 16, 16) and latent.dtype == np.float16
    norm2 = np.empty(1000, dtype=np.float64)
    for start in range(0, 1000, 16):
        value = np.asarray(latent[start:start+16], dtype=np.float64).reshape(-1, DIMENSION)
        assert np.isfinite(value).all()
        norm2[start:start+len(value)] = np.einsum('ij,ij->i', value, value, optimize=False) / DIMENSION
    zero_risk = 1. + norm2
    delta = r2 - zero_risk
    legacy_delta = legacy - zero_risk
    assert np.isfinite(r2).all() and np.isfinite(delta).all()
    np.savez(OUT / 'by_image_and_class.npz', ids=ids, labels=labels, source_rows=rows,
             residual_mse_fp64=r2, residual_mse_legacy_fp32=legacy,
             clean_norm_squared_per_dimension=norm2, analytic_zero_field_clean_mse=zero_risk,
             endpoint_risk_difference=delta, legacy_endpoint_risk_difference=legacy_delta)
    summary = {'protocol': PROTOCOL, 'request': rec(OUT / 'request.json'),
               'samples': 1000, 'classes': 1000, 'time': 1., 'dimension': DIMENSION,
               'baseline_residual_mse': describe(r2), 'analytic_zero_field_clean_mse': describe(zero_risk),
               'clean_norm_squared_per_dimension': describe(norm2), 'endpoint_risk_difference': describe(delta),
               'legacy_endpoint_risk_difference': describe(legacy_delta),
               'max_per_image_residual_fp32_fp64_reduction_difference': float(np.max(np.abs(r2 - legacy))),
               'all_1000_legacy_residuals_bitwise_equal_to_replay': True,
               'interpretation': 'Negative values oppose the zero-is-more-accurate boundary mechanism at t=1; no downstream quality or general checkpoint-convergence claim.',
               'artifact': rec(OUT / 'by_image_and_class.npz'), 'cost': cost()}
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    {'prepare': prepare, 'compute': compute}[sys.argv[1]]()
