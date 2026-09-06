#!/usr/bin/env python3
"""Frozen, CPU-only cross-bank linear endpoint-observable audit.

Prepare records metadata, exclusions and source hashes before any feature math.
Compute reads the frozen request; it never trains, samples, evaluates FID or uses CUDA.
"""
from __future__ import annotations

import time

START_WALL = time.perf_counter()
START_CPU = time.process_time()

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path('/home/zhoushunyu/data/eqvae/experiments')
OUT = EXPERIMENTS / 'raev2_guidance_restart_20260906/endpoint_crossprototype_witness_v1'
SEEDS = (20260801, 20260802)
BRANCHES = {'source': 'source', 'reconstruction': 'real',
            'full': 'scale_s1p000000', 'ig': 'scale_s1p780000'}
D, C, N = 2048, 1000, 5000
PROTOCOL = 'raev2_endpoint_crossprototype_witness_v1'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'sha256': digest(path)}


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def costs():
    return {'wall_seconds': time.perf_counter() - START_WALL,
            'cpu_seconds': time.process_time() - START_CPU,
            'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'gpu_calls': 0, 'model_calls': 0, 'fid_calls': 0,
            'timing_scope': 'From import time; excludes final JSON write, interpreter exit and parent shell startup.'}


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    maps = {}
    banks = []
    old = json.loads((EXPERIMENTS / 'raev2_guidance_restart_20260906/raw_latent_class_moments_v1/request.json').read_text())
    for seed in SEEDS:
        bank = EXPERIMENTS / 'raev2_ig_scale_response' / f'n5000_seed{seed}_scales7_v1'
        manifest_record, protocol_record = record(bank / 'manifest.json'), record(bank / 'sample_protocol.npz')
        previous = next(item for item in old['banks'] if item['seed'] == seed)
        assert manifest_record['sha256'] == previous['manifest_sha256']
        assert protocol_record['sha256'] == previous['protocol_sha256']
        manifest = json.loads((bank / 'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['seed'] == seed
        assert manifest['samples'] == N and manifest['world_size'] == 4
        assert manifest['same_noise_and_labels_across_scales']
        with np.load(bank / 'sample_protocol.npz', allow_pickle=False) as stored:
            ids, labels, rows = (stored[key].copy() for key in ('sample_ids', 'labels', 'real_source_rows'))
        assert np.array_equal(ids, np.arange(N))
        assert np.array_equal(labels, ids % C)
        assert rows.shape == (N,) and np.unique(rows).size == N
        maps[seed] = {'ids': ids, 'labels': labels, 'rows': rows}
        features = {name: [record(bank / 'inception' / f'{filename}_rank{rank:02d}.npy')
                           for rank in range(4)] for name, filename in BRANCHES.items()}
        banks.append({'seed': seed, 'manifest': manifest_record,
                      'sample_protocol': protocol_record, 'features': features})
    folds, mapping_arrays = [], {}
    for train, evaluate in ((SEEDS[0], SEEDS[1]), (SEEDS[1], SEEDS[0])):
        a, b = maps[train], maps[evaluate]
        excluded = np.isin(b['rows'], a['rows'])
        kept_ids = b['ids'][~excluded]
        counts = np.bincount(b['labels'][~excluded], minlength=C)
        assert counts.shape == (C,) and np.all(counts > 0)
        name = f'train{train}_eval{evaluate}'
        folds.append({'name': name, 'train_seed': train, 'eval_seed': evaluate,
                      'train_samples': N, 'eval_kept_samples': int((~excluded).sum()),
                      'excluded_eval_ids': b['ids'][excluded].tolist(),
                      'excluded_source_rows': b['rows'][excluded].tolist(),
                      'eval_class_counts': counts.tolist()})
        mapping_arrays[name + '_kept_eval_ids'] = kept_ids
        mapping_arrays[name + '_excluded_eval_ids'] = b['ids'][excluded]
        mapping_arrays[name + '_eval_class_counts'] = counts
    for seed, mapping in maps.items():
        for key, array in mapping.items():
            mapping_arrays[f'seed{seed}_{key}'] = array
    np.savez(out / 'source_maps.npz', **mapping_arrays)
    source = Path(__file__).read_bytes()
    (out / 'runner_source.py').write_bytes(source)
    request = {
        'protocol': PROTOCOL, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'phase': 'Frozen before loading feature values or computing prototypes/observables.',
        'seeds': list(SEEDS), 'branches': BRANCHES, 'dimension': D, 'classes': C,
        'observable': '<(mean of train source class c - mean of all train source), (eval feature - mean of all train source)> / 2048',
        'prototype': 'Exactly all five original source features per class; no trimming, projection, normalization beyond division by 2048, or prototype-number search.',
        'evaluation': 'Remove eval IDs whose source rows occur anywhere in train source, identically in all four eval arms. Keep all train images. Average kept observations within each class, then all 1000 classes equally.',
        'primary': 'IG minus source, in both fixed train/eval directions.',
        'secondary': ['IG minus reconstruction', 'IG minus Full'],
        'descriptive_rule': 'Report both means, descriptive class-cluster SEM and mean +/- 1.96 SEM. Stable negative evidence requires both primary upper endpoints below zero; neither intervals nor two folds are independent sampling experiments.',
        'pairing': 'All contrasts use identical kept IDs within each eval bank; source/reconstruction share source images; Full/IG share generation noise under old manifest, without archived per-image noise SHA. Source and generated samples are independent given class.',
        'arithmetic': 'Original float32 features promoted to float64; no feature modification.',
        'limits': ['Retrospective historical feature reuse; two directional folds reuse banks and are dependent.',
                   'Fixed-class SEM describes heterogeneity, not an exact conditional sampling confidence interval.',
                   'A linear endpoint alignment deficit is not causal evidence for FID improvement and grants no training or guidance authorization.',
                   'A feature-coordinate input gradient exists. Pullback to image/latent requires a differentiable feature/decoder map; exact historical uint8 quantization is not differentiable guidance.',
                   'The current live 5K sampling/evaluation bank is excluded from every read.'],
        'device': 'CPU only', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'banks': banks, 'folds': folds, 'source_maps': record(out / 'source_maps.npz'),
        'runner': record(Path(__file__)), 'runner_archive': record(out / 'runner_source.py'),
    }
    write_json(out / 'request.json', request)
    write_json(out / 'prepare_cost.json', costs())
    print(json.dumps({'request': record(out / 'request.json'),
                      'excluded_per_fold': [len(fold['excluded_eval_ids']) for fold in folds]}), flush=True)


def validate_record(rec):
    if digest(rec['path']) != rec['sha256']:
        raise ValueError(f"source SHA mismatch: {rec['path']}")


def load_features(bank, name):
    values = np.empty((N, D), dtype=np.float64)
    for rank, rec in enumerate(bank['features'][name]):
        validate_record(rec)
        stored = np.load(rec['path'], allow_pickle=False)
        if stored.shape != (N // 4, D) or stored.dtype != np.float32 or not np.isfinite(stored).all():
            raise ValueError(f"invalid feature cache: {rec['path']}")
        values[rank::4] = stored
    return values


def describe(values):
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(C))
    return {'class_equal_weighted_mean': mean, 'descriptive_class_sem': sem,
            'descriptive_mean_plus_minus_1p96_sem': [mean - 1.96 * sem, mean + 1.96 * sem],
            'negative_classes': int((values < 0).sum()), 'classes': C}


def compute(out):
    request = json.loads((out / 'request.json').read_text())
    assert request['protocol'] == PROTOCOL
    if (out / 'summary.json').exists():
        raise FileExistsError(out / 'summary.json')
    for key in ('runner', 'runner_archive', 'source_maps'):
        validate_record(request[key])
    for bank in request['banks']:
        validate_record(bank['manifest'])
        validate_record(bank['sample_protocol'])
    banks = {bank['seed']: bank for bank in request['banks']}
    prototype_vectors, global_means = {}, {}
    saved = {'classes': np.arange(C)}
    for seed in SEEDS:
        source = load_features(banks[seed], 'source')
        means = source.reshape(5, C, D).mean(axis=0)
        global_mean = source.mean(axis=0)
        np.testing.assert_allclose(means.mean(axis=0), global_mean, atol=1e-13, rtol=1e-13)
        prototype_vectors[seed] = means - global_mean
        global_means[seed] = global_mean
        saved[f'seed{seed}_prototype_vectors'] = prototype_vectors[seed]
        saved[f'seed{seed}_source_global_mean'] = global_mean
    del source
    fold_results = []
    for fold in request['folds']:
        train, evaluate, name = fold['train_seed'], fold['eval_seed'], fold['name']
        kept = np.ones(N, dtype=bool)
        kept[fold['excluded_eval_ids']] = False
        ids, counts = np.flatnonzero(kept), np.asarray(fold['eval_class_counts'])
        labels = ids % C
        vectors, center = prototype_vectors[train], global_means[train]
        branch_means = {}
        for branch in BRANCHES:
            values = load_features(banks[evaluate], branch)
            # Evaluate all rows directly, then apply the preregistered common mask.
            # Blocks follow occurrence x class x feature, preserving global IDs.
            observations = np.einsum('cd,kcd->kc', vectors, values.reshape(5, C, D) - center,
                                     optimize=False).reshape(N) / D
            means = np.bincount(labels, weights=observations[kept], minlength=C) / counts
            assert np.isfinite(means).all()
            branch_means[branch] = means
            saved[name + '_' + branch + '_class_mean'] = means
            saved[name + '_' + branch + '_kept_observations'] = observations[kept]
        contrasts = {}
        for reference in ('source', 'reconstruction', 'full'):
            key = 'ig_minus_' + reference
            difference = branch_means['ig'] - branch_means[reference]
            saved[name + '_' + key + '_class_difference'] = difference
            contrasts[key] = describe(difference)
            observations_delta = (saved[name + '_ig_kept_observations'] -
                                  saved[name + '_' + reference + '_kept_observations'])
            np.testing.assert_allclose(difference,
                np.bincount(labels, weights=observations_delta, minlength=C) / counts,
                atol=1e-15, rtol=1e-11)
        fold_results.append({'name': name, 'train_seed': train, 'eval_seed': evaluate,
                             'kept_samples': len(ids), 'removed_samples': N - len(ids),
                             'class_count_histogram': {str(int(k)): int(v) for k, v in zip(*np.unique(counts, return_counts=True))},
                             'branches': {key: describe(value) for key, value in branch_means.items()},
                             'contrasts': contrasts})
    np.savez(out / 'class_observables.npz', **saved)
    stable = all(fold['contrasts']['ig_minus_source']['descriptive_mean_plus_minus_1p96_sem'][1] < 0
                 for fold in fold_results)
    summary = {'protocol': PROTOCOL, 'request': record(out / 'request.json'),
               'results': fold_results, 'both_primary_means_negative': all(
                   fold['contrasts']['ig_minus_source']['class_equal_weighted_mean'] < 0 for fold in fold_results),
               'stable_negative_under_preregistered_descriptive_rule': stable,
               'interpretation': 'Only a cross-source-prototype linear endpoint observable; dependent folds and descriptive fixed-class SEM. No FID, causal, whole-law, or guidance guarantee.',
               'artifacts': {'class_observables': record(out / 'class_observables.npz')},
               'cost': costs()}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'compute'))
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Set CUDA_VISIBLE_DEVICES to empty for this CPU-only audit')
    {'prepare': prepare, 'compute': compute}[args.phase](args.output_dir)


if __name__ == '__main__':
    main()
