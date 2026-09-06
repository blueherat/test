#!/usr/bin/env python3
"""Independent numerical path: direct raw-shard per-class dot products.

Does not import producer code. Verifies every saved prototype, observation,
class mean and paired difference, plus frozen source maps and source hashes.
"""
import time
wall_start, cpu_start = time.perf_counter(), time.process_time()
import hashlib
import json
import os
from pathlib import Path
import resource
import numpy as np

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
out = Path(__file__).resolve().parent
request = json.loads((out / 'request.json').read_text())
summary = json.loads((out / 'summary.json').read_text())
result_arrays = np.load(out / 'class_observables.npz', allow_pickle=False)
source_maps = np.load(out / 'source_maps.npz', allow_pickle=False)
checks = 0
numeric_comparisons = 0
max_abs_error = 0.0


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def same(a, b):
    global checks, numeric_comparisons, max_abs_error
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, atol=2e-15, rtol=1e-10)
    checks += 1
    numeric_comparisons += a.size
    if a.size:
        max_abs_error = max(max_abs_error, float(np.max(np.abs(a - b))))


banks = {}
hash_count = 0
for bank in request['banks']:
    for record in [bank['manifest'], bank['sample_protocol']] + [r for rs in bank['features'].values() for r in rs]:
        assert sha(record['path']) == record['sha256']
        hash_count += 1
    seed = bank['seed']
    with np.load(bank['sample_protocol']['path'], allow_pickle=False) as data:
        ids, labels, rows = [data[k].copy() for k in ('sample_ids', 'labels', 'real_source_rows')]
    same(ids, source_maps[f'seed{seed}_ids'])
    same(labels, source_maps[f'seed{seed}_labels'])
    same(rows, source_maps[f'seed{seed}_rows'])
    features = {name: [np.load(rec['path'], mmap_mode='r', allow_pickle=False) for rec in records]
                for name, records in bank['features'].items()}
    banks[seed] = {'ids': ids, 'labels': labels, 'rows': rows, 'features': features}


def get_rows(shards, ids):
    return np.stack([shards[int(i) % 4][int(i) // 4] for i in ids]).astype(np.float64)


recomputed_prototypes = {}
for seed, bank in banks.items():
    means = np.stack([get_rows(bank['features']['source'], bank['ids'][bank['labels'] == c]).mean(axis=0)
                      for c in range(1000)])
    center = means.mean(axis=0)
    vectors = means - center
    same(center, result_arrays[f'seed{seed}_source_global_mean'])
    same(vectors, result_arrays[f'seed{seed}_prototype_vectors'])
    recomputed_prototypes[seed] = vectors, center

for fold, archived in zip(request['folds'], summary['results']):
    a, b = banks[fold['train_seed']], banks[fold['eval_seed']]
    assert fold['name'] == archived['name']
    common_rows = set(a['rows'].tolist()).intersection(b['rows'].tolist())
    kept_ids = np.array([i for i, row in zip(b['ids'], b['rows']) if int(row) not in common_rows])
    excluded_ids = np.array([i for i, row in zip(b['ids'], b['rows']) if int(row) in common_rows])
    assert len(common_rows) == 21 and len(kept_ids) == 4979
    same(excluded_ids, fold['excluded_eval_ids'])
    same(kept_ids, source_maps[fold['name'] + '_kept_eval_ids'])
    assert not set(b['rows'][kept_ids]).intersection(a['rows'])
    vectors, center = recomputed_prototypes[fold['train_seed']]
    means = {}
    for branch in ('source', 'reconstruction', 'full', 'ig'):
        per_class = []
        row_results = {}
        for c in range(1000):
            selected = kept_ids[b['labels'][kept_ids] == c]
            x = get_rows(b['features'][branch], selected)
            # Alternative contraction: one dot product with each empirical class mean.
            per_class.append(float(vectors[c] @ (x.mean(axis=0) - center) / 2048))
            for i, value in zip(selected, (x - center) @ vectors[c] / 2048):
                row_results[int(i)] = value
        means[branch] = np.array(per_class)
        same(means[branch], result_arrays[fold['name'] + '_' + branch + '_class_mean'])
        same(np.array([row_results[int(i)] for i in kept_ids]),
             result_arrays[fold['name'] + '_' + branch + '_kept_observations'])
        same(means[branch].mean(), archived['branches'][branch]['class_equal_weighted_mean'])
        same(means[branch].std(ddof=1) / np.sqrt(1000), archived['branches'][branch]['descriptive_class_sem'])
    for reference in ('source', 'reconstruction', 'full'):
        key = 'ig_minus_' + reference
        delta = means['ig'] - means[reference]
        same(delta, result_arrays[fold['name'] + '_' + key + '_class_difference'])
        report = archived['contrasts'][key]
        same(delta.mean(), report['class_equal_weighted_mean'])
        same(delta.std(ddof=1) / np.sqrt(1000), report['descriptive_class_sem'])
        same([delta.mean() - 1.96 * delta.std(ddof=1) / np.sqrt(1000),
              delta.mean() + 1.96 * delta.std(ddof=1) / np.sqrt(1000)],
             report['descriptive_mean_plus_minus_1p96_sem'])
        assert int((delta < 0).sum()) == report['negative_classes']

for record in [summary['request'], summary['artifacts']['class_observables'], request['source_maps'],
               request['runner'], request['runner_archive']]:
    assert sha(record['path']) == record['sha256']
    hash_count += 1

result = {'passed': True, 'method': 'Direct per-class means and raw-shard row dot products; no producer imports.',
          'numeric_checks': checks, 'numeric_values_compared': numeric_comparisons,
          'max_absolute_error': max_abs_error, 'source_or_artifact_hash_checks': hash_count,
          'script_sha256': sha(__file__), 'summary_sha256': sha(out / 'summary.json'),
          'wall_seconds': time.perf_counter() - wall_start,
          'cpu_seconds': time.process_time() - cpu_start,
          'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
          'gpu_calls': 0, 'model_calls': 0, 'fid_calls': 0,
          'timing_scope': 'From import time to result construction, excluding final JSON write and interpreter exit.'}
with (out / 'verification.json').open('x') as handle:
    json.dump(result, handle, indent=2, allow_nan=False)
    handle.write('\n')
print(json.dumps(result, indent=2))
