#!/usr/bin/env python3
"""Independent NumPy readout recomputation from finite-retention snapshots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stats(a, b):
    a, b = a.astype(np.float64).reshape(8, -1), b.astype(np.float64).reshape(8, -1)
    aa, bb, ab = np.sum(a*a, axis=1), np.sum(b*b, axis=1), np.sum(a*b, axis=1)
    return {'projection': ab/bb, 'rms': np.sqrt(aa/a.shape[1]),
            'relative_norm': np.sqrt(aa/bb), 'cosine': ab/np.maximum(np.sqrt(aa*bb), 1e-300)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    args = p.parse_args()
    root, began = args.input.resolve(), time.perf_counter()
    load = lambda path: json.loads(path.read_text())
    cohort, result = load(root/'cohort.json'), load(root/'analysis.json')
    grid, numerical, hashes, max_error, max_recurrence_ratio = cohort['time_grid'], 0, 0, 0., 0.
    for c in cohort['selected']:
        path = root/f"rank{c['ordinal'] % 4}"/f"batch{c['ordinal']:02d}"
        summary = load(path/'summary.json')
        for name, expected in summary['files'].items():
            assert sha(path/name) == expected
            hashes += 1
        data = np.load(path/'snapshots.npz')
        assert np.array_equal(data['ids'], c['ids']) and np.array_equal(data['labels'], c['labels'])
        message = data['source_guided']-data['source_full']
        impulse = data['successor_once']-data['successor_none']
        with (path/'readouts.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == (99-c['query_index'])*8
        initial = rows[:8]
        expectations = {'read': stats(data['first_read_once']-data['first_read_none'], message),
                        'state': stats(impulse, message)}
        terminal = {'terminal': stats(data['endpoint_once']-data['endpoint_none'], message),
                    'all_guidance_terminal': stats(data['endpoint_all']-data['endpoint_none'], message),
                    'one_relative_to_all': stats(data['endpoint_once']-data['endpoint_none'], data['endpoint_all']-data['endpoint_none'])}
        for group, reference in ((expectations, initial), (terminal, summary['terminal'])):
            for prefix, metrics in group.items():
                for metric, expected in metrics.items():
                    actual = np.array([float(row[prefix+'_'+metric]) for row in reference])
                    error = np.max(np.abs(expected-actual))
                    assert np.allclose(expected, actual, rtol=1e-10, atol=1e-12)
                    max_error = max(max_error, float(error))
                    numerical += 8
        mrms = np.sqrt(np.mean(message.astype(np.float64).reshape(8, -1)**2, axis=1))
        for j in range(c['query_index']+1, 100):
            offset = 8*(j-c['query_index']-1)
            current = rows[offset:offset+8]
            assert [int(r['id']) for r in current] == c['ids']
            assert all(int(r['read_query']) == j and float(r['read_time']) == grid[j] for r in current)
            following = rows[offset+8:offset+16] if j < 99 else summary['terminal']
            a, q = grid[j+1]/grid[j], (grid[j]-grid[j+1])/grid[j]
            for i, (row, nex) in enumerate(zip(current, following)):
                predicted = a*float(row['state_projection'])+q*float(row['read_projection'])
                actual = float(nex['state_projection'] if j < 99 else nex['terminal_projection'])
                # Measured native rounding plus a conservative FP32 multiply/add
                # envelope, followed by Cauchy--Schwarz projection onto message.
                bound = (float(row['finite_recurrence_roundoff_rms']) + 1e-6*(a*float(row['state_rms'])+q*float(row['read_rms'])))/mrms[i]+1e-10
                assert abs(actual-predicted) <= bound
                max_recurrence_ratio = max(max_recurrence_ratio, abs(actual-predicted)/bound)
                numerical += 1
    assert len(cohort['selected']) == 16 and result['samples'] == 128
    audit = {'complete': True, 'sample_count': 128, 'numerical_comparisons': numerical,
             'artifact_hashes': hashes, 'max_snapshot_statistic_error': max_error,
             'max_finite_recurrence_error_over_rounding_bound': max_recurrence_ratio,
             'scope': 'All initial and terminal readout metrics independently recomputed from saved arrays; every suffix projection recurrence checked. No independent model rerun or FID claim.',
             'analysis_sha256': sha(root/'analysis.json'), 'audit_source_sha256': sha(Path(__file__).resolve()),
             'seconds': time.perf_counter()-began}
    (root/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
