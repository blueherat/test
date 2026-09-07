"""Recompute the finite writer's constraints and terminal statistics on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def norm(x):
    return np.sqrt(np.sum(x.astype(np.float64).reshape(8, -1)**2, axis=1))


def dot(x, y):
    return np.sum(x.astype(np.float64).reshape(8, -1)*y.astype(np.float64).reshape(8, -1), axis=1)


def stats(x, y):
    a, b, ab = norm(x), norm(y), dot(x, y)
    return {'projection': ab/b**2, 'cosine': ab/np.maximum(a*b, 1e-300), 'rms': a/512}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    source, out, began = args.source.resolve(), args.input.resolve(), time.perf_counter()
    read = lambda p: json.loads(p.read_text())
    groups, comparisons, max_error, changed = {}, 0, 0., 0
    for c in read(source/'cohort.json')['selected']:
        old_dir = source/f"rank{c['ordinal']%4}"/f"batch{c['ordinal']:02d}"
        new_dir = out/f"rank{c['ordinal']%4}"/f"batch{c['ordinal']:02d}"
        old_meta, meta = read(old_dir/'summary.json'), read(new_dir/'summary.json')
        assert sha(old_dir/'snapshots.npz') == old_meta['files']['snapshots.npz']
        assert sha(new_dir/'snapshots.npz') == meta['snapshot_sha256']
        old, new = np.load(old_dir/'snapshots.npz'), np.load(new_dir/'snapshots.npz')
        assert np.array_equal(old['ids'], new['ids'])
        message = old['source_guided']-old['source_full']
        r0 = norm(old['successor_once']-old['successor_none'])
        r1 = norm(new['successor_writer']-old['successor_none'])
        a0, a1 = old['first_read_once']-old['first_read_none'], new['read_writer']-old['first_read_none']
        rho0, rho1 = norm(a0), norm(a1)
        # Cross-backend reductions get only 1e-12 relative tolerance; producer
        # accepts using strict FP64 norm inequalities without this tolerance.
        assert np.all(r1 <= r0*(1+1e-12)) and np.all(rho1 <= rho0*(1+1e-12))
        p0, p1 = dot(a0, message)/norm(message), dot(a1, message)/norm(message)
        assert np.all(p1 >= p0-1e-12)
        positive = p0 >= 0
        assert np.all((rho1*rho1-p1*p1)[positive] <= (rho0*rho0-p0*p0)[positive]+1e-10)
        changed += int(np.count_nonzero(np.any(new['successor_writer'] != old['successor_once'], axis=(1, 2, 3))))
        actual = {'terminal': stats(new['endpoint_writer']-old['endpoint_none'], message),
                  'terminal_native': stats(old['endpoint_once']-old['endpoint_none'], message),
                  'to_all': stats(new['endpoint_writer']-old['endpoint_none'], old['endpoint_all']-old['endpoint_none']),
                  'to_all_native': stats(old['endpoint_once']-old['endpoint_none'], old['endpoint_all']-old['endpoint_none'])}
        for group, values in actual.items():
            for key, expected in values.items():
                assert np.allclose(expected, meta[group][key], atol=1e-12, rtol=1e-10)
                max_error = max(max_error, float(np.max(np.abs(expected-np.array(meta[group][key])))))
                comparisons += 8
        groups.setdefault(c['query_index'], []).append(actual)
    improved = int(sum(np.mean([b['terminal']['projection'] for b in rs]) > np.mean([b['terminal_native']['projection'] for b in rs]) for rs in groups.values()))
    result = read(out/'analysis.json')
    assert improved == result['terminal_projection_improved_groups'] == 4
    assert not result['mechanism_triage_passed']
    audit = {'complete': True, 'samples': 128, 'changed_samples': changed,
        'numerical_comparisons': comparisons, 'maximum_metric_error': max_error,
        'native_input_and_read_response_budgets_verified': True, 'finite_projection_monotonicity_verified': True,
        'terminal_groups_improved': improved, 'mechanism_triage_passed': False, 'quality_evaluated': False,
        'audit_source_sha256': sha(Path(__file__).resolve()), 'analysis_sha256': sha(out/'analysis.json'),
        'seconds': time.perf_counter()-began}
    (out/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
