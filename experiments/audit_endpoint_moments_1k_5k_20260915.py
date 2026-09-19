"""Read-only CPU audit of the completed endpoint-moments 600-step evaluation.

Reuse saved ADM activations; never sample, train, or modify experiment outputs.
Run with OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import linalg


WORK = Path(__file__).resolve().parents[1]
EXPS = Path('/home/zhoushunyu/data/eqvae/experiments')
OLD = EXPS / 'guidance_dynamic_50k_20260915/sit_small'
NEW = EXPS / 'adversarial_weak_training_20260915/endpoint_moments_v1'
OUTPUT = WORK / 'docs/data/endpoint_moments_1k_5k_audit_20260915.json'
POINTS = {
    'moments_a1.2': NEW / 'quality/step000600_head/c0048',
    'moments_a1.175': NEW / 'quality/step000600_head/c0047',
    'guided_weak_a1.075': OLD / 'points/guided_weak__c0043',
    'guided_weak_a1.05': OLD / 'points/guided_weak__c0042',
    'real_a1.0': OLD / 'points/real__c0040',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    ref = EXPS.parent / 'imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'
    with np.load(ref) as f:
        mu_ref, cov_ref = f['mu'], f['sigma']
    # Equivalent PSD formulation of the covariance trace. Validate against every
    # recorded full-5K ADM sqrtm result before interpreting the new 4K slice.
    eigenvalues, eigenvectors = linalg.eigh(cov_ref, check_finite=True)
    assert eigenvalues.min() > -1e-8
    root_ref = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T

    def fid(features):
        # Retain ADM's float32 mean and float64, ddof=1 covariance conventions.
        mean = features.mean(axis=0)
        cov = np.cov(features, rowvar=False)
        product = root_ref @ cov @ root_ref
        values = linalg.eigvalsh((product + product.T) * .5)
        assert values.min() > -1e-7
        mean_term = float(np.square(mean - mu_ref).sum())
        cov_term = float(np.trace(cov) + np.trace(cov_ref)
                         - 2 * np.sqrt(np.maximum(values, 0)).sum())
        return dict(fid=mean_term + cov_term, mean=mean_term, covariance=cov_term)

    bank = OLD / 'quality_inputs'
    bank_receipt = read(bank / 'complete.json')
    for filename, digest in bank_receipt['files'].items():
        assert sha(Path(filename)) == digest
    noise = np.load(bank / 'noise.npy', mmap_mode='r')
    labels = np.load(bank / 'labels.npy')
    noise_hashes = {start: array_sha(noise[start:start + 8])
                    for start in range(0, 5000, 8)}
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha(Path(__file__)), reference=str(ref),
                  reference_sha256=sha(ref), bank_files_verified=True,
                  bank_receipt_sha256=sha(bank / 'complete.json'),
                  baseline_parity=read(NEW / 'quality/step000600_head/baseline_parity.json'),
                  class_balance={}, points={})
    for name, subset in [('first_1k', labels[:1000]),
                         ('additional_4k', labels[1000:]), ('all_5k', labels)]:
        classes, counts = np.unique(subset, return_counts=True)
        report['class_balance'][name] = dict(n=len(subset), classes=len(classes),
            min_per_class=int(counts.min()), max_per_class=int(counts.max()))

    for name, point in POINTS.items():
        summaries = {n: read(point / f'n{n}/summary.json') for n in (1000, 5000)}
        s1, s5 = summaries[1000], summaries[5000]
        assert s1['records'] == s5['records'][:125]
        assert len(s1['records']) == 125 and len(s5['records']) == 625
        if 'checkpoint_sha256' in s1:
            assert s1['checkpoint_sha256'] == s5['checkpoint_sha256']
            assert s1['weights'] == s5['weights'] == 'head'
            assert s1['step'] == s5['step'] == 600
        else:
            assert s1['head_provenance'] == s5['head_provenance']
            assert s1['bank_sha256'] == s5['bank_sha256'] == report['bank_receipt_sha256']
        for start in range(0, 5000, 8):
            path = point / 'batches' / f'{start:05d}.npz'
            receipt = read(path.with_suffix('.json'))
            assert receipt['sha256'] == s5['records'][start // 8]['sha256']
            with np.load(path) as batch:
                assert int(batch['start']) == start
                np.testing.assert_array_equal(batch['labels'], labels[start:start + 8])
                if 'checkpoint_sha256' in s1:
                    assert receipt['checkpoint_sha256'] == s1['checkpoint_sha256']
                    assert receipt['noise_sha256'] == noise_hashes[start]
                    assert receipt['coefficient'] == s1['coefficient'] == s5['coefficient']
                else:
                    assert str(batch['noise_sha256']) == noise_hashes[start]
                    assert receipt['head_provenance'] == s1['head_provenance']
        with np.load(point / 'n1000/activations.npz') as f:
            first = f['pool_3']
        with np.load(point / 'n5000/activations.npz') as f:
            full = f['pool_3']
        prefix_error = float(np.max(np.abs(first - full[:1000])))
        np.testing.assert_allclose(first, full[:1000], atol=1e-4, rtol=1e-4)
        row = dict(path=str(point), prefix_batch_receipts_identical=True,
                   all_5000_noise_and_labels_verified=True,
                   prefix_activations_bitwise_equal=bool(np.array_equal(first, full[:1000])),
                   prefix_activations_max_abs_error=prefix_error, recorded={})
        for n in (1000, 5000):
            adm = read(point / f'n{n}/adm.json')
            assert adm['reference'] == str(ref)
            assert adm['sample_count'] == n and adm['batch_size'] == 32
            row['recorded'][n] = {k: adm[k] for k in (
                'fid', 'fid_mean_component', 'fid_covariance_component', 'inception_score')}
        row['recomputed_5k'] = fid(full)
        row['recomputed_5k_abs_error'] = abs(row['recomputed_5k']['fid'] - row['recorded'][5000]['fid'])
        assert row['recomputed_5k_abs_error'] < 1e-5
        row['additional_4k_only'] = fid(full[1000:])
        report['points'][name] = row
        print(name, json.dumps({k: row[k] for k in (
            'prefix_activations_max_abs_error', 'recomputed_5k_abs_error', 'additional_4k_only')}), flush=True)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    selection = read(NEW / 'search_step000600_head_full/selected.json')
    result = read(NEW / 'search_step000600_head_full/result.json')
    report['selection'] = dict(frozen_utc=selection['frozen_before_5k_utc'],
        coefficients=[x['coefficient'] for x in selection['selected']],
        unique_coefficients_scanned=len({t for stage in result['stages'] for t in stage['ticks']}))
    report['complete'] = True
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    print('report', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
