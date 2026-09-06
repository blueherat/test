#!/usr/bin/env python3
"""Fixed CPU-only full-spectrum covariance-shape transfer audit.

No model, sampling, FID, covariance inverse, selected rank, or guidance fit.
"""
import time
START = time.perf_counter()
CPU_START = time.process_time()
import hashlib
import json
import os
from pathlib import Path
import resource
import numpy as np
from scipy.linalg import eigh

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments')
PRIOR = ROOT / 'raev2_guidance_restart_20260906/decoded_class_moments_v1/request.json'
SEEDS = (20260801, 20260802)
FILES = {'source': 'source', 'reconstruction': 'real',
         'full': 'scale_s1p000000', 'ig': 'scale_s1p780000'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def record(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}

def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def covariance(x):
    xc = x - x.mean(0)
    return xc.T @ xc / (len(x) - 1)

def scalar_and_influence(x, labels, k):
    # Centering-mean derivatives cancel in covariance's first-order influence.
    xc = x - x.mean(0)
    num = np.einsum('ni,ni->n', xc @ k, xc)
    den = np.einsum('ni,ni->n', xc, xc)
    value = num.mean() / den.mean()
    classes = np.unique(labels)
    clusters = np.stack([np.array([num[labels == c].mean(), den[labels == c].mean()])
                         for c in classes])
    influence = (clusters[:, 0] - value * clusters[:, 1]) / den.mean()
    np.testing.assert_allclose(influence.mean(), 0, atol=2e-15)
    return float(value), influence, classes

def describe(value, influence):
    sem = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    return {'value': float(value), 'descriptive_class_influence_sem': sem,
            'descriptive_95_interval': [float(value - 1.96*sem), float(value + 1.96*sem)],
            'classes': len(influence)}

def main():
    if (OUT / 'request.json').exists():
        raise FileExistsError('refusing to overwrite an existing run')
    prior = json.loads(PRIOR.read_text())
    request = {
        'protocol': 'raev2_decoded_covariance_shape_transfer_v1',
        'question': 'Does the complete covariance-shape discrepancy of native IG versus original source transfer from old train800 classes to heldout200 classes in both existing seeds?',
        'seeds': SEEDS, 'branches': FILES, 'dimension': 2048, 'images_per_class': 5,
        'fit': 'For each seed use only train800 to set D = cov(IG)/tr(cov(IG)) - cov(source)/tr(cov(source)), K = D/||D||_F. Keep complete 2048-dimensional spectrum, no rank or direction selection.',
        'primary': 'In heldout200 compute T_A = tr(K cov(A))/tr(cov(A)); primary paired class contrast T_IG - T_source.',
        'fixed_additional_contrasts': ['ig - reconstruction', 'full - source', 'ig - full', 'reconstruction - source'],
        'full_spectrum': 'Save every eigenvalue of training D and every heldout IG-source shape diagonal in its ordered eigenbasis; no threshold, inverse, whitening, clipping, or spectral truncation.',
        'secondary': 'Frobenius cosine between training D and heldout IG-source shape discrepancy, and reconstruction-reference discrepancy; descriptive, not used to select anything.',
        'uncertainty': 'At fixed fitted K, first-order cluster influence for covariance-trace ratio: IF_c = (mean_c[(x-mu)^T K (x-mu)] - T_A mean_c[||x-mu||^2])/mean[||x-mu||^2]. Difference aligned class influences, SEM=sd(IF_c)/sqrt(200). Mean +/-1.96 SEM is descriptive, not exact fixed-label repeated-noise CI.',
        'identity': 'global_id=rank+4*local_row; sample_protocol supplies labels/test_mask. Check all original feature, protocol and manifest SHA against completed decoded-class-moment audit.',
        'arithmetic': 'All stored float32 features promoted to float64; centered ddof=1 covariance; trace normalization removes pooled covariance scale.',
        'retrospective_scope': 'Both banks and old class split already used for research. K is newly estimated on training classes only; heldout is retrospective transfer evidence, not unseen method or quality confirmation.',
        'limits': ['Positive shape discrepancy is not a quality-improving direction or guidance formula.',
                   'Full spectrum transfer does not establish class-conditional density, semantic defects, rollout cause, or FID gain.',
                   'Historical float-before-clamp decoding differs from current native BF16; no baseline protocol change.',
                   'No generated/source image pairing beyond shared class. Full/IG and source/reconstruction pairings retained at class level.'],
        'new_fid': False, 'gpu_calls': 0, 'model_calls': 0,
        'thread_settings': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','CUDA_VISIBLE_DEVICES')},
        'source': record(__file__), 'prior_identity_request': record(PRIOR),
        'expected_inputs': prior['banks']}
    # This immutable request is written before reading any numerical features.
    save(OUT / 'request.json', request)
    results, files, influence_output = [], [], {}
    for seed in SEEDS:
        phase = time.perf_counter()
        bank = ROOT / 'raev2_ig_scale_response' / f'n5000_seed{seed}_scales7_v1'
        old = next(x for x in prior['banks'] if x['seed'] == seed)
        for name in ('manifest', 'sample_protocol'):
            path = bank / Path(old[name]['path']).name
            rec = record(path)
            assert rec['sha256'] == old[name]['sha256']
            files.append(rec)
        manifest = json.loads((bank / 'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['samples'] == 5000
        assert manifest['seed'] == seed and manifest['world_size'] == 4
        with np.load(bank / 'sample_protocol.npz', allow_pickle=False) as p:
            ids, labels, test = (p[key].copy() for key in ('sample_ids','labels','test_mask'))
        assert np.array_equal(ids, np.arange(5000))
        assert np.array_equal(labels, ids % 1000)
        assert np.array_equal(test, np.isin(labels, np.random.default_rng(seed+17).permutation(1000)[:200]))
        assert test.sum() == 1000
        values = {}
        for branch, prefix in FILES.items():
            x = np.empty((5000, 2048), np.float64)
            for rank in range(4):
                path = bank / 'inception' / f'{prefix}_rank{rank:02d}.npy'
                rec = record(path)
                expected = next(r for r in old['features'] if Path(r['path']).name == path.name)
                assert rec['sha256'] == expected['sha256']
                files.append(rec)
                raw = np.load(path, allow_pickle=False)
                assert raw.shape == (1250, 2048) and raw.dtype == np.float32
                assert np.isfinite(raw).all()
                x[rank::4] = raw
            values[branch] = x
        sr = covariance(values['source'][~test])
        sg = covariance(values['ig'][~test])
        d = sg / np.trace(sg) - sr / np.trace(sr)
        norm = np.linalg.norm(d)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('training shape discrepancy is zero/nonfinite; no stabilizer allowed')
        k = d / norm
        np.testing.assert_allclose(k, k.T, atol=1e-15)
        np.testing.assert_allclose(np.trace(d), 0, atol=1e-15)
        eigenvalues, eigvec = eigh(d, check_finite=True, driver='evd')
        np.testing.assert_allclose(np.dot(eigenvalues,eigenvalues), norm**2, rtol=2e-12)
        test_cov = {branch: covariance(x[test]) for branch, x in values.items()}
        d_test = test_cov['ig']/np.trace(test_cov['ig']) - test_cov['source']/np.trace(test_cov['source'])
        d_recon = test_cov['ig']/np.trace(test_cov['ig']) - test_cov['reconstruction']/np.trace(test_cov['reconstruction'])
        diagonal = np.einsum('ij,ij->j', eigvec, d_test @ eigvec)
        direct = float(np.sum(k * d_test))
        np.testing.assert_allclose(direct, eigenvalues @ diagonal / norm, rtol=2e-12, atol=1e-15)
        scalar, influences = {}, {}
        for branch, x in values.items():
            scalar[branch], influences[branch], classes = scalar_and_influence(x[test], labels[test], k)
            np.testing.assert_allclose(scalar[branch], np.sum(k*test_cov[branch])/np.trace(test_cov[branch]), rtol=2e-12, atol=1e-15)
            influence_output[f'seed{seed}_{branch}'] = influences[branch]
        contrasts = {}
        for a,b in (('ig','source'),('ig','reconstruction'),('full','source'),('ig','full'),('reconstruction','source')):
            contrasts[f'{a}_minus_{b}'] = describe(scalar[a]-scalar[b], influences[a]-influences[b])
        np.testing.assert_allclose(contrasts['ig_minus_source']['value'], direct, atol=1e-15)
        spectral_path = OUT / f'full_spectrum_seed{seed}.npz'
        np.savez(spectral_path, eigenvalue_train=eigenvalues,
                 diagonal_heldout_ig_minus_source=diagonal, heldout_classes=classes)
        files.append(record(spectral_path))
        result = {'seed': seed, 'training_classes': 800, 'heldout_classes': 200,
                  'training_shape_frobenius_norm': float(norm),
                  'heldout_shape_frobenius_norm': float(np.linalg.norm(d_test)),
                  'shape_train_test_cosine_source': float(np.sum(d*d_test)/(norm*np.linalg.norm(d_test))),
                  'shape_train_test_cosine_reconstruction': float(np.sum(d*d_recon)/(norm*np.linalg.norm(d_recon))),
                  'heldout_scalar_all_branches': {a: describe(scalar[a], influences[a]) for a in FILES},
                  'heldout_contrasts': contrasts,
                  'training_eigenvalue_quantiles': {str(q):float(np.quantile(eigenvalues,q)) for q in (0,.01,.1,.25,.5,.75,.9,.99,1)},
                  'full_spectrum_artifact': record(spectral_path),
                  'phase_wall_seconds': time.perf_counter()-phase}
        results.append(result)
        print(json.dumps({'seed':seed, 'primary':contrasts['ig_minus_source'],
                          'shape_cosine':result['shape_train_test_cosine_source'],
                          'phase_wall_seconds':result['phase_wall_seconds']}), flush=True)
    np.savez(OUT/'class_influences.npz', **influence_output)
    files.append(record(OUT/'class_influences.npz'))
    summary = {'protocol':request['protocol'], 'complete':True, 'results':results,
               'request':record(OUT/'request.json'), 'files':files,
               'source':record(__file__), 'new_fid':False, 'gpu_calls':0, 'model_calls':0,
               'cost':{'wall_seconds':time.perf_counter()-START,
                       'cpu_seconds':time.process_time()-CPU_START,
                       'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'boundary':'from time import; includes input hashes, feature reads, full spectrum, all covariance and influence calculations and output hashes; excludes interpreter startup before time import and final summary serialization/exit'}}
    save(OUT/'summary.json', summary)
    save(OUT/'artifact_hashes.json', {p.name:record(p) for p in OUT.iterdir() if p.is_file() and p.name!='artifact_hashes.json'})

if __name__ == '__main__':
    main()
