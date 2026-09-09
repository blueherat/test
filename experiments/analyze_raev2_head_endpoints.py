"""Recompute source-distribution moments from saved raw endpoint arrays."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_head_endpoint_distribution_20260908')
BANK = ROOT.parent/'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1'
REFERENCE = ROOT.parent/'raev2_pfr_working_point_20260908/full/quality'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def moments(path, projection):
    data = np.load(path, mmap_mode='r')
    assert data.shape == (1000, 1024, 16, 16)
    pooled = np.empty((1000, 1024), dtype=np.float64)
    norm2, spatial = np.empty(1000), np.empty(1000)
    total = np.zeros((1024, 16, 16), dtype=np.float64)
    for i in range(0, 1000, 8):
        block = np.array(data[i:i+8], dtype=np.float64)
        assert np.isfinite(block).all()
        pool = block.mean((2, 3))
        pooled[i:i+8] = pool
        norm2[i:i+8] = np.square(block).sum((1, 2, 3))
        spatial[i:i+8] = np.square(block-pool[:, :, None, None]).mean((1, 2, 3))
        total += block.sum(0)
    mean = total/1000
    centered = pooled-pooled.mean(0)
    cov = centered.T@centered/999
    projected = pooled@projection
    pcov = np.cov(projected, rowvar=False)
    stats = dict(latent_mean_squared_norm=float(np.square(mean).sum()),
                 latent_second_moment=float(norm2.mean()),
                 latent_covariance_trace=float((norm2.sum()-1000*np.square(mean).sum())/999),
                 pooled_mean_squared_norm=float(np.square(pooled.mean(0)).sum()),
                 pooled_covariance_trace=float(np.trace(cov)),
                 spatial_centered_energy_per_coordinate=float(spatial.mean()),
                 projected_covariance_trace=float(np.trace(pcov)))
    return stats, dict(pooled=pooled, covariance=cov, mean=mean, norm_squared=norm2,
                       spatial_energy=spatial, projected_covariance=pcov)


def main():
    started = time.perf_counter()
    summaries = {}
    for arm in ['full', 'base']:
        dest = ROOT/arm
        s = json.loads((dest/'summary.json').read_text())
        assert s['complete'] and s['samples'] == 1000
        assert sha(dest/'request.json') == s['request_sha256']
        for name, value in s['files'].items():
            assert sha(dest/name) == value
        request = json.loads((dest/'request.json').read_text())
        for name, value in request['sources'].items():
            assert sha(name) == value
        assert request['checkpoint_sha256'] == '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
        summaries[arm] = s
    reference = json.loads((REFERENCE/'summary.json').read_text())
    for key in ['noise_sha256', 'label_sha256']:
        assert summaries['full'][key] == summaries['base'][key] == reference[key]
    np.testing.assert_array_equal(np.load(ROOT/'full/samples.npz')['arr_0'], np.load(REFERENCE/'samples.npz')['arr_0'])
    bank_summary = json.loads((BANK/'summary.json').read_text())
    assert bank_summary['complete']
    for kind, filename in [('latents', 'latents.npy'), ('metadata', 'metadata.npz')]:
        assert sha(BANK/'validation'/filename) == bank_summary['banks']['validation'][kind]['sha256']
    labels = np.load(BANK/'validation/metadata.npz')['labels']
    classes, counts = np.unique(labels, return_counts=True)
    np.testing.assert_array_equal(classes, np.arange(1000))
    assert np.all(counts == 1)
    projection, _ = np.linalg.qr(np.random.default_rng(202609491).standard_normal((1024, 64)))
    arrays, stats = {}, {}
    for arm, path in [('full', ROOT/'full/latents.npy'), ('base', ROOT/'base/latents.npy'),
                      ('real', BANK/'validation/latents.npy')]:
        stats[arm], arrays[arm] = moments(path, projection)
    contrasts = {}
    for name, a, b in [('base_minus_full', 'base', 'full'), ('full_minus_real', 'full', 'real'),
                       ('base_minus_real', 'base', 'real')]:
        difference = arrays[a]['covariance']-arrays[b]['covariance']
        spectrum = np.linalg.eigvalsh(arrays[a]['projected_covariance']-arrays[b]['projected_covariance'])
        contrasts[name] = dict(pooled_variance_difference=np.diag(difference).tolist(),
                               projected_covariance_difference_spectrum=spectrum.tolist(),
                               pooled_mean_difference_squared_norm=float(np.square(arrays[a]['pooled'].mean(0)-arrays[b]['pooled'].mean(0)).sum()),
                               latent_covariance_trace_difference=stats[a]['latent_covariance_trace']-stats[b]['latent_covariance_trace'],
                               pooled_covariance_trace_difference=float(np.trace(difference)))
    # The same generated rows share initial noise; this paired endpoint difference is not a transport density ratio.
    paired = arrays['base']['pooled']-arrays['full']['pooled']
    result = dict(complete=True, statistics=stats, contrasts=contrasts,
                  paired_pooled_endpoint_difference_mse=float(np.square(paired).mean()),
                  classes_per_arm=1000, images_per_class=1, full_reference_pixel_parity=True,
                  finite_cohort_only=True, quality_improvement_claim=False,
                  sources={str(Path(__file__)):sha(Path(__file__)), str(BANK/'summary.json'):sha(BANK/'summary.json')},
                  endpoint_summary_sha256={arm:sha(ROOT/arm/'summary.json') for arm in summaries},
                  seconds=time.perf_counter()-started)
    np.savez(ROOT/'moments.npz', projection=projection,
             **{arm+'_'+key:value for arm, values in arrays.items() for key, value in values.items()})
    result['moments_sha256'] = sha(ROOT/'moments.npz')
    (ROOT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=True, statistics=stats,
                         trace_differences={k:v['latent_covariance_trace_difference'] for k,v in contrasts.items()},
                         seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
