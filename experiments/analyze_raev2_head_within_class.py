"""Paired two-noise estimates of average class-conditional endpoint spread."""
import json
from pathlib import Path
import time
import numpy as np
from experiments.analyze_raev2_head_endpoints import ROOT, sha

SECOND = ROOT.parent/'ig_head_endpoint_distribution_replica2_20260908'


def main():
    started = time.perf_counter()
    source = Path('experiments/sample_raev2_head_endpoints.py').read_text()
    expected = source.replace('ig_head_endpoint_distribution_20260908\')', 'ig_head_endpoint_distribution_replica2_20260908\')')
    expected = expected.replace('seed=202609413', 'seed=202609492')
    expected = expected.replace('docs/IG_HEAD_ENDPOINT_DISTRIBUTION_20260908_ZH.md', 'docs/IG_HEAD_ENDPOINT_REPLICA_20260908_ZH.md')
    expected = expected.replace("    assert noise_hash.hexdigest() == reference['noise_sha256']\n", '')
    expected = expected.replace("    if args.arm == 'full':\n        np.testing.assert_array_equal(pixels, np.load(REFERENCE/'samples.npz')['arr_0'])\n        parity = True\n", '')
    assert expected == Path('experiments/sample_raev2_head_endpoints_replica2.py').read_text()
    summaries, arrays = {}, {}
    for replica, directory in enumerate([ROOT, SECOND]):
        summaries[replica], arrays[replica] = {}, {}
        for arm in ['full', 'base']:
            dest = directory/arm
            summary = json.loads((dest/'summary.json').read_text())
            assert summary['complete'] and summary['samples'] == 1000
            assert sha(dest/'request.json') == summary['request_sha256']
            request = json.loads((dest/'request.json').read_text())
            for filename, value in request['sources'].items():
                assert sha(filename) == value
            for filename, value in summary['files'].items():
                assert sha(dest/filename) == value
            assert request['seed'] == [202609413, 202609492][replica]
            summaries[replica][arm] = summary
            arrays[replica][arm] = np.load(dest/'latents.npy', mmap_mode='r')
        assert summaries[replica]['full']['noise_sha256'] == summaries[replica]['base']['noise_sha256']
        assert summaries[replica]['full']['label_sha256'] == summaries[replica]['base']['label_sha256']
    assert summaries[0]['full']['noise_sha256'] != summaries[1]['full']['noise_sha256']
    assert summaries[0]['full']['label_sha256'] == summaries[1]['full']['label_sha256']
    assert summaries[0]['full']['full_reference_pixel_parity'] is True
    projection, _ = np.linalg.qr(np.random.default_rng(202609491).standard_normal((1024, 64)))
    raw, pooled, projected, results = {}, {}, {}, {}
    for arm in ['full', 'base']:
        raw[arm], pooled[arm] = np.empty(1000), np.empty((1000, 1024))
        for i in range(0, 1000, 8):
            delta = np.array(arrays[0][arm][i:i+8], dtype=np.float64)-np.array(arrays[1][arm][i:i+8], dtype=np.float64)
            assert np.isfinite(delta).all()
            raw[arm][i:i+8] = np.square(delta).sum((1, 2, 3))/2
            pooled[arm][i:i+8] = delta.mean((2, 3))
        projected[arm] = pooled[arm]@projection
        results[arm] = dict(latent_within_class_covariance_trace=float(raw[arm].mean()),
                            pooled_within_class_covariance_trace=float(np.square(pooled[arm]).sum()/2000),
                            projected_within_class_covariance_trace=float(np.square(projected[arm]).sum()/2000))
    differences = dict(latent=raw['base']-raw['full'],
                       pooled=(np.square(pooled['base']).sum(1)-np.square(pooled['full']).sum(1))/2,
                       projected=(np.square(projected['base']).sum(1)-np.square(projected['full']).sum(1))/2)
    # Independent replicas remove the variance bias in squared conditional mean shifts.
    mean_shift = {key:np.empty(1000) for key in ['latent', 'pooled', 'projected']}
    for i in range(0, 1000, 8):
        d0 = np.array(arrays[0]['base'][i:i+8], dtype=np.float64)-np.array(arrays[0]['full'][i:i+8], dtype=np.float64)
        d1 = np.array(arrays[1]['base'][i:i+8], dtype=np.float64)-np.array(arrays[1]['full'][i:i+8], dtype=np.float64)
        mean_shift['latent'][i:i+8] = (d0*d1).sum((1, 2, 3))
        p0, p1 = d0.mean((2, 3)), d1.mean((2, 3))
        mean_shift['pooled'][i:i+8] = (p0*p1).sum(1)
        mean_shift['projected'][i:i+8] = ((p0@projection)*(p1@projection)).sum(1)
    measured = dict(differences, **{key+'_conditional_mean_shift_squared':value for key,value in mean_shift.items()})
    rng = np.random.default_rng(202609493)
    boot = {k:np.empty(2000) for k in measured}
    for j in range(2000):
        idx = rng.integers(0, 1000, 1000)
        for key, values in measured.items():
            boot[key][j] = values[idx].mean()
    covariance_difference = (projected['base'].T@projected['base']-projected['full'].T@projected['full'])/2000
    result = dict(complete=True, statistics=results,
                  base_minus_full={key:dict(mean=float(value.mean()), positive_class_pairs=int((value>0).sum()),
                                           class_bootstrap_percentiles_2_5_97_5=np.quantile(boot[key],[.025,.975]).tolist())
                                   for key,value in differences.items()},
                  conditional_mean_shift_squared={key:dict(estimate=float(value.mean()),
                      class_bootstrap_percentiles_2_5_97_5=np.quantile(boot[key+'_conditional_mean_shift_squared'],[.025,.975]).tolist())
                      for key,value in mean_shift.items()},
                  projected_within_covariance_difference_spectrum=np.linalg.eigvalsh(covariance_difference).tolist(),
                  verified_replica_source_difference=True, paired_inputs=True,
                  limitation='Two noises per fixed class; bootstrap describes class resampling, not independent training or all-class population certainty.',
                  sources={str(Path(__file__)):sha(Path(__file__)),
                           **{str(directory/arm/'summary.json'):sha(directory/arm/'summary.json')
                              for directory in [ROOT, SECOND] for arm in ['full','base']}},
                  seconds=time.perf_counter()-started)
    np.savez(ROOT/'within_class_arrays.npz', projection=projection,
             **{arm+'_latent_trace':v for arm,v in raw.items()},
             **{arm+'_pooled_difference':v for arm,v in pooled.items()},
             **{key+'_conditional_mean_shift_cross':v for key,v in mean_shift.items()},
             **{key+'_bootstrap':v for key,v in boot.items()})
    result['arrays_sha256'] = sha(ROOT/'within_class_arrays.npz')
    (ROOT/'within_class.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
