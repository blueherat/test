"""A second CPU numerical path; no producer imports or random-number draws."""
import time
start, cpu_start = time.perf_counter(), time.process_time()
from pathlib import Path
import json
import hashlib
import os
import numpy as np

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
out = Path(__file__).resolve().parent
request = json.loads((out / 'request.json').read_text())
summary = json.loads((out / 'summary.json').read_text())
hash_checks = 0
for record in list(request['sources'].values()) + [summary['request'], summary['artifact']]:
    digest = hashlib.sha256()
    with Path(record['path']).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    assert digest.hexdigest() == record['sha256']
    hash_checks += 1

x = np.load(request['sources']['validation_latents']['path'], mmap_mode='r', allow_pickle=False)
result = np.load(out / 'by_image_and_class.npz', allow_pickle=False)
old = np.load(request['sources']['legacy_step']['path'], allow_pickle=False)
replay = np.load(request['sources']['fp64_step']['path'], allow_pickle=False)
metadata = np.load(request['sources']['validation_metadata']['path'], allow_pickle=False)
for key in ('ids', 'labels'):
    assert np.array_equal(result[key], old[key])
    assert np.array_equal(result[key], replay[key])
    assert np.array_equal(result[key], metadata[key])
assert np.array_equal(result['source_rows'], metadata['rows'])
assert np.array_equal(np.sort(metadata['labels']), np.arange(1000))
assert np.array_equal(old['residual_mse'], replay['residual_mse_fp32_parity'])

norms = np.array([np.mean(np.asarray(row, dtype=np.float64) ** 2, dtype=np.float64) for row in x])
delta = replay['residual_energy_fp64'] - norms - 1.
np.testing.assert_allclose(norms, result['clean_norm_squared_per_dimension'], atol=3e-14, rtol=1e-13)
np.testing.assert_allclose(delta, result['endpoint_risk_difference'], atol=3e-14, rtol=1e-13)
np.testing.assert_allclose(delta.mean(), summary['endpoint_risk_difference']['mean'], atol=3e-14, rtol=1e-13)
np.testing.assert_allclose(delta.std(ddof=1) / np.sqrt(1000),
                           summary['endpoint_risk_difference']['descriptive_class_sem'], atol=3e-14, rtol=1e-13)
assert np.sum(delta < 0) == summary['endpoint_risk_difference']['negative_classes']
verification = {'passed': True, 'method': 'Per-image direct FP64 mean of squared latent; producer uses block einsum. All 1000 identities, r2 parity, norms and deltas checked.',
    'max_norm_absolute_difference': float(np.max(np.abs(norms-result['clean_norm_squared_per_dimension']))),
    'max_risk_difference_absolute_difference': float(np.max(np.abs(delta-result['endpoint_risk_difference']))),
    'hash_checks': hash_checks, 'samples': 1000,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'wall_seconds': time.perf_counter()-start, 'cpu_seconds': time.process_time()-cpu_start,
    'gpu_calls': 0, 'model_calls': 0, 'fid_calls': 0, 'noise_draws': 0}
with (out / 'verification.json').open('x') as handle:
    json.dump(verification, handle, indent=2)
    handle.write('\n')
print(json.dumps(verification, indent=2))
