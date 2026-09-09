"""Descriptive endpoint nonadditivity; no fitted correction or extra sampling."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae/experiments')
OUT = ROOT / 'experiments/results/terminal_defect_20260908'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    # The complete source, parity and FID audit must have passed first.
    assert (OUT / 'pfr_condition_component.csv').exists()
    result = {'scope': 'Same-bank endpoint mean shifts. Field additivity does not imply endpoint additivity; no population inference.', 'models': {}}
    for model in ['sit', 'raev2']:
        if model == 'sit':
            paths = {a: DATA / 'official_sit_pfr_20260908' / p / 'quality'
                     for a, p in [('native', 'ordinary100'), ('raw', 'pfr')]}
        else:
            paths = dict(native=DATA / 'raev2_fsg_clock_transfer_20260908/quality/ordinary100',
                         raw=DATA / 'raev2_canonical_pfr_query_20260908/time_only/quality')
        paths.update({a: DATA / 'pfr_condition_component_20260908' / model / a / 'quality'
                      for a in ['interaction', 'unconditional']})
        means, metadata, evidence = {}, {}, {}
        for arm, path in paths.items():
            m = json.loads((path / 'summary.json').read_text())
            metric = json.loads((path / 'fid.json').read_text())[0]
            assert m['complete'] and m['samples'] == 1000
            assert sha(path / 'samples.npz') == m['pixel_sha256'] == metric['sample_sha256']
            f, = (path / 'features').glob('*.features.pt')
            x = torch.load(f, map_location='cpu', weights_only=True).numpy().astype(np.float64)
            assert x.shape == (1000, 2048) and np.isfinite(x).all()
            means[arm], metadata[arm] = x.mean(0), m
            evidence[arm] = dict(pixel_sha256=m['pixel_sha256'], feature_sha256=sha(f))
        for arm in paths:
            for key in ['noise_sha256', 'label_sha256', 'checkpoint_sha256']:
                assert metadata[arm][key] == metadata['native'][key]
        d = {a: means[a] - means['native'] for a in ['raw', 'interaction', 'unconditional']}
        def dot(a, b):
            return float(d[a] @ d[b])
        norms = {a: dot(a, a) for a in d}
        closure = d['raw'] - d['interaction'] - d['unconditional']
        result['models'][model] = dict(
            evidence=evidence,
            shift_norm_squared=norms,
            shift_dot_products={a + '__' + b: dot(a, b) for a, b in
                                [('raw', 'interaction'), ('raw', 'unconditional'), ('interaction', 'unconditional')]},
            endpoint_mean_closure_squared=float(closure @ closure),
            endpoint_mean_closure_over_raw_squared=float(closure @ closure) / norms['raw'],
        )
    with (OUT / 'pfr_component_mean_closure.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
