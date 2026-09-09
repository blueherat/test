"""Decompose the completed paired CFG controls; no fitting or sample selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def main():
    base = Path('/home/zhoushunyu/data/eqvae/experiments')
    refpath = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
    ref = np.load(refpath)
    mu, cov = ref['mu'].astype(float), ref['sigma'].astype(float)
    rows = []
    baseline = None
    inputs = set()
    for name, folder in [
        ('ordinary110', 'raev2_cfg_fsg_20260908/ordinary110'),
        ('asynchronous', 'raev2_cfg_fsg_20260908/asynchronous'),
        ('consistent', 'raev2_cfg_fsg_consistent_20260908/asynchronous'),
    ]:
        root = base / folder / 'quality'
        m = json.loads((root / 'summary.json').read_text())
        inputs.add((m['noise_sha256'], m['label_sha256'], m['checkpoint_sha256']))
        feature, = (root / 'features').glob('*.features.pt')
        x = torch.load(feature, map_location='cpu', weights_only=True).numpy().astype(float)
        assert x.shape == (1000, 2048) and np.isfinite(x).all()
        if baseline is None:
            baseline = x.copy()
        mean = x.mean(0)
        centered = x - mean
        gram = centered @ cov @ centered.T / 999
        eig = np.linalg.eigvalsh((gram + gram.T) / 2)
        assert eig.min() > -1e-7
        mean_term = float(np.square(mean - mu).sum())
        trace = float(np.square(centered).sum() / 999)
        covariance_term = float(trace + np.trace(cov) - 2*np.sqrt(np.maximum(eig, 0)).sum())
        fid = json.loads((root / 'fid.json').read_text())[0]['fid']
        assert abs(mean_term + covariance_term - fid) < 2e-4
        shift = x - baseline
        rows.append(dict(arm=name, mean_term=mean_term, covariance_term=covariance_term,
                         fid=mean_term+covariance_term, feature_covariance_trace=trace,
                         paired_feature_displacement_squared=float(np.square(shift).sum(1).mean()),
                         squared_mean_displacement=float(np.square(shift.mean(0)).sum()),
                         feature_sha256=hashlib.sha256(feature.read_bytes()).hexdigest()))
    assert len(inputs) == 1
    output = dict(rows=rows, reference_covariance_trace=float(np.trace(cov)),
                  reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation='Pooled finite-sample feature moments; not class accuracy or a causal diversity measure.')
    path = Path('experiments/results/terminal_defect_20260908/raev2_cfg_fsg_moments.json')
    with path.open('x') as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
