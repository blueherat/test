"""Paired held-out risk decomposition; descriptive, never selects a checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_sit_subspace_gate import SiTGate


@torch.inference_mode()
def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    ck = torch.load(a.gates, map_location='cpu', weights_only=False)
    meta = json.loads((a.cache / 'manifest.json').read_text())
    assert meta == ck['cache_manifest']
    lo = meta['fit_count']
    def load(name):
        return torch.from_numpy(np.load(a.cache / (name + '.npy'))[lo:]).to(a.device)
    z, t, y, q = load('states'), load('times'), load('labels').long(), load('quadratics')
    risks, gates = {}, {}
    for kind in ['scalar', 'pca']:
        m = SiTGate().to(a.device).eval()
        m.load_state_dict(ck['models'][kind])
        g = torch.cat([m(z[i:i+256], t[i:i+256], y[i:i+256], kind)
                       for i in range(0, len(t), 256)]).double()
        gates[kind] = g
        risks[kind] = (q[:, :, 0]*g.square()+2*q[:, :, 1]*g+q[:, :, 2])/4096
    delta = risks['pca']-risks['scalar']
    native_delta = risks['pca'].sum(1)-load('native_risk')/4096
    def stats(v):
        return {'mean': v.mean().item(), 'paired_se': (v.std(unbiased=True)/len(v)**.5).item()}
    rows = []
    for i in range(10):
        mask = (t >= i/10) & (t < (i+1)/10)
        d = delta[mask]
        rows.append({'t_low': i/10, 't_high': (i+1)/10, 'count': int(mask.sum()),
                     'pca_minus_scalar': stats(d.sum(1)), 'parallel': stats(d[:, 0]),
                     'complement': stats(d[:, 1]),
                     'scalar_gate_mean': gates['scalar'][mask].mean(0).tolist(),
                     'pca_gate_mean': gates['pca'][mask].mean(0).tolist()})
    result = {'complete': True, 'count': len(t), 'sign': 'negative favors PCA',
              'scope': 'paired training-image holdout, one bridge draw per image; no endpoint-quality inference',
              'pca_minus_scalar': stats(delta.sum(1)), 'parallel': stats(delta[:, 0]),
              'complement': stats(delta[:, 1]), 'pca_minus_native': stats(native_delta), 'time_bins': rows}
    (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='time_bins'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--gates', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:3')
    run(p.parse_args())
