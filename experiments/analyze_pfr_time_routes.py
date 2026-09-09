"""Independent CPU source/Gram/old-bank audit of finite time-route attribution."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'experiments/results/terminal_defect_20260908'
DATA = Path('/home/zhoushunyu/data/eqvae/experiments/pfr_time_routes_20260908')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    output = dict(complete=True, scope='Captured sources and stored Gram/statistic audit; no independent neural forward or full-vector recomputation.', models={})
    for model in ['sit', 'raev2']:
        p = RESULT / f'pfr_time_routes_{model}.json'
        d = json.loads(p.read_text())
        assert d == json.loads((DATA/model/'summary.json').read_text())
        assert d['complete'] and d['native_corner_parity'] and d['unchanged_endpoint_parity']
        for i, (path, digest) in enumerate(d['sources'].items()):
            assert sha(DATA/model/f'{i}_{Path(path).name}') == digest
        assert (d['full_calls'], d['prefix_calls'], d['explicit_readout_calls']) == (832,64,128)
        old = json.loads((RESULT/f'pfr_head_kinematic_response_{model}.json').read_text())
        for k in ['noise_sha256', 'endpoint_sha256', 'checkpoint_sha256']:
            assert d[k] == old[k]
        n = 3 if model=='raev2' else 2
        assert len(d['rows']) == 128
        assert {(r['sample'], r['step']) for r in d['rows']} == {(r['sample'], r['step']) for r in old['rows']}
        for row in d['rows']:
            g = np.asarray(row['gram'], dtype=float)
            assert g.shape == (n+1,n+1) and np.isfinite(g).all()
            tol = 1e-10*max(float(np.abs(g).max()),1e-12)
            assert np.allclose(g,g.T,rtol=0,atol=tol)
            assert np.linalg.eigvalsh(g).min() > -tol
            assert abs(g[0,0]-g[1:,1:].sum()) < tol
            assert np.allclose(g[0],g[1:].sum(0),rtol=1e-10,atol=tol)
            prior = next(r for r in old['rows'] if r['sample']==row['sample'] and r['step']==row['step'])
            assert abs(g[0,0]-prior['weak_energy']) < tol
        for s in d['summary']:
            g = sum(np.asarray(r['gram']) for r in d['rows'] if r['step']==s['step'])
            names = d['rows'][0]['names'][1:]
            for j,name in enumerate(names,1):
                assert abs(s['energy_ratio'][name]-g[j,j]/g[0,0]) < 1e-12
                assert abs(s['signed_raw_dot_ratio'][name]-g[0,j]/g[0,0]) < 1e-12
            assert abs(sum(s['signed_raw_dot_ratio'].values())-1) < 1e-10
        output['models'][model] = dict(result_sha256=sha(p), rows=128, seconds=d['seconds'], summary=d['summary'])
    output['total_sampling_seconds'] = sum(v['seconds'] for v in output['models'].values())
    with (RESULT/'pfr_time_routes_audit.json').open('x') as f:
        json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
