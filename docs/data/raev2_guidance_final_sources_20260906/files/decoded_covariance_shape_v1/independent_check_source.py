"""Independently recheck saved full-spectrum/statistical identities and hashes.

This check does not independently reconstruct covariance matrices from features.
"""
import time
START, CPU_START = time.perf_counter(), time.process_time()
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):
            h.update(c)
    return h.hexdigest()
def rec(p):
    p=Path(p)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}

summary=json.loads((ROOT/'summary.json').read_text())
manifest=json.loads((ROOT/'artifact_hashes.json').read_text())
checks, residuals = 0, []
for x in [*summary['files'],*manifest.values()]:
    p=Path(x['path'])
    assert p.stat().st_size==x['bytes']
    assert digest(p)==x['sha256']
    checks+=1
with np.load(ROOT/'class_influences.npz',allow_pickle=False) as f:
    for r in summary['results']:
        seed=r['seed']
        with np.load(ROOT/f'full_spectrum_seed{seed}.npz',allow_pickle=False) as p:
            lam=p['eigenvalue_train']
            diag=p['diagonal_heldout_ig_minus_source']
            assert lam.shape==diag.shape==(2048,)
            assert np.all(np.diff(lam)>=0)
            residuals.append(abs(lam.sum()))
            norm=float(np.sqrt(sum(v*v for v in lam)))
            residuals.append(abs(norm-r['training_shape_frobenius_norm']))
            projected=float(sum(a*b for a,b in zip(lam,diag))/norm)
            residuals.append(abs(projected-r['heldout_contrasts']['ig_minus_source']['value']))
            residuals.append(abs(projected/r['heldout_shape_frobenius_norm']-r['shape_train_test_cosine_source']))
            checks+=6
        for name,obj in r['heldout_contrasts'].items():
            a,b=name.split('_minus_')
            val=r['heldout_scalar_all_branches'][a]['value']-r['heldout_scalar_all_branches'][b]['value']
            x=f[f'seed{seed}_{a}']-f[f'seed{seed}_{b}']
            sem=float(np.sqrt(sum((v-x.mean())**2 for v in x)/(len(x)-1)/len(x)))
            residuals.extend([abs(val-obj['value']),abs(sem-obj['descriptive_class_influence_sem']),
                              abs(val-1.96*sem-obj['descriptive_95_interval'][0]),
                              abs(val+1.96*sem-obj['descriptive_95_interval'][1])])
            checks+=4
assert max(residuals)<1e-12
out={'complete':True,'checks':checks,'max_absolute_numeric_residual':max(residuals),
     'scope':'Independent saved-output identities, all artifact/input hashes, full-spectrum primary contrast, and every paired class SEM/interval. Does not independently reconstruct covariance from raw features.',
     'source':rec(__file__),'wall_seconds':time.perf_counter()-START,
     'cpu_seconds':time.process_time()-CPU_START,'gpu_calls':0,'model_calls':0}
(ROOT/'independent_check.json').write_text(json.dumps(out,indent=2)+'\n')
(ROOT/'independent_check_hashes.json').write_text(json.dumps({name:rec(ROOT/name) for name in ('independent_check_source.py','independent_check.json')},indent=2)+'\n')
print(json.dumps(out))
