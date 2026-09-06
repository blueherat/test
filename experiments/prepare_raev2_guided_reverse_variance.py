#!/usr/bin/env python3
"""Compile all cached native-guided MSEs into the unique fixed-mean variance."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

SOURCE=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_validation_v1')
OUT=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/guided_reverse_variance')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    ref=json.loads((SOURCE/'shard0/request.json').read_text())
    grid=ref['time_grid']
    rows=[]
    ids0=None
    for i,t in enumerate(grid[:-1]):
        files=list(SOURCE.glob(f'shard*/step{i:03d}.npz'))
        if len(files)!=1: raise ValueError(f'expected one complete cached timestep: {files}')
        p=files[0]
        x=np.load(p)
        ids=x['ids']; mse=x['residual_mse']
        if len(mse)!=1000 or not np.isfinite(mse).all() or np.any(mse<0): raise ValueError('invalid MSE')
        if ids0 is None: ids0=ids
        elif not np.array_equal(ids,ids0): raise ValueError('cohort mismatch')
        rows.append({'step':i,'time':t,'next_time':grid[i+1], 'mse':float(mse.mean()),
            'mse_standard_error':float(mse.std(ddof=1)/np.sqrt(len(mse))),
            'source':str(p),'source_sha256':sha(p),'samples':len(mse)})
    obj={'complete':True,'protocol':'raev2_fixed_guided_mean_optimal_spherical_reverse_variance_v1',
        'eta':0.,'extra_strength':None,'fid_used_for_fit':False,'rows':rows,
        'baseline_checkpoint':ref['baseline_checkpoint'],'config':ref['config'],
        'source_request':str(SOURCE/'shard0/request.json'),'source_request_sha256':sha(SOURCE/'shard0/request.json'),
        'source_precision':ref['precision'],'source_batch_size':ref['batch_size'],'seconds':time.perf_counter()-started}
    (OUT/'calibration.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({'steps':len(rows),'first_mse':rows[0]['mse'],'last_mse':rows[-1]['mse'],'path':str(OUT/'calibration.json')}))


if __name__=='__main__': main()
