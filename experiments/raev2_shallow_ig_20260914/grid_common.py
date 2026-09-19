"""Shared fixed0.2/0.05 grid and lossless image storage for large5K scans."""
from pathlib import Path
import numpy as np
from experiments.guidance_pasted_20260912 import common as c
from experiments.weak_reference_20260914.cross_run import save_npz as raw_save
COARSE=tuple(i/100 for i in range(100,201,20))


def best_intervals(rows):
    by={round(row['w']*100):row for row in rows};intervals=[]
    for lo in range(100,200,20):
        hi=lo+20
        if lo not in by or hi not in by:continue
        a,b=by[lo],by[hi]
        if not np.isfinite([a['fid'],b['fid']]).all():continue
        intervals.append(dict(left=lo/100,right=hi/100,mean_fid=(a['fid']+b['fid'])/2,
            endpoint_best=min(a['fid'],b['fid']),endpoint_rows=[a,b]))
    selected=sorted(intervals,key=lambda x:(x['mean_fid'],x['endpoint_best'],x['left']))[:3]
    if not selected:raise RuntimeError('No interval with two valid5K endpoints')
    fine=sorted({i/100 for row in selected for i in range(round(row['left']*100)+5,round(row['right']*100),5)})
    return selected,fine


def writer(tuning):
    def save(path,**values):
        if 'latents' in values:
            z=values['latents'];assert np.isfinite(z).all()
            values.update(latent_sha256=c.array_sha(z),latent_shape=np.asarray(z.shape),latent_dtype=str(z.dtype),latent_finite=True)
            if tuning and int(values['start'])!=0:values.pop('latents')
        raw_save(path,**values)
    return save


def check_latents(data,shape):
    if 'latents' in data:
        z=data['latents'];assert z.shape==tuple(shape) and np.isfinite(z).all()
        if 'latent_sha256' in data:assert c.array_sha(z)==str(data['latent_sha256'])
    else:
        assert bool(data['latent_finite']) and tuple(data['latent_shape'])==tuple(shape)
        assert str(data['latent_dtype'])=='float32' and len(str(data['latent_sha256']))==64
