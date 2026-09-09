"""Audit selected completed transfer arms with an independent feature-Gram FID."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms',default='ordinary100,ordinary110,short,asynchronous,time_only')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_fsg_clock_transfer_20260908/quality')
    ref=np.load('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
    mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
    hashes=set();sources=set();checkpoints=set();rows=[]
    for arm in a.arms.split(','):
        d=root/arm
        assert json.loads((d/'evaluation_complete.json').read_text())['complete']
        m=json.loads((d/'summary.json').read_text());assert m['complete'] and m['samples']==1000
        assert m['full_calls']==250*(100 if arm=='ordinary100' else 110)
        assert m['calibration_calls']==(0 if arm.startswith('ordinary') else 2500)
        hashes.add((m['noise_sha256'],m['label_sha256']))
        checkpoints.add(m['checkpoint_sha256']);sources.add(tuple(sorted(m['sources'].items())))
        pixels=np.load(d/'samples.npz')['arr_0'];assert pixels.shape==(1000,256,256,3) and pixels.dtype==np.uint8
        del pixels
        pixel_hash=hashlib.sha256((d/'samples.npz').read_bytes()).hexdigest();assert pixel_hash==m['pixel_sha256']
        f=json.loads((d/'fid.json').read_text())[0];assert f['sample_sha256']==pixel_hash
        paths=list((d/'features').glob('*.features.pt'));assert len(paths)==1
        x=torch.load(paths[0],map_location='cpu',weights_only=True).numpy().astype(float)
        assert x.shape==(1000,2048) and np.isfinite(x).all()
        sm=x.mean(0);x=x-sm
        gram=x@cov@x.T/999
        values=np.linalg.eigvalsh((gram+gram.T)/2);assert values.min()>-1e-7
        independent=float(((sm-mu)**2).sum()+(x*x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(values,0)).sum())
        assert abs(independent-f['fid'])<2e-4,(arm,independent,f['fid'])
        row=dict(arm=arm,samples=1000,fid=f['fid'],independent_fid=independent,
                 IS=f['inception_score'],sampling_seconds=m['seconds'],full_forwards_per_sample=m['full_calls']/250,
                 noise_sha256=m['noise_sha256'],labels_sha256=m['label_sha256'],pixel_sha256=pixel_hash)
        rows.append(row);print(row,flush=True)
    assert len(hashes)==len(sources)==len(checkpoints)==1
    with a.output.open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
