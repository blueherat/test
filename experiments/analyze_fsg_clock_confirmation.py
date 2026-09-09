"""Audit all fixed 5K arms and independently reconstruct ADM FID from features."""
import csv
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_clock_quality_20260908')
    ref=np.load('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz')
    mu=ref['mu'].astype(np.float64);cov=ref['sigma'].astype(np.float64)
    w,u=np.linalg.eigh(cov)
    assert w.min()>-1e-7
    root_cov=(u*np.sqrt(np.maximum(w,0)))@u.T
    rows=[];hashes=set();sources=set()
    for family in ['cfg','ig']:
        expected_calls=100 if family=='cfg' else 50
        screen=json.loads((root/family/'asynchronous/sampling_manifest.json').read_text())
        for arm in ['closed50','short','asynchronous']:
            out=root/'confirmation5k'/family/arm
            assert json.loads((out/'confirmation_complete.json').read_text())['complete']
            m=json.loads((out/'sampling_manifest.json').read_text())
            cm=json.loads((out/'clock_manifest.json').read_text());assert cm['complete']
            assert m['requested_samples']==5000 and m['global_seed']==202609412
            assert m['noise_sha256']!=screen['noise_sha256']
            hashes.add((m['noise_sha256'],m['label_sha256']))
            sources.add(tuple(sorted(cm['sources'].items())))
            calls=sum(m['model_forward_totals'].values())*m['batch_size']/5000
            assert calls==expected_calls
            pixels=np.load(out/'samples_n5000.npz')['arr_0']
            assert pixels.shape==(5000,256,256,3) and pixels.dtype==np.uint8
            del pixels
            acts=np.load(out/'activations.npz')['pool_3'].astype(np.float64)
            assert acts.shape==(5000,2048) and np.isfinite(acts).all()
            sm=acts.mean(0);sc=np.cov(acts,rowvar=False)
            product=root_cov@sc@root_cov
            values=np.linalg.eigvalsh((product+product.T)/2)
            assert values.min()>-1e-7
            fid=float(((mu-sm)**2).sum()+np.trace(cov)+np.trace(sc)-2*np.sqrt(np.maximum(values,0)).sum())
            d=json.loads((out/'fid.json').read_text())
            assert abs(fid-d['fid'])<2e-4,(family,arm,fid,d['fid'])
            rows.append(dict(family=family,arm=arm,samples=5000,fid=d['fid'],independent_fid=fid,
                             inception_score=d['inception_score'],sampling_seconds=m['elapsed_seconds'],
                             model_forwards_per_sample=calls,noise_sha256=m['noise_sha256'],labels_sha256=m['label_sha256']))
            print(family,arm,fid,flush=True)
    assert len(hashes)==1 and len(sources)==1
    out=Path('experiments/results/terminal_defect_20260908/fsg_clock_quality_confirmation5k.csv')
    with out.open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
