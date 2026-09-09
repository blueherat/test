"""Reconstruct reference FIDs and record full/prefix model costs separately."""
import csv
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_clock_quality_20260908')
    ref=np.load('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz')
    mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
    w,u=np.linalg.eigh(cov);assert w.min()>-1e-7
    rc=(u*np.sqrt(np.maximum(w,0)))@u.T
    rows=[]
    for arm in ['cfg_ordinary','ig_ordinary','ig_pfr','ig_pfr_euler42']:
        d=root/'matched_references'/arm
        assert json.loads((d/'complete.json').read_text())['complete']
        p=d/'quality';m=json.loads((p/'sampling_manifest.json').read_text())
        assert json.loads((p/'reference_manifest.json').read_text())['complete']
        family=arm.split('_')[0]
        original=json.loads((root/'confirmation5k'/family/'closed50/sampling_manifest.json').read_text())
        for key in ['noise_sha256','label_sha256','strong','weak','precision','allow_tf32']:
            assert m[key]==original[key],(arm,key)
        pixels=np.load(p/'samples_n5000.npz')['arr_0']
        assert pixels.shape==(5000,256,256,3) and pixels.dtype==np.uint8
        del pixels
        acts=np.load(p/'activations.npz')['pool_3'].astype(float)
        assert acts.shape==(5000,2048) and np.isfinite(acts).all()
        sm=acts.mean(0);sc=np.cov(acts,rowvar=False)
        prod=rc@sc@rc;values=np.linalg.eigvalsh((prod+prod.T)/2)
        assert values.min()>-1e-7
        value=float(((sm-mu)**2).sum()+np.trace(sc)+np.trace(cov)-2*np.sqrt(np.maximum(values,0)).sum())
        f=json.loads((p/'fid.json').read_text());assert abs(value-f['fid'])<2e-4
        counts=m['model_forward_totals'];prefix=counts.get('pfr_depth4_prefix_forwards',0)*8/5000
        full=(sum(counts.values())-counts.get('pfr_depth4_prefix_forwards',0))*8/5000
        row=dict(arm=arm,samples=5000,fid=f['fid'],independent_fid=value,IS=f['inception_score'],
                 full_forwards_per_sample=full,prefix4_forwards_per_sample=prefix,
                 transformer_blocks_per_sample=12*full+4*prefix,sampling_seconds=m['elapsed_seconds'],
                 noise_sha256=m['noise_sha256'],labels_sha256=m['label_sha256'])
        rows.append(row);print(row,flush=True)
    out=Path('experiments/results/terminal_defect_20260908/fsg_pfr_matched_bank.csv')
    with out.open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
