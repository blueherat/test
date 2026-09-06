"""Report every fixed balanced 1K block of the completed independent 5K.

This diagnoses sample-count/seed sensitivity; no block is selected as a new
quality result or used to alter guidance. All 15 block scores are preserved.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,'/home/zhoushunyu/data/eqvae/external_sources/nanogen-evals/fd_evaluator')
from fd_evaluator.stats import load_moments,resolve

ROOT=Path(__file__).resolve().parents[1]
STUDY=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/weak_confirm5k')


def fid(x,mr,sr):
    mean=x.mean(0);centered=x-mean
    vals=np.linalg.eigvalsh(centered@sr@centered.T/(len(x)-1))
    return float(np.square(mean-mr).sum()+np.square(centered).sum()/(len(x)-1)+np.trace(sr)-2*np.sqrt(np.maximum(vals,0)).sum())


def main():
    execution=json.loads((STUDY/'execution.json').read_text())
    if not execution['complete']:raise ValueError('only analyze completed 5K')
    source=Path(resolve('imagenet_256_fid_stats'))
    mr,sr=load_moments(str(source))
    results=[];sources={}
    for mode in ['official','piecewise','stochastic_weak']:
        path=next((STUDY/'official_feature_cache').glob(mode+'-*-inception.features.pt'))
        x=torch.load(path,map_location='cpu',weights_only=True).double().numpy()
        if x.shape!=(5000,2048):raise ValueError('requires full sorted 5K features')
        sources[mode]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for block in range(5):
            score=fid(x[1000*block:1000*(block+1)],mr,sr)
            results.append({'mode':mode,'block':block,'first_id':1000*block,'last_id':1000*(block+1)-1,'samples':1000,'fid':score})
            print(json.dumps(results[-1]),flush=True)
    baseline={r['block']:r['fid'] for r in results if r['mode']=='official'}
    for row in results:row['relative_improvement_percent']=100*(1-row['fid']/baseline[row['block']])
    out=ROOT/'experiments/results/raev2_guidance_20260907/weak_5k_all_balanced_blocks.json'
    out.write_text(json.dumps({'complete':True,'purpose':'diagnostic only; all five fixed blocks, no selection or new success claim',
        'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sources':sources,
        'official_full5k_metrics':json.loads((STUDY/'metrics.json').read_text()),'rows':results},indent=2)+'\n')


if __name__=='__main__':main()
