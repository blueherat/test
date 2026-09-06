"""Independent rank-N FID reconstruction and fixed-critic diagnostics.

Uses cached Inception features only, never updates samples or guidance. The
rank-N Gram formula is independent of nanogen's D-by-D scipy sqrtm path.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907')
sys.path.insert(0,'/home/zhoushunyu/data/eqvae/external_sources/nanogen-evals/fd_evaluator')
os.environ['HF_HUB_OFFLINE']='1'
from fd_evaluator.stats import load_moments,resolve


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ref=Path(resolve('imagenet_256_fid_stats'))
    mr,sr=load_moments(str(ref))
    records=[]
    for study in ['ancestral_screen1k','calibrated_screen1k','projection_screen1k','weak_screen1k']:
        folder=DATA/study
        for metric in json.loads((folder/'metrics.json').read_text()):
            name=metric['branch']
            path=next((folder/'official_feature_cache').glob(name+'-*-inception.features.pt'))
            x=torch.load(path,map_location='cpu',weights_only=True).double().numpy()
            mean=x.mean(0); centered=x-mean
            eig=np.linalg.eigvalsh(centered@sr@centered.T/(len(x)-1))
            mean_term=float(np.sum((mean-mr)**2))
            trace_sample=float(np.sum(centered**2)/(len(x)-1))
            covariance_term=float(trace_sample+np.trace(sr)-2*np.sqrt(np.maximum(eig,0)).sum())
            fid=mean_term+covariance_term
            if abs(fid-metric['fid'])>1e-3:raise ValueError('FID reconstruction mismatch')
            logits=torch.load(path.with_name(path.name.replace('.features.pt','.logits.pt')),map_location='cpu',weights_only=True).double().numpy()
            logp=logits-logsumexp(logits,axis=1,keepdims=True)
            p=np.exp(logp); marginal=p.mean(0)
            entropy=float(-(marginal*np.log(np.maximum(marginal,1e-300))).sum())
            conditional=float(-(p*logp).sum(1).mean())
            records.append({'study':study,'mode':name,'fid':fid,'official_fid':metric['fid'],
                'mean_term':mean_term,'covariance_term':covariance_term,'sample_covariance_trace':trace_sample,
                'feature_sha256':sha(path),'marginal_entropy':entropy,'conditional_entropy':conditional,
                'whole_sample_is_not_ten_split_is':float(np.exp(entropy-conditional))})
    trace_ref=float(np.trace(sr))
    base=records[0]
    trace_base=base['sample_covariance_trace']
    affinity_base=(trace_base+trace_ref-base['covariance_term'])/(2*math.sqrt(trace_base*trace_ref))
    for r in records:
        tr=r['sample_covariance_trace']
        affinity=(tr+trace_ref-r['covariance_term'])/(2*math.sqrt(tr*trace_ref))
        r['covariance_decomposition_relative_official']={
            'trace_effect_at_baseline_shape':tr-trace_base-2*math.sqrt(trace_ref)*affinity_base*(math.sqrt(tr)-math.sqrt(trace_base)),
            'normalized_shape_effect_at_new_trace':-2*math.sqrt(tr*trace_ref)*(affinity-affinity_base),
            'normalized_root_affinity':affinity}
    result={'complete':True,'reference':str(ref),'reference_sha256':sha(ref),'formula':'rank-N Gram eigenvalues, FP64, Bessel sample covariance',
        'max_fid_error':max(abs(r['fid']-r['official_fid']) for r in records),'rows':records,
        'reference_covariance_trace':trace_ref}
    out=ROOT/'experiments/results/raev2_guidance_20260907/fid_audit.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
