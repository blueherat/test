"""Describe existing PFR endpoint changes in the original official FID coordinates."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/zhoushunyu/data/eqvae/experiments')
OUT=ROOT/'experiments/results/terminal_defect_20260908/pfr_endpoint_moments.json'
REF=Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 start=time.perf_counter();torch.set_num_threads(4)
 assert sha(REF)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
 ref=np.load(REF);mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
 v,u=np.linalg.eigh(cov);assert v.min()>-1e-7
 root=(u*np.sqrt(np.maximum(v,0)))@u.T
 rae=DATA/'raev2_pfr_retiming_v1/fid5k_seed20260905'
 old={x['branch']:x for x in json.loads((rae/'official_metrics.json').read_text())}
 cases=[('sit5k','ordinary115',DATA/'official_sit_pfr_5k_20260908/ordinary115/quality',None),
        ('sit5k','pfr',DATA/'official_sit_pfr_5k_20260908/pfr/quality',None),
        ('rae5k','ordinary',rae/'ig_s1p78',old['ig_s1p78']),
        ('rae5k','pfr',rae/'pfr_h0p03125_r0p05',old['pfr_h0p03125_r0p05']),
        ('rae1k','ordinary',DATA/'raev2_fsg_clock_transfer_20260908/quality/ordinary100',None),
        ('rae1k','pfr',DATA/'raev2_canonical_pfr_query_20260908/time_only/quality',None)]
 rows=[];means={};covterms={}
 for group,arm,d,metric in cases:
  if metric is None:metric=json.loads((d/'fid.json').read_text())[0]
  assert metric['evaluator_commit']=='19dfb4c2705333eb8b97e454fb354d47d1fe135b' and metric['fid_reference']=='imagenet_256_fid_stats'
  pixel=sha(d/'samples.npz');assert pixel==metric['sample_sha256']
  if group=='rae5k':
   f,=(rae/'official_feature_cache').glob(f'{d.name}-{pixel[:16]}-inception.features.pt')
   meta=json.loads((d/'sampling_summary.json').read_text());assert meta['archive_sha256']==pixel and meta['samples']==5000
  else:f,=(d/'features').glob('*.features.pt')
  x=torch.load(f,map_location='cpu',weights_only=True).numpy().astype(float)
  n=1000 if group=='rae1k' else 5000
  assert x.shape==(n,2048) and np.isfinite(x).all()
  m=x.mean(0);x-=m;c=x.T@x/(n-1);g=root@c@root
  eig=np.linalg.eigvalsh((g+g.T)/2);assert eig.min()>-1e-7
  mt=float(np.square(m-mu).sum());ct=float(np.trace(c)+np.trace(cov)-2*np.sqrt(np.maximum(eig,0)).sum())
  assert abs(mt+ct-metric['fid'])<2e-4
  row=dict(group=group,arm=arm,n=n,reported_fid=metric['fid'],mean_term=mt,covariance_term=ct,recomputed_fid=mt+ct,
           sample_path=str(d/'samples.npz'),sample_sha256=pixel,feature_path=str(f),feature_sha256=sha(f))
  rows.append(row);means[group,arm]=m;covterms[group,arm]=ct
  print({k:row[k] for k in ['group','arm','mean_term','covariance_term']},flush=True)
 changes=[]
 for group,control in [('sit5k','ordinary115'),('rae5k','ordinary'),('rae1k','ordinary')]:
  b=means[group,control]-mu;delta=means[group,'pfr']-means[group,control]
  cross=float(b@delta);energy=float(delta@delta)
  change=float((b+delta)@(b+delta)-b@b)
  assert abs(change-(2*cross+energy))<1e-12
  changes.append(dict(group=group,mean_term_change=change,covariance_term_change=covterms[group,'pfr']-covterms[group,control],
                      mean_shift_energy=energy,mean_shift_dot_baseline_error=cross,
                      mean_shift_cosine_to_target=float(-cross/np.sqrt((b@b)*energy))))
 result=dict(complete=True,rows=rows,changes=changes,seconds=time.perf_counter()-start,source_sha256=sha(Path(__file__)),reference_sha256=sha(REF),
             scope='Existing unmodified Inception features; descriptive moment decomposition, no controller fitting or quality claim. RAE5k rho.05 and RAE1k rho1 differ; SiT control115 versus RAE100 steps.')
 with OUT.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(changes),flush=True)
if __name__=='__main__':main()
